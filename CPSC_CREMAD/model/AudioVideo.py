from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from .Resnet import resnet18, resnet34, resnet50
import torch.distributions as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

class HighOrderGenerator(nn.Module):
    def __init__(self, input_dim=512, num_variants=10, hidden_dim=1024):
        super(HighOrderGenerator, self).__init__()
        self.num_variants = num_variants
        self.input_dim = input_dim
        
        self.audio_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variants * input_dim)  
        )
        
        self.video_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variants * input_dim) 
        )
        
    def forward(self, audio_features, video_features):
        batch_size = audio_features.shape[0]
        
        audio_flat = self.audio_mlp(audio_features)  
        audio_variants = audio_flat.view(batch_size, self.num_variants, self.input_dim)  
        
        video_flat = self.video_mlp(video_features)  # [batch_size, 10 * 512]
        video_variants = video_flat.view(batch_size, self.num_variants, self.input_dim)  # [batch_size, 10, 512]
        
        return audio_variants, video_variants


class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')

    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class JSInformationBottleneck(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(JSInformationBottleneck, self).__init__()
        self.alpha = alpha  # consistency
        self.beta = beta    # diversity
        
    def js_divergence(self, p, q):

        p_dist = F.softmax(p, dim=-1)
        q_dist = F.softmax(q, dim=-1)
        m = 0.5 * (p_dist + q_dist)

        kl_p_m = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean')
        kl_q_m = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean')
        
        return 0.5 * (kl_p_m + kl_q_m)
    
    def forward(self, original_features, high_order_variants, task_loss=None):
        """
        Information bottleneck
        """
        batch_size, num_variants, feat_dim = high_order_variants.shape
        
        # 1. consistency
        consistency_loss = 0
        for i in range(num_variants):
            variant = high_order_variants[:, i, :] 
            js_consistency = self.js_divergence(variant, original_features)
            consistency_loss += js_consistency
        
        consistency_loss /= num_variants
        
        # 2. diversity
        diversity_loss = 0
        count = 0
        for i in range(num_variants):
            for j in range(i + 1, num_variants):
                variant_i = high_order_variants[:, i, :]
                variant_j = high_order_variants[:, j, :]
                js_diversity = self.js_divergence(variant_i, variant_j)
                diversity_loss += js_diversity
                count += 1
        
        diversity_loss = diversity_loss / count if count > 0 else 0
        
        # 3. loss
        ib_loss = self.alpha * consistency_loss - self.beta * diversity_loss
        return ib_loss, consistency_loss, diversity_loss

class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512

        self.feature_generator = HighOrderGenerator(
            input_dim=self.hidden_dim, 
            num_variants=10, 
            hidden_dim=1024
        )
        self.ib_constraint = JSInformationBottleneck(alpha=0.8, beta=0.2)

        
        self.audio_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.video_residual_scale = nn.Parameter(torch.tensor(0.1))

        self.cls_a = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        self.cls_v = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        # self.cls_b = nn.Linear(self.hidden_dim * 2 , config['setting']['num_class'])
    
    def match_magnitude(self, original_feat, variants):

        batch_size, num_variants, feat_dim = variants.shape
        eps = 1e-8

        original_magnitude = torch.norm(original_feat, p=2, dim=1, keepdim=True)  # [batch_size, 1]

        variants_flat = variants.reshape(batch_size * num_variants, feat_dim)
        variants_magnitude = torch.norm(variants_flat, p=2, dim=1, keepdim=True)
        variants_magnitude = variants_magnitude.reshape(batch_size, num_variants, 1)  # [batch_size, num_variants, 1]
        variants_magnitude = torch.clamp(variants_magnitude, min=eps)

        magnitude_ratio = original_magnitude.unsqueeze(1) / variants_magnitude
        matched_variants = variants * magnitude_ratio
        
        return matched_variants

    def apply_residual_connection(self, original_feat, matched_variants,epoch):
        original_expanded = original_feat.unsqueeze(1).repeat(1, matched_variants.shape[1], 1)
        if epoch < 70 :
            residual_variants = original_expanded
        else:
            residual_variants = 0.95 * original_expanded + 0.05 * matched_variants
        
        return residual_variants


    def forward(self, audio, video,epoch):
        # 1. features generate
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        
        
        # 2. high order features generate
        audio_variants, video_variants = self.feature_generator(a_feature, v_feature)  #[64, 10, 512]

        # 3. information bottlenck limitation
        audio_ib_loss, audio_consistency, audio_diversity = self.ib_constraint(
                a_feature, audio_variants
            )
            
        video_ib_loss, video_consistency, video_diversity = self.ib_constraint(
                v_feature, video_variants
            )

        # 4. residual connnect
        audio_variants_matched = self.match_magnitude(a_feature, audio_variants)
        video_variants_matched = self.match_magnitude(v_feature, video_variants)

        audio_variants = self.apply_residual_connection(
            a_feature, audio_variants_matched,epoch
        )
        video_variants = self.apply_residual_connection(
            v_feature, video_variants_matched,epoch
        )


        result_a = self.cls_a(a_feature)
        result_v = self.cls_v(v_feature)
        # result_b = self.cls_b(torch.cat((a_feature, v_feature), dim=1))
        result_b = result_v + result_a
        a_v_ib_loss = video_ib_loss + audio_ib_loss

        return result_b, result_a, result_v, a_feature, v_feature, audio_variants, video_variants, a_v_ib_loss
    
    def getFeature(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature
