#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torchvision
from pytorch_pretrained_bert.modeling import BertModel
import torch.nn.functional as F

# ===== Feature Generation =====
class EfficientHighOrderGenerator(nn.Module):
    def __init__(self, txt_input_dim=768, img_input_dim=6144 , num_variants=10, txt_hidden_dim=1024, img_hidden_dim=4096):  # 增加隐藏层维度
        super(EfficientHighOrderGenerator, self).__init__()
        self.num_variants = num_variants
        self.txt_input_dim = txt_input_dim
        self.img_input_dim = img_input_dim
        
        self.txt_mlp = nn.Sequential(
            nn.Linear(txt_input_dim, txt_hidden_dim),
            nn.ReLU(),
            nn.Linear(txt_hidden_dim, num_variants * txt_input_dim)
        )
        
        self.img_mlp = nn.Sequential(
            nn.Linear(img_input_dim, img_hidden_dim),
            nn.ReLU(),
            nn.Linear(img_hidden_dim, num_variants * img_input_dim)
        )
        
    def forward(self, txt_features, img_features):
        batch_size = txt_features.shape[0]
        
        txt_flat = self.txt_mlp(txt_features)
        txt_variants = txt_flat.view(batch_size, self.num_variants, self.txt_input_dim)
        
        img_flat = self.img_mlp(img_features)
        img_variants = img_flat.view(batch_size, self.num_variants, self.img_input_dim)
        
        return txt_variants, img_variants


class JSInformationBottleneck(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(JSInformationBottleneck, self).__init__()
        self.alpha = alpha  # 一致性权重
        self.beta = beta    # 多样性权重
        
    def js_divergence(self, p, q):
        """计算JS散度"""
        p_dist = F.softmax(p, dim=-1)
        q_dist = F.softmax(q, dim=-1)
        m = 0.5 * (p_dist + q_dist)
        
        kl_p_m = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean')
        kl_q_m = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean')
        
        return 0.5 * (kl_p_m + kl_q_m)
    
    def forward(self, original_features, high_order_variants):
        """
        信息瓶颈约束
        """
        batch_size, num_variants, feat_dim = high_order_variants.shape
        
        # 1. 一致性约束：高阶特征与原始特征的一致性
        consistency_loss = 0
        for i in range(num_variants):
            variant = high_order_variants[:, i, :]
            js_consistency = self.js_divergence(variant, original_features)
            consistency_loss += js_consistency
        
        consistency_loss /= num_variants
        
        # 2. 多样性约束：高阶特征之间的多样性
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
        
        # 3. 组合损失
        ib_loss = self.alpha * consistency_loss + self.beta * (1 - torch.sigmoid(diversity_loss))
        
        return ib_loss, consistency_loss, diversity_loss


# ===== Text =====
class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out


class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, 3)
        self.clf.apply(self.enc.bert.init_bert_weights)

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        return self.clf(x), x  


# ===== Image =====
class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(args.img_hidden_sz * args.num_image_embeds, 3)

    def forward(self, x):
        img_features = self.img_encoder(x)  
        img_features = torch.flatten(img_features, start_dim=1)
        out = self.clf(img_features)
        return out, img_features 


# ===== Multimodal Late Fusion (Pure CPSC) =====
class MultimodalLateFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)

        self.feature_generator = EfficientHighOrderGenerator(txt_input_dim=768, img_input_dim=6144, num_variants=10, txt_hidden_dim=1024, img_hidden_dim=4096)
        self.ib_constraint = JSInformationBottleneck(alpha=0.8, beta=0.2)

    def match_magnitude(self, original_feat, variants):
        batch_size, num_variants, feat_dim = variants.shape
        eps = 1e-8
    
        original_magnitude = torch.norm(original_feat, p=2, dim=1, keepdim=True)
    
        variants_flat = variants.reshape(batch_size * num_variants, feat_dim)
        variants_magnitude = torch.norm(variants_flat, p=2, dim=1, keepdim=True)
        variants_magnitude = variants_magnitude.reshape(batch_size, num_variants, 1)
        
        variants_magnitude = torch.clamp(variants_magnitude, min=eps)
        
        magnitude_ratio = original_magnitude.unsqueeze(1) / variants_magnitude
        matched_variants = variants * magnitude_ratio
        
        return matched_variants

    def apply_residual_connection(self, original_feat, matched_variants, gamma):
        batch_size, num_variants, feat_dim = matched_variants.shape
        original_expanded = original_feat.unsqueeze(1).expand(batch_size, num_variants, feat_dim)
        residual_variants = (1 - gamma) * original_expanded + matched_variants * gamma
        
        return residual_variants

    def forward(self, txt, mask, segment, img, gamma):

        txt_out, txt_features = self.txtclf(txt, mask, segment)
        img_out, img_features = self.imgclf(img)


        txt_img_out = 0.5 * txt_out + 0.5 * img_out


        txt_variants, img_variants = self.feature_generator(txt_features, img_features)

        txt_variants_matched = self.match_magnitude(txt_features, txt_variants)
        img_variants_matched = self.match_magnitude(img_features, img_variants)

        txt_ib_loss, _, _ = self.ib_constraint(txt_features, txt_variants_matched)
        img_ib_loss, _, _ = self.ib_constraint(img_features, img_variants_matched)

        txt_variants = self.apply_residual_connection(txt_features, txt_variants_matched, gamma)
        img_variants = self.apply_residual_connection(img_features, img_variants_matched, gamma)

        ib_loss = txt_ib_loss + img_ib_loss

        return txt_img_out, txt_out, img_out, txt_variants, img_variants, ib_loss