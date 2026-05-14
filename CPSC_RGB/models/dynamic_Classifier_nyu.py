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
from models.image import ImageEncoder
import torch.nn.functional as F


class EfficientHighOrderGenerator(nn.Module):
    def __init__(self, input_dim, num_variants=10, hidden_dim=256):
        super(EfficientHighOrderGenerator, self).__init__()
        self.num_variants = num_variants
        self.input_dim = input_dim
        

        self.rgb_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variants * input_dim)
        )
        

        self.depth_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variants * input_dim)
        )
        
    def forward(self, rgb_features, depth_features):
        batch_size = rgb_features.shape[0]
        

        rgb_flat = self.rgb_mlp(rgb_features)
        rgb_variants = rgb_flat.view(batch_size, self.num_variants, self.input_dim)
        
        depth_flat = self.depth_mlp(depth_features)
        depth_variants = depth_flat.view(batch_size, self.num_variants, self.input_dim)
        
        return rgb_variants, depth_variants


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
        ib_loss = self.alpha * consistency_loss - self.beta * diversity_loss
        
        return ib_loss, consistency_loss, diversity_loss


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds

        #print(rgb_last_size)

        self.unimodal_transform = nn.Sequential(
            nn.Linear(depth_last_size, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        self.rgb_clf = nn.Linear(128, args.n_classes)
        self.depth_clf = nn.Linear(128, args.n_classes)
        
        self.feature_generator = EfficientHighOrderGenerator(
            input_dim = 128,  
            num_variants = 10,
            hidden_dim = 256
        )
        
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

    def apply_residual_connection(self, original_feat, matched_variants,gamma):
        batch_size, num_variants, feat_dim = matched_variants.shape
        original_expanded = original_feat.unsqueeze(1).expand(batch_size, num_variants, feat_dim)
        residual_variants = original_expanded + matched_variants * gamma
        
        return residual_variants

    def forward(self, rgb, depth,gamma):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        

        depth_uni = self.unimodal_transform(depth)  
        rgb_uni = self.unimodal_transform(rgb)


        rgb_variants, depth_variants = self.feature_generator(rgb_uni, depth_uni)
        #print(rgb_variants.shape)
        rgb_ib_loss, rgb_consistency, rgb_diversity = self.ib_constraint(rgb_uni, rgb_variants)
        depth_ib_loss, depth_consistency, depth_diversity = self.ib_constraint(depth_uni, depth_variants)


        rgb_variants_matched = self.match_magnitude(rgb_uni, rgb_variants)
        depth_variants_matched = self.match_magnitude(depth_uni, depth_variants)
        
        rgb_variants = self.apply_residual_connection(
            rgb_uni, rgb_variants_matched, gamma
        )
        depth_variants = self.apply_residual_connection(
            depth_uni, depth_variants_matched, gamma
        )

        ib_loss = rgb_ib_loss + depth_ib_loss
        rgb_out = self.rgb_clf(rgb_uni)
        depth_out = self.depth_clf(depth_uni)     

        both_output = 0.5 * (rgb_out + depth_out)
        
        return both_output, rgb_out, depth_out, rgb_uni, depth_uni,rgb_variants,depth_variants,ib_loss
        #return both_output







        
