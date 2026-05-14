#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.CREMA import CramedDataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import weight_init
import copy
from torchinfo import summary

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def Alignment(p, q,beta=0.1):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * p + 0.5 * q
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)

    kl_a = F.kl_div(p.log(), p, reduction='batchmean')
    kl_v = F.kl_div(q.log(), q, reduction='batchmean')

    return js_score + beta * (kl_a + kl_v)

class CPThresholdCalibrator:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  
        self.threshold = None 
    
    def calibrate(self, cp_model, val_loader):
        cp_model.eval()
        audio_nonconformity_scores = []
        video_nonconformity_scores = []
    
        with torch.no_grad():
            for spectrogram, image, y in val_loader:
            
                image = image.float().cuda()
                y = y.cuda()
                spectrogram = spectrogram.unsqueeze(1).float().cuda()
                y_labels = torch.argmax(y, dim=1).long()
                #print(image.shape)
                o_b, o_a, o_v, _, _,_,_,_ = cp_model(spectrogram, image, epoch=1)
            
                probs_a = torch.softmax(o_a, dim=1)
                true_prob_a = probs_a[torch.arange(len(y_labels)), y_labels]
                nonconformity_a = 1 - true_prob_a
                audio_nonconformity_scores.extend(nonconformity_a.cpu().numpy())

                probs_v = torch.softmax(o_v, dim=1)
                true_prob_v = probs_v[torch.arange(len(y_labels)), y_labels]
                nonconformity_v = 1 - true_prob_v
                video_nonconformity_scores.extend(nonconformity_v.cpu().numpy())

        if audio_nonconformity_scores:
            sorted_scores = np.sort(audio_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            audio_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            audio_threshold = 0.0

        if video_nonconformity_scores:
            sorted_scores = np.sort(video_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            video_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            video_threshold = 0.0
    
        return audio_threshold, video_threshold

def get_top3_features_by_reliability(features_with_variants, reliability_scores):
    # Get indices of top-3 features based on reliability scores
    _, top_indices = torch.topk(reliability_scores, k=3, dim=1)
    
    # Extract corresponding feature vectors
    batch_size, _, feature_dim = features_with_variants.shape
    expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, feature_dim)

    top3_features = torch.gather(features_with_variants, 1, expanded_indices)
    
    return (
        top3_features[:, 0, :],  
        top3_features[:, 1, :],  
        top3_features[:, 2, :]   
    )




def train_audio_video(epoch, train_loader, model, optimizer, logger,cp_model,cp_threshold_audio,cp_threshold_video):
    model.train()
    tl = Averager()

    tl_align = Averager()
    tl_ib = Averager()

    tl_supervise = Averager()
    tl_unsupervise = Averager()
    reliability_avg = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()


    #select the feature,if the cp model exist
    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()

        # image:[64,3,3,224,224]
        
        if cp_model is not None and cp_threshold_audio is not None and cp_threshold_video is not None and epoch >= 70:
            #print('here')
            #selected_audio, selected_video features
            # get the variants features
            # get the variants features
            o_b, o_a, o_v, a_f, v_f,audio_variants,video_variants,loss_ib = model(spectrogram, image,epoch)
            batch_size = y.shape[0]
            audio_reliability = torch.zeros((batch_size, 10)).cuda()
            video_reliability = torch.zeros((batch_size, 10)).cuda()

            for variant_idx in range(10):
                current_audio_features = audio_variants[:, variant_idx, :]
            
                with torch.no_grad():
                    results_a = cp_model.cls_a(current_audio_features)
                    pretrain_probs_a = torch.softmax(results_a, dim=1)
                    true_a = torch.argmax(y,dim=1)
                
                    cp_set_a = []
                    for probs in pretrain_probs_a:
                        valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_audio]
                        valid_classes.sort(key=lambda x: x[1], reverse=True)
                        sorted_classes = [idx for idx, _ in valid_classes]
                        cp_set_a.append(sorted_classes)

                    for i, pred_label in enumerate(true_a):
                        if pred_label.item() in cp_set_a[i]:
                            sorted_classes = torch.argsort(pretrain_probs_a[i], descending=True)
                            rank = (sorted_classes == pred_label).nonzero().item()
                            reliability = 1 / (rank + 1)
                        else:
                            reliability = 0.0

                        audio_reliability[i, variant_idx] = reliability

            for variant_idx in range(10):
                current_video_features = video_variants[:, variant_idx, :]
            
                with torch.no_grad():
                    results_v = cp_model.cls_v(current_video_features)
                
                    pretrain_probs_v = torch.softmax(results_v, dim=1)
                    true_v = torch.argmax(y,dim=1)
                
                    cp_set_v = []
                    for probs in pretrain_probs_v:
                        valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_video]
                        valid_classes.sort(key=lambda x: x[1], reverse=True)
                        sorted_classes = [idx for idx, _ in valid_classes]
                        cp_set_v.append(sorted_classes)
                
                    for i, pred_label in enumerate(true_v):
                        if pred_label.item() in cp_set_v[i]:
                            sorted_classes = torch.argsort(pretrain_probs_v[i], descending=True)
                            rank = (sorted_classes == pred_label).nonzero().item()
                            reliability = 1 / (rank + 1)
                        else:
                            reliability = 0.0
                    
                        video_reliability[i, variant_idx] = reliability

            audio_feature_1, audio_feature_2, audio_feature_3 = get_top3_features_by_reliability(audio_variants, audio_reliability)
            audio_features_stack = torch.stack([audio_feature_1, audio_feature_2, audio_feature_3], dim=1)  # [64, 3, 512]
            fused_audio = torch.mean(audio_features_stack, dim=1)  # [64, 512]

            video_feature_1, video_feature_2, video_feature_3 = get_top3_features_by_reliability(video_variants, video_reliability)
            video_features_stack = torch.stack([video_feature_1, video_feature_2, video_feature_3], dim=1)  # [64, 3, 512]
            fused_video = torch.mean(video_features_stack, dim=1)

            #print(audio_feature_1.shape)
            #print(fused_audio.shape)
            selected_audio = fused_audio
            selected_video = fused_video
            #print('need to select the features')
            
        else:
            #print('does not using cp')
            o_b, o_a, o_v, a_f, v_f,audio_variants,video_variants,loss_ib = model(spectrogram, image,epoch)
            selected_audio = a_f
            selected_video = v_f

        
        o_a = model.cls_a(selected_audio)
        o_v = model.cls_v(selected_video)

        loss_alignment = Alignment(o_a, o_v)
        loss_cls_mm = criterion( 0.5 * o_v+  0.5 *o_a, y).mean()
        loss_cls_a = criterion(o_a, y).mean()
        loss_cls_v = criterion(o_v, y).mean()
        loss_supervise = 0.8 * loss_cls_mm + 0.2 * (loss_cls_a + loss_cls_v)

        loss_unsupervise = (loss_alignment + loss_ib)
        #loss_unsupervise = (loss_alignment)

        loss =   2 * (loss_supervise +  loss_unsupervise)

        if cp_model is not None and cp_threshold_audio is not None and cp_threshold_video is not None and epoch >= 70:
            with torch.no_grad():
                results_a = cp_model.cls_a(selected_audio)
                results_v = cp_model.cls_v(selected_audio)
                results_mm = 0.5 * results_a +  0.5 *results_v
                #print(results_a)
                pretrain_probs_mm = torch.softmax(results_mm , dim=1)
                pred_mm = torch.argmax(0.5 * o_v+  0.5 *o_a,dim=1)
                cp_set_mm = []
                for probs in pretrain_probs_mm:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_audio]
                    # sort as probs
                    valid_classes.sort(key=lambda x: x[1], reverse=True)
                    # retain index
                    sorted_classes = [idx for idx, _ in valid_classes]
                    cp_set_mm.append(sorted_classes)
            reliability_mm = []

            for i, pred in enumerate(pred_mm):
                if pred.item() in cp_set_mm[i]:
                
                    sorted_classes = torch.argsort(pretrain_probs_mm[i], descending=True)
                    rank = (sorted_classes == pred).nonzero().item()
                    reliability_mm.append(1 / (rank + 1))  
                else:
                    reliability_mm.append(0.0)  
            reliability_mm = torch.tensor(reliability_mm).cuda()
            reliability_mm = torch.mean(reliability_mm)

            batch_reliability = torch.mean(reliability_mm)
            reliability_avg.add(batch_reliability.item())

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'cls_' in name:
                        continue
                    if 'audio_encoder' in name:
                        fractor = torch.mean(reliability_mm)
                        #print(fractor)
                        fractor = 0.6 * fractor + 0.6
                        param.grad *= fractor

                    if 'video_encoder' in name:
                        fractor = torch.mean(reliability_mm)
                        fractor = 0.6 * fractor + 0.6
                        param.grad *= fractor
        else:
            loss.backward()
        
        optimizer.step()
        tl.add(loss.item())
        tl_align.add(loss_alignment.item())
        tl_ib.add(loss_ib.item())
        tl_unsupervise.add(loss_unsupervise.item())
        tl_supervise.add(loss_supervise.item())


    loss_ave = tl.item()
    loss_align_all = tl_align.item()
    loss_ib_all = tl_ib.item()
    loss_supervise_all = tl_supervise.item()
    loss_unsupervise_all = tl_unsupervise.item()

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , Average AlignLoss : {loss_align_all:.3f},Average ibloss : {loss_ib_all:.3f},Average unsuperviseLoss : {loss_unsupervise_all:.3f},Average superviseLoss : {loss_supervise_all:.3f}').format(epoch=epoch, loss_ave=loss_ave, loss_align_all = loss_align_all, loss_ib_all = loss_ib_all,loss_unsupervise_all = loss_unsupervise_all,loss_supervise_all =loss_supervise_all))

    return model


best_acc = 0.0
best_epoch = 0
best_log = ""
best_metrics = {}

def val(epoch, val_loader, model, logger):
    global best_acc, best_epoch, best_log, best_metrics
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []
    score_a = 0.0
    score_v = 0.0
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

            o_b, o_a, o_v, a_f, v_f,_,_,_ = model(spectrogram, image,epoch)

            soft_pred_a = soft_pred_a + (F.softmax(o_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(o_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(0.5 * o_v+ 0.5 * o_a, dim=1)).tolist()
            pred = (F.softmax( 0.5 * o_v+ 0.5 * o_a, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(o_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(o_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    log_message = (f'Epoch {epoch}: f1:{f1:.4f}, acc:{acc:.4f}, mAP:{mAP:.4f}, '
                   f'f1_a:{f1_a:.4f}, acc_a:{acc_a:.4f}, mAP_a:{mAP_a:.4f}, '
                   f'f1_v:{f1_v:.4f}, acc_v:{acc_v:.4f}, mAP_v:{mAP_v:.4f}')
    
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(log_message)
    
    
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_log = log_message
        best_metrics = {
            'f1': f1, 'acc': acc, 'mAP': mAP,
            'f1_a': f1_a, 'acc_a': acc_a, 'mAP_a': mAP_a,
            'f1_v': f1_v, 'acc_v': acc_v, 'mAP_v': mAP_v
        }
        logger.info(f'🔥 NEW BEST ACC 🔥 Epoch {epoch} | Acc: {acc:.4f}')                                                                                                                
    return acc, score_a, score_v


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='./data/crema.json')

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)
                        
    # ----- SET DATALOADER -----                          
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')

    num_samples = len(train_dataset)
    
    val_indices = torch.randperm(num_samples)[:16]

    val_dataset = Subset(train_dataset, val_indices)

    train_indices = torch.tensor([i for i in range(num_samples) if i not in val_indices.tolist()])
    train_dataset = Subset(train_dataset, train_indices)

    print(len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                            num_workers=cfg['train']['num_workers'], pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                            num_workers=8, pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                            num_workers=cfg['test']['num_workers'], pin_memory=True)
    
    
    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)


     # ----- Initial CP_MODEL -----
    
    cp_model = copy.deepcopy(model)
    cp_model.eval()  
    cp_calibrator = CPThresholdCalibrator(alpha=0.2)
    cp_threshold_audio = None  
    cp_threshold_video = None 
    
    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=config['train']['optimizer']['momentum'],
                          weight_decay=config['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger,cp_model,cp_threshold_audio,cp_threshold_video)
        acc, v_a, v_v = val(epoch, test_loader, model, logger)
         # SAVE THE LAST MODEL PARAM (USED AS CP)
        
        cp_model.load_state_dict(model.state_dict())
        cp_model.eval()  
        
        # compute the threshold
        cp_threshold_audio,cp_threshold_video = cp_calibrator.calibrate(cp_model, val_loader)
        logger.info(f'Epoch {epoch}: Calibrated CP threshold_audio = {cp_threshold_audio:.4f},Calibrated CP threshold_video = {cp_threshold_video:.4f}')
        