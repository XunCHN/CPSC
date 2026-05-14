#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from src.data.helpers1 import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *
import copy
import json

import logging
logging.getLogger("pytorch_pretrained_bert").setLevel(logging.WARNING)

def get_args(parser: argparse.ArgumentParser) -> None:
    """Defines and parses command line arguments"""
    parser.add_argument("--batch_sz", type=int, default=32, help="Batch size")
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased", help="Pre-trained BERT model path")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Dataset directory path")
    parser.add_argument("--drop_img_percent", type=float, default=0.0, help="Percentage of images to drop")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--embed_sz", type=int, default=300, help="Embedding size")
    parser.add_argument("--freeze_img", type=int, default=3, help="Epochs to freeze image encoder")
    parser.add_argument("--freeze_txt", type=int, default=5, help="Epochs to freeze text encoder")
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24, help="Gradient accumulation steps")
    parser.add_argument("--hidden", nargs="*", type=int, default=[], help="Hidden layer sizes")
    parser.add_argument("--hidden_sz", type=int, default=768, help="Main hidden size")
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048, help="Image encoder hidden size")
    parser.add_argument("--include_bn", type=int, default=True, help="Include batch normalization")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reduction factor")
    parser.add_argument("--lr_patience", type=int, default=2, help="Patience for learning rate scheduler")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--model", type=str, default="latefusion", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt", "latefusion"])
    parser.add_argument("--n_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--name", type=str, default="cpsc", help="Experiment name")
    parser.add_argument("--num_image_embeds", type=int, default=3, help="Number of image embeddings")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--savedir", type=str, default="./checkpoint", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["food101", "MVSA_Single"], help="Task name")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification"])
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup proportion for BERT training")
    parser.add_argument("--weight_classes", type=int, default=1, help="Apply class weighting")
    parser.add_argument("--df", type=bool, default=True, help="Use dynamic fusion")
    
    # Noise Testing Args
    parser.add_argument("--noise_level", type=float, default=0.0, help="Noise level for testing")
    parser.add_argument("--noise_type", type=str, default='Gaussian', help="Noise type for testing")
    
    # CPSC Specific Args
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--epoch_threshold", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.6)

def get_criterion(args: argparse.Namespace) -> nn.Module:
    return nn.CrossEntropyLoss()

def get_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=args.lr)

def get_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace) -> optim.lr_scheduler._LRScheduler:
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, factor=args.lr_factor)

class CPThresholdCalibrator:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  
        
    def calibrate(self, cp_model, val_loader, args):
        cp_model.eval()
        txt_nonconformity_scores = []
        img_nonconformity_scores = []
    
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc="Calibrating CP")):
                text, segment, mask, image, target, indices = batch
                text, mask, segment, image, target = text.cuda(), mask.cuda(), segment.cuda(), image.cuda(), target.cuda()
                
                txt_img_logits, text_logits, image_logits, _, _, _ = cp_model(text, mask, segment, image, args.gamma)

                probs_txt = torch.softmax(text_logits, dim=1)
                true_prob_txt = probs_txt[torch.arange(len(target)), target]
                txt_nonconformity_scores.extend((1 - true_prob_txt).cpu().numpy())

                probs_img = torch.softmax(image_logits, dim=1)
                true_prob_img = probs_img[torch.arange(len(target)), target]
                img_nonconformity_scores.extend((1 - true_prob_img).cpu().numpy())

        if txt_nonconformity_scores:
            sorted_scores = np.sort(txt_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            txt_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            txt_threshold = 0.0

        if img_nonconformity_scores:
            sorted_scores = np.sort(img_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            img_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            img_threshold = 0.0
    
        return txt_threshold, img_threshold

def get_top3_features_by_reliability(features_with_variants, reliability_scores):
    _, top_indices = torch.topk(reliability_scores, k=3, dim=1)
    batch_size, _, feature_dim = features_with_variants.shape
    expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    top3_features = torch.gather(features_with_variants, 1, expanded_indices)
    return top3_features[:, 0, :], top3_features[:, 1, :], top3_features[:, 2, :]

def model_forward(epoch, model, args, criterion, batch, cp_model=None, cp_threshold_txt=None, cp_threshold_img=None, mode='train'):
    text, segment, mask, image, target, _ = batch
    device = next(model.parameters()).device
    text, mask, segment, image, target = text.to(device), mask.to(device), segment.to(device), image.to(device), target.to(device)

    if mode == 'train': 
        # ================== CPSC 触发阶段 ==================
        if cp_model is not None and cp_threshold_txt is not None and cp_threshold_img is not None and epoch >= args.epoch_threshold:
            txt_img_logits, text_logits, image_logits, txt_variants, img_variants, ib_loss = model(text, mask, segment, image, args.gamma)
            batch_size = target.shape[0]

            txt_reliability = torch.zeros((batch_size, 10)).cuda()
            img_reliability = torch.zeros((batch_size, 10)).cuda()

            # RSC Text
            for variant_idx in range(10):
                current_txt_features = txt_variants[:, variant_idx, :]
                with torch.no_grad():
                    results_txt = cp_model.txtclf.clf(current_txt_features)
                    pretrain_probs_txt = torch.softmax(results_txt, dim=1)
                
                cp_set_txt = []
                for probs in pretrain_probs_txt:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_txt]
                    valid_classes.sort(key=lambda x: x[1], reverse=True)
                    cp_set_txt.append([idx for idx, _ in valid_classes])

                for i, pred_label in enumerate(target):
                    if pred_label.item() in cp_set_txt[i]:
                        sorted_classes = torch.argsort(pretrain_probs_txt[i], descending=True)
                        rank = (sorted_classes == pred_label).nonzero().item()
                        txt_reliability[i, variant_idx] = 1 / (rank + 1)
            
            # RSC Image
            for variant_idx in range(10):
                current_img_features = img_variants[:, variant_idx, :]
                with torch.no_grad():
                    results_img = cp_model.imgclf.clf(current_img_features)
                    pretrain_probs_img = torch.softmax(results_img, dim=1)
                
                cp_set_img = []
                for probs in pretrain_probs_img:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_img]
                    valid_classes.sort(key=lambda x: x[1], reverse=True)
                    cp_set_img.append([idx for idx, _ in valid_classes])

                for i, pred_label in enumerate(target):
                    if pred_label.item() in cp_set_img[i]:
                        sorted_classes = torch.argsort(pretrain_probs_img[i], descending=True)
                        rank = (sorted_classes == pred_label).nonzero().item()
                        img_reliability[i, variant_idx] = 1 / (rank + 1)
            
            txt_feature_1, txt_feature_2, txt_feature_3 = get_top3_features_by_reliability(txt_variants, txt_reliability)
            fused_txt = torch.mean(torch.stack([txt_feature_1, txt_feature_2, txt_feature_3], dim=1), dim=1) 
            
            img_feature_1, img_feature_2, img_feature_3 = get_top3_features_by_reliability(img_variants, img_reliability)
            fused_img = torch.mean(torch.stack([img_feature_1, img_feature_2, img_feature_3], dim=1), dim=1)

            # 使用筛选出的鲁棒特征重新计算 Logits
            text_logits = model.txtclf.clf(fused_txt)
            image_logits = model.imgclf.clf(fused_img)
            text_img_logits = 0.5 * text_logits + 0.5 * image_logits

            text_clf_loss = criterion(text_logits, target)
            image_clf_loss = criterion(image_logits, target)
            joint_clf_loss = criterion(text_img_logits, target)
            
            total_loss = text_clf_loss + image_clf_loss + joint_clf_loss + args.lamb * ib_loss
            return total_loss, fused_txt, fused_img

        # ================== Warmup 阶段 (Epoch 0 -> threshold) ==================
        else:
            # 直接接收模型输出，不再错误地将 variants 送入分类器
            txt_img_logits, text_logits, image_logits, txt_variants, img_variants, ib_loss = model(text, mask, segment, image, args.gamma)
            
            text_clf_loss = criterion(text_logits, target)
            image_clf_loss = criterion(image_logits, target)
            joint_clf_loss = criterion(txt_img_logits, target)
            
            total_loss = text_clf_loss + image_clf_loss + joint_clf_loss + args.lamb * ib_loss
            return total_loss, None, None

    # ================== Eval 测试阶段 ==================
    else:
        txt_img_logits, _, _, _, _, _ = model(text, mask, segment, image, args.gamma)
        joint_clf_loss = criterion(txt_img_logits, target)
        return joint_clf_loss, txt_img_logits, target

def model_eval(epoch, dataloader, model, args, criterion, store_preds=False):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating epoch {epoch}"):
            loss, output, target = model_forward(epoch, model, args, criterion, batch, mode='eval')
            losses.append(loss.item())
            if args.task_type == "multilabel":
                pred = (torch.sigmoid(output).cpu().detach().numpy() > 0.5)
            else:
                pred = torch.nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            targets.append(target.cpu().detach().numpy())

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        targets = np.vstack(targets)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(targets, preds, average="macro")
        metrics["micro_f1"] = f1_score(targets, preds, average="micro")
    else:
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        metrics["acc"] = accuracy_score(targets, preds)

    if store_preds:
        store_preds_to_disk(targets, preds, args)
    return metrics

def train(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    
    args.noise_level = 0.0
    args.noise_type = 'Gaussian' 
    train_loader, val_loader, cp_loader, test_loaders = get_data_loaders(args)
    
    model = get_model(args)
    model.cuda()

    cp_model = copy.deepcopy(model)
    cp_model.eval()
    for param in cp_model.parameters():
        param.requires_grad = False

    cp_calibrator = CPThresholdCalibrator(alpha=args.alpha)
    cp_threshold_txt = None  
    cp_threshold_img = None 

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    logger = create_logger(os.path.join(args.savedir, "logfile.log"), args)
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, no_improve_count, best_metric = 0, 0, 0, -np.inf
    logger.info("Starting training process with CPSC...")

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            loss, selected_txt, selected_img = model_forward(epoch, model, args, criterion, batch, cp_model, cp_threshold_txt, cp_threshold_img, mode='train')
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_losses.append(loss.item())

            if cp_model is not None and cp_threshold_txt is not None and cp_threshold_img is not None and epoch >= args.epoch_threshold:
                with torch.no_grad():
                    results_txt = cp_model.txtclf.clf(selected_txt)
                    results_img = cp_model.imgclf.clf(selected_img)
                    results_mm = 0.5 * results_txt + 0.5 * results_img
                    pretrain_probs_mm = torch.softmax(results_mm , dim=1)
                    
                    txt_out = model.txtclf.clf(selected_txt)
                    img_out = model.imgclf.clf(selected_img)

                    pred_mm = torch.argmax(0.5 * txt_out + 0.5 * img_out, dim=1)
                    cp_set_mm = []
                    for probs in pretrain_probs_mm:
                        valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_txt]
                        valid_classes.sort(key=lambda x: x[1], reverse=True)
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

                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if 'txtclf.enc' in name:
                            fractor = torch.mean(reliability_mm)
                            fractor = args.eta * fractor + args.sigma
                            param.grad *= fractor
                            
                        if 'imgclf.img_encoder' in name:
                            fractor = torch.mean(reliability_mm)
                            fractor = args.eta * fractor + args.sigma
                            param.grad *= fractor
            else:
                loss.backward()
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        avg_train_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")
        
        logger.info(f"Evaluating on Validation Set...")
        val_metrics = model_eval(epoch, val_loader, model, args, criterion)
        log_metrics(f"Validation", val_metrics, args, logger)

        cp_model.load_state_dict(model.state_dict())
        cp_threshold_txt, cp_threshold_img = cp_calibrator.calibrate(cp_model, cp_loader, args)
        logger.info(f'Epoch {epoch}: Calibrated CP threshold_txt = {cp_threshold_txt:.4f}, CP threshold_img = {cp_threshold_img:.4f}')

        tuning_metric = val_metrics["micro_f1"] if args.task_type == "multilabel" else val_metrics["acc"]
        scheduler.step(tuning_metric)
        
        improvement = tuning_metric > best_metric
        if improvement:
            best_metric = tuning_metric
            no_improve_count = 0
            logger.info("\n" + "="*50)
            logger.info(f"🚀 NEW BEST MODEL FOUND! 🚀")
            logger.info(f"🏆 Metric: {best_metric:.4f}")
            logger.info("="*50 + "\n")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": no_improve_count,
                    "best_metric": best_metric,
                },
                improvement,
                args.savedir,
            )
        else:
            no_improve_count += 1
        
        if no_improve_count >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs. Stopping early.")
            break

    # ==========================================
    # FINAL ROBUSTNESS EVALUATION 
    # ==========================================
    logger.info("Loading best model for final robust evaluation on Test sets")
    model_path = os.path.join(args.savedir, "model_best.pt")
    if os.path.exists(model_path):
        load_checkpoint(model, model_path)
    model.eval()
    
    test_scenarios = [

        {"name": "Clean Test", "noise_level": 0.0, "noise_type": "Gaussian"},

        {"name": "Gaussian (Lvl 5.0)", "noise_level": 5.0, "noise_type": "Gaussian"},

        {"name": "Gaussian (Lvl 10.0)", "noise_level": 10.0, "noise_type": "Gaussian"},

        {"name": "Salt & Pepper (Lvl 5.0)", "noise_level": 5.0, "noise_type": "Salt"},

        {"name": "Salt & Pepper (Lvl 10.0)", "noise_level": 10.0, "noise_type": "Salt"},

    ]



    final_results = {}



    for scenario in test_scenarios:

        args.noise_level = scenario["noise_level"]

        args.noise_type = scenario["noise_type"]

        

        logger.info(f"--- Generating dataset for: {scenario['name']} ---")

        _, _, _, current_test_loaders = get_data_loaders(args)

        

        current_test_loader = current_test_loaders["test"]

        

        scenario_metrics = model_eval(np.inf, current_test_loader, model, args, criterion, store_preds=True)

        metric_val = scenario_metrics.get("micro_f1") if args.task_type == "multilabel" else scenario_metrics.get("acc")

        final_results[scenario["name"]] = metric_val

        

        log_metrics(scenario["name"], scenario_metrics, args, logger)



    logger.info("\n" + "="*60)

    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS ON TEST SET':^60}")

    logger.info("="*60)

    logger.info(f"| {'Test Scenario':<35} | {'Score (Acc / F1)':<18} |")

    logger.info("|" + "-"*37 + "|" + "-"*20 + "|")

    

    for name, score in final_results.items():

        logger.info(f"| {name:<35} | {score:>18.4f} |")

        

    logger.info("="*60 + "\n")



    result_file = os.path.join(args.savedir, "final_results.json")

    with open(result_file, "w") as f:

        json.dump({

            "best_clean_model": {"clean_acc": final_results["Clean Test"]},

            "robustness": final_results

        }, f, indent=4)

    logger.info(f"Saved final results to {result_file}")

def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Model Trainer")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert not remaining_args, f"Unrecognized arguments: {remaining_args}"
    train(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()