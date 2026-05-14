#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import json

import torch
import torch.nn as nn

from src.data.helpers1 import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *

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
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--model", type=str, default="latefusion", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt", "latefusion", "cadp_latefusion"])
    parser.add_argument("--n_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--name", type=str, default="cpsc", help="Experiment name")
    parser.add_argument("--num_image_embeds", type=int, default=3, help="Number of image embeddings")
    parser.add_argument("--savedir", type=str, default="./checkpoint", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["food101", "MVSA_Single"], help="Task name")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification"])
    parser.add_argument("--weight_classes", type=int, default=1, help="Apply class weighting")
    parser.add_argument("--df", type=bool, default=True, help="Use dynamic fusion")
    
    # Noise Testing Args
    parser.add_argument("--noise_level", type=float, default=0.0, help="Noise level for testing")
    parser.add_argument("--noise_type", type=str, default='Gaussian', help="Noise type for testing")
    
    # CPSC Specific Args (Need these to initialize the model properly)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--lamb", type=float, default=0.1)
    
    # Model path explicitly passed here
    parser.add_argument("--model_path", type=str, default="/data/hdd2/postgraduates/2025/guyufan/QMF_TIC/checkpoint/cpsc/model_best.pt", help="Path to the trained model")

def model_forward_eval(model, args, criterion, batch):
    """Simplified model_forward strictly for evaluation"""
    text, segment, mask, image, target, _ = batch
    device = next(model.parameters()).device
    text, mask, segment, image, target = text.to(device), mask.to(device), segment.to(device), image.to(device), target.to(device)

    txt_img_logits, _, _, _, _, _ = model(text, mask, segment, image, args.gamma)
    joint_clf_loss = criterion(txt_img_logits, target)
    return joint_clf_loss, txt_img_logits, target

def model_eval(dataloader, model, args, criterion, store_preds=False):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating"):
            loss, output, target = model_forward_eval(model, args, criterion, batch)
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

def evaluate(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Init logger
    logger = create_logger(os.path.join(model_dir, "eval_logfile.log"), args)
    logger.info(f"Starting Evaluation Pipeline...")

    # Load previously saved arguments if they exist to match architecture
    args_file = os.path.join(model_dir, "args.pt")
    if os.path.exists(args_file):
        logger.info(f"Found training arguments at {args_file}. Loading configuration to match architecture...")
        saved_args = torch.load(args_file, weights_only=False) # <--- 修改这行
        # Keep our evaluation specific arguments
        saved_args.model_path = args.model_path
        saved_args.noise_level = args.noise_level
        saved_args.noise_type = args.noise_type
        args = saved_args

    set_seed(args.seed)

    # Load Model
    logger.info(f"Loading best model from {args.model_path}...")
    model = get_model(args)
    model.cuda()
    
    if os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path)
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"Could not find model file at {args.model_path}! Please check the path.")
        return

    criterion = nn.CrossEntropyLoss()
    
    # Define scenarios
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
        
        scenario_metrics = model_eval(current_test_loader, model, args, criterion, store_preds=True)
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

    result_file = os.path.join(model_dir, "eval_final_results.json")
    with open(result_file, "w") as f:
        json.dump({
            "best_clean_model": {"clean_acc": final_results["Clean Test"]},
            "robustness": final_results
        }, f, indent=4)
    logger.info(f"Saved final evaluation results to {result_file}")

def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Model Evaluation")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert not remaining_args, f"Unrecognized arguments: {remaining_args}"
    evaluate(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()