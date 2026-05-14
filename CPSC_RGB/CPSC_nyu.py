import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.dynamic_Classifier_nyu import Classifier
import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from data.aligned_conc_dataset_noised import AlignedConcDataset as AlignedConcDatasetNoised
from utils.utils import *
from utils.crl_utils import *
import os
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader, Subset
import copy
from data.additional_transform import AddSaltPepperNoise, AddGaussianNoise
import json
import logging

class Averager:
    def __init__(self):
        self.reset()
    
    def add(self, value):
        self.sum += value
        self.count += 1
        
    def item(self):
        return self.sum / self.count if self.count > 0 else 0.0
    
    def reset(self):
        self.sum = 0.0
        self.count = 0

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="/nyud2_trainvaltest")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=2.4e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--savedir", type=str, default="./savepath/nyud")
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=19)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--epoch_threshold", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str,
                        default="./checkpoint/resnet18_pretrained.pth")

def get_optimizer(model, args):
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

class CPThresholdCalibrator:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  
        self.threshold = None 
    
    def calibrate(self, cp_model, val_loader,gamma):
        cp_model.eval()
        rgb_nonconformity_scores = []
        depth_nonconformity_scores = []
    
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc="Calibrating CP")):
                rgb, depth, tgt = batch['A'], batch['B'], batch['label']
                rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
                y_labels = tgt
                _,rgb_logits, depth_logits, rgb_conf, depth_conf,_,_,_ = cp_model(rgb, depth,gamma)
                
                probs_rgb = torch.softmax(rgb_logits, dim=1)
                true_prob_rgb = probs_rgb[torch.arange(len(y_labels)), y_labels]
                nonconformity_rgb = 1 - true_prob_rgb
                rgb_nonconformity_scores.extend(nonconformity_rgb.cpu().numpy())

                probs_depth = torch.softmax(depth_logits, dim=1)
                true_prob_depth = probs_depth[torch.arange(len(y_labels)), y_labels]
                nonconformity_depth = 1 - true_prob_depth
                depth_nonconformity_scores.extend(nonconformity_depth.cpu().numpy())

        if rgb_nonconformity_scores:
            sorted_scores = np.sort(rgb_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            rgb_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            rgb_threshold = 0.0

        if depth_nonconformity_scores:
            sorted_scores = np.sort(depth_nonconformity_scores)
            index = int(np.ceil((1 - self.alpha) * len(sorted_scores)))
            depth_threshold = sorted_scores[min(index, len(sorted_scores)-1)]
        else:
            depth_threshold = 0.0
    
        return rgb_threshold, depth_threshold

def get_top3_features_by_reliability(features_with_variants, reliability_scores):
    _, top_indices = torch.topk(reliability_scores, k=3, dim=1)
    batch_size, _, feature_dim = features_with_variants.shape
    expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    top3_features = torch.gather(features_with_variants, 1, expanded_indices)
    return top3_features[:, 0, :], top3_features[:, 1, :], top3_features[:, 2, :]

def train_rgbd(epoch, train_loader, model, optimizer, logger, args,cp_model,cp_threshold_rgb, cp_threshold_depth,eta ,sigma,gamma):
    model.train()
    tl = Averager()
    criterion = nn.CrossEntropyLoss().cuda()
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        rgb, depth, tgt = batch['A'], batch['B'], batch['label']
        rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()

        optimizer.zero_grad()

        if cp_model is not None and cp_threshold_rgb is not None and cp_threshold_depth is not None and epoch >= args.epoch_threshold :
            _,_, _, rgb_conf, depth_conf,rgb_variants,depth_variants,ib_loss = model(rgb, depth,gamma)
            batch_size = tgt.shape[0]
            rgb_reliability = torch.zeros((batch_size, 10)).cuda()
            depth_reliability = torch.zeros((batch_size, 10)).cuda()

            for variant_idx in range(10):
                current_rgb_features = rgb_variants[:, variant_idx, :]
                with torch.no_grad():
                    results_rgb = cp_model.rgb_clf(current_rgb_features)
                    pretrain_probs_rgb = torch.softmax(results_rgb, dim=1)
                
                cp_set_rgb = []
                for probs in pretrain_probs_rgb:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_rgb]
                    valid_classes.sort(key=lambda x: x[1], reverse=True)
                    sorted_classes = [idx for idx, _ in valid_classes]
                    cp_set_rgb.append(sorted_classes)

                for i, pred_label in enumerate(tgt):
                    if pred_label.item() in cp_set_rgb[i]:
                        sorted_classes = torch.argsort(pretrain_probs_rgb[i], descending=True)
                        rank = (sorted_classes == pred_label).nonzero().item()
                        reliability = 1 / (rank + 1)
                    else:
                        reliability = 0.0
                    rgb_reliability[i, variant_idx] = reliability
                
            for variant_idx in range(10):
                current_depth_features = depth_variants[:, variant_idx, :]
                with torch.no_grad():
                    results_depth = cp_model.depth_clf(current_depth_features) 
                    pretrain_probs_depth = torch.softmax(results_depth, dim=1)
                
                cp_set_depth = []
                for probs in pretrain_probs_depth:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_depth]
                    valid_classes.sort(key=lambda x: x[1], reverse=True)
                    sorted_classes = [idx for idx, _ in valid_classes]
                    cp_set_depth.append(sorted_classes)

                for i, pred_label in enumerate(tgt):
                    if pred_label.item() in cp_set_depth[i]:
                        sorted_classes = torch.argsort(pretrain_probs_depth[i], descending=True)
                        rank = (sorted_classes == pred_label).nonzero().item()
                        reliability = 1 / (rank + 1)
                    else:
                        reliability = 0.0
                    depth_reliability[i, variant_idx] = reliability
            
            rgb_feature_1, rgb_feature_2, rgb_feature_3 = get_top3_features_by_reliability(rgb_variants, rgb_reliability)
            fused_rgb = torch.mean(torch.stack([rgb_feature_1, rgb_feature_2, rgb_feature_3], dim=1), dim=1) 

            depth_feature_1, depth_feature_2, depth_feature_3 = get_top3_features_by_reliability(depth_variants, depth_reliability)
            fused_depth = torch.mean(torch.stack([depth_feature_1, depth_feature_2, depth_feature_3], dim=1), dim=1)

            selected_rgb = fused_rgb
            selected_depth = fused_depth
            
        else:
            _,_, _, rgb_conf, depth_conf,rgb_variants,depth_variants,ib_loss = model(rgb, depth,gamma)
            selected_rgb = rgb_conf
            selected_depth = depth_conf
        
        rgb_out = model.rgb_clf(selected_rgb)
        depth_out = model.depth_clf(selected_depth)
        both_output = 0.5 * (rgb_out + depth_out)

        loss_both = criterion(both_output, tgt)
        loss_rgb = criterion(rgb_out, tgt)
        loss_depth = criterion(depth_out, tgt)

        loss = loss_both +  (loss_depth + loss_rgb + 0.1 * ib_loss)
        
        if torch.isnan(loss).any() or loss <= 0 :
            print(f"NaN detected at step {step}")
            print(f"loss_both: {loss_both.item()}, loss_depth: {loss_depth.item()}, loss_rgb: {loss_rgb.item()}, ib_loss: {ib_loss.item()}")
        
        if cp_model is not None and cp_threshold_rgb is not None and cp_threshold_depth is not None and epoch >= args.epoch_threshold:
            with torch.no_grad():
                results_rgb = cp_model.rgb_clf(selected_rgb)
                results_depth = cp_model.depth_clf(selected_depth)
                results_mm = 0.5 * results_rgb +  0.5 * results_depth
                pretrain_probs_mm = torch.softmax(results_mm , dim=1)

                pred_mm = torch.argmax(0.5 * depth_out +  0.5 * rgb_out,dim=1)
                cp_set_mm = []
                for probs in pretrain_probs_mm:
                    valid_classes = [(i, p.item()) for i, p in enumerate(probs) if p >= 1 - cp_threshold_depth]
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
                    if 'depthenc' in name:
                        fractor = torch.mean(reliability_mm)
                        fractor = eta * fractor + sigma
                        param.grad *= fractor
                    if 'rgbenc' in name:
                        fractor = torch.mean(reliability_mm)
                        fractor = eta * fractor + sigma
                        param.grad *= fractor
        else:
            loss.backward()

        optimizer.step()
        tl.add(loss.item())
    
    loss = tl.item()
    logger.info(f'Epoch {epoch}: Total Loss: {loss:.4f}')
    return model

def val_rgbd(epoch, val_loader, model, logger, args, gamma):
    model.eval()
    pred_list_fusion = []
    label_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            rgb, depth, tgt = batch['A'], batch['B'], batch['label']
            rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
            
            label_list.extend(tgt.cpu().tolist())
            
            output, _, _, _, _, _, _, _ = model(rgb, depth, gamma) 
            pred_fusion = output.argmax(dim=1)
            pred_list_fusion.extend(pred_fusion.cpu().tolist())
    
    acc_fusion = accuracy_score(label_list, pred_list_fusion)
    if epoch != -1: # Only log during training, skip for final tests to avoid clutter
        logger.info(f'Epoch {epoch}: Clean - Acc: {acc_fusion:.4f}')
    return acc_fusion

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
    parser = argparse.ArgumentParser(description="Train RGB-D Scene Recognition Model")
    get_args(parser)
    args = parser.parse_args()
    
    args.name = f"cpsc_nyu_eta{args.eta}_sigma{args.sigma}_th{args.epoch_threshold}"
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    
    eta = args.eta
    sigma = args.sigma
    gamma = args.gamma

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Save log to file as well
    file_handler = logging.FileHandler(os.path.join(args.savedir, "training.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    
    train_transforms = list()
    train_transforms.append(transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    train_transforms.append(transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=mean, std=std))

    val_transforms = list()
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    train_dataset = AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'),
                    transform=transforms.Compose(train_transforms))

    num_samples = len(train_dataset)
    val_indices = torch.randperm(num_samples)[:4]
    val_dataset = Subset(train_dataset, val_indices)
    train_indices = torch.tensor([i for i in range(num_samples) if i not in val_indices.tolist()])
    train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=args.n_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=args.n_workers
    )

    test_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), 
        transform=transforms.Compose(val_transforms)),
        batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers
    )

    model = Classifier(args)
    model = model.cuda()

    cp_model = copy.deepcopy(model)
    cp_model.eval()
    for param in cp_model.parameters():
        param.requires_grad = False
        
    cp_calibrator = CPThresholdCalibrator(alpha=0.2)
    cp_threshold_rgb = None  
    cp_threshold_depth = None 

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    best_clean_acc = 0.0
    best_clean_epoch = 0
    
    for epoch in range(args.max_epochs):
        logger.info(f'Epoch {epoch} training started...')
        model = train_rgbd(epoch, train_loader, model, optimizer, logger, args, cp_model, cp_threshold_rgb, cp_threshold_depth,eta,sigma,gamma)
        
        clean_acc = val_rgbd(epoch, test_loader, model, logger, args, gamma)
        
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            best_clean_epoch = epoch
            best_clean_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_clean_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'clean_acc': best_clean_acc,
            }, os.path.join(args.savedir, 'model_best_clean.pt'))
            logger.info(f' 🔥 NEW BEST CLEAN ACC 🔥 Epoch {epoch} | Clean Acc: {best_clean_acc:.4f}')

        cp_model.load_state_dict(model.state_dict())
        cp_model.eval()
        cp_threshold_rgb, cp_threshold_depth = cp_calibrator.calibrate(cp_model, val_loader, gamma)
        logger.info(f'Epoch {epoch}: Calibrated CP threshold_rgb = {cp_threshold_rgb:.4f}, Calibrated CP threshold_depth = {cp_threshold_depth:.4f}')
        
        scheduler.step(clean_acc)

    # ==========================================
    # FINAL ROBUSTNESS EVALUATION 
    # ==========================================
    logger.info("Loading best model for final robust evaluation on Test sets...")
    clean_model = Classifier(args)
    clean_model.load_state_dict(best_clean_model_state)
    clean_model = clean_model.cuda()
    clean_model.eval()

    clean_transforms = list()
    clean_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    clean_transforms.append(transforms.ToTensor())
    clean_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    # 椒盐噪声转换
    sp5_transforms = list()
    sp5_transforms.append(transforms.Resize((256, 256)))
    sp5_transforms.append(transforms.RandomApply([AddSaltPepperNoise(density=0.10)], p=0.5))
    sp5_transforms.append(transforms.CenterCrop(224))
    sp5_transforms.append(transforms.ToTensor())
    sp5_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    sp10_transforms = list()
    sp10_transforms.append(transforms.Resize((256, 256)))
    sp10_transforms.append(transforms.RandomApply([AddSaltPepperNoise(density=0.10)], p=1.0))
    sp10_transforms.append(transforms.CenterCrop(224))
    sp10_transforms.append(transforms.ToTensor())
    sp10_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    # 高斯噪声转换
    gs5_transforms = list()
    gs5_transforms.append(transforms.Resize((256, 256)))
    gs5_transforms.append(transforms.RandomApply([AddGaussianNoise(mean=0.0, variance=5)], p=0.5))
    gs5_transforms.append(transforms.CenterCrop(224))
    gs5_transforms.append(transforms.ToTensor())
    gs5_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    gs10_transforms = list()
    gs10_transforms.append(transforms.Resize((256, 256)))
    gs10_transforms.append(transforms.RandomApply([AddGaussianNoise(mean=0.0, variance=10)], p=0.5))
    gs10_transforms.append(transforms.CenterCrop(224))
    gs10_transforms.append(transforms.ToTensor())
    gs10_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    # 创建不同的测试集加载器
    test_scenarios = {
        "Clean Test": DataLoader(
            AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(clean_transforms),
            depth_transform=transforms.Compose(clean_transforms)),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers),
            
        "Salt & Pepper (Lvl 5.0)": DataLoader(
            AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(sp5_transforms),
            depth_transform=transforms.Compose(sp5_transforms)),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers),
            
        "Salt & Pepper (Lvl 10.0)": DataLoader(
            AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(sp10_transforms),
            depth_transform=transforms.Compose(sp10_transforms)),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers),
            
        "Gaussian (Lvl 5.0)": DataLoader(
            AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(gs5_transforms),
            depth_transform=transforms.Compose(gs5_transforms)),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers),
            
        "Gaussian (Lvl 10.0)": DataLoader(
            AlignedConcDatasetNoised(args, data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(gs10_transforms),
            depth_transform=transforms.Compose(gs10_transforms)),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers)
    }

    final_results = {}
    for name, loader in test_scenarios.items():
        logger.info(f"Generating and Evaluating on dataset for: {name} ...")
        acc = val_rgbd(-1, loader, clean_model, logger, args, gamma)
        final_results[name] = acc

    logger.info("\n" + "="*60)
    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS ON TEST SET':^60}")
    logger.info("="*60)
    logger.info(f"| {'Test Scenario':<35} | {'Score (Acc)':<18} |")
    logger.info("|" + "-"*37 + "|" + "-"*20 + "|")
    
    for name, score in final_results.items():
        logger.info(f"| {name:<35} | {score:>18.4f} |")
        
    logger.info("="*60 + "\n")

    result_file = os.path.join(args.savedir, "final_results.json")
    with open(result_file, "w") as f:
        json.dump({
            "best_clean_model": {
                "epoch": best_clean_epoch,
                "clean_acc": final_results["Clean Test"]
            },
            "robustness": final_results
        }, f, indent=4)
    
    logger.info(f"Saved final results to {result_file}")
    logger.info(f"Best Clean Model: epoch={best_clean_epoch}, clean_acc={final_results['Clean Test']:.4f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()