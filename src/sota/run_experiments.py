import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from dataset import get_dataloaders
from models import BaselineCNNPANN, CNNLSTMPANN

# Configurations mapping
CONFIGS = {
    'A': {'channels': [0], 'in_channels': 1, 'name': 'Mel'},
    'B': {'channels': [0, 1], 'in_channels': 2, 'name': 'Mel+CQT'},
    'C': {'channels': [0, 2], 'in_channels': 2, 'name': 'Mel+CWT'},
    'D': {'channels': [0, 1, 2], 'in_channels': 3, 'name': 'Stacked (All)'}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "training_logs"
OUTPUT_DIR = "evaluation_results"

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = torch.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()
            
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -targets * focal_weight * log_probs
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = loss * alpha.unsqueeze(0)
            
        loss = torch.sum(loss, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.best_val_score = -float('inf')
        self.counter = 0

    def __call__(self, val_score, model, path):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_score, model, path):
        if self.verbose:
            print(f"Validation Metric Improved ({self.best_val_score:.4f} --> {val_score:.4f}). Saving model...")
        torch.save(model.state_dict(), path)
        self.best_val_score = val_score

def calculate_icbhi_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    normal_total = np.sum(y_true == 0)
    normal_correct = np.sum((y_true == 0) & (y_pred == 0))
    specificity = normal_correct / (normal_total + 1e-8)
    
    abnormal_recalls = []
    for c in [1, 2, 3]:
        total_c = np.sum(y_true == c)
        correct_c = np.sum((y_true == c) & (y_pred == c))
        recall_c = correct_c / (total_c + 1e-8)
        abnormal_recalls.append(recall_c)
        
    sensitivity = np.mean(abnormal_recalls)
    icbhi_score = (sensitivity + specificity) / 2.0
    
    return sensitivity, specificity, icbhi_score

def train_epoch(model, loader, criterion, optimizer, scaler, mixup=False, mixup_alpha=0.2, mixup_prob=0.8, label_smoothing_factor=0.1, multitask=False, pathology_weight=1.0, criterion_pathology=None):
    model.train()
    running_loss = 0.0
    running_cycle_loss = 0.0
    running_pathology_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    all_pathology_preds = []
    all_pathology_labels = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if multitask:
            batch_x, batch_y, batch_pathology = batch
            batch_x, batch_y, batch_pathology = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_pathology.to(DEVICE)
        else:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
        optimizer.zero_grad()
        
        if mixup and np.random.rand() < mixup_prob:
            # Sample mixup weight
            lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
            index = torch.randperm(batch_x.size(0), device=DEVICE)
            
            # Mix inputs and targets (one-hot soft targets)
            mixed_x = lam * batch_x + (1.0 - lam) * batch_x[index]
            y_a = torch.nn.functional.one_hot(batch_y, num_classes=4).float()
            y_b = torch.nn.functional.one_hot(batch_y[index], num_classes=4).float()
            targets = lam * y_a + (1.0 - lam) * y_b
            
            if multitask:
                yp_a = torch.nn.functional.one_hot(batch_pathology, num_classes=3).float()
                yp_b = torch.nn.functional.one_hot(batch_pathology[index], num_classes=3).float()
                targets_pathology = lam * yp_a + (1.0 - lam) * yp_b
        else:
            mixed_x = batch_x
            targets = torch.nn.functional.one_hot(batch_y, num_classes=4).float()
            if multitask:
                targets_pathology = torch.nn.functional.one_hot(batch_pathology, num_classes=3).float()
            
        if label_smoothing_factor > 0:
            targets = targets * (1.0 - label_smoothing_factor) + (label_smoothing_factor / 4.0)
            if multitask:
                targets_pathology = targets_pathology * (1.0 - label_smoothing_factor) + (label_smoothing_factor / 3.0)
            
        with autocast():
            if multitask:
                outputs, outputs_pathology = model(mixed_x)
                loss_cycle = criterion(outputs, targets)
                loss_pathology = criterion_pathology(outputs_pathology, targets_pathology)
                loss = loss_cycle + pathology_weight * loss_pathology
            else:
                outputs = model(mixed_x)
                loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * batch_x.size(0)
        if multitask:
            running_cycle_loss += loss_cycle.item() * batch_x.size(0)
            running_pathology_loss += loss_pathology.item() * batch_x.size(0)
            
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        if multitask:
            _, pathology_preds = torch.max(outputs_pathology, 1)
            all_pathology_preds.extend(pathology_preds.cpu().numpy())
            all_pathology_labels.extend(batch_pathology.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    _, _, epoch_icbhi = calculate_icbhi_score(all_labels, all_preds)
    
    if multitask:
        epoch_pathology_loss = running_pathology_loss / len(loader.dataset)
        epoch_pathology_acc = accuracy_score(all_pathology_labels, all_pathology_preds)
        return epoch_loss, epoch_acc, epoch_icbhi, epoch_pathology_loss, epoch_pathology_acc
        
    return epoch_loss, epoch_acc, epoch_icbhi

@torch.no_grad()
def validate(model, loader, criterion, multitask=False, pathology_weight=1.0, criterion_pathology=None):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    
    all_pathology_preds = []
    all_pathology_labels = []
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        if multitask:
            batch_x, batch_y, batch_pathology = batch
            batch_x, batch_y, batch_pathology = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_pathology.to(DEVICE)
        else:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
        with autocast():
            if multitask:
                outputs, outputs_pathology = model(batch_x)
                loss_cycle = criterion(outputs, batch_y)
                loss_pathology = criterion_pathology(outputs_pathology, batch_pathology)
                loss = loss_cycle + pathology_weight * loss_pathology
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
        running_loss += loss.item() * batch_x.size(0)
        all_logits.append(outputs.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        if multitask:
            _, pathology_preds = torch.max(outputs_pathology, 1)
            all_pathology_preds.extend(pathology_preds.cpu().numpy())
            all_pathology_labels.extend(batch_pathology.cpu().numpy())
        
    val_loss = running_loss / len(loader.dataset)
    val_logits = np.concatenate(all_logits, axis=0)
    
    exp_logits = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
    val_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(all_labels, val_preds)
    val_se, val_sp, val_score = calculate_icbhi_score(all_labels, val_preds)
    
    if multitask:
        val_pathology_acc = accuracy_score(all_pathology_labels, all_pathology_preds)
        return val_loss, val_acc, val_se, val_sp, val_score, val_probs, np.array(all_labels), val_pathology_acc
        
    return val_loss, val_acc, val_se, val_sp, val_score, val_probs, np.array(all_labels)

def optimize_thresholds(val_probs, val_labels):
    print("Optimizing decision thresholds on validation split using Random Sweep...")
    best_score = 0.0
    best_thresholds = np.ones(4)
    
    np.random.seed(42)
    num_trials = 20000
    
    for _ in range(num_trials):
        thresholds = np.random.uniform(0.1, 2.0, size=4)
        adjusted_probs = val_probs / thresholds
        preds = np.argmax(adjusted_probs, axis=1)
        
        _, _, score = calculate_icbhi_score(val_labels, preds)
        
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            
    print(f"Optimal Thresholds Found: {best_thresholds}")
    print(f"Best Validation ICBHI Score (Calibrated): {best_score*100:.2f}%")
    return best_thresholds, best_score

def evaluate_test(model, test_loader, thresholds=None, multitask=False):
    model.eval()
    all_logits = []
    all_labels = []
    
    all_pathology_logits = []
    all_pathology_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if multitask:
                batch_x, batch_y, batch_pathology = batch
                batch_x = batch_x.to(DEVICE)
                with autocast():
                    outputs, outputs_pathology = model(batch_x)
                all_pathology_logits.append(outputs_pathology.cpu().numpy())
                all_pathology_labels.extend(batch_pathology.numpy())
            else:
                batch_x, batch_y = batch
                batch_x = batch_x.to(DEVICE)
                with autocast():
                    outputs = model(batch_x)
            
            all_logits.append(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    end_time = time.time()
    
    latency_ms = ((end_time - start_time) / len(test_loader.dataset)) * 1000.0
    
    logits = np.concatenate(all_logits, axis=0)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    labels = np.array(all_labels)
    
    preds_std = np.argmax(probs, axis=1)
    std_se, std_sp, std_score = calculate_icbhi_score(labels, preds_std)
    std_acc = accuracy_score(labels, preds_std)
    
    results = {
        'std': {
            'accuracy': std_acc,
            'se': std_se,
            'sp': std_sp,
            'score': std_score,
            'preds': preds_std.tolist()
        },
        'latency_ms': latency_ms
    }
    
    if thresholds is not None:
        adjusted_probs = probs / thresholds
        preds_cal = np.argmax(adjusted_probs, axis=1)
        cal_se, cal_sp, cal_score = calculate_icbhi_score(labels, preds_cal)
        cal_acc = accuracy_score(labels, preds_cal)
        
        results['calibrated'] = {
            'accuracy': cal_acc,
            'se': cal_se,
            'sp': cal_sp,
            'score': cal_score,
            'preds': preds_cal.tolist()
        }
        
    if multitask:
        pathology_logits = np.concatenate(all_pathology_logits, axis=0)
        exp_pathology = np.exp(pathology_logits - np.max(pathology_logits, axis=1, keepdims=True))
        pathology_probs = exp_pathology / np.sum(exp_pathology, axis=1, keepdims=True)
        pathology_preds = np.argmax(pathology_probs, axis=1)
        pathology_labels = np.array(all_pathology_labels)
        pathology_acc = accuracy_score(pathology_labels, pathology_preds)
        
        results['pathology'] = {
            'accuracy': pathology_acc,
            'preds': pathology_preds.tolist(),
            'labels': pathology_labels.tolist()
        }
        
    return results, labels

def plot_curves(history, model_type, config_key, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.8)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='orange', alpha=0.8)
    ax1.set_title(f'Loss Curves ({model_type.upper()} - Config {config_key})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(epochs, [s * 100 for s in history['train_score']], label='Train ICBHI', color='green', alpha=0.8)
    ax2.plot(epochs, [s * 100 for s in history['val_score']], label='Val ICBHI', color='red', alpha=0.8)
    ax2.set_title(f'ICBHI Score Curves ({model_type.upper()} - Config {config_key})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_experiment(model_type, config_key, epochs=50, batch_size=32, learning_rate=1e-4, patience=10, 
                   tune_only=False, eval_only=False, use_augmentations=True, 
                   mixup=False, mixup_alpha=0.2, mixup_prob=0.8,
                   label_smoothing=0.1, focal_gamma=2.0, multitask=False, pathology_weight=1.0):
    config_info = CONFIGS[config_key]
    channels = config_info['channels']
    in_channels = config_info['in_channels']
    config_name = config_info['name']
    
    # Save SOTA checkpoints separately by using the _sota suffix
    suffix = "_sota_multitask" if multitask else "_sota"
    checkpoint_name = f"{model_type}_config_{config_key}{suffix}.pth"
    model_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    thresholds_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_config_{config_key}{suffix}_thresholds.npy")
    log_path = os.path.join(LOG_DIR, f"{model_type}_config_{config_key}{suffix}_history.json")
    plot_path = os.path.join(LOG_DIR, f"{model_type}_config_{config_key}{suffix}_curves.png")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f" RUNNING SOTA EXPERIMENT | Model: {model_type.upper()} | Config: {config_key} ({config_name})")
    print(f"=======================================================")
    print(f"Device: {DEVICE}")
    print(f"Input Channels: {in_channels} (Indices: {channels})")
    print(f"SOTA Augmentations: {use_augmentations} | Mixup: {mixup} (alpha={mixup_alpha}, prob={mixup_prob})")
    print(f"Label Smoothing: {label_smoothing} | Focal Loss Gamma: {focal_gamma}")
    print(f"Multi-Task Learning: {multitask} (Weight: {pathology_weight})")
    
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, 
        channels_to_use=channels,
        use_augmentations=use_augmentations,
        multitask=multitask
    )
    
    pretrained = not (eval_only or tune_only)
    if model_type == 'cnn':
        model = BaselineCNNPANN(in_channels=in_channels, num_classes=4, pretrained=pretrained, multitask=multitask).to(DEVICE)
    else:
        model = CNNLSTMPANN(in_channels=in_channels, num_classes=4, pretrained=pretrained, multitask=multitask).to(DEVICE)
        
    if focal_gamma > 0:
        criterion = FocalLoss(gamma=focal_gamma)
        criterion_pathology = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()
        criterion_pathology = nn.CrossEntropyLoss()
    
    if not (eval_only or tune_only):
        print(f"Initializing Adam optimizer (lr={learning_rate}) and CosineAnnealingLR...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()
        
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        history = {
            'train_loss': [], 'train_acc': [], 'train_score': [],
            'val_loss': [], 'val_acc': [], 'val_score': []
        }
        if multitask:
            history['train_pathology_loss'] = []
            history['train_pathology_acc'] = []
            history['val_pathology_acc'] = []
        
        print("\nStarting training loop...")
        for epoch in range(1, epochs + 1):
            if multitask:
                train_loss, train_acc, train_score, train_path_loss, train_path_acc = train_epoch(
                    model, train_loader, criterion, optimizer, scaler,
                    mixup=mixup, mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
                    label_smoothing_factor=label_smoothing, multitask=True,
                    pathology_weight=pathology_weight, criterion_pathology=criterion_pathology
                )
                val_loss, val_acc, val_se, val_sp, val_score, _, _, val_path_acc = validate(
                    model, val_loader, criterion, multitask=True,
                    pathology_weight=pathology_weight, criterion_pathology=criterion_pathology
                )
            else:
                train_loss, train_acc, train_score = train_epoch(
                    model, train_loader, criterion, optimizer, scaler,
                    mixup=mixup, mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
                    label_smoothing_factor=label_smoothing
                )
                val_loss, val_acc, val_se, val_sp, val_score, _, _ = validate(model, val_loader, criterion)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_score'].append(train_score)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_score'].append(val_score)
            if multitask:
                history['train_pathology_loss'].append(train_path_loss)
                history['train_pathology_acc'].append(train_path_acc)
                history['val_pathology_acc'].append(val_path_acc)
            
            if multitask:
                print(f"Epoch {epoch:02d}/{epochs:02d} | "
                      f"Train Loss: {train_loss:.4f} (Path Loss: {train_path_loss:.4f}) - Cycle Acc: {train_acc*100:.2f}% - Path Acc: {train_path_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} - Cycle ICBHI: {val_score*100:.2f}% - Path Acc: {val_path_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch:02d}/{epochs:02d} | "
                      f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% - Train ICBHI: {train_score*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val ICBHI: {val_score*100:.2f}%")
            
            early_stopping(val_score, model, model_path)
            if early_stopping.early_stop:
                print("Early stopping triggered! Training stopped.")
                break
                
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
        plot_curves(history, model_type, config_key, plot_path)
        print(f"Training log and curves saved to {LOG_DIR}/")
        
    print(f"Loading best model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    thresholds = None
    if not eval_only:
        if multitask:
            val_loss, val_acc, val_se, val_sp, val_score, val_probs, val_labels, val_path_acc = validate(
                model, val_loader, criterion, multitask=True,
                pathology_weight=pathology_weight, criterion_pathology=criterion_pathology
            )
        else:
            val_loss, val_acc, val_se, val_sp, val_score, val_probs, val_labels = validate(model, val_loader, criterion)
        print(f"\nDefault Validation ICBHI Score: {val_score*100:.2f}% (Se: {val_se*100:.2f}%, Sp: {val_sp*100:.2f}%)")
        
        thresholds, calibrated_val_score = optimize_thresholds(val_probs, val_labels)
        np.save(thresholds_path, thresholds)
        print(f"Optimized thresholds saved to {thresholds_path}")
    else:
        if os.path.exists(thresholds_path):
            print(f"Loading saved thresholds from {thresholds_path}...")
            thresholds = np.load(thresholds_path)
        else:
            print("No saved thresholds found. Evaluating using standard argmax only.")
            
    print("\nEvaluating on Test Split...")
    test_results, test_labels = evaluate_test(model, test_loader, thresholds, multitask=multitask)
    
    print("\n" + "="*50)
    print("               TEST SPLIT METRICS")
    print("="*50)
    print(f"Inference Latency: {test_results['latency_ms']:.2f} ms/cycle")
    print("\n[Standard Predictions (Argmax)]:")
    print(f"  Accuracy    : {test_results['std']['accuracy']*100:.2f}%")
    print(f"  Sensitivity : {test_results['std']['se']*100:.2f}%")
    print(f"  Specificity : {test_results['std']['sp']*100:.2f}%")
    print(f"  ICBHI Score : {test_results['std']['score']*100:.2f}%")
    
    if 'calibrated' in test_results:
        print("\n[Calibrated Predictions (Tuned Thresholds)]:")
        print(f"  Accuracy    : {test_results['calibrated']['accuracy']*100:.2f}%")
        print(f"  Sensitivity : {test_results['calibrated']['se']*100:.2f}%")
        print(f"  Specificity : {test_results['calibrated']['sp']*100:.2f}%")
        print(f"  ICBHI Score : {test_results['calibrated']['score']*100:.2f}%")
        
    if multitask and 'pathology' in test_results:
        print("\n[Pathology Classification (Disease State)]:")
        print(f"  Accuracy    : {test_results['pathology']['accuracy']*100:.2f}%")
        
    summary_path = os.path.join(OUTPUT_DIR, f"{model_type}_config_{config_key}{suffix}_results.json")
    with open(summary_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Results summary saved to {summary_path}")
    return test_results

def main():
    parser = argparse.ArgumentParser(description="Run SOTA Lung Sound Classification Experiments")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'hybrid'], help='Model type: cnn or hybrid')
    parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C', 'D'], help='Ablation Configuration: A, B, C, D')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--tune_only', action='store_true', help='Only tune thresholds using saved checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate saved model')
    parser.add_argument('--no_aug', action='store_false', dest='use_augmentations', help='Disable SOTA data augmentations')
    parser.add_argument('--mixup', action='store_true', help='Enable Mixup data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.8, help='Mixup probability per batch')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter (set 0.0 to use standard cross-entropy)')
    parser.add_argument('--multitask', action='store_true', help='Enable multi-task pathology classification')
    parser.add_argument('--pathology_weight', type=float, default=1.0, help='Weight factor for pathology loss')
    args = parser.parse_args()
    
    run_experiment(
        model_type=args.model,
        config_key=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        tune_only=args.tune_only,
        eval_only=args.eval_only,
        use_augmentations=args.use_augmentations,
        mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        multitask=args.multitask,
        pathology_weight=args.pathology_weight
    )

if __name__ == "__main__":
    main()
