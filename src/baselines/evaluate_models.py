import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import get_dataloaders
from train_hybrid_model import CNNLSTM

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
CNN_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_baseline_cnn.pth")
HYBRID_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_hybrid_model.pth")
OUTPUT_DIR = "evaluation_results"

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())
        
    return np.array(all_labels), np.array(all_preds)

def calculate_icbhi_score(y_true, y_pred):
    # ICBHI evaluation logic:
    # 0: Normal, 1: Crackle, 2: Wheeze, 3: Both
    
    # Specificity (Sp): Recall of the Normal class (0)
    normal_total = np.sum(y_true == 0)
    normal_correct = np.sum((y_true == 0) & (y_pred == 0))
    specificity = normal_correct / (normal_total + 1e-8)
    
    # Sensitivity (Se): Average recall of the abnormal classes (1, 2, 3)
    abnormal_recalls = []
    for c in [1, 2, 3]:
        total_c = np.sum(y_true == c)
        correct_c = np.sum((y_true == c) & (y_pred == c))
        recall_c = correct_c / (total_c + 1e-8)
        abnormal_recalls.append(recall_c)
        
    sensitivity = np.mean(abnormal_recalls)
    icbhi_score = (sensitivity + specificity) / 2.0
    
    return sensitivity, specificity, icbhi_score

def profile_latency(model, sample_tensor):
    model.eval()
    
    # Warm-up run
    for _ in range(10):
        _ = model(sample_tensor)
        
    # Profile CPU/GPU execution time over 100 runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_tensor)
    end_time = time.time()
    
    # Average latency in milliseconds
    latency_ms = ((end_time - start_time) / 100.0) * 1000.0
    return latency_ms

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Normal', 'Crackle', 'Wheeze', 'Both']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    
    # 1. Load Test Dataloader
    print("Loading test dataset...")
    _, _, test_loader = get_dataloaders(batch_size=32)
    
    # Get a sample tensor for latency profiling
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[:1].to(DEVICE) # Single sample batch
    
    # 2. Evaluate Baseline CNN (ResNet-18)
    print("\nEvaluating Baseline 2D CNN (ResNet-18)...")
    cnn_model = models.resnet18()
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    cnn_model = cnn_model.to(DEVICE)
    
    y_true_cnn, y_pred_cnn = evaluate_model(cnn_model, test_loader)
    se_cnn, sp_cnn, score_cnn = calculate_icbhi_score(y_true_cnn, y_pred_cnn)
    latency_cnn = profile_latency(cnn_model, sample_x)
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, "Baseline CNN (ResNet-18) Confusion Matrix", "confusion_matrix_cnn.png")
    
    # 3. Evaluate Proposed CNN-LSTM Hybrid Model
    print("\nEvaluating Proposed CNN-LSTM Hybrid Model...")
    hybrid_model = CNNLSTM(num_classes=4)
    hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH))
    hybrid_model = hybrid_model.to(DEVICE)
    
    y_true_hyb, y_pred_hyb = evaluate_model(hybrid_model, test_loader)
    se_hyb, sp_hyb, score_hyb = calculate_icbhi_score(y_true_hyb, y_pred_hyb)
    latency_hyb = profile_latency(hybrid_model, sample_x)
    plot_confusion_matrix(y_true_hyb, y_pred_hyb, "Proposed CNN-LSTM Confusion Matrix", "confusion_matrix_hybrid.png")
    
    # 4. Print Summary Comparison Table
    print("\n" + "="*70)
    print("                      METRIC COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} | {'Baseline ResNet-18':<20} | {'Proposed CNN-LSTM':<20}")
    print("-"*70)
    print(f"{'Overall Test Accuracy':<25} | {np.mean(y_true_cnn == y_pred_cnn)*100:6.2f}% {' ':<12} | {np.mean(y_true_hyb == y_pred_hyb)*100:6.2f}%")
    print(f"{'ICBHI Sensitivity (Se)':<25} | {se_cnn*100:6.2f}% {' ':<12} | {se_hyb*100:6.2f}%")
    print(f"{'ICBHI Specificity (Sp)':<25} | {sp_cnn*100:6.2f}% {' ':<12} | {sp_hyb*100:6.2f}%")
    print(f"{'Official ICBHI Score (S)':<25} | {score_cnn*100:6.2f}% {' ':<12} | {score_hyb*100:6.2f}%")
    print(f"{'Inference Latency/Cycle':<25} | {latency_cnn:6.2f} ms {' ':<12} | {latency_hyb:6.2f} ms")
    print("="*70)
    print(f"Confusion matrices saved as images in directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
