import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
from dataset import get_dataloaders

# Configuration
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_baseline_cnn.pth")

def calculate_metrics(y_true, y_pred):
    # Compute precision, recall, and F1-score for multiclass (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(loader, desc="Training", leave=False):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    _, _, epoch_f1 = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(loader, desc="Validating", leave=False):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        running_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
    val_loss = running_loss / len(loader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_precision, val_recall, val_f1 = calculate_metrics(all_labels, all_preds)
    
    return val_loss, val_acc, val_f1, all_labels, all_preds

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data Loaders
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Instantiate Model
    print("Initializing ResNet-18 model...")
    # Load ResNet-18 with default ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify classification head (output size = 4 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_f1 = 0.0
    
    print("\nStarting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% - Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val F1: {val_f1:.4f}")
              
        # Checkpoint if F1 score improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> Best model saved with Val F1: {best_val_f1:.4f}")
            
    # 4. Final Evaluation on Test Split
    print("\nTraining completed. Loading best checkpoint for evaluation on test split...")
    best_model = models.resnet18()
    best_model.fc = nn.Linear(num_ftrs, 4)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    best_model = best_model.to(DEVICE)
    
    test_loss, test_acc, test_f1, test_labels, test_preds = validate(best_model, test_loader, criterion)
    print(f"\nFinal Test Results | Loss: {test_loss:.4f} - Accuracy: {test_acc*100:.2f}% - Macro F1: {test_f1:.4f}")
    
    print("\nDetailed Test Classification Report:")
    target_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
