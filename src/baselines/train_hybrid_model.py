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
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_hybrid_model.pth")

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTM, self).__init__()
        # 1. Spatial Feature Extractor: ResNet-18 Backbone
        # Load backbone with default ImageNet weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove average pooling and final fully connected layer to obtain raw feature maps
        self.spatial_extractor = nn.Sequential(*list(resnet.children())[:-2])
        # Output shape from spatial_extractor for a (128, 128) input: (B, 512, 4, 4)
        
        # 2. Sequential Modeling: Bidirectional LSTM
        # Treat width (time axis = 4 steps) as the sequence dimension
        # Treat channel (512) * height (4) = 2048 as the feature dimension
        self.lstm = nn.LSTM(
            input_size=512 * 4, 
            hidden_size=256, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        
        # 3. Classifier Head (BiLSTM output size is hidden_size * 2 = 512)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, 3, 128, 128)
        
        # Extract spatial feature maps: output shape is (B, 512, 4, 4)
        spatial_maps = self.spatial_extractor(x)
        b, c, h, w = spatial_maps.shape
        
        # Re-arrange dimensions to format sequence: (B, Time_Steps, Feature_Dim)
        # Treat width (axis 3) as the sequence length (4 steps)
        # Permute shape to: (B, w, c, h) -> (B, 4, 512, 4)
        seq_features = spatial_maps.permute(0, 3, 1, 2).contiguous()
        # Flatten features at each step: (B, 4, 512 * 4) -> (B, 4, 2048)
        seq_features = seq_features.view(b, w, c * h)
        
        # Pass sequence to Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(seq_features) # Output shape: (B, 4, 512)
        
        # Take output of the final time step
        final_state = lstm_out[:, -1, :] # Shape: (B, 512)
        
        # Classification
        logits = self.fc(final_state)
        return logits

def calculate_metrics(y_true, y_pred):
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
    print("Initializing CNN-LSTM hybrid model...")
    model = CNNLSTM(num_classes=4).to(DEVICE)
    
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
    best_model = CNNLSTM(num_classes=4)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    best_model = best_model.to(DEVICE)
    
    test_loss, test_acc, test_f1, test_labels, test_preds = validate(best_model, test_loader, criterion)
    print(f"\nFinal Test Results | Loss: {test_loss:.4f} - Accuracy: {test_acc*100:.2f}% - Macro F1: {test_f1:.4f}")
    
    print("\nDetailed Test Classification Report:")
    target_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
