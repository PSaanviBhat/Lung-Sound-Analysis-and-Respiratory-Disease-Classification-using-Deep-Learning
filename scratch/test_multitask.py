import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
import torch.nn as nn
from sota.dataset import get_dataloaders
from sota.models import BaselineCNNPANN, CNNLSTMPANN
from sota.run_experiments import FocalLoss

def test_multitask():
    print("Testing multi-task dataset loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_file="metadata.csv",
        batch_size=8,
        channels_to_use=[0, 1, 2],
        use_augmentations=False,
        multitask=True
    )
    
    # Get a batch
    batch_x, batch_y, batch_pathology = next(iter(train_loader))
    print(f"Batch loaded successfully!")
    print(f"Inputs shape: {batch_x.shape}")
    print(f"Cycle Labels shape: {batch_y.shape}, values: {batch_y}")
    print(f"Pathology Labels shape: {batch_pathology.shape}, values: {batch_pathology}")
    
    assert batch_x.shape == (8, 3, 128, 128)
    assert batch_y.shape == (8,)
    assert batch_pathology.shape == (8,)
    assert torch.all(batch_pathology >= 0) and torch.all(batch_pathology < 3)
    print("Dataset verification passed!")
    
    print("\nTesting models with multitask=True...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.to(device)
    
    cnn = BaselineCNNPANN(in_channels=3, num_classes=4, pretrained=False, multitask=True).to(device)
    cycle_logits, pathology_logits = cnn(batch_x)
    print(f"BaselineCNNPANN Cycle Logits shape: {cycle_logits.shape}")
    print(f"BaselineCNNPANN Pathology Logits shape: {pathology_logits.shape}")
    assert cycle_logits.shape == (8, 4)
    assert pathology_logits.shape == (8, 3)
    
    hybrid = CNNLSTMPANN(in_channels=3, num_classes=4, pretrained=False, multitask=True).to(device)
    cycle_logits, pathology_logits = hybrid(batch_x)
    print(f"CNNLSTMPANN Cycle Logits shape: {cycle_logits.shape}")
    print(f"CNNLSTMPANN Pathology Logits shape: {pathology_logits.shape}")
    assert cycle_logits.shape == (8, 4)
    assert pathology_logits.shape == (8, 3)
    print("Model verification passed!")
    
    print("\nTesting multi-task FocalLoss and Mixup target shapes...")
    # Targets for cycle classification (smoothed/mixed)
    lam = 0.8
    y_cycle_onehot = torch.nn.functional.one_hot(batch_y, num_classes=4).float().to(device)
    targets_cycle = lam * y_cycle_onehot + (1 - lam) * y_cycle_onehot.roll(1, dims=0)
    
    y_pathology_onehot = torch.nn.functional.one_hot(batch_pathology, num_classes=3).float().to(device)
    targets_pathology = lam * y_pathology_onehot + (1 - lam) * y_pathology_onehot.roll(1, dims=0)
    
    criterion_cycle = FocalLoss(gamma=2.0)
    criterion_pathology = FocalLoss(gamma=2.0)
    
    loss_cycle = criterion_cycle(cycle_logits, targets_cycle)
    loss_pathology = criterion_pathology(pathology_logits, targets_pathology)
    joint_loss = loss_cycle + 1.0 * loss_pathology
    print(f"Loss Cycle: {loss_cycle.item():.4f}")
    print(f"Loss Pathology: {loss_pathology.item():.4f}")
    print(f"Joint Loss: {joint_loss.item():.4f}")
    
    assert not torch.isnan(joint_loss)
    print("Loss verification passed!")
    print("\nAll multi-task unit tests passed successfully!")

if __name__ == '__main__':
    test_multitask()
