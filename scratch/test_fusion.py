import sys
import os
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sota.models import BaselineCNNResNet, CNNLSTMResNet, BaselineCNNPANN, CNNLSTMPANN
from src.sota.loss import SupervisedContrastiveLoss, DynamicMultiTaskLoss

def test_models_and_losses():
    # Test parameters
    batch_size = 4
    in_channels = 3
    num_classes = 4
    height, width = 128, 128
    
    # Mock input (stacked feature maps: Mel, CQT, CWT)
    mock_input = torch.randn(batch_size, in_channels, height, width)
    print(f"Mock input shape: {mock_input.shape}\n")
    
    # 1. Test BaselineCNNResNet with cross-attention and shared features extraction
    print("--- Testing BaselineCNNResNet (cross_attention=True) ---")
    model_resnet_cnn = BaselineCNNResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True
    )
    model_resnet_cnn.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_resnet_cnn(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    assert hasattr(model_resnet_cnn, 'last_shared_features'), "Model is missing last_shared_features attribute!"
    print(f"last_shared_features shape: {model_resnet_cnn.last_shared_features.shape} (Expected: [{batch_size}, 512])")
    print("Success!\n")
    
    # 2. Test CNNLSTMResNet with cross-attention, sequence_len=32 and shared features extraction
    print("--- Testing CNNLSTMResNet (cross_attention=True, sequence_len=32, use_conformer=True) ---")
    model_resnet_conformer = CNNLSTMResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True,
        sequence_len=32,
        use_conformer=True
    )
    model_resnet_conformer.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_resnet_conformer(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    assert hasattr(model_resnet_conformer, 'last_shared_features'), "Model is missing last_shared_features attribute!"
    print(f"last_shared_features shape: {model_resnet_conformer.last_shared_features.shape} (Expected: [{batch_size}, 256])")
    print("Success!\n")
    
    # 3. Test BaselineCNNPANN with cross-attention and shared features extraction
    print("--- Testing BaselineCNNPANN (cross_attention=True) ---")
    model_pann_cnn = BaselineCNNPANN(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True
    )
    model_pann_cnn.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_pann_cnn(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    assert hasattr(model_pann_cnn, 'last_shared_features'), "Model is missing last_shared_features attribute!"
    print(f"last_shared_features shape: {model_pann_cnn.last_shared_features.shape} (Expected: [{batch_size}, 2048])")
    print("Success!\n")
    
    # 4. Test CNNLSTMPANN with cross-attention, sequence_len=64 and shared features extraction
    print("--- Testing CNNLSTMPANN (cross_attention=True, sequence_len=64, use_conformer=True) ---")
    model_pann_conformer = CNNLSTMPANN(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True,
        sequence_len=64,
        use_conformer=True
    )
    model_pann_conformer.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_pann_conformer(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    assert hasattr(model_pann_conformer, 'last_shared_features'), "Model is missing last_shared_features attribute!"
    print(f"last_shared_features shape: {model_pann_conformer.last_shared_features.shape} (Expected: [{batch_size}, 256])")
    print("Success!\n")

    # 5. Test SupervisedContrastiveLoss computations
    print("--- Testing SupervisedContrastiveLoss ---")
    contrastive_criterion = SupervisedContrastiveLoss(temperature=0.07)
    features_a = torch.randn(batch_size, 256)
    
    # Case A: Batch with duplicate patient IDs (valid positive pairs exist)
    patient_ids_a = torch.tensor([101, 102, 101, 102], dtype=torch.long)
    loss_a = contrastive_criterion(features_a, patient_ids_a)
    print(f"Supervised Contrastive Loss (with duplicates): {loss_a.item():.4f} (Expected: non-zero finite value)")
    assert loss_a.item() > 0, "Loss should be positive!"
    
    # Case B: Batch with all unique patient IDs (no positive pairs exist)
    patient_ids_b = torch.tensor([101, 102, 103, 104], dtype=torch.long)
    loss_b = contrastive_criterion(features_a, patient_ids_b)
    print(f"Supervised Contrastive Loss (all unique patients): {loss_b.item():.4f} (Expected: 0.0000)")
    assert loss_b.item() == 0.0, f"Loss should be exactly 0.0 when no duplicate patients exist in batch, got {loss_b.item()}!"
    print("Success!\n")

    # 6. Test DynamicMultiTaskLoss weighting parameters optimization
    print("--- Testing DynamicMultiTaskLoss (Homoscedastic Uncertainty) ---")
    dynamic_loss = DynamicMultiTaskLoss(num_tasks=3)
    print(f"Initial log_vars parameters: {dynamic_loss.log_vars.data}")
    
    # Dummy active losses: cycle_loss, pathology_loss, contrastive_loss
    l_cycle = torch.tensor(1.5, requires_grad=True)
    l_path = torch.tensor(0.8, requires_grad=True)
    l_contr = torch.tensor(2.1, requires_grad=True)
    
    # Setup optimizer to include both loss parameters and model params
    optimizer = optim.Adam(dynamic_loss.parameters(), lr=0.1)
    
    # Forward pass through dynamic loss weighting
    combined_loss = dynamic_loss([l_cycle, l_path, l_contr])
    print(f"Combined Dynamic Loss: {combined_loss.item():.4f}")
    
    # Backward pass and optimization step
    optimizer.zero_grad()
    combined_loss.backward()
    optimizer.step()
    
    print(f"Updated log_vars parameters after backward pass: {dynamic_loss.log_vars.data}")
    assert not torch.allclose(dynamic_loss.log_vars.data, torch.zeros(3)), "Parameters log_vars should have updated after optimization!"
    print("Success!\n")

if __name__ == "__main__":
    test_models_and_losses()
