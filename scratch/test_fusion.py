import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sota.models import BaselineCNNResNet, CNNLSTMResNet, BaselineCNNPANN, CNNLSTMPANN
from src.sota.loss import SupervisedContrastiveLoss

def test_models_and_contrastive_loss():
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
    
    # Case A: Batch with duplicate patient IDs (valid positive pairs exist)
    features_a = torch.randn(batch_size, 256)
    # Patients: Patient 101 (2 cycles), Patient 102 (2 cycles)
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

if __name__ == "__main__":
    test_models_and_contrastive_loss()
