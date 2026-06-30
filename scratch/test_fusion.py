import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sota.models import BaselineCNNResNet, CNNLSTMResNet, BaselineCNNPANN, CNNLSTMPANN

def test_models():
    # Test parameters
    batch_size = 2
    in_channels = 3
    num_classes = 4
    height, width = 128, 128
    
    # Mock input (stacked feature maps: Mel, CQT, CWT)
    mock_input = torch.randn(batch_size, in_channels, height, width)
    print(f"Mock input shape: {mock_input.shape}\n")
    
    # 1. Test BaselineCNNResNet with cross-attention
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
    print("Success!\n")
    
    # 2. Test CNNLSTMResNet with cross-attention
    print("--- Testing CNNLSTMResNet (cross_attention=True) ---")
    model_resnet_hybrid = CNNLSTMResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True
    )
    model_resnet_hybrid.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_resnet_hybrid(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    print("Success!\n")
    
    # 3. Test BaselineCNNPANN with cross-attention
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
    print("Success!\n")
    
    # 4. Test CNNLSTMPANN with cross-attention
    print("--- Testing CNNLSTMPANN (cross_attention=True) ---")
    model_pann_hybrid = CNNLSTMPANN(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        multitask=True,
        cross_attention=True
    )
    model_pann_hybrid.eval()
    with torch.no_grad():
        cycle_logits, path_logits = model_pann_hybrid(mock_input)
    print(f"Cycle logits shape: {cycle_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Pathology logits shape: {path_logits.shape} (Expected: [{batch_size}, 3])")
    print("Success!\n")

if __name__ == "__main__":
    test_models()
