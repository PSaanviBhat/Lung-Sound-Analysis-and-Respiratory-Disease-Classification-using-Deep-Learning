import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
from sota.models import BaselineCNNPANN, CNNLSTMPANN

def test_models():
    print("Testing shapes with pretrained=False...")
    for in_channels in [1, 2, 3]:
        print(f"\n--- Channels: {in_channels} ---")
        x = torch.randn(4, in_channels, 128, 128)
        
        # BaselineCNNPANN
        cnn = BaselineCNNPANN(in_channels=in_channels, num_classes=4, pretrained=False)
        out_cnn = cnn(x)
        print(f"BaselineCNNPANN output shape: {out_cnn.shape}")
        assert out_cnn.shape == (4, 4), f"Expected (4, 4), got {out_cnn.shape}"
        
        # CNNLSTMPANN
        hybrid = CNNLSTMPANN(in_channels=in_channels, num_classes=4, pretrained=False)
        out_hybrid = hybrid(x)
        print(f"CNNLSTMPANN output shape: {out_hybrid.shape}")
        assert out_hybrid.shape == (4, 4), f"Expected (4, 4), got {out_hybrid.shape}"

    print("\nAll shape tests passed successfully with pretrained=False!")

    print("\nTesting weights download and loading (pretrained=True) for in_channels=3...")
    cnn_pt = BaselineCNNPANN(in_channels=3, num_classes=4, pretrained=True)
    x = torch.randn(2, 3, 128, 128)
    out_cnn_pt = cnn_pt(x)
    print(f"Pretrained BaselineCNNPANN output shape: {out_cnn_pt.shape}")
    assert out_cnn_pt.shape == (2, 4)
    
    hybrid_pt = CNNLSTMPANN(in_channels=3, num_classes=4, pretrained=True)
    out_hybrid_pt = hybrid_pt(x)
    print(f"Pretrained CNNLSTMPANN output shape: {out_hybrid_pt.shape}")
    assert out_hybrid_pt.shape == (2, 4)
    print("\nAll pretrained load tests passed successfully!")

if __name__ == '__main__':
    test_models()
