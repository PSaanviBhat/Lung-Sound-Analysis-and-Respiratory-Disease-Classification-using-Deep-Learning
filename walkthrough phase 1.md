# Walkthrough: Phase 1 (Cross-Attention Fusion) Implementation

This document walks through the modifications completed in Phase 1 of the publication roadmap and lists the terminal commands you can execute to run and verify the changes.

---

## What was Accomplished

1.  **Multi-Branch Encoder Modules**:
    *   Implemented `ResNetBranch` and `MultiBranchResNet` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py). This splits stacked inputs of shape `(B, in_channels, H, W)` into `in_channels` separate single-channel tensors of shape `(B, 1, H, W)`. Each channel is processed by its own ResNet-18 feature extractor.
    *   Implemented `PANNSBranch` and `MultiBranchPANNs` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to apply the same multi-branch feature separation for the PANNs Cnn14 backbone.
2.  **Cross-Attention Fusion Layer**:
    *   Implemented `CrossAttentionFusion` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py). This class:
        *   Learns branch embeddings (to identify Mel, CQT, and CWT features) and spatial position embeddings.
        *   Concatenates sequences from all branches to a shape of `(B, num_branches * 16, C)`.
        *   Applies a Multi-Head Self-Attention layer across all combined spatial locations and branch domains.
        *   Splits features back to branches and projects them via a linear layer back to a single feature map of shape `(B, C, 4, 4)`.
3.  **Model Integrations**:
    *   Updated `BaselineCNNResNet`, `CNNLSTMResNet`, `BaselineCNNPANN`, and `CNNLSTMPANN` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to support the `cross_attention` parameter. When enabled, models default to their multi-branch encoder + cross-attention fusion block rather than naive early channel stacking.
4.  **CLI argument support**:
    *   Updated `run_experiments.py` in [src/sota/](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py) to accept the `--cross_attention` command-line flag. 
    *   Modified checkpoint naming to append `_crossattn` to avoid overwriting your baseline checkpoints (e.g., `hybrid_config_D_resnet_sota_crossattn.pth`).
5.  **Verification Script**:
    *   Created `test_fusion.py` in [scratch/](file:///d:/Internship%20'26/Lung%20Disease/scratch/test_fusion.py) to mock inputs and perform shape checking on all models.

---

## Commands to Run

> [!NOTE]
> Run all commands from the root directory of the workspace (`d:\Internship '26\Lung Disease`).

### 1. Run Unit Tests (Shape & Execution Verification)
To verify that all new models compile, initialize weights, and perform the forward pass correctly:
```bash
python scratch/test_fusion.py
```

### 2. Run SOTA Single-Task ResNet-18 Training with Cross-Attention
To train the hybrid CNN-LSTM model under feature Config D with Mixup, Focal Loss, and Cross-Attention Fusion enabled:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention
```

### 3. Run SOTA Multi-Task Joint Training with Cross-Attention
To train the joint cycle classification and pathology diagnosis model with Cross-Attention Fusion enabled:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --pathology_weight 1.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention
```
