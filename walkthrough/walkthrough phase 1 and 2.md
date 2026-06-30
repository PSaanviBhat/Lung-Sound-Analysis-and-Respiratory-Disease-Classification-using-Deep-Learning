# Walkthrough: Phase 1 & Phase 2 Implementation

This document walks through the modifications completed in Phase 1 (Cross-Attention Fusion) and Phase 2 (High-Resolution Temporal Modeling) of the publication roadmap, and lists the terminal commands you can execute to run and verify the changes.

---

## Phase 1 Accomplishments: Cross-Attention Fusion

1.  **Multi-Branch Encoder Modules**:
    *   Implemented `ResNetBranch` and `MultiBranchResNet` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py). This splits stacked inputs of shape `(B, in_channels, H, W)` into `in_channels` separate single-channel tensors of shape `(B, 1, H, W)`. Each channel is processed by its own ResNet-18 feature extractor.
    *   Implemented `PANNSBranch` and `MultiBranchPANNs` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to apply the same multi-branch feature separation for the PANNs Cnn14 backbone.
2.  **Cross-Attention Fusion Layer**:
    *   Implemented `CrossAttentionFusion` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py). This class learns branch embeddings (to identify Mel, CQT, and CWT features) and spatial position embeddings, concatenating all branches to shape `(B, num_branches * 16, C)` and performing multi-head self-attention.
3.  **Model Integrations**:
    *   Updated `BaselineCNNResNet`, `CNNLSTMResNet`, `BaselineCNNPANN`, and `CNNLSTMPANN` in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to support the `cross_attention` parameter.

---

## Phase 2 Accomplishments: High-Resolution Temporal Modeling

1.  **High-Resolution Sequence Projection**:
    *   Updated sequence-level hybrid models (`CNNLSTMResNet` and `CNNLSTMPANN`) to accept custom sequence lengths (e.g. `--sequence_len 32` or `--sequence_len 64`). 
    *   When `sequence_len > 4`, the model projects the flattened spatial grid dimension ($H \times W = 16$) using a linear layer `self.time_project = nn.Linear(16, sequence_len)`. This yields an expanded temporal sequence of shape `(B, sequence_len, C)`, resolving the 4-step sequence bottleneck.
    *   Maintained complete backward compatibility: when `sequence_len = 4` (default), the model falls back to the baseline average pooling sequence modeling.
2.  **SOTA Conformer Architecture Integration**:
    *   Implemented `FeedForward`, `ConformerConvModule`, and `ConformerBlock` classes in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py). This provides a state-of-the-art alternative to the BiLSTM, combining Macaron-style feed-forward layers, Multi-Head Self-Attention, and 1D Depthwise Separable Convolutions to learn both local and global temporal dynamics.
3.  **Expanded CLI argument support**:
    *   Added `--sequence_len` and `--conformer` command-line flags to [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py).
    *   Dynamic checkpoint suffix logic now appends `_seqlen{sequence_len}` and `_conformer` to log and model filenames to prevent name collisions.
4.  **Unit Tests Expansion**:
    *   Updated the verification script [test_fusion.py](file:///d:/Internship%20'26/Lung%20Disease/scratch/test_fusion.py) to cover shape assertions and dry-run forwards on the new high-resolution projection (sequence length 32/64) and Conformer modeling.

---

## Commands to Run

> [!NOTE]
> Run all commands from the root directory of the workspace (`d:\Internship '26\Lung Disease`).

### 1. Run Unit Tests (Shape & Execution Verification)
To verify that the high-resolution sequences, Conformer blocks, and cross-attention branches execute correctly:
```bash
python scratch/test_fusion.py
```

### 2. Run Cross-Attention + Conformer (Sequence Length 32)
To train the Conformer hybrid model on the 3-channel Stacked features (Config D) using a 32-step sequence length:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer
```

### 3. Run Multi-Task Cross-Attention + Conformer (Sequence Length 32)
To jointly train cycle classification and pathology diagnosis using Conformer blocks and cross-attention fusion:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --pathology_weight 1.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer
```

### 4. Run Cross-Attention + BiLSTM (Sequence Length 32)
To evaluate the effect of expanding the sequence length from 4 to 32 while retaining the standard BiLSTM:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32
```
