# Walkthrough: Phase 1, Phase 2, & Phase 3 Implementation

This document walks through the modifications completed in Phase 1 (Cross-Attention Fusion), Phase 2 (High-Resolution Temporal Modeling), and Phase 3 (Patient-Invariant Contrastive Loss) of the publication roadmap, and lists the terminal commands you can execute to run and verify the changes.

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

---

## Phase 3 Accomplishments: Patient-Invariant Contrastive Loss

1.  **Modularized Loss Implementations**:
    *   Created [loss.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/loss.py) containing `FocalLoss` and `SupervisedContrastiveLoss`.
    *   `SupervisedContrastiveLoss` implements patient-invariant InfoNCE loss. It calculates Cosine similarity between feature embeddings in a batch, temperature-scales them, and computes logs over same-patient pairings. It is mathematically stable and dynamically skips rows that do not have duplicate patient representations in the batch.
2.  **Features Extraction Hooking**:
    *   Modified all four models in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to save their pre-classification embeddings to `self.last_shared_features` during forward passes. This avoids breaking model return shapes.
3.  **Dataloader Updates for Patient IDs**:
    *   Updated `LungSoundDatasetSOTA` in [dataset.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/dataset.py) to return the integer `patient_id` as the last element of the dataset tuple.
    *   Modified all batch unpackings in [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py) (training, validation, and test loops) to use explicit indexing (`batch[0]`, `batch[1]`, etc.), keeping backward compatibility intact.
4.  **CLI Argument for Contrastive Weight**:
    *   Added the `--contrastive_weight` CLI parameter in [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py). If `--contrastive_weight > 0.0`, Supervised Contrastive Loss is computed on patient IDs and combined with the classification loss during training. Suffixes now append `_contrast{weight}` to log and checkpoint paths.
5.  **Unit Tests Expansion**:
    *   Updated the verification script [test_fusion.py](file:///d:/Internship%20'26/Lung%20Disease/scratch/test_fusion.py) to assert `last_shared_features` shapes on model forwards, and verify `SupervisedContrastiveLoss` output behavior under duplicate-patient vs. unique-patient scenarios.

---

## Commands to Run

> [!NOTE]
> Run all commands from the root directory of the workspace (`d:\Internship '26\Lung Disease`).

### 1. Run Unit Tests (Shape & Contrastive loss Verification)
To verify that high-resolution sequences, Conformer blocks, cross-attention branches, patient ID returns, and Supervised Contrastive Loss execute correctly:
```bash
python scratch/test_fusion.py
```

### 2. Run Cross-Attention + Conformer + Patient-Invariant Contrastive Loss (SeqLen 32)
Trains the Conformer hybrid model on 3-channel Stacked features (Config D) using a 32-step sequence length and a Patient-Invariant Contrastive Loss weight of 0.1:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer --contrastive_weight 0.1
```

### 3. Run Multi-Task Cross-Attention + Conformer + Patient-Invariant Contrastive Loss (SeqLen 32)
Jointly trains cycle classification, pathology diagnosis, and patient-invariant contrastive regularization:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --pathology_weight 1.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer --contrastive_weight 0.1
```

### 4. Run Cross-Attention + BiLSTM + Patient-Invariant Contrastive Loss (SeqLen 32)
Trains the BiLSTM hybrid model with cross-attention, sequence length 32, and patient-invariant contrastive loss:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --contrastive_weight 0.1
```
