# Walkthrough: SOTA Codebase Implementation (Phases 1-4)

This document walks through the modifications completed in all 4 SOTA phases of the publication roadmap and lists the terminal commands you can execute to run and verify the changes.

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
    *   `SupervisedContrastiveLoss` implements patient-invariant InfoNCE loss. It calculates Cosine similarity between feature embeddings in a batch, temperature-scales them, and computes logs over same-patient pairings.
2.  **Features Extraction Hooking**:
    *   Modified all four models in [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py) to save their pre-classification embeddings to `self.last_shared_features` during forward passes. This avoids breaking model return shapes.
3.  **Dataloader Updates for Patient IDs**:
    *   Updated `LungSoundDatasetSOTA` in [dataset.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/dataset.py) to return the integer `patient_id` as the last element of the dataset tuple.
    *   Modified all batch unpackings in [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py) to use explicit indexing, keeping backward compatibility intact.
4.  **CLI Argument for Contrastive Weight**:
    *   Added the `--contrastive_weight` CLI parameter in [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py). If `--contrastive_weight > 0.0`, Supervised Contrastive Loss is computed on patient IDs and combined with the classification loss during training.

---

## Phase 4 Accomplishments: Dynamic Loss Weighting

1.  **Homoscedastic Uncertainty Weighting**:
    *   Implemented `DynamicMultiTaskLoss` in [loss.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/loss.py). It registers learnable parameters representing task log-variances (uncertainty). It scales active losses dynamically using task precisions and adds regularizers to keep the optimization balanced.
2.  **Optimizer & Training Loop Integration**:
    *   Modified [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py) to register the dynamic loss module's parameters to the Adam optimizer, allowing log-variances to be optimized via backpropagation.
    *   Updated `train_epoch` to dynamically aggregate cycle loss, pathology loss, and contrastive loss based on model/task configurations and optimize them as a single dynamically-weighted loss term.
3.  **CLI Argument for Dynamic Weighting**:
    *   Added the `--dynamic_loss` CLI flag to [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py). When enabled, checkpoint filenames automatically append `_dynloss` to prevent conflicts.
4.  **Unit Tests Expansion**:
    *   Updated [test_fusion.py](file:///d:/Internship%20'26/Lung%20Disease/scratch/test_fusion.py) to verify that `DynamicMultiTaskLoss` computes gradients, updates its learnable parameters during backpropagation, and changes value after optimization steps.

---

## Commands to Run

> [!NOTE]
> Run all commands from the root directory of the workspace (`d:\Internship '26\Lung Disease`).

### 1. Run SOTA Unit Tests (Shape, Hook, Contrastive, and Dynamic Loss Verification)
To verify that all model structures, loss computations, and learnable dynamic parameter updates compile and function correctly:
```bash
python scratch/test_fusion.py
```

### 2. Run Cross-Attention + Conformer + Contrastive + Dynamic Loss Weighting (SeqLen 32)
Trains the Conformer model using a 32-step sequence length, patient-invariant contrastive regularization, and learnable homoscedastic uncertainty loss balancing:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer --contrastive_weight 0.1 --dynamic_loss
```

### 3. Run Multi-Task Cross-Attention + Conformer + Contrastive + Dynamic Loss Weighting (SeqLen 32)
Trains the multi-task model with learnable homoscedastic loss weighting, dynamically balancing cycle classification, pathology diagnosis, and contrastive regularization:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer --contrastive_weight 0.1 --dynamic_loss
```

### 4. Run Multi-Task Cross-Attention + BiLSTM + Contrastive + Dynamic Loss Weighting (SeqLen 32)
Trains the BiLSTM hybrid model with learnable homoscedastic loss weighting, dynamically balancing cycle classification, pathology diagnosis, and contrastive regularization:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --contrastive_weight 0.1 --dynamic_loss
```
