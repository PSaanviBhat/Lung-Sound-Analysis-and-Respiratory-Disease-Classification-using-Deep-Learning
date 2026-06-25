# Implementation Plan — SOTA Loss Function Upgrade (Phase 14)

This document details the implementation plan for Phase 14: Loss Function Upgrade. We will introduce multiclass Focal Loss and Label Smoothing into our SOTA training pipeline inside the `src/sota/` subdirectory.

---

## User Review Required

> [!NOTE]
> **Mathematical Compatibility of SOTA loss modules**:
> 1. **Label Smoothing**: Instead of predicting hard one-hot targets (e.g. $[1.0, 0.0, 0.0, 0.0]$), we smooth them to prevent model overconfidence:
>    $$y_{\text{smoothed}} = y \cdot (1 - \epsilon) + \frac{\epsilon}{C}$$
>    where $\epsilon = 0.1$ (smoothing factor) and $C = 4$ (number of classes).
> 2. **Focal Loss**: To focus training on hard/rare classes (e.g., crackle/wheeze), we down-weight easy "Normal" samples using a focal scaling factor $(1 - p_c)^\gamma$.
> 3. **Composition (Mixup + Label Smoothing + Focal Loss)**:
>    If Mixup is active, the soft mixed targets $y_{\text{mix}}$ will first be smoothed using the Label Smoothing formula, and then the Focal Loss will be computed over these smoothed soft targets:
>    $$L = -\sum_{c=1}^{C} y_{\text{smoothed}, c} \cdot (1 - p_c)^\gamma \cdot \log(p_c)$$
>    This composition is mathematically robust and runs fully on the GPU.

---

## Open Questions

None. The mathematical composition of Mixup, Label Smoothing, and Focal Loss is standard and resolves class imbalance and overfitting simultaneously.

---

## Proposed Changes

### [Component: SOTA Training Pipeline]

#### [MODIFY] [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py)
*   **Implement `FocalLoss` PyTorch Module**:
    *   Create a custom `FocalLoss(nn.Module)` class that handles both 1D integer targets and 2D soft probability targets (enabling composition with Mixup and Label Smoothing).
*   **Integrate Label Smoothing & Focal Loss**:
    *   Add `--focal_gamma` (default 2.0, set to 0.0 for standard cross-entropy) and `--label_smoothing` (default 0.1, set to 0.0 to disable) arguments to the argument parser.
    *   Update `train_epoch` to:
        1. Apply label smoothing to targets.
        2. Instantiate the model loss using the new `FocalLoss` module.
        3. Optimize parameters against the joint loss.

---

## Verification Plan

### Automated Tests
1.  **Focal Loss Math & Shape Test**: Run a verification script `scratch/test_focal_loss.py` to confirm:
    *   `FocalLoss` outputs correct loss values for toy examples (higher loss for wrong/hard predictions, lower loss for correct/easy predictions).
    *   Supports input shapes of $(B, 4)$ for logits and targets (soft targets).
2.  **Pipeline Verification Run**: Run a 3-epoch dry training sweep with both Focal Loss ($\gamma = 2.0$) and Label Smoothing ($\epsilon = 0.1$) enabled to verify epoch duration and saved checkpoints:
    ```bash
    python src/sota/run_experiments.py --model cnn --config D --epochs 3 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0
    ```

### Manual Verification
1.  **Log Inspection**: Inspect the training CLI output to confirm that train/val losses decrease and the model evaluates successfully on the test split.
