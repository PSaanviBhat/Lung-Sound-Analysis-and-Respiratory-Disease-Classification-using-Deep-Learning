# Implementation Plan — SOTA Optimization & Directory Restructuring

This document outlines the organization and advanced optimization roadmap. To prepare for implementing advanced techniques, we will keep all new SOTA developments separate from our previous baseline and experiments.

---

## User Review Required

> [!IMPORTANT]
> **Performance Bottleneck of Waveform Augmentations**:
> Profiling shows that loading raw audio and computing Mel + CQT + CWT (specifically the Continuous Wavelet Transform) on-the-fly takes **~627 ms per sample**. Performing this during training would increase epoch duration from 25 seconds to **38.5 minutes per epoch**, which is computationally intractable.
> 
> **Proposed Solution (Spectrogram-Level Equivalents)**:
> To maintain fast training speeds (under 30s/epoch), we propose applying mathematically equivalent or highly correlated augmentations directly to the 3-channel pre-computed feature maps using fast GPU/PyTorch operations:
> 1. **Time-Shifting**: Implement via random horizontal rolling (circular shift) of the feature maps along the time axis (width).
> 2. **Pitch-Shifting**: Implement via random vertical rolling of the feature maps along the frequency axis (height).
> 3. **White Noise Injection**: Inject Gaussian noise directly into the feature maps.
> 4. **SpecAugment**: Continue using random frequency/time masking (zeroing blocks).
> 5. **Mixup**: Linearly interpolate input feature tensors and soft labels in the training loop.
> 
> **Separate SOTA Subdirectory**:
> To satisfy the request to keep files separate and preserve baseline access, all new SOTA code will live in a new subdirectory: `src/sota/`. The previous `src/baselines/` and `src/experiments/` folders will remain untouched.

---

## Open Questions

None. The folder structure separation cleanly addresses model versioning, and the spectrogram-level augmentations avoid training bottlenecks.

---

## Proposed Changes

### [Component: SOTA Pipeline]
We will create a new SOTA pipeline folder `src/sota/` containing versions of the dataset, runner, and models optimized for advanced training.

#### [NEW] [dataset.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/dataset.py)
*   Create a new dataset module containing:
    *   `LungSoundDatasetSOTA`: Loads pre-computed features from `metadata.csv`.
    *   `SOTAAugmentation`: A pipeline of PyTorch tensor-based augmentations:
        *   **Time Shift**: Randomly rolls the tensor along the width axis (time dimension) by up to $\pm 10$ pixels.
        *   **Frequency Shift (Pitch)**: Randomly rolls the tensor along the height axis (frequency dimension) by up to $\pm 2$ pixels.
        *   **Gaussian Noise**: Adds standard normal noise to the tensor scaled by a random factor.
        *   **SpecAugment**: Applies frequency and time masking blocks.

#### [NEW] [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py)
*   Create a new training runner containing:
    *   **Mixup Implementation**: In the training loop, mix batch items:
        $$\lambda \sim \text{Beta}(\alpha, \alpha)$$
        $$x_{\text{mix}} = \lambda x_1 + (1 - \lambda) x_2$$
        $$y_{\text{mix}} = \lambda y_1 + (1 - \lambda) y_2$$
    *   Modify `train_epoch` to compute cross-entropy loss against soft targets:
        $$L = -\sum y_{\text{mix}} \log(p)$$
    *   Option to toggle Mixup on/off to perform ablation comparisons.

---

## Verification Plan

### Automated Tests
1.  **Augmentation Shape & Target Test**: Run a verification script `scratch/test_sota_augmentations.py` to confirm:
    *   Dataloader outputs inputs of shape $(B, C, 128, 128)$ and soft targets of shape $(B, 4)$ when Mixup is enabled.
    *   Outputs match standard targets of shape $(B,)$ when Mixup is disabled.
2.  **Pipeline Verification Run**: Run a 3-epoch dry training run using the new SOTA runner to verify epoch duration, loss reduction, and stable checkpoint saving:
    ```bash
    python src/sota/run_experiments.py --model cnn --config D --epochs 3 --batch_size 32
    ```

### Manual Verification
1.  **Visual Check of Augmentations**: Save augmented spectrogram patches to disk as images to visually confirm time-rolling, frequency-rolling, noise injection, and masking.
