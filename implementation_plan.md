# Implementation Plan — SOTA Optimization & Directory Restructuring

This document outlines the organization and advanced optimization roadmap. To prepare for implementing advanced techniques (Focal Loss, Mixup, PANNs backbones, and Multi-task learning), we will first restructure the repository to separate legacy baseline scripts from the new experimental pipeline.

---

## User Review Required

> [!IMPORTANT]
> **Unified `src/` Layout Import Resolution**: Moving code files into subdirectories (`src/baselines/` and `src/experiments/`) changes Python's import context. We must add parent-directory path injections (`sys.path.append`) to the sub-folder scripts so they can import core data modules (like `dataset.py`) from the parent `src/` directory without breaking.
> 
> **Running Commands**: Scripts must always be run from the repository root directory (e.g., `d:\Internship '26\Lung Disease`), which keeps generated metadata, checkpoints, and logs located at the root level.

---

## Proposed Phase-Wise Plan

### Phase 12: Directory Restructuring & Organization (Days 1-2)
Restructure all files into a unified `src/` folder divided into `baselines/` and `experiments/` subfolders.
*   **Target Directory Structure**:
    ```text
    ├── baselines/                         # Legacy SVM, CNN, and CNN-LSTM baselines
    │   ├── train_baseline_svm.py
    │   ├── train_baseline_cnn.py
    │   ├── train_hybrid_model.py
    │   └── evaluate_models.py
    ├── experiments/                       # Active pipeline scripts (runner, curves, models)
    │   ├── models.py
    │   ├── run_experiments.py
    │   ├── run_all_experiments.py
    │   └── plot_comparison_curves.py
    ├── dataset.py                         # Core PyTorch dataset & augmentations
    ├── preprocess.py                      # Resampling & segmentation pipeline
    ├── extract_features.py                # Time-frequency feature extraction
    ```
*   **Tasks**:
    *   Create the directories: `src/`, `src/baselines/`, `src/experiments/`.
    *   Move core utilities to `src/` (`preprocess.py`, `extract_features.py`, `dataset.py`).
    *   Move baseline scripts to `src/baselines/`.
    *   Move experiments and active model code to `src/experiments/`.
    *   Update baseline and experiment scripts to prepend `src/` to `sys.path` dynamically.
    *   Update `.gitignore` to reflect the new paths.
*   **Verification**: Run a validation check of `src/experiments/run_experiments.py --eval_only` to verify all imports and configurations function.

---

### Phase 13: Data Augmentation Enhancements (Days 3-4)
Implement Mixup and waveform-level augmentations to reduce overfitting and smooth decision boundaries.
*   **Tasks**:
    *   **Audio Mixup**: Implement Mixup for stacked feature tensors in the training loop. Mix pairs of inputs $x = \lambda x_1 + (1-\lambda) x_2$ and their labels $y = \lambda y_1 + (1-\lambda) y_2$.
    *   **On-the-Fly Waveform Augmentations**: Implement waveform-level perturbations in `dataset.py` (applied before time-frequency transform):
        *   Random time-shifting ($\pm 100\text{ ms}$).
        *   White Gaussian noise injection (SNR range: $15\text{ dB} - 30\text{ dB}$).
        *   Pitch shifting ($\pm 2$ semitones) and time-stretching ($0.9 - 1.1\times$).
*   **Verification**: Run dataloader test script to verify mixed inputs and check that targets sum to 1.0.

---

### Phase 14: Loss Function Upgrade (Day 5)
Incorporate specialized loss functions to address the severe class imbalance.
*   **Tasks**:
    *   **Multiclass Focal Loss**: Implement Focal Loss to down-weight easy "Normal" samples and scale up focus on rare crackles/wheezes/both classes:
        $$L_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$
    *   **Label Smoothing**: Add label smoothing to target vectors to prevent model overconfidence and improve decision boundary calibration.
*   **Verification**: Compare a 5-epoch run using standard cross-entropy vs. Focal Loss to confirm that the model predicts rare classes more frequently.

---

### Phase 15: Pretrained Audio Backbones Integration (Days 6-7)
Replace the ImageNet-pretrained ResNet-18 with an audio-pretrained feature extractor.
*   **Tasks**:
    *   **PANNs (Pretrained Audio Neural Networks)**: Integrate a CNN14 or MobileNetV2 backbone pretrained on AudioSet.
    *   Modify input convolutional layers to accept custom-channeled stacked tensors (1, 2, or 3 channels depending on ablation configuration).
    *   Align sequence representation output dimensionality for the BiLSTM sequence modeling layer.
*   **Verification**: Run forward-pass validation checking that output tensor dimensions align with targets.

---

### Phase 16: Multi-Task Learning Framework (Days 8-9)
Leverage clinical metadata (patient diagnoses) to guide acoustic feature representation.
*   **Tasks**:
    *   Update `dataset.py` to output both cycle-level labels (Normal/Crackle/Wheeze/Both) and patient-level pathology labels (COPD, Pneumonia, URTI, Healthy).
    *   Add an auxiliary classification head to `models.py` for patient diagnosis.
    *   Modify the training loss to be a weighted combination of both tasks:
        $$L_{\text{total}} = L_{\text{cycle}} + \beta L_{\text{pathology}}$$
*   **Verification**: Confirm that multi-task heads train jointly and optimize together.

---

## Verification Plan

### Automated Tests
1.  **Restructured Import Test**: Run imports check for all scripts in their new subdirectories to verify there are no `ModuleNotFoundError` crashes.
2.  **Mixup Shape Test**: Verify that the mixed batch tensor matches batch dimensions $(B, C, 128, 128)$ and labels are soft-target distributions $(B, 4)$.
3.  **PANNs Backbone Integration Test**: Pass a dummy tensor of shape $(2, 3, 128, 128)$ through the new CNN14-LSTM hybrid model and check that it outputs shape $(2, 4)$ without crashing.

### Manual Verification
1.  **Validation Matrix Review**: Manually inspect the final ablation table to confirm if the combination of audio pretraining and Focal Loss elevates the overall ICBHI Score past $55\%$ under Stacked (Config D).
