# Implementation Plan — Lung Sound Analysis & Respiratory Disease Classification

This document outlines a phase-wise, 14-day implementation plan tailored for execution on a consumer laptop equipped with an **NVIDIA RTX 3050 6GB GPU**, running PyTorch, Librosa, and associated scientific Python packages.

---

## User Review Required

> [!IMPORTANT]
> **GPU VRAM Constraint (6GB)**: We must avoid large neural network backbones (such as ResNet-50/101 or Vision Transformers) and keep training batch sizes to $16$ or $32$ to prevent Out-Of-Memory (OOM) crashes.
> 
> **Subject-Level Split**: To ensure clinical validity, the dataset split *must* be structured at the subject level. Do not perform random cycle splits, as this leaks patient data and invalidates test results.
> 
> **Offline Feature Cache**: Computing Continuous Wavelet Transforms (CWT) during the training dataloader loop is computationally intensive. We must pre-compute and serialize all features to disk as `.pt` tensors to speed up training.

---

## Proposed Phase-Wise Plan

### Phase 1: Environment Setup & Raw Data Ingestion (Days 1–2)
Establish a robust, reproducible Python virtual environment and set up the dataset repository.
*   **Tasks**:
    *   Set up a Python 3.10 virtual environment and install PyTorch (with CUDA 11.8/12.1 support), `librosa`, `soundfile`, `pywt` (PyWavelets), `scikit-learn`, `matplotlib`, and `pandas`.
    *   Download and extract the **ICBHI 2017 Respiratory Sound Database**.
    *   Write a parser to link audio `.wav` files with their cycle annotation `.txt` files (containing onset, offset, crackle, and wheeze markers) and subject metadata.
*   **Laptop Optimization**: Keep raw data compressed; write clean path mapping scripts to avoid copying files repeatedly.

---

### Phase 2: Preprocessing & Segmentation Pipeline (Days 3–4)
Filter noise from the raw waveforms and segment continuous recordings into individual breathing cycles.
*   **Tasks**:
    *   Implement standard resampling to a uniform sampling rate (e.g., $16\,\text{kHz}$ or $4\,\text{kHz}$ depending on high-frequency analysis requirements).
    *   Apply a 4th-order Butterworth bandpass filter ($100\,\text{Hz} - 2000\,\text{Hz}$) to eliminate muscle movement and ambient noise.
    *   Segment raw audio into individual respiratory cycles using the parsed timestamps.
    *   Pad/truncate all segmented cycles to a uniform duration of $3.0\,\text{s}$ (using zero-padding or cyclic replication).
*   **Laptop Optimization**: Use vectorization in NumPy/Scipy to speed up CPU filtering.

---

### Phase 3: Multi-Branch Feature Extraction & Offline Serialization (Days 5–6)
Extract distinct time-frequency representations tailored for different disease classes.
*   **Tasks**:
    *   **Branch 1 (Mel Spectrogram)**: Extract standard time-frequency maps (e.g., 128 Mel bands).
    *   **Branch 2 (Constant-Q Transform - CQT)**: Extract log-frequency representations, optimizing low-frequency resolution for continuous wheeze harmonics.
    *   **Branch 3 (Continuous Wavelet Transform - CWT)**: Extract scale-based scalograms using a Morlet or Mexican Hat mother wavelet, optimizing time resolution for transient crackles.
    *   Resize all extracted feature matrices to a standard grid size (e.g., $128 \times 128$) and stack them into a 3-channel tensor of shape $(3, 128, 128)$.
    *   Serialize these tensors directly to disk as `.pt` binary files grouped by patient ID.
*   **Laptop Optimization**: Pre-computing features offline avoids redundant CPU transform calculations during GPU training, accelerating training speeds.

---

### Phase 4: Custom PyTorch Data Loading & Augmentation (Days 7–8)
Develop clean dataloaders that handle stacked tensor inputs and class imbalance.
*   **Tasks**:
    *   Write a custom `Dataset` class that reads the pre-computed `.pt` tensors from disk.
    *   Implement **Subject-Level Split** partitioning (60% Train, 20% Val, 20% Test) using patient ID codes.
    *   Add training augmentations directly in PyTorch: random frequency masking, time masking (SpecAugment), and adding low-amplitude White Gaussian Noise.
    *   Implement a weighted data sampler (`WeightedRandomSampler`) or compute loss weights to address the severe dataset class imbalance (e.g., the high volume of COPD samples relative to other pathologies).

---

### Phase 5: Baseline Model Development (Days 9–10)
Establish classical and deep learning baseline models to measure progress.
*   **Tasks**:
    *   **Baseline 1 (SVM)**: Flatten CQT/MFCC maps into 1D mean/variance vectors and train a Support Vector Machine classifier.
    *   **Baseline 2 (CNN)**: Adapt a standard ResNet-18 model. Modify the input layer to accept 3-channel stacked tensors and change the output layer to class-wise probability distributions.
    *   Train both baselines, track their training curves, and save their best validation weights.
*   **Laptop Optimization**: Enable mixed-precision training (`torch.cuda.amp`) to reduce VRAM consumption and speed up training on the RTX 3050.

---

### Phase 6: Proposed Hybrid Spatio-Temporal Model (Days 11–12)
Implement the core neural network combining spatial feature extraction with sequence modeling.
*   **Tasks**:
    *   Build a hybrid network (CNN-LSTM) where the ResNet-18 model acts as a feature extractor (removing its final classification head).
    *   Project the 2D spatial feature maps along the time axis to create a sequence of vectors.
    *   Pass this sequence through a Bidirectional LSTM (BiLSTM) layer to capture temporal transitions in the respiration cycle.
    *   Add a fully connected output layer to predict both anomaly categories (Healthy vs. Crackle vs. Wheeze vs. Both) and disease pathologies.
*   **Laptop Optimization**: Freeze the early convolutional layers of the ResNet-18 backbone (Transfer Learning) to reduce backpropagation overhead and protect VRAM.

---

### Phase 7: Diagnostic Metrics Evaluation & Comparison (Days 13–14)
Evaluate model performance against project requirements.
*   **Tasks**:
    *   Calculate confusion matrices, class-wise F1-scores, Sensitivity, and Specificity.
    *   Compute the official **ICBHI Score** ($S = \frac{Se + Sp}{2}$) to compare performance with literature standards.
    *   Write an inference script that measures processing latency for a single raw `.wav` file to assess real-time viability.
    *   Consolidate all findings, comparison charts, and training history logs into the final walkthrough report.

---

## Verification Plan

### Automated Tests
1.  **Pipeline Verification**: Run a dry-run script testing resampling, filtering, and segmentation on a single audio file to verify output shape consistency.
2.  **Model Dry Run**: Pass a dummy tensor of shape $(1, 3, 128, 128)$ through the CNN-LSTM model to verify that output dimensions match target classes.
3.  **VRAM Profiling**: Run a 1-epoch training loop with batch size 32 on the GPU, monitoring memory utilization via `nvidia-smi` to ensure memory consumption remains under 5GB.

### Manual Verification
1.  **Audio Quality Verification**: Plot raw vs. preprocessed waveforms and listen to the filtered segments to confirm that muscle/friction noise is suppressed.
2.  **Confusion Matrix Verification**: Manually review the confusion matrix to ensure the model successfully detects minority classes (e.g., Pneumonia) and does not default to classifying all inputs as COPD.
