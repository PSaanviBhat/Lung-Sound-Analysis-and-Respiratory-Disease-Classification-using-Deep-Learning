# Lung Sound Analysis and Respiratory Disease Classification using SOTA Deep Learning

This repository contains the source code, experimental protocols, and documentation for an automated, clinical-grade respiratory disease screening and classification pipeline. The system utilizes advanced time-frequency representations, multi-branch spatial encoders, Cross-Attention Late Fusion, Conformer sequence networks, Patient-Invariant Contrastive regularization, and learnable Homoscedastic Multi-Task loss balancing to classify chest auscultation recordings under strict patient-wise partitioning.

---

## 1. Project Overview & Clinical Objectives

Traditional chest auscultation using acoustic stethoscopes is highly subjective, exhibits low inter-observer agreement, and is limited by clinicians' auditory acuity. This project automates the diagnostic workflow using the **ICBHI 2017 Respiratory Sound Database**. 

### Core Objectives:
1.  **Acoustic Denoising & Segmentation**: Resample raw recordings to 4,000 Hz, apply a 4th-order bandpass Butterworth filter (100 Hz - 1,999 Hz) to eliminate cardiac and muscle friction noise, and segment signals into uniform 3.0s respiration cycles.
2.  **Multi-Branch Representation**: Stack Mel Spectrograms (envelope), Constant-Q Transforms (CQT, resolving wheezing harmonics), and Continuous Wavelet Transforms (CWT, capturing transient crackle clicks) into a 3-channel time-frequency tensor.
3.  **Validation Rigor (No Data Leakage)**: Implement a strict, leakage-free **Patient-Aware Split** (complete patient separation between train and test sets) to prevent models from memorizing patient-specific acoustics.
4.  **Clinical Pathology Diagnosis**: Jointly train models to classify cycle-level anomalies (Normal, Crackle, Wheeze, Both) and diagnose patient-level chronic/acute pathology (COPD, URTI, Healthy) at low latency.

---

## 2. Directory Structure

```text
├── checkpoints/                         # Saved weights (.pth) and thresholds (.npy) (gitignored)
├── data/                                # Local directory for raw datasets (gitignored)
│   ├── ICBHI_final_database/            # Raw .wav recordings and cycle annotations (.txt)
│   ├── ICBHI_Challenge_diagnosis.txt    # Mapping of Patient IDs to clinical diagnoses
│   └── ICBHI_challenge_train_test.txt   # Official train/test split definitions
├── evaluation_results/                  # Compiled evaluation summaries and plots
├── processed_data/                      # Segmented 3.0s cycle WAV files (gitignored)
├── processed_features/                  # Stacked 3-channel PyTorch feature tensors (gitignored)
├── scratch/                             # Python helpers, manuscripts, and reports
│   ├── generate_journal_paper.py        # Generates formatted journal paper Word document
│   └── test_fusion.py                   # Custom verification and shape test script
├── src/                                 # Project source code
│   ├── baselines/                       # Legacy baseline classifiers (SVM, CNN, CNN-LSTM)
│   │   ├── train_baseline_svm.py
│   │   ├── train_baseline_cnn.py
│   │   ├── train_hybrid_model.py
│   │   └── evaluate_models.py
│   ├── experiments/                     # Baseline ablation sweep scripts (Configs A-D)
│   │   ├── models.py
│   │   ├── run_experiments.py
│   │   ├── run_all_experiments.py
│   │   └── plot_comparison_curves.py
│   ├── sota/                            # SOTA Methodological Pipeline (Phases 1-4)
│   │   ├── dataset.py                   # Dataloader returning patient IDs
│   │   ├── loss.py                      # FocalLoss, ContrastiveLoss, DynamicMultiTaskLoss
│   │   ├── models.py                    # MultiBranch, CrossAttention, Conformer models
│   │   └── run_experiments.py           # Unified SOTA runner (dynamic loss, mixup)
│   ├── dataset.py                       # Core baseline dataset loader
│   ├── preprocess.py                    # Audio resampling & Butterworth filtering
│   └── extract_features.py              # Mel, CQT, CWT extraction & tensor stacking
├── metadata.csv                         # Generated metadata index CSV file
├── .gitignore                           # Git exclusions configuration
└── README.md                            # Project documentation
```

---

## 3. Installation and Setup

### Prerequisites
* Windows OS
* Python 3.10
* NVIDIA GPU with CUDA support

### Dependencies
Install the required packages using pip:
```bash
pip install torch torchvision torchaudio librosa soundfile scipy pywavelets scikit-learn pandas numpy tqdm matplotlib python-docx
```

---

## 4. Execution Pipeline

### Step 1: Preprocessing & Segmentation
Resample raw audio to 4000 Hz, apply the bandpass filter, and segment recordings into 3-second cycle files:
```bash
python src/preprocess.py
```

### Step 2: Multi-Branch Feature Extraction
Calculate Mel, CQT, and CWT features for each segment, stack them into `(3, 128, 128)` tensors, and generate `metadata.csv`:
```bash
python src/extract_features.py
```

### Step 3: Run Baseline Ablation Sweep
To run the legacy configurations sweep across Mel, Mel+CQT, Mel+CWT, and Stacked features (Configs A-D) using standard CNN and CNN-LSTM models:
```bash
python src/experiments/run_all_experiments.py --epochs 50 --batch_size 32
```

### Step 4: Run SOTA Model Verification
Dry-run shape checks and gradient backpropagation tests on Cross-Attention, Conformer, Contrastive Loss, and learnable Homoscedastic balancing:
```bash
python scratch/test_fusion.py
```

### Step 5: Run SOTA Model Training (Phase 4 Setup)
To train the Conformer hybrid model with Cross-Attention, Patient-Invariant Contrastive Loss, and Dynamic Multi-Task Loss Weighting:
```bash
python src/sota/run_experiments.py --model hybrid --backbone resnet --config D --epochs 50 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask --no_early_stopping --weighted_loss --min_se 0.40 --cross_attention --sequence_len 32 --conformer --contrastive_weight 0.1 --dynamic_loss
```

---

## 5. Experimental Results (Test Split)

### Baseline Ablation Sweep (Mel vs. Multi-Domain Stacking)
The table below illustrates the test metrics for standard and calibrated baselines on 4-Class Cycle Classification:

| Model | Feature Config | Test Accuracy | Sensitivity ($S_e$) | Specificity ($S_p$) | Official ICBHI Score ($S$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline ResNet-18** | Config A (Mel, Calibrated) | 50.80% | 26.84% | 66.62% | 46.73% |
| | Config D (Stacked, Calibrated) | **54.03%** | 18.03% | 79.61% | **48.82%** |
| **Proposed CNN-LSTM** | Config A (Mel, Calibrated) | 50.73% | 25.24% | 68.84% | 47.04% |
| | Config D (Stacked, Standard) | **53.08%** | 29.32% | 68.21% | **48.76%** |

---

### SOTA Publication Pipeline Results (Phases 1-4)
The table below contrasts our final model configurations (incorporating Cross-Attention, Conformer, Contrastive Loss, and Dynamic Loss Weighting) under strict patient-wise partition:

| Model & Phase Configuration | Calibration | Accuracy | Sensitivity ($S_e$) | Specificity ($S_p$) | ICBHI Score ($S$) | Pathology Acc. | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Task Baseline** | Standard (Argmax) | 49.85% | 35.77% | 55.39% | 45.58% | 92.36% | **2.35 ms** |
| **Multi-Task + Cross-Attention (P1)** | Standard (Argmax) | 46.08% | 38.65% | 48.60% | 43.62% | 92.96% | 18.15 ms |
| **Multi-Task + Conformer (P2)** | Standard (Argmax) | 48.53% | 35.48% | 55.98% | 45.73% | 92.62% | 4.39 ms |
| **Conformer + Contrastive (P3)** | Standard (Argmax) | 48.48% | 35.56% | 55.54% | 45.55% | — | 7.78 ms |
| **BiLSTM + Contrastive (P3)** | Standard (Argmax) | **49.82%** | 34.02% | **60.48%** | **47.25%** | — | 4.72 ms |
| | Calibrated (Product) | 46.63% | 43.97% | 42.24% | 43.11% | — | 4.72 ms |
| **Conformer + Contrastive + DynLoss (P4)**| Standard (Argmax) | 42.02% | **45.08%** | 38.21% | 41.65% | 91.19% | 4.65 ms |
| **BiLSTM + Contrastive + DynLoss (P4)** | Standard (Argmax) | 37.95% | 40.45% | 34.29% | 37.37% | **93.03%** | 4.36 ms |
| | Calibrated (Product) | 42.96% | 35.06% | 49.18% | 42.12% | **93.03%** | 4.36 ms |

---

## 6. Journal Publication Resources

A fully formatted Microsoft Word document containing a complete publication draft (Title, Abstract, Introduction, Methodology, Results, and Discussion) based on our experimental data: `Respiratory_Disease_Classification.docx` is in the reports folder.
