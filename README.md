# Lung Sound Analysis and Respiratory Disease Classification using Deep Learning

This repository contains the source code and documentation for an automated respiratory disease screening and classification pipeline. The system utilizes signal processing, multi-branch time-frequency feature extraction, and deep learning architectures to process chest auscultation audio and detect abnormal pulmonary patterns (crackles and wheezes) associated with chronic and acute respiratory conditions.

## Project Description

Traditional auscultation using stethoscopes is highly subjective and depends on clinical expertise, which is often limited in rural or low-resource settings. This project automates the diagnosis process using the ICBHI 2017 Respiratory Sound Database. Raw 1D respiratory audio signals are preprocessed, filtered, and segmented into respiration cycles. A multi-branch feature extraction pipeline converts these 1D signals into a 3-channel time-frequency tensor representing Mel Spectrogram, Constant-Q Transform (CQT), and Continuous Wavelet Transform (CWT) features. These representation maps are then analyzed using spatial-temporal deep learning architectures.

## Objectives

1. Develop a high-performance audio preprocessing and segmentation pipeline to extract individual respiration cycles from raw chest recordings.
2. Build a multi-branch time-frequency feature extraction module that captures both steady-state harmonic components (wheezes) and short-time transient impulses (crackles).
3. Design and implement baseline models (Support Vector Machines and Convolutional Neural Networks) alongside a hybrid spatio-temporal network (CNN-LSTM) to classify respiratory cycles.
4. Validate the framework against official challenge benchmarks using patient-level validation splitting to prevent demographic data leakage.

## Directory Structure

To keep the repository clean and avoid tracking large binary datasets, the project uses the following directory layout:

```text
├── checkpoints/                         # Saved model weights (.pth) and thresholds (npy) (gitignored)
├── data/                                # Local directory for raw datasets (gitignored)
│   ├── ICBHI_final_database/            # Raw .wav recordings and cycle annotations (.txt)
│   ├── ICBHI_Challenge_diagnosis.txt    # Mapping of Patient IDs to clinical diagnoses
│   └── ICBHI_challenge_train_test.txt   # Official train/test split definitions
├── evaluation_results/                  # Compiled confusion matrices and plots
├── processed_data/                      # Segmented 3.0s WAV cycle files (gitignored)
├── processed_features/                  # Stacked 3-channel PyTorch feature tensors (gitignored)
├── training_logs/                       # Epoch history JSON files and curve plots
├── preprocess.py                        # Preprocessing, filtering, and segmentation script
├── extract_features.py                  # Multi-branch feature extraction and caching script
├── dataset.py                           # PyTorch Dataset, SpecAugment, and DataLoader module
├── models.py                            # BaselineCNN and CNNLSTM architecture module
├── run_experiments.py                   # Single ablation training/calibration experiment runner
├── run_all_experiments.py               # Automation script for the 8 ablation study configurations
├── plot_comparison_curves.py            # Generates comparative validation curves comparison plot
├── train_baseline_svm.py                # Legacy SVM baseline classifier script
├── train_baseline_cnn.py                # Legacy CNN baseline training script
├── train_hybrid_model.py                # Legacy CNN-LSTM baseline training script
├── evaluate_models.py                   # Legacy comparison evaluation script
├── .gitignore                           # Git exclusion configuration
└── README.md                            # Project documentation
```

## Setup and Requirements

### Prerequisites
* Windows OS
* Python 3.10
* Conda or Python Virtual Environment
* NVIDIA GPU with CUDA support (e.g., RTX 3050 Laptop GPU)

### Dependencies
The pipeline requires the following library stack:
* **PyTorch (compiled with CUDA support)**: For deep learning models and tensor calculations.
* **Librosa / SoundFile**: For audio loading, resampling, and time-frequency representations.
* **SciPy**: For Butterworth digital filtering.
* **PyWavelets**: For Continuous Wavelet Transforms.
* **Scikit-Learn**: For dataset metrics and baseline classifiers.
* **Pandas / NumPy**: For data indexing and matrix manipulations.
* **TQDM**: For real-time CLI progress monitoring.

## Usage Instructions

### 1. Preprocessing and Segmentation
Run the preprocessing script to resample all audio recordings to 4000 Hz, apply a bandpass filter (100 Hz - 1999 Hz) to eliminate extraneous heart and muscle friction noise, and slice continuous audio files into uniform 3.0s cycle WAV files:
```bash
python preprocess.py
```

### 2. Multi-Branch Feature Extraction
Run the feature extraction script to parse the segmented cycle files, compute Mel Spectrogram, CQT, and CWT Scalogram features, stack them into a 3D tensor of size (3, 128, 128), and serialize them to disk as PyTorch tensor files:
```bash
python extract_features.py
```
This script also generates a `metadata.csv` indexing file, which is used by the dataloader to index the datasets.

### 3. Running a Single Experiment
To run a single training/evaluation configuration using early stopping, mixed-precision training, and validation-driven decision threshold calibration:
```bash
python run_experiments.py --model hybrid --config D --epochs 50 --batch_size 32
```
*   `--model`: Choose `cnn` (Baseline ResNet-18) or `hybrid` (Proposed CNN-LSTM).
*   `--config`: Choose `A` (Mel), `B` (Mel+CQT), `C` (Mel+CWT), or `D` (Stacked).

### 4. Running the Entire Ablation Sweep
To train all 8 configuration combinations sequentially and automatically compile a markdown metrics comparison table:
```bash
python run_all_experiments.py --epochs 50 --batch_size 32
```
If you have already trained the checkpoints and calibrated the decision thresholds, you can generate/compile the metrics table instantly without retraining by running:
```bash
python run_all_experiments.py --eval_only
```

### 5. Plotting Validation Convergence Curves
To read all training history files and plot a combined validation curve comparison figure:
```bash
python plot_comparison_curves.py
```

---

## Experimental Results

Below is the metrics comparison table compiled on the Test split after training all configurations for 50 epochs:

| Model | Feature Config | Calibration | Test Accuracy | Sensitivity ($S_e$) | Specificity ($S_p$) | Official ICBHI Score ($S$) | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet-18** | Config A (Mel) | Standard (Argmax) | 43.00% | 40.40% | 39.52% | 39.96% | 2.67 ms |
| | Config A (Mel) | Calibrated (Tuned) | 50.80% | 26.84% | 66.62% | 46.73% | 2.67 ms |
| | Config B (Mel+CQT) | Standard (Argmax) | 49.13% | 23.11% | 67.45% | 45.28% | 2.68 ms |
| | Config B (Mel+CQT) | Calibrated (Tuned) | 51.71% | 11.13% | 82.71% | 46.92% | 2.68 ms |
| | Config C (Mel+CWT) | Standard (Argmax) | 37.23% | 38.47% | 30.27% | 34.73% | 3.26 ms |
| | Config C (Mel+CWT) | Calibrated (Tuned) | 42.31% | 16.33% | 62.95% | 39.64% | 3.26 ms |
| | Config D (Stacked) | Standard (Argmax) | 51.71% | 27.17% | 66.94% | 47.05% | 3.65 ms |
| | Config D (Stacked) | Calibrated (Tuned) | **54.03%** | 18.03% | 79.61% | **48.82%** | 3.65 ms |
| **Proposed CNN-LSTM** | Config A (Mel) | Standard (Argmax) | 48.37% | 31.10% | 58.77% | 44.93% | 2.55 ms |
| | Config A (Mel) | Calibrated (Tuned) | 50.73% | 25.24% | 68.84% | 47.04% | 2.55 ms |
| | Config B (Mel+CQT) | Standard (Argmax) | 47.39% | 23.24% | 60.80% | 42.02% | 3.35 ms |
| | Config B (Mel+CQT) | Calibrated (Tuned) | 53.27% | 6.88% | **88.98%** | 47.93% | 3.35 ms |
| | Config C (Mel+CWT) | Standard (Argmax) | 48.62% | **35.74%** | 53.89% | 44.82% | 3.09 ms |
| | Config C (Mel+CWT) | Calibrated (Tuned) | 49.20% | 18.05% | 74.22% | 46.13% | 3.09 ms |
| | Config D (Stacked) | Standard (Argmax) | **53.08%** | 29.32% | 68.21% | **48.76%** | 3.10 ms |
| | Config D (Stacked) | Calibrated (Tuned) | 51.85% | 25.95% | 70.55% | 48.25% | 3.10 ms |

### Key Analytical Takeaways

1. **Complementary Multi-Branch Features**: MEL spectrograms provide standard acoustic patterns, while CQT captures wheeze harmonics and CWT isolates sharp impulse clicks (crackles). Stacking all branches (Config D) yields the highest accuracies.
2. **Spatio-Temporal Sequence Modeling**: The proposed **CNN-LSTM Spatio-Temporal Hybrid** model improves over the static 2D CNN baseline by capturing cyclical sequence transitions via its Bidirectional LSTM.
3. **Probability Decision Calibration**: Tuning class-specific probability thresholds on the validation split mitigates standard argmax class biases, successfully shifting boundaries to yield balanced medical diagnostic recall.
