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
├── data/                                # Local directory for raw datasets (gitignored)
│   ├── ICBHI_final_database/            # Raw .wav recordings and cycle annotations (.txt)
│   ├── ICBHI_Challenge_diagnosis.txt    # Mapping of Patient IDs to clinical diagnoses
│   └── ICBHI_challenge_train_test.txt   # Official train/test split definitions
├── processed_data/                      # Segmented 3.0s WAV cycle files (gitignored)
├── processed_features/                  # Stacked 3-channel PyTorch feature tensors (gitignored)
├── preprocess.py                        # Preprocessing, filtering, and segmentation script
├── extract_features.py                  # Multi-branch feature extraction and caching script
├── dataset.py                           # PyTorch Dataset, SpecAugment, and DataLoader module
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

## Usage

### 1. Preprocessing and Segmentation
Run the preprocessing script to resample all audio recordings to 4000 Hz, apply a bandpass filter (100 Hz - 1999 Hz) to eliminate extraneous heart and friction noise, and slice continuous audio files into uniform 3.0s cycle WAV files:
```bash
python preprocess.py
```

### 2. Multi-Branch Feature Extraction
Run the feature extraction script to parse the segmented cycle files, compute Mel Spectrogram, CQT, and CWT Scalogram features, stack them into a 3D tensor of size (3, 128, 128), and serialize them to disk as PyTorch tensor files:
```bash
python extract_features.py
```
This script also generates a `metadata.csv` indexing file, which is used by the dataloader to index the datasets.
