# Implementation Plan — SOTA Pretrained Audio Backbones (Phase 15)

This document details the implementation plan for Phase 15: Pretrained Audio Backbones (PANNs Cnn14). We will define the PANNs Cnn14 backbone and its model wrappers (`BaselineCNNPANN` and `CNNLSTMPANN`) and load the pretrained weights while adapting the input channels for multi-channel input configurations.

---

## User Review Required

> [!IMPORTANT]
> **Pretrained Weights Download & Storage**:
> We will download the `Cnn14` weights (`Cnn14_mAP=0.431.pth`) from the official Zenodo link (312 MB). The download will be cached locally in `checkpoints/Cnn14_mAP=0.431.pth`. This is done automatically on first run via `urllib.request` with a `tqdm` progress bar.
>
> **Weight Mapping and Channel Scaling**:
> The pre-trained PANNs Cnn14 was trained on mono audio (1 channel). Our pipeline supports input configurations with up to 3 channels (Mel, CQT, CWT stacked). To use the pre-trained weights for multi-channel inputs:
> 1. We repeat the weights of the first convolutional layer `conv_block1.conv1.weight` across the channel dimension (`in_channels`).
> 2. We scale the weights by dividing by `in_channels` to maintain the scale of the output activations:
>    $$\text{weights}_{\text{new}} = \frac{\text{weights}_{\text{pretrained}}.\text{repeat}(1, C_{\text{in}}, 1, 1)}{C_{\text{in}}}$$
> This is mathematically correct and preserves the pre-trained features without exploding activations.

---

## Open Questions

None. The PANNs Cnn14 mapping and channel repeat/scaling technique is a standard transfer learning procedure for multi-channel audio spectrograms.

---

## Proposed Changes

### [Component: SOTA Model Architectures]

#### [NEW] [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py)
*   Create a clean model file for SOTA containing:
    *   `ConvBlock`: PyTorch modules matching the PANNs layers.
    *   `Cnn14Backbone`: Integrates 6 convolutional blocks resulting in a `(B, 2048, 4, 4)` output feature map.
    *   `download_panns_weights()`: Automatic downloader using a tqdm progress bar callback.
    *   `load_panns_state_dict()`: Maps state dict keys, repeats and scales the first layer's weights for `in_channels != 1`, and loads them strictly for matching layers.
    *   `BaselineCNNPANN`: Wraps the backbone with a fully connected head (`fc1` + `fc_final`).
    *   `CNNLSTMPANN`: Wraps the backbone with a sequence modeling head (BiLSTM with `input_size=8192` + classification head).

### [Component: SOTA Training Runner]

#### [MODIFY] [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py)
*   Import `BaselineCNNPANN` and `CNNLSTMPANN` from the new `sota.models` instead of `experiments.models`.
*   Support launching with these new SOTA models.

---

## Verification Plan

### Automated Tests
1.  **Model & Weight Loading Verification**: Run verification script `scratch/test_panns.py` to confirm:
    *   Weights download and load correctly.
    *   Models produce correct shape `(B, 4)` for different channel configurations (A, B, C, D).
    *   State dictionary is mapped successfully with no critical missing weights in the convolutional layers.
2.  **Pipeline Verification Run**: Run a 3-epoch dry training run using the new `CNNLSTMPANN` model to check convergence:
    ```bash
    python src/sota/run_experiments.py --model hybrid --config D --epochs 3 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0
    ```

### Manual Verification
1.  **Logs and Metrics**: Inspect training CLI output to check for validation loss reduction and correctness of prediction outputs.
