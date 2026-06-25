# Implementation Plan — SOTA Multi-Task Learning (Phase 16)

This document details the implementation plan for Phase 16: Multi-Task Learning. We will introduce joint cycle sound classification (4 classes) and patient-level pathology classification (3 classes: COPD, URTI, Healthy) using shared representations in our SOTA training pipeline.

---

## User Review Required

> [!IMPORTANT]
> **Pathology Label Mapping and Filtering**:
> The ICBHI 2017 dataset includes several diagnoses. To align with a 3-class pathology target (COPD, URTI, Healthy):
> - If `--multitask` is active, the dataset will filter out patients with other diagnoses (e.g., Pneumonia, Bronchiolitis, Asthma) representing < 10% of total cycles, leaving 6,311 cycles.
> - The pathology mapping will be:
>   - **Class 0**: COPD
>   - **Class 1**: URTI
>   - **Class 2**: Healthy
> - If `--multitask` is inactive, the full dataset (all 6,898 cycles) will be used to maintain backward compatibility.
>
> **Joint Loss Function**:
> The training loop will optimize a joint multi-task loss:
> $$L_{\text{joint}} = L_{\text{cycle}} + \alpha \cdot L_{\text{pathology}}$$
> where $\alpha$ (default 1.0) is the weight parameter for the pathology task. Both losses will utilize `FocalLoss` or standard CrossEntropy depending on command-line configurations, and both will seamlessly compose with Mixup and Label Smoothing.

---

## Open Questions

None. The multi-task learning approach is standard and leverages shared feature extraction (from pre-trained PANNs backbones) to regularize representation learning.

---

## Proposed Changes

### [Component: SOTA Dataset Loader]

#### [MODIFY] [dataset.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/dataset.py)
*   Add a `multitask=False` parameter to `LungSoundDatasetSOTA` and `get_dataloaders`.
*   If `multitask=True`:
    *   Filter the metadata dataframe to include only diagnoses in `['COPD', 'URTI', 'Healthy']`.
    *   Map `diagnosis` to `pathology_label` integers (`0, 1, 2`).
    *   Modify `__getitem__` to return `(tensor, cycle_label, pathology_label)`.

### [Component: SOTA Model Architectures]

#### [MODIFY] [models.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/models.py)
*   Add a `multitask=False` flag to `BaselineCNNPANN` and `CNNLSTMPANN` initializers.
*   If `multitask=True`:
    *   Instantiate a second classification head `self.fc_pathology` outputting 3 logits.
    *   In the forward pass, return both `logits_cycle` and `logits_pathology`.

### [Component: SOTA Training Runner]

#### [MODIFY] [run_experiments.py](file:///d:/Internship%20'26/Lung%20Disease/src/sota/run_experiments.py)
*   Add CLI arguments:
    *   `--multitask`: Enable multi-task training.
    *   `--pathology_weight`: Float value representing task weight $\alpha$ (default 1.0).
*   Update `train_epoch` to unpack pathology labels, mix them if Mixup is active, apply label smoothing, compute joint loss, and optimize.
*   Update `validate` and `evaluate_test` to evaluate both cycle and pathology classification accuracies.

---

## Verification Plan

### Automated Tests
1.  **Multi-Task Pipeline & Shape Verification**: Create `scratch/test_multitask.py` to verify:
    *   `dataset.py` correctly filters and returns both labels when `multitask=True`.
    *   `models.py` outputs both cycle and pathology logits with correct shapes `(B, 4)` and `(B, 3)`.
    *   Multi-task loss and mixup functions compute successfully.
2.  **Pipeline Verification Run**: Run a 3-epoch dry training run with multi-task learning enabled:
    ```bash
    python src/sota/run_experiments.py --model hybrid --config D --epochs 3 --batch_size 32 --mixup --mixup_prob 1.0 --label_smoothing 0.1 --focal_gamma 2.0 --multitask
    ```

### Manual Verification
1.  **Logs Inspection**: Confirm that both cycle loss and pathology loss are decreasing during training and that metrics are reported for both tasks.
