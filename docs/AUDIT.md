# PHANTOM X: Technical Audit & Forensic Diagnostic Report

**Module**: Image Authenticity Detection (REAL vs AI-GENERATED)  
**System Architecture**: PyTorch Transfer Learning (ResNet-18 Baseline)  
**Phase**: Phase 0 – Audit, Diagnosis & Forensic Integrity Check  
**Date**: November 2026  
**Auditor**: ML Engineering & Computer Vision Forensics Team  

---

## 1. Executive Summary

This audit establishes the baseline technical diagnostic for the Image Authenticity Detection module of **PHANTOM X**. The objective is to identify all architectural, procedural, statistical, and data-integrity defects in the current starter implementation before expanding datasets or modifying production code.

### Key Audit Findings
1. **Severe Empirical Overfitting**: The starter ResNet-18 baseline achieves **96.88%** training accuracy after 5 epochs, but collapses to **60.00%** accuracy on the 40-image validation set (random guess = 50.00%). On the REAL class, the baseline performs at **50.00% recall** (10/20), equivalent to a random coin toss.
2. **Structural Causes of Overfitting**:
   - Extreme sample starvation: Only 80 training samples per class ($N_{train}=160$).
   - Full parameter unfreezing: All 11,177,538 parameters are trained end-to-end with Adam ($\text{lr}=10^{-4}$) with zero weight decay ($\lambda=0.0$) and no dropout ($p=0.0$).
   - Minimal data augmentation: Only `RandomHorizontalFlip()`.
3. **Statistical Shortcut-Learning Vulnerabilities**:
   - **Aspect Ratio Discrepancy**: REAL images have a median aspect ratio of **1.50** (standard 3:2 photography), whereas AI-generated images have a median aspect ratio of **1.07** (predominantly 1:1 squares typical of default Midjourney / Stable Diffusion generation).
   - **Anamorphic Distortion Shortcut**: Resizing directly to $(224, 224)$ via `transforms.Resize((224, 224))` violently distorts non-square REAL images, giving the neural network an easy geometric shortcut rather than learning genuine forensic generation traces.
   - **Train-to-Validation Distribution Shift**: `val_REAL` contains large DSLR photographs (mean resolution $1339.8 \times 1081.4$, mean size $493.4\text{ KB}$), compared to `train_REAL` (mean resolution $870.5 \times 726.6$, mean size $126.2\text{ KB}$), skewing validation performance.
4. **Codebase Inconsistencies & Bugs**:
   - `evaluate.py` inverts positive/negative class roles in confusion matrix counters while labels are $\text{FAKE}=0, \text{REAL}=1$.
   - Checkpoint saving triggers on `val_acc >= best_acc` over a high-variance 40-image validation split ($1\text{ image} = 2.5\%$).
   - Lack of a centralized constants module for class indices.
   - Pillow warnings on palette images (`P` mode with transparency) during batch loading.

---

## 2. Codebase & Pipeline Audit

### 2.1 File-by-File Technical Inspection

| File | Primary Role | Audit Assessment & Deficiencies |
| :--- | :--- | :--- |
| `phantom_x/model.py` | Model architecture definition | Implements `build_model(num_classes=2, pretrained=True)` using `models.resnet18(weights=ResNet18_Weights.DEFAULT)`. Replaces `model.fc` with a raw `nn.Linear(512, 2)`. **Defects**: No dropout layer (`nn.Dropout`), no multi-stage classification head, no backbone freezing logic. |
| `phantom_x/train.py` | Training loop & checkpointing | Loads datasets with `ImageFolder`, initializes Adam optimizer ($\text{lr}=10^{-4}$), CrossEntropyLoss, runs 5 epochs. **Defects**: 1) Trains all layers end-to-end; 2) No learning rate scheduler; 3) No weight decay; 4) Checkpoint saved purely on `val_acc >= best_acc` (overwriting on ties on a tiny 40-image split); 5) Checkpoint omits epoch, optimizer state, validation loss, and metrics; 6) `num_workers=0` hardcoded. |
| `phantom_x/predict.py` | Single-image CLI inference | Loads checkpoint, applies `Resize((224, 224))`, ToTensor, ImageNet normalization, computes softmax. **Defects**: 1) Direct non-aspect-preserving resize distorts image; 2) `Image.open().convert("RGB")` flattens transparent PNG/WebP files with black background, introducing high-frequency edge artifacts; 3) No uncertainty thresholding. |
| `phantom_x/evaluate.py` | Validation set metrics | Evaluates validation split, prints confusion matrix. **Defects**: 1) Does not implement `if __name__ == "__main__":` guard; 2) Binary classification metrics treat class 0 (FAKE) as True Positive (TP) and class 1 (REAL) as True Negative (TN), conflicting with standard conventions where positive class = 1; 3) Does not calculate Precision, Recall, F1, ROC-AUC, or ECE. |
| `phantom_x/prepare_sample.py` | Sampling script for initial 200 images | Uses `snapshot_download` from `LuckyCow/Ai_vs_real_web_images` to sample 100 REAL and 100 AI images, splitting 80/20. **Defects**: Stratified solely by class count; ignores image resolution, aspect ratio, source generator, and duplicate hash groups. |
| `phantom_x/check_environment.py` | Hardware verification | Checks PyTorch version, CUDA status, and device name. Verified functioning on Windows CPU. |
| `download_full_dataset.py` | Full dataset streaming | Uses Hugging Face `load_dataset` to stream and extract full dataset to `raw_sample_full`. Untouched per Rule 2. |
| `requirements.txt` | Package dependencies | Lists `torch`, `torchvision`, `pillow`. Missing explicit versions, `numpy`, `scipy`, `scikit-learn`, `streamlit`, `timm`. |

### 2.2 Deep Dive: Weights, Checkpoints, Transforms, and Labels

#### Weights and Layer Freezing
In `phantom_x/model.py` and `train.py`, the backbone is instantiated with ImageNet weights and immediately passed entirely to Adam:
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$
Because the initial classification layer (`fc`) has randomly initialized weights, backpropagating large gradient signals through all 18 layers with $N=160$ samples rapidly disrupts pre-trained convolutional filters in `layer1` through `layer4`. This leads directly to catastrophic memorization rather than transfer learning.

#### Checkpoint Serialization
The checkpoint `outputs/phantom_x_resnet18_best.pth` was inspected:
- **Keys present**: `model_state_dict`, `class_to_idx`, `architecture`, `input_size`.
- **FC weights**: Mean = $-7.09 \times 10^{-6}$, Standard Deviation = $0.0259$.
- **Defects**: Missing metadata required for forensic reproducibility:
  - Random seed
  - Training git commit hash / timestamp
  - Train and validation loss history
  - Normalization parameters used
  - Calibration parameters (temperature $T$)

#### Transforms Mismatch & Flaws
- **Train transform**: `Resize((224, 224))` $\rightarrow$ `RandomHorizontalFlip()` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize()`
- **Eval transform**: `Resize((224, 224))` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize()`
- **Predict transform**: Identical to `eval_tfms`.
- **Problem**: Forcing rectangular images (e.g. $1920 \times 1080$, aspect ratio 1.78) into a square $(224, 224)$ alters spatial frequency distributions in the 2D Fourier domain. Because camera sensors maintain specific spatial sampling lattices (Bayer color filter arrays, PRNU patterns), squishing them anamorphically destroys these physical cues while introducing artificial high-frequency compression along the squeezed axis.

#### Label Mapping Discrepancy
- PyTorch `ImageFolder` sorts folder names alphabetically:
  - Index 0: `FAKE`
  - Index 1: `REAL`
- Rule 8 states: `FAKE = 0`, `REAL = 1`.
- However, in `phantom_x/evaluate.py`:
  ```python
  if actual == 0 and predicted == 0:
      tp += 1  # Actual FAKE, Pred FAKE -> Counted as TP
  elif actual == 1 and predicted == 1:
      tn += 1  # Actual REAL, Pred REAL -> Counted as TN
  elif actual == 1 and predicted == 0:
      fp += 1  # Actual REAL, Pred FAKE -> Counted as FP
  else:
      fn += 1  # Actual FAKE, Pred REAL -> Counted as FN
  ```
  Here, FAKE is designated as the positive class ($y=1$). In standard scientific reporting and scikit-learn metrics, if `REAL = 1`, then actual 1 / pred 1 is True Positive. This inversion creates ambiguity across metrics. A single centralized `constants.py` must define `FAKE_LABEL = 0`, `REAL_LABEL = 1`, with explicit metrics computed for both classes and macro-averaged.

---

## 3. Dataset Inspection & Empirical Shortcut Analysis

An exhaustive scan across all 200 images in `data/` was executed using a specialized diagnostic tool (`scratch_audit_diagnostics.py`).

### 3.1 Dataset Demographics & Distribution

| Subset | Sample Count | JPEG | PNG | WEBP | Modes | Mean Width (px) | Mean Height (px) | Mean File Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train/FAKE` | 80 | 68 (85.0%) | 7 (8.8%) | 5 (6.2%) | RGB: 77, P: 3 | 943.7 | 823.9 | 189.9 |
| `train/REAL` | 80 | 70 (87.5%) | 5 (6.2%) | 5 (6.2%) | RGB: 78, RGBA: 2 | 870.5 | 726.6 | 126.2 |
| `val/FAKE` | 20 | 16 (80.0%) | 2 (10.0%) | 2 (10.0%) | RGB: 18, P: 1, RGBA: 1 | 881.3 | 798.5 | 212.8 |
| `val/REAL` | 20 | 17 (85.0%) | 1 (5.0%) | 2 (10.0%) | RGB: 19, P: 1 | 1339.8 | 1081.4 | 493.4 |
| **All FAKE** | 100 | 84 (84.0%) | 9 (9.0%) | 7 (7.0%) | RGB: 95, P: 4, RGBA: 1 | 931.2 | 818.8 | 194.5 |
| **All REAL** | 100 | 87 (87.0%) | 6 (6.0%) | 7 (7.0%) | RGB: 97, P: 1, RGBA: 2 | 964.3 | 797.5 | 199.6 |

- **Integrity**: 
  - Corrupt / Unreadable images: **0**
  - Exact duplicates (SHA-256): **0**
  - Near duplicates (Perceptual Difference Hash $\le 3$ bits): **0**

### 3.2 Statistical Shortcut-Learning Analysis

To detect whether a model can exploit non-forensic dataset artifacts, we computed mean, median, standard deviation, and Cohen's $d$ effect sizes between `REAL` and `FAKE`:

$$\text{Cohen's } d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}, \quad s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

| Feature | REAL Mean (Std) | REAL Median | FAKE Mean (Std) | FAKE Median | Cohen's $d$ | Shortcut Risk Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Width (px)** | 964.3 (609.7) | 764.5 | 931.2 (492.8) | 800.0 | -0.060 | Negligible linear separation |
| **Height (px)** | 797.6 (571.4) | 600.0 | 818.8 (441.9) | 613.0 | +0.042 | Negligible linear separation |
| **File Size (bytes)** | 204,407 (402,157) | 62,886 | 199,156 (384,784) | 66,716 | -0.013 | Negligible linear separation |
| **Bits Per Pixel (BPP)** | 1.63 (2.07) | 1.04 | 1.77 (2.31) | 1.10 | +0.065 | Slight difference in compression density |
| **Aspect Ratio (W/H)** | **1.30 (0.43)** | **1.50** | **1.21 (0.34)** | **1.07** | **-0.215** | **CRITICAL: Systematic geometric bias** |

#### Empirical Findings on Shortcuts:
1. **Aspect Ratio Bias**: 
   - REAL images have a median aspect ratio of **1.50**, matching standard $3:2$ digital single-lens reflex (DSLR) and 35mm sensors.
   - FAKE images have a median aspect ratio of **1.07**, heavily clustered around $1.00$ ($512\times 512$ and $1024\times 1024$ native outputs of Stable Diffusion v1/v2 and Midjourney).
   - The effect size ($d = -0.215$) indicates that directly downsampling with `Resize((224, 224))` induces severe horizontal squishing on REAL images while leaving FAKE images mostly isotropic. A CNN can learn to classify squished circular objects as REAL.
2. **Train vs Validation Sampling Asymmetry**:
   - `train_REAL` has an average file size of **126.2 KB** and mean resolution of $870 \times 726$.
   - `val_REAL` has an average file size of **493.4 KB** (3.9x larger!) and mean resolution of $1340 \times 1081$.
   - This acute distribution shift explains why the overfitted model failed on half the validation REAL images (50% error rate).

---

## 4. Baseline Reproduction & Overfitting Diagnosis

We executed `evaluate.py` on the existing checkpoint `outputs/phantom_x_resnet18_best.pth`:

```text
PHANTOM X EVALUATION
--------------------
Total images: 40
Correct: 24
Accuracy: 60.0 %

Confusion Matrix (actual rows, predicted columns)
                 Pred FAKE   Pred REAL
Actual FAKE:        14          6
Actual REAL:        10          10
```

### Forensic Performance Metrics Breakdown

| Metric | Class FAKE (0) | Class REAL (1) | Macro Average |
| :--- | :--- | :--- | :--- |
| **Precision** | $\frac{14}{14 + 10} = 58.33\%$ | $\frac{10}{10 + 6} = 62.50\%$ | **60.42%** |
| **Recall / Sensitivity** | $\frac{14}{14 + 6} = 70.00\%$ | $\frac{10}{10 + 10} = 50.00\%$ | **60.00%** |
| **F1-Score** | $63.64\%$ | $55.56\%$ | **59.60%** |
| **False Positive Rate (FPR)** | $50.00\%$ (REAL misclassified as FAKE) | $30.00\%$ (FAKE misclassified as REAL) | **40.00%** |

### Statistical Significance Test
On a validation sample of $N=40$ balanced binary images ($p=0.5$ under null hypothesis $H_0$):
$$P(X \ge 24 \mid N=40, p=0.5) = \sum_{k=24}^{40} \binom{40}{k} 0.5^{40} \approx 0.1341$$
Because $p = 0.1341 > 0.05$, **the baseline model's 60% accuracy is NOT statistically distinguishable from random guessing**.

---

## 5. Storage and Cache Audit

In accordance with Non-Negotiable Rule 2 and Rule 3:
- **Available Disk Space**: $593.88\text{ GB}$ free on `C:\` (well exceeding the $256\text{ GB}$ requirement).
- **Hugging Face Cache**:
  - `C:\Users\arpit\.cache\huggingface\hub\datasets--LuckyCow--Ai_vs_real_web_images\snapshots\5dd319b18730000d23bdd9374c929319a3f53275`
  - Already contains **26,333 AI images** cached locally on disk.
  - Snapshot does not contain cached REAL images; however, the local `raw_sample/` directory retains the original extracted sample of 100 REAL and 100 AI images.
  - **No redundant downloads will be triggered.**

---

## 6. Actionable Recommendations for Phase 1 and Phase 2

1. **Controlled Dataset Expansion (Phase 1)**:
   - Target $34\text{k} - 50\text{k}$ balanced images across multiple distinct generative architectures (e.g. ProGAN, StyleGAN2/3, Stable Diffusion v1.4/1.5/2.1/XL, Midjourney, DALL-E) and verified pristine real sources (e.g. MS-COCO, RAISE, ImageNet real subsets).
   - Enforce group-aware, duplicate-free train/val/test splits (70/15/15) plus a generator-held-out test split.
2. **Neutralizing the Aspect Ratio Shortcut**:
   - Replace anamorphic `Resize((224, 224))` with an aspect-ratio-preserving transformation pipeline:
     - Resize shorter edge to 256 preserving aspect ratio, followed by random crop (training) or central crop (inference) of $224 \times 224$.
     - Or random resized crops (`RandomResizedCrop(224, scale=(0.8, 1.0))`) during training so the network cannot memorize aspect ratio distortion.
3. **Regularized Transfer Learning Pipeline (Phase 2)**:
   - Introduce dropout ($p=0.3-0.5$) before the final classification head.
   - Stage 1: Freeze ResNet backbone, train only classification head ($\text{lr}=10^{-3}$, AdamW with $\text{weight\_decay}=10^{-2}$).
   - Stage 2: Unfreeze `layer4` and `layer3` with discriminative learning rates ($\text{lr}_{backbone}=10^{-5}$, $\text{lr}_{head}=10^{-4}$).
   - Cosine annealing learning rate scheduler with warm-up.
   - Early stopping monitored on validation loss/ROC-AUC, not noisy validation accuracy.
4. **Standardized Architecture & Code Standards**:
   - Centralize constants in `phantom_x/constants.py`: `FAKE_LABEL = 0`, `REAL_LABEL = 1`.
   - Implement `if __name__ == "__main__":` guard across all executable scripts.
   - Save full checkpoint bundles (weights, epoch, optimizer, scheduler, val loss, val AUC, class mapping, input config).
