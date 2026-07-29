---
license: cc-by-nc-4.0
language:
- en
library_name: pytorch
tags:
- time-series-classification
- stress-detection
- biomedical-signal-processing
- gsr
- galvanic-skin-response
- heart-rate-variability
- cnn
- lstm
- attention
- wearable-sensors
- healthcare
- industrial-safety
pipeline_tag: tabular-classification
---

# GSR + HRV Stress Detection — CNN + BiLSTM (Attention) + MLP Hybrid Classifier

A subject-independent deep learning model that classifies a person's physiological state — **Relaxed** vs. **Stressed** — from two low-cost wearable signals: Galvanic Skin Response (GSR / skin conductance) and Heart Rate Variability (RR intervals). Developed as an applied research project at **Almutlaq United Company**, targeting real-time fatigue/stress monitoring for field workers via wearable sensors reporting to a central server.

**Full training pipeline, data-cleaning scripts, and experiment history:** [github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML](https://github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML/tree/main)

---

## TL;DR

| | |
|---|---|
| **Task** | Binary classification: Relaxed (0) vs. Stressed (1) |
| **Input** | 30-second windows of GSR + RR-interval signal (1 Hz) + 5 engineered HRV/GSR features |
| **Architecture** | CNN (Conv1D ×2) + BiLSTM + attention pooling, concatenated with an MLP feature branch into a dense classifier |
| **Cross-validated accuracy** | **73.59% ± 3.90%** (5-fold, subject-wise, N = 104 volunteers, 2,592 windows) |
| **Shipped checkpoint accuracy** | 76.13% on its own held-out split (507 windows, 21 volunteers) |

---

## Model Description

The model is a hybrid architecture with two branches that are combined before classification:

1. **Sequence branch (CNN + BiLSTM + attention):** takes a 30-second, 3-channel window — baseline-normalized GSR, baseline-normalized RR interval, and the first difference of GSR (rate of change) — through two Conv1D layers (with BatchNorm + MaxPool) and a bidirectional LSTM. Instead of using only the LSTM's final timestep as the sequence summary (which discards whatever happened earlier in the window), an additive **attention-pooling** layer learns a softmax weight over every remaining timestep and returns their weighted sum. A GSR stress response can peak anywhere in a 30-second window, not necessarily right at the end, so this lets the model draw on the full window rather than just its last moment.
2. **Engineered-feature branch (MLP):** takes 5 classical HRV/GSR statistics computed per window — RMSSD, SDNN, GSR slope, GSR range, and GSR peak count — through a small dense layer.

The two representations are concatenated and passed through a final classifier head (Dense → 2 classes).

```
Input Sequence (30×3)  →  Conv1D×2 + BatchNorm + MaxPool  →  BiLSTM  →  Attention Pool  ─┐
                                                                                            ├─→ Concatenate → Dense → Relaxed / Stressed
Engineered Features (5) →  Dense + BatchNorm  ──────────────────────────────────────────────┘
```

**Why CNN+LSTM instead of a classical model (SVM / Logistic Regression)?** Classical models operate on a fixed, hand-crafted feature vector per window and cannot learn how the signal evolves second-by-second. The CNN+LSTM branch adds that temporal modeling while remaining lightweight enough for fast CPU inference — importantly, in the target deployment, inference runs on a central server (not the wearable itself), so model complexity does not affect sensor battery life.

---

## Intended Uses & Limitations

**Intended for:**
- Research and prototyping of physiological stress/fatigue monitoring pipelines
- A reference implementation for applying CNN+LSTM+attention hybrid architectures to GSR/HRV time series
- A supervised pilot deployment (with human-reviewed alerts) in an industrial field-monitoring context

**Not intended for:**
- Direct clinical diagnosis of any medical or psychiatric condition
- Fully autonomous safety-critical decisions without human oversight
- Deployment on populations meaningfully different from the training cohort without re-validation

**Known limitations:**
- Moderate sample size (N = 104 volunteers after data-quality exclusions) — broader external validation is recommended before large-scale rollout.
- Cross-validated accuracy varies noticeably by fold (66.3%–77.2%), reflecting real inter-subject variability rather than a single stable number — treat 73.59% ± 3.90% as the honest range, not a guarantee per new subject.
- 1 Hz sampling limits detection of fine-grained phasic GSR responses (SCRs); a higher-frequency sensor could improve accuracy further.
- Some borderline windows sit near 50% prediction confidence — a production system should surface these as "uncertain" rather than forcing a hard decision.
- Trained on a single-institution dataset collected under a controlled lab protocol (a scripted "time-pressure" task), which may not capture the full variability of real workplace stressors.

---

## Training Data

Custom dataset of 150+ volunteers following an identical 4-phase protocol (Calibration ≈30s → Normal ≈4min → Stress ≈3min "time-pressure task" → Relaxation ≈1min), recorded at 1 Hz. Two GSR/HRV channels were logged per second: `Conductance_microS` and `RR_Interval_ms`.

**Data-quality issues found and corrected** (full detail and code in the linked GitHub repo):
- A subset of files (volunteers 35–144) had electrical **resistance** mislabeled as **conductance**; corrected via `conductance = 1 / resistance`.
- 14 volunteers had `RR_Interval_ms` populated for fewer than 10 rows out of 500+, found via a dedicated zero-window diagnostic rather than a simple NaN count.
- 20 volunteers were excluded in total (6 known sensor-disconnect/incomplete sessions + 14 with missing heart-rate data), leaving **N = 104** volunteers used for training and evaluation.

---

## Training Procedure

- **Windowing:** 30-second windows, 15-second step (50% overlap); a window never crosses a protocol-stage boundary.
- **Normalization:** two-stage — (1) per-volunteer baseline z-scoring using only each volunteer's Calibration + Normal phases (avoids the Stress phase diluting its own signal), then (2) a global fixed normalization fit on the training split only and saved for reuse at inference time.
- **Loss:** class-weighted cross-entropy (Relaxed/Stressed windows are imbalanced).
- **Regularization:** dropout = 0.45, weight decay = 2e-4.
- **Split:** subject-wise (volunteer-level) train/validation/test — no volunteer ever appears in more than one split.
- **Early stopping:** patience of 10 epochs on validation loss.

---

## Evaluation

### Cross-validated result (primary, reported figure)

5-fold subject-wise cross-validation — 5 independently trained models, each with a different held-out group of volunteers:

| Fold | Test Accuracy |
|---|---|
| 1 | 76.57% |
| 2 | 77.18% |
| 3 | 66.28% |
| 4 | 74.41% |
| 5 | 73.53% |
| **Mean** | **73.59% ± 3.90%** |

Fold 3's lower accuracy is a real signal of inter-subject variability (a harder held-out group of volunteers), not an error — this is exactly the kind of spread a single train/val/test split would hide, and part of why subject-wise cross-validation is reported here as the primary figure instead of one split's number.

**Pooled classification report (all 5 folds, 2,592 windows):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Relaxed | 0.754 | 0.812 | 0.782 | 1,510 |
| Stressed | 0.706 | 0.630 | 0.666 | 1,082 |

### Shipped checkpoint (`best_model_stress.pt`)

The specific checkpoint included in this repository was trained once on a single train/val/test split and reached **76.13% accuracy** on its own 507-window held-out test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Relaxed | 0.818 | 0.772 | 0.795 | 303 |
| Stressed | 0.688 | 0.745 | 0.715 | 204 |

Confusion matrix and training curves for this checkpoint are included as `confusion_matrix.png` and `training_curves.png` in this repository. Per-fold accuracy from cross-validation is plotted in `kfold_accuracy.png`.

---

## How to Use

```python
import torch
import numpy as np

# model.py defines CNNLSTMHybrid — see the GitHub repo for the full class definition
from model import CNNLSTMHybrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMHybrid(n_channels=3, n_scalar_features=5).to(device)
model.load_state_dict(torch.load("best_model_stress.pt", map_location=device))
model.eval()

# Load the saved normalization stats (fit on the training split only)
seq_mean = np.load("global_mean.npy")
seq_std = np.load("global_std.npy")
feat_mean = np.load("global_feat_mean.npy")
feat_std = np.load("global_feat_std.npy")

# seq: [1, 30, 3] -> (GSR_z, RR_z, dGSR_z) per second, baseline-normalized per subject first
# feat: [1, 5]    -> (RMSSD, SDNN, GSR slope, GSR range, GSR peak count)
seq_norm = (seq - seq_mean) / seq_std
feat_norm = (feat - feat_mean) / feat_std

with torch.no_grad():
    logits = model(torch.from_numpy(seq_norm).float().to(device),
                    torch.from_numpy(feat_norm).float().to(device))
    pred = logits.argmax(dim=1).item()  # 0 = Relaxed, 1 = Stressed
```

Full preprocessing (per-volunteer baseline normalization, windowing, feature extraction) and a ready-to-run inference script (`predict.py`) are in the linked GitHub repository.

---

## Files in This Repository

| File | Description |
|---|---|
| `best_model_stress.pt` | Trained PyTorch model weights |
| `global_mean.npy`, `global_std.npy` | Global normalization stats for the 3 sequence channels (fit on train split) |
| `global_feat_mean.npy`, `global_feat_std.npy` | Global normalization stats for the 5 engineered features (fit on train split) |
| `confusion_matrix.png` | Confusion matrix for the shipped checkpoint's held-out test split |
| `training_curves.png` | Train/validation loss and accuracy curves for the shipped checkpoint |
| `kfold_accuracy.png` | Per-fold accuracy from 5-fold subject-wise cross-validation |
| `classification_report.txt`, `kfold_summary.txt` | Full text metrics |

---

## Ethical Considerations

This model was trained on physiological data collected from consenting volunteers under a controlled research protocol. It is **not** a diagnostic tool and should not be used to make unsupervised decisions about an individual's health, employment status, or fitness for duty. Any field deployment should retain human review of model outputs, especially for borderline-confidence predictions, and should be paired with per-worker calibration (a short relaxed-baseline recording) rather than applied as a one-size-fits-all threshold.

---

## Author

**Mohammed Saleh**
Almutlaq United Company

Code, data-cleaning scripts, and full experiment history: [github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML](https://github.com/Alansi775/GSR-COSSINUS_VOLUNTEERS_DATA_FOR_ML/tree/main)
