# WeatherSense-VAD — Implementation Status

**Project**: Weather-Robust, Context-Aware and Continually Adaptive Video Anomaly Detection System  
**Last Updated**: 2026-10-23  
**Environment**: Windows, Python 3.12.7, Intel UHD (CPU-only), Anaconda

---

## Repository Structure

```
E:\WeatherSense-VAD\
├── main.py                          # EMPTY — entry point stub
├── data/
│   ├── sequences/part_A/test_data/  # Pre-created 16-frame sequences
│   └── shanghaitech/
│       ├── gt/test_frame_mask/      # 107 ground-truth .npy masks
│       ├── part_A/{train,test}_data/{images,ground-truth}/
│       ├── part_B/{train,test}_data/{images,ground-truth}/
│       └── pose/test/               # AlphaPose JSON results
├── preprocessing/
│   ├── zero_dce/                    # Zero-DCE model, losses, training, enhancement
│   ├── create_sequences.py          # Sequence creation scripts
│   ├── ground_truth_loader.py       # GT annotation loader
│   └── outputs/zero_dce/            # Enhanced images + zero_dce_model.pth
├── models/
│   ├── swin_feature_extractor.py    # Swin3D-T feature extractor
│   ├── anomaly_detector.py          # Normal center calculation
│   └── extract_*.py                 # Feature extraction scripts (6 variants)
├── training/
│   ├── dataset.py                   # ShanghaiTechDataset (original + enhanced)
│   └── test_dataloader.py           # DataLoader test
├── evaluation/
│   ├── baseline_prediction.py       # Threshold-based anomaly detection
│   ├── mahalanobis_detector.py      # Mahalanobis distance scoring
│   ├── compare_scores.py            # Original vs enhanced comparison
│   └── *.py                         # Various scoring/GT scripts
├── outputs/
│   ├── features/                    # Pre-extracted .pt feature files
│   ├── scores/                      # Anomaly score .pt files
│   ├── baseline/                    # Baseline results CSV
│   └── evaluation/                  # Evaluation comparison CSVs
└── temp_stgnf/                      # EMPTY directory (STG-NF placeholder)
```

---

## Component Inventory

### A. IMPLEMENTED AND VERIFIED

| Component | Files | Notes |
|-----------|-------|-------|
| Zero-DCE Model | `preprocessing/zero_dce/model.py` | 7-layer CNN, 24-ch curve output, 8-iter enhancement. Correct architecture. |
| Zero-DCE Losses | `preprocessing/zero_dce/losses.py` | 4 losses: spatial consistency, illumination, color constancy, exposure control. Well-implemented. |
| Zero-DCE Training | `preprocessing/zero_dce/train.py` | Trains on ShanghaiTech Part A. Trained checkpoint exists at `preprocessing/outputs/zero_dce/zero_dce_model.pth` (323KB). |
| Zero-DCE Enhancement | `preprocessing/zero_dce/enhance.py` | Batch enhancement of images. Proper curve iteration (8 steps × 3 channels). |
| Swin3D Feature Extractor | `models/swin_feature_extractor.py` | Uses `torchvision.models.video.swin3d_t`. Removes classification head → 768-dim features. |
| ShanghaiTech Dataset | `training/dataset.py` | Supports original and enhanced modes. Sequence length=16, stride=8, 224×224. ImageNet normalization. |
| Feature Extraction Pipeline | `models/extract_features.py` + variants | 6 scripts for train/test × original/enhanced × subset/full. Shape [B,T,C,H,W] → permute → Swin3D. |
| Normal Center Calculation | `models/anomaly_detector.py` | Mean of training features → normal center (768-dim). |
| Euclidean Anomaly Scoring | `models/test_anomaly_scores.py` | L2 distance from normal center. |
| Baseline Threshold Detection | `evaluation/baseline_prediction.py`, `save_baseline.py` | mean + 2σ threshold. CSV output exists (22 clips, 3 anomalies detected). |
| Score Comparison | `evaluation/compare_scores.py`, `final_score_table.py` | Original vs Zero-DCE enhanced comparison. CSV outputs exist. |
| Mahalanobis Detector | `evaluation/mahalanobis_detector.py` | PCA(10) + Mahalanobis distance. Uses sklearn. |
| Ground Truth Loader | `preprocessing/ground_truth_loader.py` | Loads ShanghaiTech .mat annotations via scipy. |
| Sequence Creator | `preprocessing/create_sequences.py` | Creates 16-frame sequences with stride 8. |
| Pre-computed Outputs | `outputs/` | Train/test features (.pt), scores (.pt), baseline CSV, evaluation CSVs all exist. |

### B. IMPLEMENTED BUT NEEDS FIXES

| Component | Issue |
|-----------|-------|
| Zero-DCE train.py | Only trains on first 20 images (`[:20]`), EPOCHS=1. Proof-of-concept only. Indentation inconsistent at lines 128-146. |
| `enhance.py` vs `model.py` | Different enhancement logic: `enhance.py` uses 8-step channel-sliced iteration (correct for 24-ch); `model.py`'s `enhance_image()` uses simplified single-curve iteration. Inconsistency needs documenting. |
| `score_full_test.py` | Hardcoded `end_frame = min(start_frame + 15, 182)` — magic number 182 specific to one video. |
| `mahalanobis_detector.py` L107-111 | Redundant `torch.tensor(torch.from_numpy(...).numpy())` conversion. Should be simplified. |
| `build_clip_ground_truth.py` | Uses crowd-counting GT (.mat with head locations), NOT anomaly GT. This is a crowd density metric, not an anomaly label. |

### C. IMPLEMENTED BUT NOT INTEGRATED

| Component | Status |
|-----------|--------|
| All evaluation scripts | Standalone scripts, no unified evaluation pipeline |
| Feature extraction scripts (6 variants) | Duplicated logic across 6 separate files |
| Sequence creation scripts (5 variants) | Duplicated logic for original/enhanced/test variants |
| Ground truth frame masks (.npy) | 107 files exist but no script loads them for anomaly evaluation |
| Pose data (AlphaPose JSONs) | Data exists in `data/shanghaitech/pose/test/` but no code uses it |

### D. PARTIALLY IMPLEMENTED

| Component | What Exists | What's Missing |
|-----------|-------------|----------------|
| Anomaly Detection Pipeline | Feature extraction + Euclidean distance scoring | No ROC-AUC evaluation, no frame-level scoring, no multi-video support |
| temp_stgnf (STG-NF) | Empty directory placeholder | All STG-NF code, checkpoints (mentioned: ShanghaiTech_85_9.tar, UBnormal checkpoints — not in repo) |
| main.py | Empty file | No unified entry point |

### E. NOT YET IMPLEMENTED

| Component | Description | Priority |
|-----------|-------------|----------|
| **AOD-Net** | All-in-One Dehazing Network for weather preprocessing | HIGH |
| **Weather Assessment** | Classify weather/visibility conditions in frames | HIGH |
| **CLIP Scene Understanding** | Place-conditioned scene classification | HIGH |
| **Time-Conditioned Modelling** | Temporal context (time-of-day, day-of-week patterns) | MEDIUM |
| **Cold-Start Calibration** | Self-calibrating mechanism for new scenes | MEDIUM |
| **LoRA Continual Learning** | Low-Rank Adaptation for model updates | MEDIUM |
| **EWC Continual Learning** | Elastic Weight Consolidation to prevent forgetting | MEDIUM |
| **Grad-CAM Explainability** | Visual evidence for anomaly decisions | HIGH |
| **Reasoning Engine** | Human-readable anomaly explanations | MEDIUM |
| **Interactive Dashboard** | Web-based visualization and control | HIGH |
| **Unified Pipeline** | End-to-end processing from video to results | HIGH |
| **requirements.txt** | Dependency specification | HIGH |
| **ROC-AUC Evaluation** | Proper frame-level evaluation with GT masks | HIGH |
| **Multi-video Support** | Handle multiple surveillance videos | MEDIUM |
| **Video Input Handler** | Accept video files, extract frames | HIGH |

---

## Existing Pre-computed Results

### Feature Shapes
- Training features: shape depends on dataset (22 windows for Part A test based on 182 frames)
- Feature dimension: 768 (Swin3D-T)
- Temporal: 16 frames/window, stride 8

### Baseline Results (Part A Test, 22 clips)
- Threshold: 7.788 (mean + 2σ)
- Predicted anomalies: 3/22 clips
- Predicted normal: 19/22 clips

### Previous STG-NF Result (from user)
- ~85.94% ROC-AUC over 40,791 samples
- **Note**: This is a COMPONENT result for the pose-based STG-NF pipeline only, NOT the integrated system AUC.

---

## Dependencies

### Currently Available
- Python 3.12.7 (Anaconda)
- numpy 1.26.4
- pandas 2.2.2
- Pillow 10.4.0
- scipy 1.13.1
- Flask 3.0.3

### Required (Not Installed)
- torch (CPU-only)
- torchvision
- scikit-learn
- transformers (for CLIP)
- opencv-python
- flask-cors
- grad-cam
- peft (for LoRA)

---

## Integration Plan

### Phase 1: Foundation (requirements.txt, environment, unified pipeline)
### Phase 2: Weather Preprocessing (AOD-Net, weather assessment)
### Phase 3: Scene Understanding (CLIP integration)
### Phase 4: Temporal Modelling (time conditioning)
### Phase 5: Cold-Start Calibration
### Phase 6: Anomaly Detection (unified scoring, ROC-AUC evaluation)
### Phase 7: Continual Learning (LoRA + EWC)
### Phase 8: Explainability (Grad-CAM + reasoning)
### Phase 9: Dashboard (Flask web interface)
### Phase 10: Integration Testing & Documentation
