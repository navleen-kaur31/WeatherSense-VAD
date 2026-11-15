# WeatherSense-VAD

**A Weather-Aware Framework for Video Anomaly Detection in Surveillance Systems**

A modular research framework that studies how environmental conditions (fog, low light, rain) and image enhancement affect spatiotemporal features and distance-based anomaly scores in surveillance video. It combines heuristic weather assessment, Zero-DCE low-light enhancement, pretrained Video Swin Transformer features, and Euclidean / PCA-Mahalanobis anomaly scoring, plus a Flask dashboard for inspecting results.

> **Status: research prototype.** The offline feature and scoring pipeline and the dashboard are separate components. Detection accuracy has **not** been established because verified clip-level anomaly labels were not available for the evaluated clips. See [Limitations](#limitations).

## What the System Actually Implements

| Module | What is implemented | Method | Notes |
|---|---|---|---|
| Weather Assessment | Heuristic indicators for fog-like degradation, low illumination, rain-like patterns and visibility | Dark channel information, average luminance and dark-pixel ratio, directional gradients, brightness / contrast / Laplacian sharpness | Image-statistics heuristics, not a trained or validated weather classifier |
| Low-Light Enhancement | Zero-DCE (DCE-Net, 8 curve iterations) | Zero-reference curve estimation | Preliminary proof-of-concept training only: first 20 images, 128×128, 1 epoch, Adam, lr 1e-4 |
| Dehazing | AOD-Net architecture and inference path present | Reformulated atmospheric scattering model | **No verified trained checkpoint**, so outputs are not validated and excluded from performance claims. The dashboard's fog handling uses a lightweight atmospheric-light / transmission approximation |
| Scene Understanding | Heuristic scene classification | Brightness, colour variance, edge density, green-region and sky-region ratios | Not a learned semantic model (CLIP is future work) |
| Temporal Context | Hour / day encoding and period labels | Cyclic sin/cos encoding, weekday / weekend adjustment, predefined expected-activity values | Values are heuristic assumptions, not learned from surveillance history |
| Feature Extraction | 768-d spatiotemporal clip features | Video Swin Transformer Tiny (Swin3D-T), Kinetics-400 pretrained, classification head removed | 16-frame clips, stride 8, 224×224. **Not fine-tuned** on ShanghaiTech. Extracted separately for original and Zero-DCE-enhanced frames |
| Anomaly Scoring (offline) | Euclidean and PCA-Mahalanobis distance to the normal feature centre | Euclidean threshold τ = μ + 2σ of training scores; PCA with 10 components fitted on training features, covariance regularisation 1e-4 | Original and enhanced features are modelled separately, so raw scores are not directly comparable |
| Anomaly Scoring (dashboard) | Heuristic image-level score adjusted by scene, time and weather context, plus precomputed clip-level baseline scores | Rule-based adjustment | Distinct from the offline Swin3D pipeline; contextual information is not fed into the offline detector |
| Explainability | Heatmap and rule-based text explanations | Image-gradient and local-intensity heatmap; predefined rules | **Not Grad-CAM** and no LLM or learned reasoning. Grad-CAM is future work |
| Cold-Start Calibration | `ColdStartCalibrator` component exists | n/a | The dashboard uses **predefined calibration values**, not live calibration |
| Dashboard | Flask backend, HTML / CSS / JS frontend | `/api/analyze`, `/api/demo` | Image-level analysis only; no video decoding or real-time stream processing |

### Not implemented (future work)

CLIP-based scene understanding, Grad-CAM / spatiotemporal attention explanations, continual learning (e.g., EWC, parameter-efficient adaptation), real-time video processing, dedicated dehazing / deraining with validated weights, and weather-conditioned feature modelling.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask flask-cors scikit-learn scipy pillow numpy pandas pyyaml opencv-python-headless

# Launch dashboard
cd E:\WeatherSense-VAD
python main.py

# Or run CLI demo
python main.py --demo

# Analyze a specific image
python main.py --analyze path/to/image.jpg
```

Then open http://127.0.0.1:5000 in your browser.

## Dashboard

The dashboard analyses **single uploaded images** and shows:

- Drag-and-drop image upload
- Heuristic weather-condition indicators (fog, low light, rain, visibility)
- Side-by-side original vs. enhanced image (Zero-DCE pathway selected when low illumination is detected)
- Heuristic anomaly-related score with threshold visualisation
- Clip timeline built from precomputed clip-level baseline scores
- Gradient / intensity-based heatmap
- Rule-based textual explanation
- Metrics and continual-learning status panels

> The metrics and continual-learning indicators shown in the dashboard are **not** independently computed experimental results and do not reflect real model adaptation.

## Project Structure

```
WeatherSense-VAD/
├── main.py                     # Entry point (dashboard / demo / analyze)
├── config.yaml                 # System configuration
├── requirements.txt            # Dependencies
├── dashboard/
│   ├── app.py                  # Flask backend (heuristic analysis modules)
│   └── templates/index.html    # Dashboard UI
├── weathersense/               # Core library
│   ├── config.py               # Configuration loader
│   └── preprocessing/
│       ├── weather_assessment.py
│       ├── aod_net.py          # AOD-Net (no verified checkpoint)
│       └── weather_pipeline.py
├── models/                     # Feature extraction
│   ├── swin_feature_extractor.py
│   ├── anomaly_detector.py
│   └── extract_*.py
├── preprocessing/              # Zero-DCE
│   └── zero_dce/
├── training/                   # Dataset
│   └── dataset.py
├── evaluation/                 # Scoring scripts
├── data/                       # ShanghaiTech dataset
└── outputs/                    # Pre-computed features & scores
```

## Dataset

Uses the **ShanghaiTech Campus** dataset (frames organised in chronological order). Feature-extraction experiments are configured mainly around **Part A**; preprocessing scripts support both Parts A and B.

- Normal training clips estimate the reference feature centre and PCA / covariance statistics.
- Pre-extracted Swin3D-T features (768-d) for original and enhanced frames, evaluated on **22 test clips**.
- Crowd-count annotations are **not** treated as anomaly labels.
- Verified clip-level anomaly ground truth aligned with the 22 evaluated clips was not available, so predictions are model outputs, not verified detections.

## Experimental Observations (22 test clips, original vs. Zero-DCE-enhanced features)

| Scoring | Original mean | Enhanced mean | Clips lower after enhancement | Clips higher |
|---|---|---|---|---|
| Euclidean | 6.1438 | 5.8438 | 21 | 1 |
| PCA-Mahalanobis | 1.8222 | 1.8000 | 12 | 10 |

- Euclidean threshold: 7.7880 (original, 3 clips flagged) and 7.5486 (enhanced, 2 clips flagged).
- Lower mean distances after enhancement **must not** be read as improved detection performance, because the original and enhanced feature spaces are modelled separately and no verified labels exist.
- A separate STG-NF checkpoint evaluation on ShanghaiTech (AUC 85.9372%, 107 test clips scored) is an independent reference experiment. **It is not a WeatherSense-VAD result.**
- Zero-DCE results are qualitative only; PSNR / SSIM were not computed.

## Limitations

- Weather and scene assessment are image-statistics heuristics, not trained classifiers.
- Zero-DCE training is a limited proof-of-concept run (20 images, 1 epoch).
- AOD-Net has no verified checkpoint, so dehazing is unvalidated.
- Swin3D-T is used as a frozen pretrained extractor with no fine-tuning or comparison to other extractors.
- Distance-based scoring is simple, and ROC-AUC, AP, precision, recall and F1 cannot be reported without aligned labels.
- The dashboard works on images only; the contextual scoring is separate from the offline Swin3D detector.

## Environment

Reported experimental setup: Windows 11, Intel Core i5-1135G7 (2.40 GHz), 8 GB RAM, Intel UHD Graphics, PyTorch 2.13.0+cpu, **CPU-only** (DataLoader workers = 0). No GPU is required.

## License

Research use only.
