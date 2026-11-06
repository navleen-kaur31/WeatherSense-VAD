# WeatherSense-VAD

**Weather-Robust, Context-Aware and Continually Adaptive Video Anomaly Detection System**

A research-oriented video anomaly detection framework for heterogeneous real-world surveillance environments. Integrates weather-robust preprocessing, scene understanding, temporal context modeling, and explainable anomaly detection into a unified pipeline with an interactive web dashboard.

---

## Key Features

| Module | Description | Method |
|--------|-------------|--------|
| Weather Assessment | Automatic fog, low-light, rain detection | Dark Channel Prior, luminance statistics |
| Visual Enhancement | Conditional image enhancement | Zero-DCE (low-light), AOD-Net (dehazing) |
| Scene Understanding | Place-conditioned scene classification | Image statistics-based scoring |
| Temporal Context | Time-of-day behavioral modeling | Cyclical encoding, activity curves |
| Feature Extraction | Spatiotemporal feature representation | Video Swin Transformer (Swin3D-T, 768-dim) |
| Anomaly Scoring | Context-aware anomaly detection | Euclidean + Mahalanobis distance |
| Cold-Start | Self-calibrating for new scenes | Adaptive threshold estimation |
| Explainability | Visual evidence + reasoning | Grad-CAM heatmaps, textual explanations |
| Dashboard | Interactive web interface | Flask-based single-page application |

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

Then open **http://127.0.0.1:5000** in your browser.

## Dashboard

The web dashboard provides:
- Drag-and-drop image upload
- Real-time weather condition assessment
- Side-by-side original vs enhanced comparison
- Anomaly score with threshold visualization
- Clip timeline with normal/anomaly segments
- Grad-CAM attention heatmap
- Human-readable anomaly explanation
- Detection metrics panel

## Project Structure

```
WeatherSense-VAD/
├── main.py                     # Entry point (dashboard / demo / analyze)
├── config.yaml                 # System configuration
├── requirements.txt            # Dependencies
├── dashboard/
│   ├── app.py                  # Flask backend (all 7 modules integrated)
│   └── templates/index.html    # Dashboard UI
├── weathersense/               # Core library
│   ├── config.py               # Configuration loader
│   └── preprocessing/
│       ├── weather_assessment.py
│       ├── aod_net.py
│       └── weather_pipeline.py
├── models/                     # Feature extraction (preserved)
│   ├── swin_feature_extractor.py
│   ├── anomaly_detector.py
│   └── extract_*.py
├── preprocessing/              # Zero-DCE (preserved)
│   └── zero_dce/
├── training/                   # Dataset (preserved)
│   └── dataset.py
├── evaluation/                 # Scoring scripts (preserved)
├── data/                       # ShanghaiTech dataset
└── outputs/                    # Pre-computed features & scores
```

## Dataset

Uses **ShanghaiTech Campus** dataset with:
- Part A/B train and test images with ground-truth
- 107 ground-truth frame masks
- Pre-extracted Swin3D-T features (768-dim, 22 temporal windows)
- Pre-computed anomaly scores and baseline results

## Environment

- Python 3.12+
- PyTorch 2.x (CPU-only supported)
- Windows / Linux
- No GPU required for inference and dashboard

## License

Research use only.
