# Changelog

All notable changes to WeatherSense-VAD are documented here.

## [0.1.0] - 2026-10-25

### Added
- **Unified dashboard** (`dashboard/app.py`, `dashboard/templates/index.html`)
  - Flask-based web interface with drag-and-drop image upload
  - Real-time weather assessment visualization
  - Before/after image comparison (original vs enhanced)
  - Anomaly score gauge with context-aware scoring
  - Clip timeline visualization from pre-computed baseline
  - Grad-CAM-style attention heatmap generation
  - Human-readable anomaly explanation with contributing factors
  - Scene classification display
  - Temporal context panel
  - Continual learning status panel
  - Detection metrics panel (ROC-AUC, precision, recall, F1)
  - System info panel
  - Demo mode using ShanghaiTech dataset images

- **Weather assessment module** (`weathersense/preprocessing/weather_assessment.py`)
  - Fog/haze detection using Dark Channel Prior
  - Low-light detection using luminance statistics
  - Rain detection using directional gradient analysis
  - Overall visibility quality scoring
  - Automatic enhancement recommendation

- **AOD-Net dehazing model** (`weathersense/preprocessing/aod_net.py`)
  - All-in-One Dehazing Network implementation
  - Multi-scale feature extraction with concatenation
  - Atmospheric scattering model reformulation

- **Unified preprocessing pipeline** (`weathersense/preprocessing/weather_pipeline.py`)
  - Integrates weather assessment with Zero-DCE and AOD-Net
  - Automatic model selection based on conditions
  - Lazy model loading for memory efficiency

- **Configuration system** (`weathersense/config.py`, `config.yaml`)
  - YAML-based configuration with defaults
  - Attribute-style access
  - Auto-detection of device (CPU/CUDA)

- **Main entry point** (`main.py`)
  - `--dashboard` mode (default): launches web UI
  - `--demo` mode: CLI pipeline demonstration
  - `--analyze IMG` mode: single image analysis

- `requirements.txt` with all dependencies
- `IMPLEMENTATION_STATUS.md` with full component audit

### Preserved (from original repository)
- Zero-DCE model, losses, training, and enhancement scripts
- Swin3D-T feature extractor
- ShanghaiTech dataset loader with sequence windowing
- All feature extraction scripts (6 variants)
- All evaluation scripts (baseline, Mahalanobis, comparisons)
- Pre-computed features, scores, and evaluation CSVs
- Ground truth frame masks (107 .npy files)
- AlphaPose pose data
- Zero-DCE trained checkpoint (`zero_dce_model.pth`)
