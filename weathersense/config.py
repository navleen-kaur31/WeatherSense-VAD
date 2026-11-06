"""
Configuration loader for WeatherSense-VAD.

Loads config.yaml and provides a structured config object
with sensible defaults for all modules.
"""

import os
from pathlib import Path

import yaml


# Default configuration values
_DEFAULTS = {
    "system": {
        "name": "WeatherSense-VAD",
        "version": "0.1.0",
        "device": "auto",
        "seed": 42,
    },
    "data": {
        "root_dir": "data/shanghaitech",
        "sequences_dir": "data/sequences",
        "gt_mask_dir": "data/shanghaitech/gt/test_frame_mask",
        "pose_dir": "data/shanghaitech/pose/test",
        "sequence_length": 16,
        "stride": 8,
        "image_size": 224,
    },
    "weather": {
        "assessment": {
            "enabled": True,
            "fog_threshold": 0.5,
            "low_light_threshold": 0.3,
            "rain_threshold": 0.4,
        },
        "zero_dce": {
            "enabled": True,
            "model_path": "preprocessing/outputs/zero_dce/zero_dce_model.pth",
            "enhancement_iterations": 8,
        },
        "aod_net": {
            "enabled": True,
            "model_path": "checkpoints/aod_net.pth",
        },
    },
    "scene": {
        "clip": {
            "enabled": True,
            "model_name": "openai/clip-vit-base-patch32",
            "scene_categories": [
                "indoor corridor",
                "outdoor street",
                "parking lot",
                "campus walkway",
                "building entrance",
                "plaza or open area",
                "staircase",
                "road intersection",
            ],
        },
    },
    "temporal": {
        "enabled": True,
        "time_embedding_dim": 32,
        "use_cyclical_encoding": True,
    },
    "features": {
        "backbone": "swin3d_t",
        "feature_dim": 768,
        "batch_size": 2,
        "output_dir": "outputs/features",
    },
    "anomaly": {
        "euclidean": {"enabled": True},
        "mahalanobis": {"enabled": True, "n_components": 10},
        "context_aware": {
            "enabled": True,
            "scene_weight": 0.3,
            "temporal_weight": 0.2,
            "visual_weight": 0.5,
        },
        "threshold_method": "adaptive",
        "threshold_sigma": 2.0,
        "output_dir": "outputs/scores",
    },
    "cold_start": {
        "enabled": True,
        "calibration_frames": 100,
        "min_confidence": 0.7,
        "warmup_epochs": 5,
    },
    "continual_learning": {
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["qkv", "proj"],
        },
        "ewc": {
            "enabled": True,
            "lambda_ewc": 1000.0,
            "fisher_samples": 200,
        },
    },
    "explainability": {
        "grad_cam": {
            "enabled": True,
            "target_layer": "model.features",
        },
        "reasoning": {
            "enabled": True,
            "generate_text": True,
            "generate_heatmap": True,
        },
    },
    "evaluation": {
        "metrics": [
            "roc_auc",
            "precision",
            "recall",
            "f1_score",
            "average_precision",
        ],
        "output_dir": "outputs/evaluation",
    },
    "dashboard": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
        "upload_dir": "uploads",
        "results_dir": "outputs/dashboard",
    },
}


def _deep_merge(base, override):
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration container with attribute-style access."""

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(config_path=None, project_root=None):
    """
    Load configuration from YAML file with defaults.

    Args:
        config_path: Path to config.yaml. If None, looks in project root.
        project_root: Project root directory. If None, auto-detected.

    Returns:
        Config object with all settings.
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    if config_path is None:
        config_path = project_root / "config.yaml"

    config_data = _DEFAULTS.copy()

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config_data = _deep_merge(config_data, user_config)

    # Resolve device
    if config_data["system"]["device"] == "auto":
        try:
            import torch
            config_data["system"]["device"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            config_data["system"]["device"] = "cpu"

    # Store project root
    config_data["system"]["project_root"] = str(project_root)

    return Config(config_data)
