"""
Unified Weather-Robust Preprocessing Pipeline.

Integrates weather assessment with appropriate enhancement models:
1. Assess weather conditions (fog, low-light, rain)
2. Apply Zero-DCE for low-light scenes
3. Apply AOD-Net for foggy/hazy scenes
4. Pass through clear images without modification

This module bridges the existing Zero-DCE implementation with
the new AOD-Net and weather assessment components.
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch

from weathersense.preprocessing.weather_assessment import WeatherAssessor


class WeatherPreprocessor:
    """
    Unified weather-robust preprocessing pipeline.

    Automatically detects weather conditions and applies
    the appropriate enhancement model(s).
    """

    def __init__(self, config, device="cpu"):
        """
        Args:
            config: Config object with weather settings.
            device: torch device.
        """
        self.config = config
        self.device = torch.device(device)

        # Weather assessor
        weather_cfg = config.weather.assessment
        self.assessor = WeatherAssessor(
            fog_threshold=weather_cfg.fog_threshold,
            low_light_threshold=weather_cfg.low_light_threshold,
            rain_threshold=weather_cfg.rain_threshold,
        )

        # Enhancement models (loaded lazily)
        self._zero_dce_model = None
        self._aod_net_model = None

    def _load_zero_dce(self):
        """Lazy-load the Zero-DCE model."""
        if self._zero_dce_model is not None:
            return self._zero_dce_model

        from preprocessing.zero_dce.model import ZeroDCE

        model = ZeroDCE().to(self.device)

        model_path = self.config.weather.zero_dce.model_path
        if Path(model_path).exists():
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Zero-DCE model loaded from {model_path}")
        else:
            print(
                f"Warning: Zero-DCE model not found at {model_path}. "
                "Using untrained model."
            )

        model.eval()
        self._zero_dce_model = model
        return model

    def _load_aod_net(self):
        """Lazy-load the AOD-Net model."""
        if self._aod_net_model is not None:
            return self._aod_net_model

        from weathersense.preprocessing.aod_net import AODNet

        model = AODNet().to(self.device)

        model_path = self.config.weather.aod_net.model_path
        if Path(model_path).exists():
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(checkpoint)
            print(f"AOD-Net model loaded from {model_path}")
        else:
            print(
                f"Note: AOD-Net checkpoint not found at {model_path}. "
                "Using randomly initialized weights. "
                "Dehazing quality will be limited until trained."
            )

        model.eval()
        self._aod_net_model = model
        return model

    def _enhance_zero_dce(self, image_tensor):
        """
        Apply Zero-DCE enhancement.

        Uses the same 8-iteration channel-sliced approach
        as the existing enhance.py to maintain consistency.

        Args:
            image_tensor: [1, 3, H, W] tensor in [0, 1].

        Returns:
            Enhanced tensor [1, 3, H, W] in [0, 1].
        """
        model = self._load_zero_dce()

        with torch.no_grad():
            curve = model(image_tensor)

            enhanced = image_tensor.clone()

            # 8 enhancement iterations with per-channel curves
            # (consistent with preprocessing/zero_dce/enhance.py)
            iterations = self.config.weather.zero_dce.enhancement_iterations
            for i in range(iterations):
                current_curve = curve[:, i * 3:(i + 1) * 3]
                enhanced = enhanced + (
                    current_curve * (enhanced * enhanced - enhanced)
                )

        return torch.clamp(enhanced, 0.0, 1.0)

    def _enhance_aod_net(self, image_tensor):
        """
        Apply AOD-Net dehazing.

        Args:
            image_tensor: [1, 3, H, W] tensor in [0, 1].

        Returns:
            Dehazed tensor [1, 3, H, W] in [0, 1].
        """
        model = self._load_aod_net()

        with torch.no_grad():
            dehazed = model(image_tensor)

        return torch.clamp(dehazed, 0.0, 1.0)

    def preprocess_image(self, image, return_assessment=False):
        """
        Preprocess a single image with weather-appropriate enhancement.

        Args:
            image: PIL Image or numpy array (H, W, 3).
            return_assessment: If True, also return the weather assessment.

        Returns:
            Enhanced PIL Image (or tuple with assessment dict).
        """
        # Assess weather conditions
        assessment = self.assessor.assess(image)

        # Convert to tensor
        if isinstance(image, Image.Image):
            image_np = np.asarray(image).astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image_np = image.astype(np.float32) / 255.0
        else:
            image_np = image.astype(np.float32)

        image_tensor = torch.from_numpy(image_np)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Apply recommended enhancement(s)
        enhanced = image_tensor
        applied_methods = []

        recommendations = assessment["recommended_enhancement"]

        if "aod_net" in recommendations and self.config.weather.aod_net.enabled:
            enhanced = self._enhance_aod_net(enhanced)
            applied_methods.append("aod_net")

        if "zero_dce" in recommendations and self.config.weather.zero_dce.enabled:
            enhanced = self._enhance_zero_dce(enhanced)
            applied_methods.append("zero_dce")

        assessment["applied_enhancement"] = (
            applied_methods if applied_methods else ["none"]
        )

        # Convert back to PIL Image
        enhanced_np = enhanced.squeeze(0).permute(1, 2, 0)
        enhanced_np = (enhanced_np.cpu().numpy() * 255.0).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_np)

        if return_assessment:
            return enhanced_image, assessment
        return enhanced_image

    def preprocess_batch(self, images, return_assessments=False):
        """
        Preprocess a batch of images.

        Args:
            images: list of PIL Images.
            return_assessments: If True, also return assessments.

        Returns:
            list of enhanced PIL Images (or tuples with assessments).
        """
        results = []
        for image in images:
            result = self.preprocess_image(
                image,
                return_assessment=return_assessments,
            )
            results.append(result)
        return results
