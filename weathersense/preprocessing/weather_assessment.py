"""
Weather Condition Assessment Module.

Classifies weather and image quality conditions from surveillance frames:
- Fog / haze detection (dark channel prior)
- Low-light detection (average luminance)
- Rain detection (streak pattern analysis)
- Overall visibility quality score

These assessments guide which enhancement models to apply.
"""

import numpy as np
from PIL import Image


class WeatherAssessor:
    """
    Assesses weather and visibility conditions from image frames.

    Uses lightweight image statistics (no deep model required)
    to determine which enhancement pipeline to apply.
    """

    def __init__(
        self,
        fog_threshold=0.5,
        low_light_threshold=0.3,
        rain_threshold=0.4,
    ):
        self.fog_threshold = fog_threshold
        self.low_light_threshold = low_light_threshold
        self.rain_threshold = rain_threshold

    def assess(self, image):
        """
        Assess weather conditions for a single image.

        Args:
            image: PIL Image or numpy array (H, W, 3) in [0, 255].

        Returns:
            dict with condition scores and recommended actions.
        """
        if isinstance(image, Image.Image):
            image_np = np.asarray(image).astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image_np = image.astype(np.float32) / 255.0
        else:
            image_np = image.astype(np.float32)

        fog_score = self._fog_score(image_np)
        low_light_score = self._low_light_score(image_np)
        rain_score = self._rain_score(image_np)
        visibility = self._visibility_quality(image_np)

        conditions = {
            "fog_score": float(fog_score),
            "low_light_score": float(low_light_score),
            "rain_score": float(rain_score),
            "visibility_quality": float(visibility),
            "is_foggy": fog_score > self.fog_threshold,
            "is_low_light": low_light_score > self.low_light_threshold,
            "is_rainy": rain_score > self.rain_threshold,
        }

        # Determine recommended enhancement
        conditions["recommended_enhancement"] = (
            self._recommend_enhancement(conditions)
        )

        return conditions

    def _fog_score(self, image_np):
        """
        Estimate fog/haze level using Dark Channel Prior.

        In foggy images, the dark channel has higher values
        because haze adds a constant brightness offset.
        """
        # Dark channel: min across RGB, then min in local patch
        dark_channel = image_np.min(axis=2)

        # Use 15×15 patches
        from scipy.ndimage import minimum_filter
        dark_channel_local = minimum_filter(
            dark_channel, size=15
        )

        # High dark channel mean → more haze
        fog_score = dark_channel_local.mean()

        return np.clip(fog_score, 0.0, 1.0)

    def _low_light_score(self, image_np):
        """
        Estimate low-light level from average luminance.

        Low luminance → high low-light score.
        """
        # Convert to grayscale using ITU-R BT.601
        gray = (
            0.299 * image_np[:, :, 0]
            + 0.587 * image_np[:, :, 1]
            + 0.114 * image_np[:, :, 2]
        )

        mean_luminance = gray.mean()

        # Invert: low luminance → high score
        low_light_score = 1.0 - mean_luminance

        # Also check the proportion of dark pixels
        dark_ratio = (gray < 0.2).mean()

        # Weighted combination
        score = 0.6 * low_light_score + 0.4 * dark_ratio

        return np.clip(score, 0.0, 1.0)

    def _rain_score(self, image_np):
        """
        Estimate rain presence using streak pattern analysis.

        Rain streaks create high-frequency directional patterns
        in the vertical direction.
        """
        gray = (
            0.299 * image_np[:, :, 0]
            + 0.587 * image_np[:, :, 1]
            + 0.114 * image_np[:, :, 2]
        )

        # Vertical gradient (rain streaks are mostly vertical)
        dy = np.abs(np.diff(gray, axis=0))
        # Horizontal gradient
        dx = np.abs(np.diff(gray, axis=1))

        # Rain creates strong vertical streaks
        vertical_energy = dy.mean()
        horizontal_energy = dx.mean()

        # Ratio of vertical to horizontal gradients
        if horizontal_energy > 1e-6:
            directional_ratio = vertical_energy / horizontal_energy
        else:
            directional_ratio = 1.0

        # High-frequency content in vertical direction
        # suggests rain if ratio is significantly > 1
        rain_score = np.clip(
            (directional_ratio - 1.0) * 2.0, 0.0, 1.0
        )

        return float(rain_score)

    def _visibility_quality(self, image_np):
        """
        Overall visibility quality score (0=poor, 1=excellent).

        Combines contrast, brightness, and edge sharpness.
        """
        gray = (
            0.299 * image_np[:, :, 0]
            + 0.587 * image_np[:, :, 1]
            + 0.114 * image_np[:, :, 2]
        )

        # Contrast (standard deviation of luminance)
        contrast = gray.std()

        # Brightness (should be moderate, not too dark or bright)
        brightness = gray.mean()
        brightness_quality = 1.0 - 2.0 * abs(brightness - 0.5)

        # Edge sharpness (Laplacian variance)
        from scipy.ndimage import laplace
        laplacian = laplace(gray)
        sharpness = min(laplacian.var() * 100, 1.0)

        quality = (
            0.4 * contrast
            + 0.3 * brightness_quality
            + 0.3 * sharpness
        )

        return np.clip(quality, 0.0, 1.0)

    def _recommend_enhancement(self, conditions):
        """
        Recommend which enhancement model(s) to apply.

        Returns list of recommended enhancement methods.
        """
        recommendations = []

        if conditions["is_foggy"]:
            recommendations.append("aod_net")

        if conditions["is_low_light"]:
            recommendations.append("zero_dce")

        if not recommendations:
            recommendations.append("none")

        return recommendations

    def assess_batch(self, images):
        """
        Assess weather conditions for a batch of images.

        Args:
            images: list of PIL Images or numpy arrays.

        Returns:
            list of condition dicts.
        """
        return [self.assess(img) for img in images]
