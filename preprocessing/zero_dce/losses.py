import torch
import torch.nn.functional as F


def illumination_loss(enhanced):
    """
    Encourages neighboring pixels to have similar illumination.
    """

    gray = (
        0.299 * enhanced[:, 0:1]
        + 0.587 * enhanced[:, 1:2]
        + 0.114 * enhanced[:, 2:3]
    )

    dx = torch.abs(gray[:, :, :, :-1] - gray[:, :, :, 1:])
    dy = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])

    return dx.mean() + dy.mean()


def spatial_consistency_loss(original, enhanced):
    """
    Encourages the enhanced image to preserve spatial structure
    from the original image.
    """

    original_gray = (
        0.299 * original[:, 0:1]
        + 0.587 * original[:, 1:2]
        + 0.114 * original[:, 2:3]
    )

    enhanced_gray = (
        0.299 * enhanced[:, 0:1]
        + 0.587 * enhanced[:, 1:2]
        + 0.114 * enhanced[:, 2:3]
    )

    original_dx = original_gray[:, :, :, :-1] - original_gray[:, :, :, 1:]
    original_dy = original_gray[:, :, :-1, :] - original_gray[:, :, 1:, :]

    enhanced_dx = enhanced_gray[:, :, :, :-1] - enhanced_gray[:, :, :, 1:]
    enhanced_dy = enhanced_gray[:, :, :-1, :] - enhanced_gray[:, :, 1:, :]

    loss_x = F.l1_loss(enhanced_dx, original_dx)
    loss_y = F.l1_loss(enhanced_dy, original_dy)

    return loss_x + loss_y


def color_constancy_loss(enhanced):
    """
    Encourages balanced RGB channels.
    """

    mean_rgb = enhanced.mean(dim=(2, 3))

    r = mean_rgb[:, 0]
    g = mean_rgb[:, 1]
    b = mean_rgb[:, 2]

    loss_rg = torch.abs(r - g).mean()
    loss_rb = torch.abs(r - b).mean()
    loss_gb = torch.abs(g - b).mean()

    return loss_rg + loss_rb + loss_gb


def exposure_control_loss(enhanced, target=0.6):
    """
    Encourages the enhanced image to have a reasonable
    average exposure.
    """

    gray = (
        0.299 * enhanced[:, 0:1]
        + 0.587 * enhanced[:, 1:2]
        + 0.114 * enhanced[:, 2:3]
    )

    mean = F.avg_pool2d(
        gray,
        kernel_size=16,
        stride=16
    )

    target_tensor = torch.full_like(mean, target)

    return torch.mean(torch.abs(mean - target_tensor))


def total_loss(original, enhanced):
    """
    Combined Zero-DCE training loss.
    """

    loss_spatial = spatial_consistency_loss(
        original,
        enhanced
    )

    loss_illumination = illumination_loss(
        enhanced
    )

    loss_color = color_constancy_loss(
        enhanced
    )

    loss_exposure = exposure_control_loss(
        enhanced
    )

    total = (
        1.0 * loss_spatial
        + 1.0 * loss_illumination
        + 5.0 * loss_color
        + 10.0 * loss_exposure
    )

    return total, {
        "spatial": loss_spatial.item(),
        "illumination": loss_illumination.item(),
        "color": loss_color.item(),
        "exposure": loss_exposure.item(),
    }