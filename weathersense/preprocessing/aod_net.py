"""
AOD-Net: All-in-One Dehazing Network.

Implements the AOD-Net architecture from:
"AOD-Net: All-in-One Dehazing Network" (Li et al., ICCV 2017)

The atmospheric scattering model is:
    I(x) = J(x) * t(x) + A * (1 - t(x))

where:
    I(x) = observed hazy image
    J(x) = clean scene radiance (dehazed)
    t(x) = transmission map
    A = global atmospheric light

AOD-Net reformulates this as:
    J(x) = K(x) * I(x) - K(x) + b

where K(x) is a learned transformation and b is a bias.
"""

import torch
import torch.nn as nn


class AODNet(nn.Module):
    """
    All-in-One Dehazing Network.

    Lightweight CNN that directly estimates the dehazed image
    from a hazy input without explicitly computing the
    transmission map or atmospheric light.
    """

    def __init__(self):
        super().__init__()

        # Feature extraction layers with concatenation
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input hazy image tensor [B, 3, H, W] in [0, 1].

        Returns:
            Dehazed image tensor [B, 3, H, W] in [0, 1].
        """
        # Multi-scale feature extraction with concatenation
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.relu(self.conv3(cat1))

        cat2 = torch.cat([x2, x3], dim=1)
        x4 = self.relu(self.conv4(cat2))

        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = self.relu(self.conv5(cat3))

        # AOD-Net reformulation: J(x) = K(x) * I(x) - K(x) + b
        # where b = 1 (bias)
        output = k * x - k + 1.0

        return torch.clamp(output, 0.0, 1.0)


def dehaze_image(image_tensor, model, device="cpu"):
    """
    Dehaze a single image using AOD-Net.

    Args:
        image_tensor: [1, 3, H, W] or [3, H, W] tensor in [0, 1].
        model: Loaded AODNet model.
        device: Device to run on.

    Returns:
        Dehazed tensor, same shape as input.
    """
    was_3d = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        was_3d = True

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        dehazed = model(image_tensor)

    if was_3d:
        dehazed = dehazed.squeeze(0)

    return dehazed
