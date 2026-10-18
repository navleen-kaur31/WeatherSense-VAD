import torch
import torch.nn as nn
from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights
)


class Swin3DFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        weights = Swin3D_T_Weights.DEFAULT

        self.model = swin3d_t(weights=weights)

        # Remove Kinetics-400 classification head
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    print("Loading pretrained Swin3D...")

    model = Swin3DFeatureExtractor()
    model.eval()

    x = torch.randn(1, 3, 16, 224, 224)

    with torch.no_grad():
        features = model(x)

    print("Input shape:", x.shape)
    print("Feature shape:", features.shape)