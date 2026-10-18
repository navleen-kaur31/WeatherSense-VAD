import torch
from torchvision.models.video import swin3d_t


print("Loading Swin3D...")

model = swin3d_t(weights=None)
model.eval()

# TorchVision expects [B, C, T, H, W]
x = torch.randn(1, 3, 16, 224, 224)

print("Input shape:", x.shape)

with torch.no_grad():
    output = model(x)

print("Output shape:", output.shape)