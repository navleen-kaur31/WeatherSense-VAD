import torch
from torch.utils.data import DataLoader
from torchvision.models.video import swin3d_t

from training.dataset import ShanghaiTechDataset


# -----------------------------
# 1. Load ShanghaiTech dataset
# -----------------------------

dataset = ShanghaiTechDataset(
    root_dir="data/shanghaitech",
    part="part_A",
    split="train_data",
    sequence_length=16,
    stride=8,
    image_size=224
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("Dataset size:", len(dataset))


# -----------------------------
# 2. Get one batch
# -----------------------------

clips = next(iter(loader))

print("Original batch shape:", clips.shape)


# -----------------------------
# 3. Change dimension order
# -----------------------------
# Dataset:
# [B, T, C, H, W]
#
# Swin3D:
# [B, C, T, H, W]

clips = clips.permute(0, 2, 1, 3, 4)

print("Swin3D input shape:", clips.shape)


# -----------------------------
# 4. Load model
# -----------------------------

model = swin3d_t(weights=None)
model.eval()


# -----------------------------
# 5. Forward pass
# -----------------------------

with torch.no_grad():
    output = model(clips)


print("Swin3D output shape:", output.shape)