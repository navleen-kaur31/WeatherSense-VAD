from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights
)

from training.dataset import ShanghaiTechDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    "outputs/features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Enhanced TEST dataset
# --------------------------------------------------

dataset = ShanghaiTechDataset(
    root_dir="data/shanghaitech",
    part="part_A",
    split="test_data",
    sequence_length=16,
    stride=8,
    image_size=224,
    enhanced=True
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("Device:", DEVICE)
print("Enhanced test dataset size:", len(dataset))


# --------------------------------------------------
# Load Swin3D
# --------------------------------------------------

weights = Swin3D_T_Weights.DEFAULT

model = swin3d_t(
    weights=weights
)

# Remove classification head
model.head = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Extract features
# --------------------------------------------------

all_features = []

with torch.no_grad():

    for batch_idx, clips in enumerate(loader):

        print(
            f"Batch {batch_idx + 1}/{len(loader)} "
            f"→ {clips.shape}"
        )

        # Dataset:
        # [B, T, C, H, W]

        # Swin3D:
        # [B, C, T, H, W]

        clips = clips.permute(
            0, 2, 1, 3, 4
        )

        clips = clips.to(DEVICE)

        features = model(clips)

        print(
            "   Features →",
            features.shape
        )

        all_features.append(
            features.cpu()
        )


# --------------------------------------------------
# Combine features
# --------------------------------------------------

features = torch.cat(
    all_features,
    dim=0
)

print()
print("Final enhanced test feature shape:")
print(features.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    OUTPUT_DIR /
    "part_A_test_enhanced_features.pt"
)

torch.save(
    features,
    output_path
)

print()
print("Enhanced test features saved to:")
print(output_path)