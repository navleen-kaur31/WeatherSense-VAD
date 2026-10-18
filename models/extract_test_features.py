import torch
from torch.utils.data import DataLoader
from pathlib import Path

from training.dataset import ShanghaiTechDataset
from models.swin_feature_extractor import Swin3DFeatureExtractor


device = torch.device("cpu")

print("Device:", device)


# -----------------------------
# Test dataset
# -----------------------------

dataset = ShanghaiTechDataset(
    root_dir="data/shanghaitech",
    part="part_A",
    split="test_data",
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

print("Test dataset size:", len(dataset))


# -----------------------------
# Pretrained Swin3D
# -----------------------------

model = Swin3DFeatureExtractor()
model.to(device)
model.eval()


# -----------------------------
# Extract features
# -----------------------------

all_features = []

with torch.no_grad():

    for batch_index, clips in enumerate(loader):

        # [B, T, C, H, W]
        clips = clips.permute(0, 2, 1, 3, 4)

        clips = clips.to(device)

        features = model(clips)

        all_features.append(features.cpu())

        print(
            f"Batch {batch_index + 1}/{len(loader)} "
            f"→ {features.shape}"
        )


# -----------------------------
# Combine
# -----------------------------

all_features = torch.cat(
    all_features,
    dim=0
)

print("\nFinal test feature shape:", all_features.shape)


# -----------------------------
# Save
# -----------------------------

output_dir = Path("outputs/features")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir / "part_A_test_features.pt"
)

torch.save(
    all_features,
    output_file
)

print("Test features saved to:")
print(output_file)