from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.video import (
    swin3d_t,
    Swin3D_T_Weights
)


# --------------------------------------------------
# Dataset for pre-created sequences
# --------------------------------------------------

class SequenceDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.sequences = sorted(
            self.root_dir.glob("sequence_*"),
            key=lambda p: int(
                p.name.split("_")[-1]
            )
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence_dir = self.sequences[index]

        frames = sorted(
            sequence_dir.glob("*.jpg"),
            key=lambda p: int(
                p.stem.split("_")[-1]
            )
        )

        images = []

        for frame in frames:

            image = Image.open(
                frame
            ).convert("RGB")

            image = self.transform(image)

            images.append(image)

        return torch.stack(images)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = (
    "outputs/evaluation/full_test_sequences"
)

OUTPUT_PATH = Path(
    "outputs/features/"
    "part_A_full_test_features.pt"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SequenceDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("Device:", DEVICE)
print("Dataset size:", len(dataset))


# --------------------------------------------------
# Load Swin3D
# --------------------------------------------------

weights = Swin3D_T_Weights.DEFAULT

model = swin3d_t(
    weights=weights
)

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

        # [B, T, C, H, W]
        # →
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
# Combine
# --------------------------------------------------

features = torch.cat(
    all_features,
    dim=0
)

print()
print("Final feature shape:")
print(features.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    features,
    OUTPUT_PATH
)

print()
print("Features saved to:")
print(OUTPUT_PATH)