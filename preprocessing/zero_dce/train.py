import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from preprocessing.zero_dce.model import ZeroDCE, enhance_image
from preprocessing.zero_dce.losses import total_loss


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path(
    "data/shanghaitech/part_A/train_data/images"
)

OUTPUT_DIR = Path(
    "preprocessing/outputs/zero_dce"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE =  1
EPOCHS = 1
LEARNING_RATE = 1e-4


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class ShanghaiTechImages(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.files = sorted(
            self.root.glob("*.jpg"),
            key=lambda p: int(
                p.stem.split("_")[1]
            )
        )[:20]

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = ShanghaiTechImages(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Images:", len(dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ZeroDCE().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    for batch_idx, images in enumerate(loader):

        images = images.to(device)

        optimizer.zero_grad()

        curve = model(images)

# Zero-DCE outputs 24 channels:
# 8 enhancement steps × 3 RGB channels
        curve = curve.view(
            images.size(0),
            8,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        enhanced = images

        for i in range(8):

            current_curve = curve[:, i, :, :, :]

            enhanced = enhanced + current_curve * (
            enhanced * enhanced - enhanced
        )

        enhanced = torch.clamp(
            enhanced,
            0,
            1
        )
        loss, components = total_loss(
            images,
            enhanced
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        print(
            f"  Batch {batch_idx + 1}/{len(loader)} "
            f"| Loss: {loss.item():.6f}"
        )

    average_loss = (
        epoch_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = (
    OUTPUT_DIR / "zero_dce_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print()
print("Training complete!")
print("Model saved to:")
print(model_path)