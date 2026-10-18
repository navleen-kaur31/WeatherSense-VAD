from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ShanghaiTechDataset(Dataset):

    def __init__(
        self,
        root_dir,
        part="part_A",
        split="train_data",
        sequence_length=16,
        stride=8,
        image_size=224,
        enhanced=False
    ):
        self.root_dir = Path(root_dir)
        self.part = part
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.enhanced = enhanced

        # --------------------------------------------------
        # Original dataset
        # --------------------------------------------------

        if not enhanced:

            self.image_dir = (
                self.root_dir
                / part
                / split
                / "images"
            )

            # Numeric ordering:
            # IMG_1, IMG_2, ..., IMG_10
            self.images = sorted(
                self.image_dir.glob("*.jpg"),
                key=lambda p: int(
                    p.stem.split("_")[1]
                )
            )

            # Create sequences
            self.sequences = []

            for start in range(
                0,
                len(self.images) - sequence_length + 1,
                stride
            ):
                self.sequences.append(
                    self.images[
                        start:start + sequence_length
                    ]
                )

        # --------------------------------------------------
        # Zero-DCE enhanced dataset
        # --------------------------------------------------

        else:

            enhanced_root = (
                Path("preprocessing")
                / "outputs"
                / "zero_dce"
                / "sequences"
                / f"{part}_{split.replace('_data', '')}"
            )

            # For part_A + train_data:
            # preprocessing/outputs/zero_dce/
            # sequences/part_A_train

            sequence_dirs = sorted(
                enhanced_root.glob("sequence_*")
            )

            self.sequences = []

            for sequence_dir in sequence_dirs:

                frames = sorted(
                    sequence_dir.glob("*.jpg"),
                    key=lambda p: int(
                        p.stem.split("_")[1]
                    )
                )

                if len(frames) == sequence_length:
                    self.sequences.append(frames)

            self.image_dir = enhanced_root

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        frame_paths = self.sequences[index]

        frames = []

        for frame_path in frame_paths:

            image = Image.open(
                frame_path
            ).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        # [T, C, H, W]
        clip = torch.stack(frames)

        return clip


if __name__ == "__main__":

    # ---------------------------------------------
    # Test original dataset
    # ---------------------------------------------

    dataset = ShanghaiTechDataset(
        root_dir="data/shanghaitech",
        part="part_A",
        split="train_data",
        enhanced=False
    )

    print("Original dataset")
    print("Dataset size:", len(dataset))

    clip = dataset[0]

    print("Clip shape:", clip.shape)
    print("Clip dtype:", clip.dtype)

    # ---------------------------------------------
    # Test enhanced dataset
    # ---------------------------------------------

    enhanced_dataset = ShanghaiTechDataset(
        root_dir="data/shanghaitech",
        part="part_A",
        split="train_data",
        enhanced=True
    )

    print()
    print("Enhanced dataset")
    print("Dataset size:", len(enhanced_dataset))

    enhanced_clip = enhanced_dataset[0]

    print(
        "Clip shape:",
        enhanced_clip.shape
    )

    print(
        "Clip dtype:",
        enhanced_clip.dtype
    )