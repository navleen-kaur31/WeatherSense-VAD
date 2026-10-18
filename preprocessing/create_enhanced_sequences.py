from pathlib import Path
from PIL import Image
import shutil


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path(
    "preprocessing/outputs/zero_dce/part_A_train"
)

OUTPUT_DIR = Path(
    "preprocessing/outputs/zero_dce/sequences/part_A_train"
)

SEQUENCE_LENGTH = 16
STRIDE = 8


# --------------------------------------------------
# Get images in NUMERICAL order
# --------------------------------------------------

images = sorted(
    INPUT_DIR.glob("IMG_*_enhanced.jpg"),
    key=lambda p: int(
        p.stem.split("_")[1]
    )
)

print("Enhanced images:", len(images))


# --------------------------------------------------
# Create sequences
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sequence_count = 0

for start in range(
    0,
    len(images) - SEQUENCE_LENGTH + 1,
    STRIDE
):

    sequence_dir = OUTPUT_DIR / (
        f"sequence_{sequence_count:04d}"
    )

    sequence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sequence_images = images[
        start:start + SEQUENCE_LENGTH
    ]

    for frame_idx, image_path in enumerate(
        sequence_images
    ):

        output_path = (
            sequence_dir /
            f"frame_{frame_idx:02d}.jpg"
        )

        shutil.copy2(
            image_path,
            output_path
        )

    print(
        f"Sequence {sequence_count:04d}: "
        f"{sequence_images[0].name} → "
        f"{sequence_images[-1].name}"
    )

    sequence_count += 1


print()
print("=" * 50)
print("Enhanced sequence creation complete!")
print("=" * 50)

print("Images:", len(images))
print("Sequences:", sequence_count)
print("Sequence length:", SEQUENCE_LENGTH)
print("Stride:", STRIDE)
print("Output:", OUTPUT_DIR)