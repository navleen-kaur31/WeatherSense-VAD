from pathlib import Path


DATASET_ROOT = Path("data/shanghaitech")
OUTPUT_ROOT = Path("data/sequences")

SEQUENCE_LENGTH = 16
STRIDE = 8


def get_images(image_dir):
    return sorted(
        image_dir.glob("*.jpg"),
        key=lambda p: int(p.stem.split("_")[1])
    )


def create_sequences(part, split):
    image_dir = DATASET_ROOT / part / split / "images"
    output_dir = OUTPUT_ROOT / part / split

    output_dir.mkdir(parents=True, exist_ok=True)

    images = get_images(image_dir)

    sequence_count = 0

    for start in range(
        0,
        len(images) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):
        sequence = images[start:start + SEQUENCE_LENGTH]

        sequence_dir = output_dir / f"sequence_{sequence_count:04d}"
        sequence_dir.mkdir(exist_ok=True)

        for frame_index, image_path in enumerate(sequence):
            destination = sequence_dir / f"frame_{frame_index:02d}.jpg"

            # Copy the image without modifying the original
            destination.write_bytes(image_path.read_bytes())

        sequence_count += 1

    print(f"{part} - {split}")
    print(f"Frames: {len(images)}")
    print(f"Sequences created: {sequence_count}")


for part in ["part_A", "part_B"]:
    for split in ["train_data", "test_data"]:
        create_sequences(part, split)