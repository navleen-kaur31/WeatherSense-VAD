from pathlib import Path
from PIL import Image
from scipy.io import loadmat


DATASET_ROOT = Path("data/shanghaitech")


def load_ground_truth(gt_file):
    """Load point annotations from a ShanghaiTech .mat file."""
    data = loadmat(
        gt_file,
        squeeze_me=True,
        struct_as_record=False
    )

    image_info = data["image_info"]
    locations = image_info.location

    return locations


def verify_split(part, split):
    images_dir = DATASET_ROOT / part / split / "images"
    gt_dir = DATASET_ROOT / part / split / "ground-truth"

    images = sorted(
        images_dir.glob("*.jpg"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    print(f"\nChecking {part} - {split}")
    print("-" * 50)

    total_points = 0
    invalid_files = 0

    for image_path in images:

        # Corresponding ground-truth filename
        image_number = image_path.stem.split("_")[1]
        gt_path = gt_dir / f"GT_IMG_{image_number}.mat"

        if not gt_path.exists():
            print(f"Missing GT: {gt_path.name}")
            invalid_files += 1
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Load annotations
        locations = load_ground_truth(gt_path)

        # Check coordinates
        if len(locations) > 0:
            x = locations[:, 0]
            y = locations[:, 1]

            invalid_points = (
                (x < 0) |
                (x >= width) |
                (y < 0) |
                (y >= height)
            )

            invalid_count = invalid_points.sum()

            if invalid_count > 0:
                print(
                    f"{image_path.name}: "
                    f"{invalid_count} invalid points "
                    f"(image size: {width}x{height})"
                )
                invalid_files += 1

            total_points += len(locations)

    print(f"Images checked: {len(images)}")
    print(f"Total annotated points: {total_points}")
    print(f"Files with problems: {invalid_files}")

    if invalid_files == 0:
        print("✓ All annotations are within image boundaries")
    else:
        print("⚠ Some annotations need inspection")


# Check all ShanghaiTech splits
for part in ["part_A", "part_B"]:
    for split in ["train_data", "test_data"]:
        verify_split(part, split)