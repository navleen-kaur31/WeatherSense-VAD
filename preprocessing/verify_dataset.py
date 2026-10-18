from pathlib import Path

DATASET_ROOT = Path("data/shanghaitech")


def check_split(part, split):
    images_dir = DATASET_ROOT / part / split / "images"
    gt_dir = DATASET_ROOT / part / split / "ground-truth"

    images = sorted(images_dir.glob("*.jpg"))
    gt_files = sorted(gt_dir.glob("*.mat"))

    print(f"\n{part} - {split}")
    print("-" * 40)
    print(f"Images:       {len(images)}")
    print(f"Ground truth: {len(gt_files)}")

    if len(images) == len(gt_files):
        print("✓ Image/ground-truth counts match")
    else:
        print("✗ Counts DO NOT match")

    if images:
        print(f"First image: {images[0].name}")

    if gt_files:
        print(f"First GT:    {gt_files[0].name}")


for part in ["part_A", "part_B"]:
    for split in ["train_data", "test_data"]:
        check_split(part, split)