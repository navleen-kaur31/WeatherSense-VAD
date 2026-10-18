from pathlib import Path
from PIL import Image

IMAGE_DIR = Path(
    "data/shanghaitech/part_A/train_data/images"
)

images = sorted(IMAGE_DIR.glob("*.jpg"))

print("Total images:", len(images))

for image_path in images[:5]:
    with Image.open(image_path) as img:
        print(f"\n{image_path.name}")
        print("  Size:", img.size)
        print("  Mode:", img.mode)
        print("  Format:", img.format)