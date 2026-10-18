from pathlib import Path

IMAGE_DIR = Path(
    "data/shanghaitech/part_A/train_data/images"
)

images = sorted(
    IMAGE_DIR.glob("*.jpg"),
    key=lambda p: int(p.stem.split("_")[1])
)

print("Total images:", len(images))

print("\nFirst 20 frames:")
for image in images[:20]:
    print(image.name)

print("\nLast 10 frames:")
for image in images[-10:]:
    print(image.name)