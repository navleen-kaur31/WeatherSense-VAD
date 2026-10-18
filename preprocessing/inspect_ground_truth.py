from pathlib import Path
from scipy.io import loadmat

GT_FILE = Path(
    "data/shanghaitech/part_A/train_data/ground-truth/GT_IMG_1.mat"
)

data = loadmat(GT_FILE, squeeze_me=True, struct_as_record=False)

image_info = data["image_info"]

print("Type:", type(image_info))
print("Shape:", getattr(image_info, "shape", "N/A"))

print("\nImage info:")
print(image_info)

print("\nAttributes:")

if hasattr(image_info, "__dict__"):
    for key, value in image_info.__dict__.items():
        print(f"{key}:")
        print(f"  Type: {type(value)}")
        print(f"  Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"  Value: {value}")