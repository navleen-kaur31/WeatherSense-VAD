import os
import torch
import numpy as np
from PIL import Image

from preprocessing.zero_dce.model import ZeroDCE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "preprocessing/outputs/zero_dce/zero_dce_model.pth"

INPUT_DIR = "data/shanghaitech/part_A/train_data/images"

OUTPUT_DIR = "preprocessing/outputs/zero_dce/part_A_train"


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Device:", DEVICE)

model = ZeroDCE().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Zero-DCE model loaded!")


# --------------------------------------------------
# Find images
# --------------------------------------------------

image_files = sorted(
    [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ],
    key=lambda x: int(
        os.path.splitext(x)[0].split("_")[-1]
    )
)

print("Images found:", len(image_files))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Process images
# --------------------------------------------------

for idx, filename in enumerate(image_files, start=1):

    input_path = os.path.join(INPUT_DIR, filename)

    output_name = (
        os.path.splitext(filename)[0]
        + "_enhanced.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    # Skip if already processed
    if os.path.exists(output_path):
        print(
            f"[{idx}/{len(image_files)}] "
            f"Already exists: {filename}"
        )
        continue

    # Load original image
    image = Image.open(input_path).convert("RGB")

    # Keep ORIGINAL resolution
    image_np = np.asarray(image).astype(
        np.float32
    ) / 255.0

    x = torch.from_numpy(image_np)
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.to(DEVICE)

    # Zero-DCE inference
    with torch.no_grad():

        curve = model(x)

        enhanced = x

        for i in range(8):

            current_curve = curve[
                :,
                i * 3:(i + 1) * 3
            ]

            enhanced = enhanced + (
                current_curve *
                (enhanced * enhanced - enhanced)
            )

    # Clamp
    enhanced = torch.clamp(
        enhanced,
        0,
        1
    )

    # Tensor → image
    enhanced = enhanced.squeeze(0)
    enhanced = enhanced.permute(1, 2, 0)

    enhanced_np = (
        enhanced.cpu().numpy() * 255.0
    ).astype(np.uint8)

    enhanced_image = Image.fromarray(
        enhanced_np
    )

    # Save
    enhanced_image.save(
        output_path,
        quality=95
    )

    print(
        f"[{idx}/{len(image_files)}] "
        f"{filename} → enhanced"
    )


print()
print("========================================")
print("Batch enhancement complete!")
print("========================================")
print("Input images:", len(image_files))
print("Output directory:")
print(OUTPUT_DIR)