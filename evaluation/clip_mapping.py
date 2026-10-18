from pathlib import Path
import torch


DATASET_ROOT = Path("data/shanghaitech")

SCORES_FILE = Path(
    "outputs/scores/part_A_test_anomaly_scores.pt"
)

SEQUENCE_LENGTH = 16
STRIDE = 8


# -----------------------------
# Load test images
# -----------------------------

image_dir = (
    DATASET_ROOT
    / "part_A"
    / "test_data"
    / "images"
)

images = sorted(
    image_dir.glob("*.jpg"),
    key=lambda p: int(p.stem.split("_")[1])
)


# -----------------------------
# Load anomaly scores
# -----------------------------

scores = torch.load(
    SCORES_FILE,
    map_location="cpu"
)


# -----------------------------
# Map scores to clips
# -----------------------------

print("Clip Mapping")
print("=" * 70)

for clip_index, score in enumerate(scores):

    start = clip_index * STRIDE
    end = start + SEQUENCE_LENGTH - 1

    if end >= len(images):
        break

    start_frame = images[start].name
    end_frame = images[end].name

    print(
        f"Clip {clip_index:02d} | "
        f"{start_frame} → {end_frame} | "
        f"Score: {score.item():.4f}"
    )