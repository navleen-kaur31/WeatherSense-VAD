import torch
from pathlib import Path


FEATURE_FILE = Path(
    "outputs/features/part_A_train_features.pt"
)

OUTPUT_FILE = Path(
    "outputs/features/part_A_normal_center.pt"
)


# -----------------------------
# Load training features
# -----------------------------

features = torch.load(
    FEATURE_FILE,
    map_location="cpu"
)

print("Feature shape:", features.shape)


# -----------------------------
# Calculate normal center
# -----------------------------

normal_center = features.mean(dim=0)

print("Normal center shape:", normal_center.shape)


# -----------------------------
# Calculate training distances
# -----------------------------

distances = torch.norm(
    features - normal_center,
    dim=1
)

print("\nTraining distance statistics:")
print("Minimum:", distances.min().item())
print("Maximum:", distances.max().item())
print("Mean:", distances.mean().item())


# -----------------------------
# Save normal center
# -----------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    normal_center,
    OUTPUT_FILE
)

print("\nNormal center saved to:")
print(OUTPUT_FILE)