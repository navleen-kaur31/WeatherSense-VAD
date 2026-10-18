import torch
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURE_PATH = Path(
    "outputs/features/part_A_train_enhanced_features.pt"
)

OUTPUT_PATH = Path(
    "outputs/features/part_A_enhanced_normal_center.pt"
)


# --------------------------------------------------
# Load features
# --------------------------------------------------

features = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

print("Enhanced training features:", features.shape)


# --------------------------------------------------
# Calculate normal center
# --------------------------------------------------

normal_center = features.mean(
    dim=0
)

print(
    "Enhanced normal center:",
    normal_center.shape
)


# --------------------------------------------------
# Calculate training distances
# --------------------------------------------------

distances = torch.norm(
    features - normal_center,
    dim=1
)

print()
print("Enhanced training distance statistics:")
print("Minimum:", distances.min().item())
print("Maximum:", distances.max().item())
print("Mean:", distances.mean().item())


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    normal_center,
    OUTPUT_PATH
)

print()
print("Enhanced normal center saved to:")
print(OUTPUT_PATH)