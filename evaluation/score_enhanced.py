import torch
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURE_PATH = Path(
    "outputs/features/part_A_test_enhanced_features.pt"
)

CENTER_PATH = Path(
    "outputs/features/part_A_enhanced_normal_center.pt"
)

OUTPUT_PATH = Path(
    "outputs/scores/part_A_test_enhanced_anomaly_scores.pt"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

features = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

normal_center = torch.load(
    CENTER_PATH,
    map_location="cpu"
)

print("Enhanced test features:", features.shape)
print("Enhanced normal center:", normal_center.shape)


# --------------------------------------------------
# Calculate anomaly scores
# --------------------------------------------------

scores = torch.norm(
    features - normal_center,
    dim=1
)


# --------------------------------------------------
# Print scores
# --------------------------------------------------

print()
print("Enhanced anomaly scores:")
print("-" * 40)

for i, score in enumerate(scores):

    print(
        f"Clip {i:02d}: "
        f"{score.item():.4f}"
    )


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print()
print("Score statistics:")
print("-" * 40)

print(
    "Minimum:",
    scores.min().item()
)

print(
    "Maximum:",
    scores.max().item()
)

print(
    "Mean:",
    scores.mean().item()
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    scores,
    OUTPUT_PATH
)

print()
print("Enhanced anomaly scores saved to:")
print(OUTPUT_PATH)