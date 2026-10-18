import torch
from pathlib import Path


TEST_FEATURES = Path(
    "outputs/features/part_A_test_features.pt"
)

NORMAL_CENTER = Path(
    "outputs/features/part_A_normal_center.pt"
)


# -----------------------------
# Load features
# -----------------------------

test_features = torch.load(
    TEST_FEATURES,
    map_location="cpu"
)

normal_center = torch.load(
    NORMAL_CENTER,
    map_location="cpu"
)

print("Test features:", test_features.shape)
print("Normal center:", normal_center.shape)


# -----------------------------
# Calculate anomaly scores
# -----------------------------

scores = torch.norm(
    test_features - normal_center,
    dim=1
)


# -----------------------------
# Display scores
# -----------------------------

print("\nAnomaly scores:")

for i, score in enumerate(scores):
    print(
        f"Clip {i:02d}: {score.item():.4f}"
    )


# -----------------------------
# Statistics
# -----------------------------

print("\nScore statistics:")
print("Minimum:", scores.min().item())
print("Maximum:", scores.max().item())
print("Mean:", scores.mean().item())


# -----------------------------
# Save scores
# -----------------------------

output_dir = Path("outputs/scores")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir / "part_A_test_anomaly_scores.pt"
)

torch.save(scores, output_file)

print("\nScores saved to:")
print(output_file)