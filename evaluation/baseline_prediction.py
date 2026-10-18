import torch
from pathlib import Path


# -----------------------------
# Files
# -----------------------------

TRAIN_FEATURES = Path(
    "outputs/features/part_A_train_features.pt"
)

TEST_FEATURES = Path(
    "outputs/features/part_A_test_features.pt"
)


# -----------------------------
# Load features
# -----------------------------

train_features = torch.load(
    TRAIN_FEATURES,
    map_location="cpu"
)

test_features = torch.load(
    TEST_FEATURES,
    map_location="cpu"
)

print("Training features:", train_features.shape)
print("Test features:", test_features.shape)


# -----------------------------
# Calculate normal center
# -----------------------------

normal_center = train_features.mean(dim=0)


# -----------------------------
# Training distances
# -----------------------------

train_scores = torch.norm(
    train_features - normal_center,
    dim=1
)


# -----------------------------
# Choose baseline threshold
# -----------------------------
# We use mean + 2 standard deviations
# as an initial unsupervised threshold.

threshold = (
    train_scores.mean()
    + 2 * train_scores.std()
)


print("\nBaseline threshold:")
print(f"{threshold.item():.4f}")


# -----------------------------
# Test anomaly scores
# -----------------------------

test_scores = torch.norm(
    test_features - normal_center,
    dim=1
)


# -----------------------------
# Predictions
# -----------------------------

predictions = test_scores > threshold


print("\nBaseline predictions:")
print("-" * 40)

for i in range(len(test_scores)):

    label = "ANOMALY" if predictions[i] else "NORMAL"

    print(
        f"Clip {i:02d} | "
        f"Score: {test_scores[i].item():.4f} | "
        f"{label}"
    )


# -----------------------------
# Summary
# -----------------------------

print("\nSummary")
print("-" * 40)

print(
    "Predicted anomalies:",
    predictions.sum().item()
)

print(
    "Predicted normal:",
    (~predictions).sum().item()
)