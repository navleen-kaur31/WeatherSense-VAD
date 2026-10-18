import torch
import csv
from pathlib import Path


# -----------------------------
# Load data
# -----------------------------

train_features = torch.load(
    "outputs/features/part_A_train_features.pt",
    map_location="cpu"
)

test_features = torch.load(
    "outputs/features/part_A_test_features.pt",
    map_location="cpu"
)


# -----------------------------
# Normal center
# -----------------------------

normal_center = train_features.mean(dim=0)


# -----------------------------
# Training scores
# -----------------------------

train_scores = torch.norm(
    train_features - normal_center,
    dim=1
)


# -----------------------------
# Threshold
# -----------------------------

threshold = (
    train_scores.mean()
    + 2 * train_scores.std()
)


# -----------------------------
# Test scores
# -----------------------------

test_scores = torch.norm(
    test_features - normal_center,
    dim=1
)

predictions = test_scores > threshold


# -----------------------------
# Save results
# -----------------------------

output_dir = Path("outputs/baseline")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_dir / "part_A_baseline_results.csv"


with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "clip",
        "anomaly_score",
        "threshold",
        "prediction"
    ])

    for i in range(len(test_scores)):

        prediction = (
            "ANOMALY"
            if predictions[i]
            else "NORMAL"
        )

        writer.writerow([
            i,
            round(test_scores[i].item(), 4),
            round(threshold.item(), 4),
            prediction
        ])


print("Baseline results saved to:")
print(output_file)

print("\nThreshold:", round(threshold.item(), 4))
print(
    "Predicted anomalies:",
    predictions.sum().item()
)
print(
    "Predicted normal:",
    (~predictions).sum().item()
)