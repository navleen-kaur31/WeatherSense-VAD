import torch
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

ORIGINAL_PATH = Path(
    "outputs/scores/part_A_test_anomaly_scores.pt"
)

ENHANCED_PATH = Path(
    "outputs/scores/part_A_test_enhanced_anomaly_scores.pt"
)

OUTPUT_PATH = Path(
    "outputs/scores/part_A_original_vs_enhanced.csv"
)


# --------------------------------------------------
# Load scores
# --------------------------------------------------

original = torch.load(
    ORIGINAL_PATH,
    map_location="cpu"
)

enhanced = torch.load(
    ENHANCED_PATH,
    map_location="cpu"
)


# --------------------------------------------------
# Convert to numpy
# --------------------------------------------------

original = original.numpy()
enhanced = enhanced.numpy()


# --------------------------------------------------
# Build comparison
# --------------------------------------------------

data = []

for i in range(len(original)):

    data.append({
        "clip": i,
        "original_score": float(original[i]),
        "enhanced_score": float(enhanced[i]),
        "change": float(
            enhanced[i] - original[i]
        ),
        "absolute_change": float(
            abs(enhanced[i] - original[i])
        )
    })


df = pd.DataFrame(data)


# --------------------------------------------------
# Print statistics
# --------------------------------------------------

print("Score comparison")
print("=" * 60)

print()
print("Original:")
print("  Mean:", original.mean())
print("  Std :", original.std())
print("  Min :", original.min())
print("  Max :", original.max())

print()
print("Zero-DCE:")
print("  Mean:", enhanced.mean())
print("  Std :", enhanced.std())
print("  Min :", enhanced.min())
print("  Max :", enhanced.max())

print()
print("Average score change:")
print(
    enhanced.mean() - original.mean()
)

print()
print("Largest absolute change:")

largest = df.loc[
    df["absolute_change"].idxmax()
]

print(
    f"  Clip {int(largest['clip'])}: "
    f"{largest['original_score']:.4f} → "
    f"{largest['enhanced_score']:.4f}"
)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Comparison saved to:")
print(OUTPUT_PATH)