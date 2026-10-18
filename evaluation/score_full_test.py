from pathlib import Path
import torch
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ORIGINAL_FEATURES = Path(
    "outputs/features/part_A_full_test_features.pt"
)

ENHANCED_FEATURES = Path(
    "outputs/features/part_A_full_enhanced_test_features.pt"
)

ORIGINAL_CENTER = Path(
    "outputs/features/part_A_normal_center.pt"
)

ENHANCED_CENTER = Path(
    "outputs/features/part_A_enhanced_normal_center.pt"
)

OUTPUT = Path(
    "outputs/evaluation/"
    "part_A_full_score_comparison.csv"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

original_features = torch.load(
    ORIGINAL_FEATURES,
    weights_only=True
)

enhanced_features = torch.load(
    ENHANCED_FEATURES,
    weights_only=True
)

original_center = torch.load(
    ORIGINAL_CENTER,
    weights_only=True
)

enhanced_center = torch.load(
    ENHANCED_CENTER,
    weights_only=True
)


print("Original features:", original_features.shape)
print("Enhanced features:", enhanced_features.shape)


# --------------------------------------------------
# Euclidean distance
# --------------------------------------------------

original_scores = torch.norm(
    original_features - original_center,
    dim=1
)

enhanced_scores = torch.norm(
    enhanced_features - enhanced_center,
    dim=1
)


# --------------------------------------------------
# Clip mapping
# --------------------------------------------------

rows = []

num_clips = len(original_scores)

for clip in range(num_clips):

    start_frame = clip * 8 + 1

    end_frame = min(
        start_frame + 15,
        182
    )

    rows.append({
        "clip": clip,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "original_score": float(
            original_scores[clip]
        ),
        "enhanced_score": float(
            enhanced_scores[clip]
        ),
        "change": float(
            enhanced_scores[clip]
            - original_scores[clip]
        ),
        "abs_change": float(
            abs(
                enhanced_scores[clip]
                - original_scores[clip]
            )
        )
    })


df = pd.DataFrame(rows)


# --------------------------------------------------
# Print
# --------------------------------------------------

print()
print("=" * 80)
print("FULL ORIGINAL vs ZERO-DCE SCORE COMPARISON")
print("=" * 80)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print()
print("Statistics")
print("-" * 40)

print(
    f"Original mean : "
    f"{original_scores.mean().item():.4f}"
)

print(
    f"Enhanced mean : "
    f"{enhanced_scores.mean().item():.4f}"
)

print(
    f"Original std  : "
    f"{original_scores.std().item():.4f}"
)

print(
    f"Enhanced std  : "
    f"{enhanced_scores.std().item():.4f}"
)

print(
    f"Mean change   : "
    f"{(enhanced_scores.mean() - original_scores.mean()).item():.4f}"
)

print(
    f"Max abs change: "
    f"{df['abs_change'].max():.4f}"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved to:")
print(OUTPUT)