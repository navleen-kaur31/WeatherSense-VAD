import torch
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Load scores
# --------------------------------------------------

original = torch.load(
    "outputs/scores/part_A_test_anomaly_scores.pt",
    map_location="cpu"
).numpy()

enhanced = torch.load(
    "outputs/scores/part_A_test_enhanced_anomaly_scores.pt",
    map_location="cpu"
).numpy()


# --------------------------------------------------
# Build clip table
# --------------------------------------------------

rows = []

for clip_id in range(len(original)):

    start_frame = 1 + clip_id * 8
    end_frame = start_frame + 15

    rows.append({
        "clip": clip_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "original_score": float(original[clip_id]),
        "enhanced_score": float(enhanced[clip_id]),
        "change": float(
            enhanced[clip_id] - original[clip_id]
        ),
        "abs_change": float(
            abs(enhanced[clip_id] - original[clip_id])
        )
    })


df = pd.DataFrame(rows)


# --------------------------------------------------
# Print
# --------------------------------------------------

print()
print("=" * 85)
print("FINAL ORIGINAL vs ZERO-DCE SCORE TABLE")
print("=" * 85)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output = Path(
    "outputs/evaluation/"
    "part_A_original_vs_zero_dce.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output,
    index=False
)

print()
print("Saved to:")
print(output)