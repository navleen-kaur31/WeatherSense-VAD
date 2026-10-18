from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GT_DIR = Path(
    "data/shanghaitech/part_A/test_data/ground-truth"
)

OUTPUT_PATH = Path(
    "outputs/evaluation/part_A_clip_ground_truth.csv"
)

SEQUENCE_LENGTH = 16
STRIDE = 8


# --------------------------------------------------
# Get GT files in numerical order
# --------------------------------------------------

gt_files = sorted(
    GT_DIR.glob("GT_IMG_*.mat"),
    key=lambda p: int(
        p.stem.split("_")[-1]
    )
)

print("Ground-truth files:", len(gt_files))


# --------------------------------------------------
# Extract annotation counts
# --------------------------------------------------

frame_counts = {}

for gt_file in gt_files:

    frame_number = int(
        gt_file.stem.split("_")[-1]
    )

    data = loadmat(
        gt_file,
        squeeze_me=True,
        struct_as_record=False
    )

    image_info = data["image_info"]

    locations = np.asarray(
        image_info.location
    )

    # Handle a single point safely
    if locations.ndim == 1:
        locations = locations.reshape(-1, 2)

    count = len(locations)

    frame_counts[frame_number] = count


# --------------------------------------------------
# Build clip-level information
# --------------------------------------------------

frame_numbers = sorted(
    frame_counts.keys()
)

clips = []

for clip_id, start_idx in enumerate(
    range(
        0,
        len(frame_numbers) - SEQUENCE_LENGTH + 1,
        STRIDE
    )
):

    clip_frames = frame_numbers[
        start_idx:start_idx + SEQUENCE_LENGTH
    ]

    counts = [
        frame_counts[f]
        for f in clip_frames
    ]

    clips.append({
        "clip": clip_id,
        "start_frame": clip_frames[0],
        "end_frame": clip_frames[-1],
        "min_people": min(counts),
        "max_people": max(counts),
        "mean_people": np.mean(counts),
        "total_annotations": sum(counts)
    })


# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(clips)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print()
print(df.to_string(index=False))

print()
print("Clip ground-truth information saved to:")
print(OUTPUT_PATH)