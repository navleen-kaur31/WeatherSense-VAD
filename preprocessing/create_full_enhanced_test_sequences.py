from pathlib import Path
import shutil


INPUT_DIR = Path(
    "preprocessing/outputs/zero_dce/part_A_test"
)

OUTPUT_DIR = Path(
    "outputs/evaluation/full_enhanced_test_sequences"
)

SEQUENCE_LENGTH = 16
STRIDE = 8


# --------------------------------------------------
# Get enhanced images in numeric order
# --------------------------------------------------

images = sorted(
    INPUT_DIR.glob("IMG_*_enhanced.jpg"),
    key=lambda p: int(
        p.stem.split("_")[1]
    )
)

print("Enhanced images:", len(images))


# --------------------------------------------------
# Determine sequence starts
# --------------------------------------------------

starts = list(
    range(
        0,
        len(images) - SEQUENCE_LENGTH + 1,
        STRIDE
    )
)

# Ensure final frames are covered
last_start = len(images) - SEQUENCE_LENGTH

if starts[-1] != last_start:
    starts.append(last_start)


# --------------------------------------------------
# Create sequences
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for clip_id, start in enumerate(starts):

    frames = images[
        start:start + SEQUENCE_LENGTH
    ]

    sequence_dir = (
        OUTPUT_DIR /
        f"sequence_{clip_id:04d}"
    )

    sequence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for frame_id, frame in enumerate(frames):

        output_path = (
            sequence_dir /
            f"frame_{frame_id:02d}.jpg"
        )

        shutil.copy2(
            frame,
            output_path
        )

    print(
        f"Sequence {clip_id:02d}: "
        f"{frames[0].name} → "
        f"{frames[-1].name}"
    )


print()
print("=" * 50)
print("Complete enhanced test sequences finished!")
print("=" * 50)

print("Images:", len(images))
print("Sequences:", len(starts))
print("Sequence length:", SEQUENCE_LENGTH)
print("Stride:", STRIDE)
print("Output:", OUTPUT_DIR)