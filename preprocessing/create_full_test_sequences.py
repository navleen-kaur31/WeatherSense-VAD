from pathlib import Path
import shutil


INPUT_DIR = Path(
    "data/shanghaitech/part_A/test_data/images"
)

OUTPUT_DIR = Path(
    "outputs/evaluation/full_test_sequences"
)

SEQUENCE_LENGTH = 16
STRIDE = 8


# --------------------------------------------------
# Images in numeric order
# --------------------------------------------------

images = sorted(
    INPUT_DIR.glob("IMG_*.jpg"),
    key=lambda p: int(
        p.stem.split("_")[1]
    )
)

print("Total images:", len(images))


# --------------------------------------------------
# Generate windows
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

starts = list(
    range(
        0,
        len(images) - SEQUENCE_LENGTH + 1,
        STRIDE
    )
)

# Make sure the final frames are covered
last_start = len(images) - SEQUENCE_LENGTH

if starts[-1] != last_start:
    starts.append(last_start)


# --------------------------------------------------
# Create sequences
# --------------------------------------------------

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

        shutil.copy2(
            frame,
            sequence_dir /
            f"frame_{frame_id:02d}.jpg"
        )

    print(
        f"Sequence {clip_id:02d}: "
        f"{frames[0].name} → "
        f"{frames[-1].name}"
    )


print()
print("=" * 50)
print("Complete test sequence creation finished!")
print("=" * 50)
print("Total images:", len(images))
print("Sequences:", len(starts))
print("Sequence length:", SEQUENCE_LENGTH)
print("Stride:", STRIDE)