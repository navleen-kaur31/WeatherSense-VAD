from pathlib import Path

import torch
import pandas as pd

from sklearn.decomposition import PCA


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_ORIGINAL = Path(
    "outputs/features/part_A_train_features.pt"
)

TEST_ORIGINAL = Path(
    "outputs/features/part_A_full_test_features.pt"
)

TRAIN_ENHANCED = Path(
    "outputs/features/part_A_train_enhanced_features.pt"
)

TEST_ENHANCED = Path(
    "outputs/features/part_A_full_enhanced_test_features.pt"
)

OUTPUT = Path(
    "outputs/evaluation/"
    "part_A_mahalanobis_comparison.csv"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

train_original = torch.load(
    TRAIN_ORIGINAL,
    map_location="cpu",
    weights_only=True
).numpy()

test_original = torch.load(
    TEST_ORIGINAL,
    map_location="cpu",
    weights_only=True
).numpy()

train_enhanced = torch.load(
    TRAIN_ENHANCED,
    map_location="cpu",
    weights_only=True
).numpy()

test_enhanced = torch.load(
    TEST_ENHANCED,
    map_location="cpu",
    weights_only=True
).numpy()


print("Original training:", train_original.shape)
print("Original test:", test_original.shape)

print("Enhanced training:", train_enhanced.shape)
print("Enhanced test:", test_enhanced.shape)


# --------------------------------------------------
# Mahalanobis function
# --------------------------------------------------

def mahalanobis_scores(
    train_features,
    test_features,
    n_components=10
):

    # PCA fitted ONLY on training data
    pca = PCA(
        n_components=n_components,
        whiten=False
    )

    train_pca = pca.fit_transform(
        train_features
    )

    test_pca = pca.transform(
        test_features
    )

    print(
        f"PCA explained variance "
        f"({n_components} components): "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # Normal center
    center = train_pca.mean(
        axis=0
    )

    # Covariance
    covariance = torch.tensor(
        torch.from_numpy(
            train_pca - center
        ).numpy()
    ).float()

    covariance = (
        covariance.T @ covariance
    ) / (len(train_pca) - 1)

    # Small regularization for numerical stability
    covariance += (
        torch.eye(n_components) * 1e-4
    )

    # Inverse covariance
    inverse_covariance = torch.linalg.inv(
        covariance
    )

    # Test differences
    test_diff = torch.tensor(
        test_pca - center
    ).float()

    # Mahalanobis squared distance
    scores_squared = torch.sum(
        (
            test_diff @ inverse_covariance
        ) * test_diff,
        dim=1
    )

    scores = torch.sqrt(
        torch.clamp(
            scores_squared,
            min=0
        )
    )

    return scores


# --------------------------------------------------
# Original
# --------------------------------------------------

print()
print("=" * 70)
print("ORIGINAL MAHALANOBIS")
print("=" * 70)

original_scores = mahalanobis_scores(
    train_original,
    test_original,
    n_components=10
)


# --------------------------------------------------
# Enhanced
# --------------------------------------------------

print()
print("=" * 70)
print("ZERO-DCE MAHALANOBIS")
print("=" * 70)

enhanced_scores = mahalanobis_scores(
    train_enhanced,
    test_enhanced,
    n_components=10
)


# --------------------------------------------------
# Build table
# --------------------------------------------------

rows = []

for i in range(len(original_scores)):

    start_frame = i * 8 + 1

    end_frame = min(
        start_frame + 15,
        182
    )

    rows.append({
        "clip": i,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "original_mahalanobis": float(
            original_scores[i]
        ),
        "enhanced_mahalanobis": float(
            enhanced_scores[i]
        ),
        "change": float(
            enhanced_scores[i]
            - original_scores[i]
        )
    })


df = pd.DataFrame(rows)


# --------------------------------------------------
# Print
# --------------------------------------------------

print()
print("=" * 80)
print("MAHALANOBIS SCORE COMPARISON")
print("=" * 80)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
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