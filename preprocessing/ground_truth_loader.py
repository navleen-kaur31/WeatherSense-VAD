from pathlib import Path
from scipy.io import loadmat


def load_ground_truth(mat_file):
    """
    Load ShanghaiTech ground-truth annotations.

    Returns:
        locations: NumPy array of shape (N, 2)
        number: Number of annotated points
    """

    mat_file = Path(mat_file)

    data = loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False
    )

    image_info = data["image_info"]

    locations = image_info.location
    number = image_info.number

    return locations, number


if __name__ == "__main__":

    gt_file = (
        Path("data/shanghaitech")
        / "part_A"
        / "train_data"
        / "ground-truth"
        / "GT_IMG_1.mat"
    )

    locations, number = load_ground_truth(gt_file)

    print("Ground-truth loaded successfully!")
    print("Number of annotated points:", number)
    print("Location array shape:", locations.shape)
    print("First 5 locations:")
    print(locations[:5])