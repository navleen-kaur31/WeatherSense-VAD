import torch
from torch.utils.data import DataLoader

from dataset import ShanghaiTechDataset


dataset = ShanghaiTechDataset(
    root_dir="data/shanghaitech",
    part="part_A",
    split="train_data",
    sequence_length=16,
    stride=8,
    image_size=224
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

batch = next(iter(loader))

print("Batch shape:", batch.shape)
print("Batch dtype:", batch.dtype)

print("\nExpected:")
print("Batch size = 2")
print("Frames = 16")
print("Channels = 3")
print("Resolution = 224 x 224")