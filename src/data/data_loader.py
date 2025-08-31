# src/data/data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(
    dataset_path: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
):
    """
    Load dataset from a local path.
    Args:
        dataset_path (str): Path to dataset root.
        split (str): "train" or "test".
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.
        shuffle (bool): Shuffle data.

    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # adjust to match your needs
        transforms.CenterCrop(112),
        transforms.ToTensor(),
    ])

    # Example using UCF101 (you can replace with VSD dataset loader later)
    try:
        dataset = datasets.UCF101(
            root=dataset_path,
            annotation_path=f"{dataset_path}/ucfTrainTestlist",
            frames_per_clip=16,
            train=(split == "train"),
            transform=transform,
        )
    except RuntimeError as e:
        raise FileNotFoundError(
            f"Could not find UCF101 dataset in {dataset_path}. "
            "Please download it manually from "
            "https://www.crcv.ucf.edu/data/UCF101.php and place it there."
        ) from e

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
