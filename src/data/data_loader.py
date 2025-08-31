import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(
    dataset_name="ucf101",
    root="./data",
    split="train",
    batch_size=4,
    num_workers=2,
    frame_size=(112, 112),
    num_frames=16,
    download=True
):
    """
    Loads a small video dataset for testing / training.

    Args:
        dataset_name (str): name of the dataset (currently supports "ucf101")
        root (str): path to store the dataset
        split (str): "train" or "test"
        batch_size (int): batch size for DataLoader
        num_workers (int): workers for DataLoader
        frame_size (tuple): (H, W) to resize frames
        num_frames (int): number of frames per video clip
        download (bool): whether to download the dataset if not found

    Returns:
        DataLoader, dataset object
    """

    # ✅ Minimal transform pipeline
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.CenterCrop(frame_size),
        transforms.ConvertImageDtype(torch.float32),  # convert to float
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1, 1]
    ])

    # ✅ TorchVision datasets: start with UCF101
    if dataset_name.lower() == "ucf101":
        dataset = datasets.UCF101(
            root=root,
            annotation_path=root,
            frames_per_clip=num_frames,
            step_between_clips=5,
            train=(split == "train"),
            transform=transform,
            num_workers=num_workers,
            download=download,
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported yet.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, dataset
