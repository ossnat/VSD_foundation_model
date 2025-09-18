# src/data/data_loader.py
import torch
from torch.utils.data import DataLoader
from .datasets import VsdVideoDataset

def load_dataset(
    cfg, # Pass the entire config dictionary
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
):
    """
    Load dataset based on configuration.
    Args:
        cfg (dict): Configuration dictionary containing dataset type and parameters.
        split (str): "train" or "test".
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.
        shuffle (bool): Shuffle data.

    Returns:
        DataLoader
    """
    if cfg["dataset"] == "vsd":
        # Use the VsdVideoDataset
        # Assuming the HDF5 file path is specified in the config
        hdf5_path = cfg["vsd_hdf5_path"]
        dataset = VsdVideoDataset(hdf5_path=hdf5_path)
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader