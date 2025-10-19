# src/data/data_loader.py
import torch
from torch.utils.data import DataLoader
from .datasets import VsdVideoDataset
from typing import Dict, Any

def load_dataset(
    cfg: Dict[str, Any], # Pass the entire config dictionary
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
        # Extract configuration parameters
        hdf5_path = cfg["vsd_hdf5_path"]
        
        # Normalization settings
        normalize = cfg.get("normalize", False)
        normalization_type = cfg.get("normalization_type", "baseline_zscore")
        baseline_frame = cfg.get("baseline_frame", 20)
        cache_dir = cfg.get("cache_dir", "cache")
        normalization_kwargs = cfg.get("normalization_kwargs", {})
        
        # Frame slicing settings
        frame_start = cfg.get("frame_start", 0)
        frame_end = cfg.get("frame_end", None)
        
        # Create dataset with all parameters
        dataset = VsdVideoDataset(
            hdf5_path=hdf5_path,
            normalize=normalize,
            normalization_type=normalization_type,
            baseline_frame=baseline_frame,
            frame_start=frame_start,
            frame_end=frame_end,
            cache_dir=cache_dir,
            normalization_kwargs=normalization_kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader