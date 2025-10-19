# ================================
# File: src/data/datasets.py
# ================================
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import h5py # Import h5py to be used in VsdVideoDataset
import numpy as np # Import numpy to be used in VsdVideoDataset
from .normalization import get_normalizer
from typing import Optional, Dict, Any


# Define the VsdVideoDataset class within this file or ensure it's imported
# Assuming the VsdVideoDataset class is defined in the previous steps and accessible
class VsdVideoDataset(Dataset):
    """PyTorch Dataset for loading VSD video data from HDF5 file."""

    def __init__(self, hdf5_path: str, 
                 normalize: bool = False,
                 normalization_type: str = "baseline_zscore",
                 baseline_frame: int = 20,
                 frame_start: int = 0,
                 frame_end: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 normalization_kwargs: Optional[Dict] = None):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            normalize (bool): Whether to apply normalization.
            normalization_type (str): Type of normalization ('baseline_zscore' or 'baseline_robust').
            baseline_frame (int): Frame index to use as baseline (default: 20).
            frame_start (int): Start frame index for slicing trials.
            frame_end (int): End frame index for slicing trials (if None, uses all frames).
            cache_dir (str): Directory to cache normalization statistics.
            normalization_kwargs (dict): Additional kwargs for normalization.
        """
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.normalization_type = normalization_type
        self.baseline_frame = baseline_frame
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.cache_dir = cache_dir
        self.normalization_kwargs = normalization_kwargs or {}
        
        # Build data structure
        self.data_structure = [] # Store tuples of (group_name, dataset_name, trial_index)

        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    num_trials = dataset.shape[-1]
                    # Assuming data is (pixels, frames, trials)
                    # Add each trial as a separate sample
                    for trial_index in range(num_trials):
                         self.data_structure.append((group_name, dataset_name, trial_index))

        self.total_samples = len(self.data_structure)
        
        # Initialize normalization if enabled
        if self.normalize:
            self._setup_normalization()
        else:
            self.normalizer = None
            self.normalization_stats = None

    def _setup_normalization(self):
        """Setup normalization by computing statistics if needed"""
        print(f"Setting up {self.normalization_type} normalization...")
        
        # Create normalizer
        self.normalizer = get_normalizer(
            normalization_type=self.normalization_type,
            baseline_frame=self.baseline_frame,
            cache_dir=self.cache_dir,
            **self.normalization_kwargs
        )
        
        # Compute normalization statistics
        self.normalization_stats = self.normalizer.compute_stats(
            hdf5_path=self.hdf5_path,
            frame_start=self.frame_start,
            frame_end=self.frame_end
        )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.total_samples

    def __getitem__(self, idx: int):
        """
        Loads and returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing 'video' and 'mask' tensors.
        """
        if idx >= self.total_samples:
            raise IndexError("Dataset index out of range")

        group_name, dataset_name, trial_index = self.data_structure[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[group_name][dataset_name]
            # Data is likely (pixels, frames, trials), slice the correct trial
            # Slice along the last dimension (trial index)
            data_slice = dataset[:, :, trial_index]
            
            # Apply frame slicing if specified
            if self.frame_end is not None:
                data_slice = data_slice[:, self.frame_start:self.frame_end + 1]
            else:
                data_slice = data_slice[:, self.frame_start:]

        # Reshape the data slice to (channels, frames, height, width)
        # Assuming channels = 1, height = 100, width = 100
        height, width = 100, 100
        frames = data_slice.shape[1]
        reshaped_data = data_slice.reshape(height, width, frames)

        # Rearrange to (channels, frames, height, width) - (1, frames, 100, 100)
        # Add a channel dimension
        tensor_data = torch.from_numpy(reshaped_data).unsqueeze(0).permute(0, 3, 1, 2)

        # Apply normalization if enabled
        if self.normalize and self.normalizer is not None:
            tensor_data = self.normalizer.normalize(tensor_data, self.normalization_stats)

        # Create a dummy mask tensor with the same spatial and temporal dimensions as the video tensor
        # Assuming mask is single channel (1, frames, height, width)
        mask_tensor = torch.zeros(1, frames, height, width, dtype=torch.float32)

        return {"video": tensor_data, "mask": mask_tensor} # Return a dictionary for consistency with DummyVideoDataset