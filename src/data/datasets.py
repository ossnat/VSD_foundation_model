# ================================
# File: src/data/datasets.py
# ================================
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import h5py # Import h5py to be used in VsdVideoDataset
import numpy as np # Import numpy to be used in VsdVideoDataset


# Define the VsdVideoDataset class within this file or ensure it's imported
# Assuming the VsdVideoDataset class is defined in the previous steps and accessible
class VsdVideoDataset(Dataset):
    """PyTorch Dataset for loading VSD video data from HDF5 file."""

    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
        """
        self.hdf5_path = hdf5_path
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
            torch.Tensor: The video data tensor.
        """
        if idx >= self.total_samples:
            raise IndexError("Dataset index out of range")

        group_name, dataset_name, trial_index = self.data_structure[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[group_name][dataset_name]
            # Data is likely (pixels, frames, trials), slice the correct trial
            # Assuming pixels are 10000, frames are 256
            # Reshape to (100, 100, 256)
            # Slice along the last dimension (trial index)
            data_slice = dataset[:, :, trial_index]

        # Reshape the data slice to (channels, frames, height, width)
        # Assuming channels = 1, height = 100, width = 100
        # Original slice shape is (10000, 256)
        # Reshape to (100, 100, 256) first, then rearrange dimensions
        height, width = 100, 100
        frames = data_slice.shape[1]
        reshaped_data = data_slice.reshape(height, width, frames)

        # Rearrange to (channels, frames, height, width) - (1, 256, 100, 100)
        # Add a channel dimension
        tensor_data = torch.from_numpy(reshaped_data).unsqueeze(0).permute(0, 3, 1, 2)

        # Create a dummy mask tensor with the same spatial and temporal dimensions as the video tensor
        # Assuming mask is single channel (1, frames, height, width)
        mask_tensor = torch.zeros(1, frames, height, width, dtype=torch.float32)

        return {"video": tensor_data, "mask": mask_tensor} # Return a dictionary for consistency with DummyVideoDataset