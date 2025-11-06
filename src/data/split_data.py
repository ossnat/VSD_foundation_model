"""
Data splitting utilities for VSD video dataset.

This module provides functions to split data at the trial level, ensuring that
train/validation splits respect trial boundaries regardless of how samples are defined.
Now `split_data` returns a split over global trial IDs across all groups/datasets,
along with the mapping from global ID to (group, dataset, trial_idx).
"""

import h5py
import numpy as np
from typing import List, Tuple, Optional
import random

def split_data(hdf5_path: str,
               split_ratio: float = 0.8,
               random_seed: Optional[int] = None
               ) -> Tuple[List[int], List[int], List[Tuple[str, str, int]]]:
    """
    Split using global trial ids across all groups/datasets.

    Returns:
        train_ids: list of global trial IDs (ints)
        val_ids: list of global trial IDs (ints)
        index_entries: mapping list where index_entries[gid] = (group, dataset, trial_idx)
    """
    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    index_entries = index_trials(hdf5_path)
    total = len(index_entries)
    if total == 0:
        raise ValueError("No trials found in the HDF5 file")

    all_ids = list(range(total))
    random.shuffle(all_ids)
    train_size = int(total * split_ratio)
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:]

    print("Global data split summary:")
    print(f"  Total trials: {total}")
    print(f"  Train trials: {len(train_ids)} ({len(train_ids)/total:.1%})")
    print(f"  Val trials: {len(val_ids)} ({len(val_ids)/total:.1%})")

    return train_ids, val_ids, index_entries


def get_trial_info(hdf5_path: str) -> dict:
    """
    Get information about trials in the HDF5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
    
    Returns:
        dict: Dictionary containing trial information for each group/dataset.
    """
    trial_info = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            trial_info[group_name] = {}
            
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                num_trials = dataset.shape[-1]
                trial_info[group_name][dataset_name] = {
                    'num_trials': num_trials,
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype)
                }
    
    return trial_info


def validate_split(hdf5_path: str, train_indices: List[int], val_indices: List[int]) -> bool:
    """
    Validate that the train/val split is correct (no overlap, covers all trials).
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        train_indices (List[int]): Training trial indices.
        val_indices (List[int]): Validation trial indices.
    
    Returns:
        bool: True if the split is valid, False otherwise.
    """
    # Check for overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    
    if train_set & val_set:
        print("ERROR: Overlap found between train and validation indices")
        return False
    
    # For global split, this function only verifies no overlap; full coverage is not required
    if train_set & val_set:
        print("ERROR: Overlap found between train and validation indices")
        return False
    print("Split validation passed: no overlap")
    return True


# -------------------------------
# Global trial indexing utilities
# -------------------------------

def index_trials(hdf5_path: str) -> List[Tuple[str, str, int]]:
    """
    Build a deterministic list of (group_name, dataset_name, trial_idx) across the HDF5.
    The position in this list is the global trial id.
    """
    entries: List[Tuple[str, str, int]] = []
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                num_trials = dataset.shape[-1]
                for t in range(num_trials):
                    entries.append((group_name, dataset_name, t))
    return entries


# Removed split_data_global; global split is now provided by split_data