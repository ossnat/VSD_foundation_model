"""
Data splitting utilities for VSD video dataset.

This module provides functions to split data at the trial level, ensuring that
train/validation splits respect trial boundaries regardless of how samples are defined.
"""

import h5py
import numpy as np
from typing import List, Tuple, Optional
import random


def split_data(hdf5_path: str, split_ratio: float = 0.8, random_seed: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """
    Split data at the trial level to ensure train/val splits respect trial boundaries.
    
    This function identifies all unique trials across all groups and datasets in the HDF5 file,
    then splits them into train and validation sets. This ensures that no trial appears in
    both train and validation sets, regardless of how samples are defined (whole trial, 
    single frame, or window of frames).
    
    Args:
        hdf5_path (str): Path to the HDF5 file containing the dataset.
        split_ratio (float): Ratio of trials to use for training (default: 0.8).
                            The remaining trials will be used for validation.
        random_seed (int, optional): Random seed for reproducible splits.
    
    Returns:
        Tuple[List[int], List[int]]: A tuple containing (train_trial_indices, val_trial_indices).
                                    These indices correspond to trial indices within each group/dataset.
    
    Example:
        >>> train_idx, val_idx = split_data('data.h5', split_ratio=0.8, random_seed=42)
        >>> print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")
    """
    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Collect all unique trial indices across all groups and datasets
    all_trial_indices = set()
    
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                num_trials = dataset.shape[-1]  # Last dimension is trials
                all_trial_indices.update(range(num_trials))
    
    # Convert to sorted list for consistent ordering
    all_trial_indices = sorted(list(all_trial_indices))
    total_trials = len(all_trial_indices)
    
    if total_trials == 0:
        raise ValueError("No trials found in the HDF5 file")
    
    # Calculate split sizes
    train_size = int(total_trials * split_ratio)
    val_size = total_trials - train_size
    
    # Shuffle the trial indices
    shuffled_indices = all_trial_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Split into train and validation
    train_trial_indices = shuffled_indices[:train_size]
    val_trial_indices = shuffled_indices[train_size:]
    
    print(f"Data split summary:")
    print(f"  Total trials: {total_trials}")
    print(f"  Train trials: {len(train_trial_indices)} ({len(train_trial_indices)/total_trials:.1%})")
    print(f"  Val trials: {len(val_trial_indices)} ({len(val_trial_indices)/total_trials:.1%})")
    
    return train_trial_indices, val_trial_indices


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
    
    # Check coverage
    all_trial_indices = set()
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                num_trials = dataset.shape[-1]
                all_trial_indices.update(range(num_trials))
    
    covered_indices = train_set | val_set
    if covered_indices != all_trial_indices:
        missing = all_trial_indices - covered_indices
        extra = covered_indices - all_trial_indices
        if missing:
            print(f"ERROR: Missing trial indices: {sorted(missing)}")
        if extra:
            print(f"ERROR: Extra trial indices: {sorted(extra)}")
        return False
    
    print("Split validation passed: no overlap, full coverage")
    return True
