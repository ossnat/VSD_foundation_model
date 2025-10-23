# ================================
# File: src/data/datasets.py
# ================================
import torch
from torch.utils.data import Dataset
import h5py # Import h5py to be used in VsdVideoDataset
import numpy as np # Import numpy to be used in VsdVideoDataset
from .normalization import get_normalizer
from typing import Optional, Dict, Any, Tuple, List


# Define the VsdVideoDataset class within this file or ensure it's imported
# Assuming the VsdVideoDataset class is defined in the previous steps and accessible
class VsdVideoDataset(Dataset):
    """PyTorch Dataset for loading VSD video data from HDF5 file."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 hdf5_path: str = "path/to/hdf5_file.hdf5",
                 normalize: bool = False,
                 normalization_type: str = "baseline_zscore",
                 baseline_frame: int = 20,
                 frame_start: int = 1,
                 frame_end: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 normalization_kwargs: Optional[Dict] = None,
                 clip_length: int = 1,
                 trial_indices: Optional[List[int]] = None):
        """
        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary. If provided, parameters will be
                                          extracted from this config, overriding individual parameters.
            hdf5_path (str): Path to the HDF5 file.
            normalize (bool): Whether to apply normalization.
            normalization_type (str): Type of normalization ('baseline_zscore' or 'baseline_robust').
            baseline_frame (int): Frame index to use as baseline (default: 20).
            frame_start (int): Start frame index for slicing trials.
            frame_end (int): End frame index for slicing trials (if None, uses all frames).
            cache_dir (str): Directory to cache normalization statistics.
            normalization_kwargs (dict): Additional kwargs for normalization.
            clip_length (int): Length of video clips to return (1 for single frames, >1 for clips).
            trial_indices (List[int], optional): List of trial indices to include in the dataset.
                                                If None, includes all trials.
        """
        # Extract parameters from config if provided
        if cfg is not None:
            # Override parameters with config values
            hdf5_path = cfg.get('hdf5_path', hdf5_path)
            normalize = cfg.get('normalize', normalize)
            normalization_type = cfg.get('normalization_type', normalization_type)
            baseline_frame = cfg.get('baseline_frame', baseline_frame)
            frame_start = cfg.get('frame_start', frame_start)
            frame_end = cfg.get('frame_end', frame_end)
            cache_dir = cfg.get('cache_dir', cache_dir)
            clip_length = cfg.get('clip_length', clip_length)  # Use clip_length directly
            if clip_length == 0:  # Handle clip_length=0 case
                clip_length = 1
            
            # Extract trial_indices from config if provided
            if 'trial_indices' in cfg:
                trial_indices = cfg.get('trial_indices', trial_indices)
            
            # Extract normalization kwargs from config
            if normalization_kwargs is None:
                normalization_kwargs = {}
            # Add any additional config parameters that might be normalization-related
            for key in ['seed', 'log_dir', 'ckpt_dir']:
                if key in cfg and key not in normalization_kwargs:
                    normalization_kwargs[key] = cfg[key]

        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.normalization_type = normalization_type
        self.baseline_frame = baseline_frame
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.cache_dir = cache_dir
        self.normalization_kwargs = normalization_kwargs or {}
        self.clip_length = int(clip_length) if clip_length is not None else 1
        self.trial_indices = trial_indices
        
        # Build data structure
        # If windowing is enabled (window_size > 0), store (group, dataset, trial_idx, window_idx)
        # Otherwise store (group, dataset, trial_idx, None)
        self.data_structure: list[Tuple[str, str, int, Optional[int]]] = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    num_trials = dataset.shape[-1]
                    total_frames = dataset.shape[1]
                    
                    start = max(0, self.frame_start)
                    end = (self.frame_end if self.frame_end is not None else total_frames - 1)
                    end = min(end, total_frames - 1)
                    if end < start:
                        start, end = 0, total_frames - 1
                    effective_frames = end - start + 1
                    
                    # Filter trials based on trial_indices if provided
                    trials_to_process = range(num_trials)
                    if self.trial_indices is not None:
                        trials_to_process = [t for t in range(num_trials) if t in self.trial_indices]
                    
                    for trial_index in trials_to_process:
                        if self.clip_length > 0 and self.clip_length <= effective_frames:
                            # Non-overlapping clips
                            num_clips = effective_frames // self.clip_length
                            for clip_idx in range(num_clips):
                                clip_start_frame = start + (clip_idx * self.clip_length)
                                self.data_structure.append((group_name, dataset_name, trial_index, clip_start_frame))
                        else:
                            # Single clip for entire range
                            self.data_structure.append((group_name, dataset_name, trial_index, start))

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
        group_name, dataset_name, trial_index, clip_start = self.data_structure[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[group_name][dataset_name]
            trial_data = dataset[:, :, trial_index]
            
            total_frames = trial_data.shape[1]
            start = max(0, self.frame_start)
            end = (self.frame_end if self.frame_end is not None else total_frames - 1)
            end = min(end, total_frames - 1)
            if end < start:
                start, end = 0, total_frames - 1
            
            if self.clip_length > 0 and (clip_start + self.clip_length <= end + 1):
                data_slice = trial_data[:, clip_start:clip_start + self.clip_length]
                abs_start_frame = clip_start
                abs_end_frame = clip_start + self.clip_length - 1
            else:
                # Return full available frames if clip would exceed end
                data_slice = trial_data[:, start:end + 1]
                abs_start_frame = start
                abs_end_frame = end
        
        # Reshape and permute as before (height=100, width=100)
        height, width = 100, 100
        frames = data_slice.shape[1]
        reshaped_data = data_slice.reshape(height, width, frames)
        tensor_data = torch.from_numpy(reshaped_data).unsqueeze(0).permute(0, 3, 1, 2)
        
        if self.normalize and self.normalizer is not None:
            tensor_data = self.normalizer.normalize(tensor_data, self.normalization_stats)

        mask_tensor = torch.zeros(1, frames, height, width, dtype=torch.float32)

        return {"video": tensor_data, "mask": mask_tensor, "start_frame": int(abs_start_frame), "end_frame": int(abs_end_frame)}

