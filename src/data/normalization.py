# src/data/normalization.py
import torch
import numpy as np
import h5py
import pickle
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


class Normalizer(ABC):
    """Abstract base class for normalization methods"""
    
    def __init__(self, baseline_frame: int = 20, cache_dir: Optional[str] = None, normalization_type: str = "unknown"):
        self.baseline_frame = baseline_frame
        self.cache_dir = cache_dir or "cache"
        self.normalization_type = normalization_type
        self._stats_cache = {}  # Cache for computed statistics
    
    @abstractmethod
    def normalize(self, data: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """Normalize the input data using pre-computed statistics"""
        pass
    
    @abstractmethod
    def compute_stats(self, hdf5_path: str, frame_start: int, frame_end: int) -> Dict[str, torch.Tensor]:
        """Compute normalization statistics from the dataset"""
        pass
    
    def _get_cache_path(self, hdf5_path: str, frame_start: int, frame_end: int) -> str:
        """Generate cache file path for statistics"""
        cache_name = f"{Path(hdf5_path).stem}_frames_{frame_start}_{frame_end}_baseline_{self.baseline_frame}_{self.normalization_type}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load statistics from cache if available"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"Loaded cached statistics from {cache_path}")
                    return cached_data
        except Exception as e:
            print(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_path: str, stats: Dict[str, torch.Tensor]):
        """Save statistics to cache"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"Cached statistics to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")


class BaselineZScoreNormalizer(Normalizer):
    """
    Global Z-score normalization to baseline frames.
    Computes mean and std of each pixel across designated baseline frames.
    Subtracts mean and divides by std for the whole dataset.
    """
    
    def __init__(self, baseline_frame: int = 20, epsilon: float = 1e-8, cache_dir: Optional[str] = None):
        super().__init__(baseline_frame, cache_dir, "baseline_zscore")
        self.epsilon = epsilon
    
    def normalize(self, data: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """
        Normalize data using pre-computed baseline statistics.
        
        Args:
            data: Input tensor of shape (C, T, H, W)
            stats: Dictionary containing 'mean' and 'std' tensors
            
        Returns:
            Normalized tensor
        """
        if stats is None:
            raise ValueError("Baseline normalization requires pre-computed statistics")
        
        mean_baseline = stats['mean']  # Shape: (C, 1, H, W)
        std_baseline = stats['std']    # Shape: (C, 1, H, W)
        
        # Normalize: (data - mean) / (std + epsilon)
        normalized = (data - mean_baseline) / (std_baseline + self.epsilon)
        
        return normalized
    
    def compute_stats(self, hdf5_path: str, frame_start: int, frame_end: int) -> Dict[str, torch.Tensor]:
        """
        Compute mean and std of each pixel across baseline frames for the entire dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            frame_start: Start frame index for slicing
            frame_end: End frame index for slicing
            
        Returns:
            Dictionary containing 'mean' and 'std' tensors
        """
        cache_path = self._get_cache_path(hdf5_path, frame_start, frame_end)
        
        # Try to load from cache first
        cached_stats = self._load_from_cache(cache_path)
        if cached_stats is not None:
            return cached_stats
        
        print(f"Computing baseline z-score statistics for frames {frame_start}-{frame_end}...")
        
        # Collect all baseline frame data
        baseline_data = []
        
        with h5py.File(hdf5_path, 'r') as f:
            # Handle case where there are no groups (direct datasets)
            groups = list(f.keys())
            if len(groups) == 0:
                raise ValueError("HDF5 file contains no groups or datasets")
            
            for group_name in groups:
                group = f[group_name]
                datasets = list(group.keys())
                
                # Handle case where there are no datasets in the group
                if len(datasets) == 0:
                    continue
                
                for dataset_name in datasets:
                    dataset = group[dataset_name]
                    dataset_shape = dataset.shape
                    dataset_ndim = len(dataset_shape)
                    
                    # Handle 2D dataset (pixels, frames) - single trial case
                    if dataset_ndim == 2:
                        # Shape: (pixels, frames)
                        trial_data = dataset[:, :]
                        
                        # Slice the specified frame range
                        trial_sliced = trial_data[:, frame_start:frame_end + 1]
                        
                        # Get baseline frame (frame 20 by default, or closest available)
                        baseline_frame_idx = min(self.baseline_frame, trial_sliced.shape[1] - 1)
                        baseline_frame_data = trial_sliced[:, baseline_frame_idx]
                        
                        # Reshape to (height, width) and add to collection
                        height, width = 100, 100
                        baseline_frame_reshaped = baseline_frame_data.reshape(height, width)
                        baseline_data.append(baseline_frame_reshaped)
                    
                    # Handle 3D dataset (pixels, frames, trials) - multiple trials case
                    elif dataset_ndim == 3:
                        # Shape: (pixels, frames, trials)
                        num_trials = dataset_shape[-1]
                        
                        for trial_idx in range(num_trials):
                            # Get trial data: shape (pixels, frames, trials)
                            trial_data = dataset[:, :, trial_idx]
                            
                            # Slice the specified frame range
                            trial_sliced = trial_data[:, frame_start:frame_end + 1]
                            
                            # Get baseline frame (frame 20 by default, or closest available)
                            baseline_frame_idx = min(self.baseline_frame, trial_sliced.shape[1] - 1)
                            baseline_frame_data = trial_sliced[:, baseline_frame_idx]
                            
                            # Reshape to (height, width) and add to collection
                            height, width = 100, 100
                            baseline_frame_reshaped = baseline_frame_data.reshape(height, width)
                            baseline_data.append(baseline_frame_reshaped)
                    else:
                        raise ValueError(f"Unsupported dataset dimensionality: {dataset_ndim}. "
                                       f"Expected 2D (pixels, frames) or 3D (pixels, frames, trials).")
        
        if len(baseline_data) == 0:
            raise ValueError("No baseline data collected. Check dataset structure and frame indices.")
        
        # Convert to tensor and stack
        all_baseline_frames = torch.stack([torch.from_numpy(frame) for frame in baseline_data])  # (N, H, W)
        
        # Compute mean and std for each pixel across all trials
        # If only one trial, mean/std will be computed from that single frame
        mean_baseline = all_baseline_frames.mean(dim=0)  # (H, W)
        std_baseline = all_baseline_frames.std(dim=0)    # (H, W)
        
        # Reshape to match input tensor format: (C, 1, H, W)
        mean_baseline = mean_baseline.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        std_baseline = std_baseline.unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
        
        stats = {
            'mean': mean_baseline,
            'std': std_baseline
        }
        
        # Cache the statistics
        self._save_to_cache(cache_path, stats)
        
        print(f"Computed baseline statistics - Mean range: [{mean_baseline.min():.4f}, {mean_baseline.max():.4f}], "
              f"Std range: [{std_baseline.min():.4f}, {std_baseline.max():.4f}]")
        
        return stats


class BaselineRobustNormalizer(Normalizer):
    """
    Global robust normalization to baseline frames.
    Computes median and IQR of each pixel across designated baseline frames.
    Uses robust statistics for normalization.
    """
    
    def __init__(self, baseline_frame: int = 20, epsilon: float = 1e-8, cache_dir: Optional[str] = None):
        super().__init__(baseline_frame, cache_dir, "baseline_robust")
        self.epsilon = epsilon
    
    def normalize(self, data: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """
        Normalize data using pre-computed robust baseline statistics.
        
        Args:
            data: Input tensor of shape (C, T, H, W)
            stats: Dictionary containing 'median' and 'iqr' tensors
            
        Returns:
            Normalized tensor
        """
        if stats is None:
            raise ValueError("Baseline robust normalization requires pre-computed statistics")
        
        median_baseline = stats['median']  # Shape: (C, 1, H, W)
        iqr_baseline = stats['iqr']        # Shape: (C, 1, H, W)
        
        # Robust normalize: (data - median) / (iqr + epsilon)
        normalized = (data - median_baseline) / (iqr_baseline + self.epsilon)
        
        return normalized
    
    def compute_stats(self, hdf5_path: str, frame_start: int, frame_end: int) -> Dict[str, torch.Tensor]:
        """
        Compute median and IQR of each pixel across baseline frames for the entire dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            frame_start: Start frame index for slicing
            frame_end: End frame index for slicing
            
        Returns:
            Dictionary containing 'median' and 'iqr' tensors
        """
        cache_path = self._get_cache_path(hdf5_path, frame_start, frame_end)
        
        # Try to load from cache first
        cached_stats = self._load_from_cache(cache_path)
        if cached_stats is not None:
            return cached_stats
        
        print(f"Computing baseline robust statistics for frames {frame_start}-{frame_end}...")
        
        # Collect all baseline frame data
        baseline_data = []
        
        with h5py.File(hdf5_path, 'r') as f:
            # Handle case where there are no groups (direct datasets)
            groups = list(f.keys())
            if len(groups) == 0:
                raise ValueError("HDF5 file contains no groups or datasets")
            
            for group_name in groups:
                group = f[group_name]
                datasets = list(group.keys())
                
                # Handle case where there are no datasets in the group
                if len(datasets) == 0:
                    continue
                
                for dataset_name in datasets:
                    dataset = group[dataset_name]
                    dataset_shape = dataset.shape
                    dataset_ndim = len(dataset_shape)
                    
                    # Handle 2D dataset (pixels, frames) - single trial case
                    if dataset_ndim == 2:
                        # Shape: (pixels, frames)
                        trial_data = dataset[:, :]
                        
                        # Slice the specified frame range
                        trial_sliced = trial_data[:, frame_start:frame_end + 1]
                        
                        # Get baseline frame (frame 20 by default, or closest available)
                        baseline_frame_idx = min(self.baseline_frame, trial_sliced.shape[1] - 1)
                        baseline_frame_data = trial_sliced[:, baseline_frame_idx]
                        
                        # Reshape to (height, width) and add to collection
                        height, width = 100, 100
                        baseline_frame_reshaped = baseline_frame_data.reshape(height, width)
                        baseline_data.append(baseline_frame_reshaped)
                    
                    # Handle 3D dataset (pixels, frames, trials) - multiple trials case
                    elif dataset_ndim == 3:
                        # Shape: (pixels, frames, trials)
                        num_trials = dataset_shape[-1]
                        
                        for trial_idx in range(num_trials):
                            # Get trial data: shape (pixels, frames, trials)
                            trial_data = dataset[:, :, trial_idx]
                            
                            # Slice the specified frame range
                            trial_sliced = trial_data[:, frame_start:frame_end + 1]
                            
                            # Get baseline frame (frame 20 by default, or closest available)
                            baseline_frame_idx = min(self.baseline_frame, trial_sliced.shape[1] - 1)
                            baseline_frame_data = trial_sliced[:, baseline_frame_idx]
                            
                            # Reshape to (height, width) and add to collection
                            height, width = 100, 100
                            baseline_frame_reshaped = baseline_frame_data.reshape(height, width)
                            baseline_data.append(baseline_frame_reshaped)
                    else:
                        raise ValueError(f"Unsupported dataset dimensionality: {dataset_ndim}. "
                                       f"Expected 2D (pixels, frames) or 3D (pixels, frames, trials).")
        
        if len(baseline_data) == 0:
            raise ValueError("No baseline data collected. Check dataset structure and frame indices.")
        
        # Convert to tensor and stack
        all_baseline_frames = torch.stack([torch.from_numpy(frame) for frame in baseline_data])  # (N, H, W)
        
        # Compute robust statistics for each pixel across all trials
        # If only one trial, median/IQR will be computed from that single frame
        median_baseline = torch.median(all_baseline_frames, dim=0)[0]  # (H, W)
        
        # Compute IQR (Interquartile Range)
        q25 = torch.quantile(all_baseline_frames, 0.25, dim=0)  # (H, W)
        q75 = torch.quantile(all_baseline_frames, 0.75, dim=0)  # (H, W)
        iqr_baseline = q75 - q25  # (H, W)
        
        # Reshape to match input tensor format: (C, 1, H, W)
        median_baseline = median_baseline.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        iqr_baseline = iqr_baseline.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)
        
        stats = {
            'median': median_baseline,
            'iqr': iqr_baseline
        }
        
        # Cache the statistics
        self._save_to_cache(cache_path, stats)
        
        print(f"Computed baseline robust statistics - Median range: [{median_baseline.min():.4f}, {median_baseline.max():.4f}], "
              f"IQR range: [{iqr_baseline.min():.4f}, {iqr_baseline.max():.4f}]")
        
        return stats


def get_normalizer(normalization_type: str, **kwargs) -> Normalizer:
    """
    Factory function to create normalizers.
    
    Args:
        normalization_type: Type of normalization ('baseline_zscore' or 'baseline_robust')
        **kwargs: Additional arguments for the normalizer
        
    Returns:
        Normalizer instance
    """
    if normalization_type == "baseline_zscore":
        return BaselineZScoreNormalizer(**kwargs)
    elif normalization_type == "baseline_robust":
        return BaselineRobustNormalizer(**kwargs)
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}. "
                        f"Supported types: 'baseline_zscore', 'baseline_robust'")
