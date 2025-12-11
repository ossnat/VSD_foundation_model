# ================================
# File: src/data/datasets.py
# ================================
import torch
from torch.utils.data import Dataset
import h5py # Import h5py to be used in VsdVideoDataset
import numpy as np # Import numpy to be used in VsdVideoDataset
import pandas as pd
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


# Define the VsdVideoDataset class within this file or ensure it's imported
# Assuming the VsdVideoDataset class is defined in the previous steps and accessible
class VsdVideoDataset(Dataset):
    """PyTorch Dataset for loading VSD video data from CSV-based split structure."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 split_csv_path: Optional[str] = None,
                 split_name: str = "train",
                 stats_json_path: Optional[str] = None,
                 processed_root: Optional[str] = None,
                 frame_start: int = 1,
                 frame_end: Optional[int] = None,
                 clip_length: int = 1,
                 # Legacy parameters for backward compatibility
                 hdf5_path: Optional[str] = None,
                 normalize: bool = False,
                 normalization_type: str = "baseline_zscore",
                 baseline_frame: int = 20,
                 cache_dir: Optional[str] = None,
                 normalization_kwargs: Optional[Dict] = None,
                 trial_indices: Optional[List[int]] = None,
                 index_entries: Optional[List[Tuple[str, str, int]]] = None
                 ):
        """
        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary. If provided, parameters will be
                                          extracted from this config, overriding individual parameters.
            split_csv_path (str): Path to the split CSV file containing trial information.
            split_name (str): Split name to use ('train', 'val', or 'test').
            stats_json_path (str): Path to the stats JSON file with mean and std.
            processed_root (str, optional): Root directory for processed data (for relative paths).
            frame_start (int): Start frame index for slicing trials.
            frame_end (int): End frame index for slicing trials (if None, uses all frames).
            clip_length (int): Length of video clips to return (1 for single frames, >1 for clips).
            
            Legacy parameters (for backward compatibility with old single HDF5 structure):
            hdf5_path (str): Path to the HDF5 file (legacy).
            normalize (bool): Whether to apply normalization (legacy).
            normalization_type (str): Type of normalization (legacy).
            baseline_frame (int): Frame index to use as baseline (legacy).
            cache_dir (str): Directory to cache normalization statistics (legacy).
            normalization_kwargs (dict): Additional kwargs for normalization (legacy).
            trial_indices (List[int], optional): List of trial indices (legacy).
            index_entries (List[Tuple], optional): Index entries mapping (legacy).
        """
        # Extract parameters from config if provided
        if cfg is not None:
            split_csv_path = cfg.get('split_csv_path', split_csv_path)
            split_name = cfg.get('split_name', split_name)
            stats_json_path = cfg.get('stats_json_path', stats_json_path)
            processed_root = cfg.get('processed_root', processed_root)
            frame_start = cfg.get('frame_start', frame_start)
            frame_end = cfg.get('frame_end', frame_end)
            clip_length = cfg.get('clip_length', clip_length)
            if clip_length == 0:
                clip_length = 1
            # Legacy parameters
            hdf5_path = cfg.get('hdf5_path', hdf5_path)
            normalize = cfg.get('normalize', normalize)
            normalization_type = cfg.get('normalization_type', normalization_type)
            baseline_frame = cfg.get('baseline_frame', baseline_frame)
            cache_dir = cfg.get('cache_dir', cache_dir)
            trial_indices = cfg.get('trial_indices', trial_indices)
            index_entries = cfg.get('index_entries', index_entries)
            if normalization_kwargs is None:
                normalization_kwargs = {}
            for key in ['seed', 'log_dir', 'ckpt_dir']:
                if key in cfg and key not in normalization_kwargs:
                    normalization_kwargs[key] = cfg[key]

        # Check if using new CSV-based structure or legacy HDF5 structure
        self.use_legacy = (split_csv_path is None and hdf5_path is not None)
        
        if self.use_legacy:
            # Legacy mode: use old single HDF5 file structure
            self._init_legacy(cfg, hdf5_path, normalize, normalization_type, baseline_frame,
                            frame_start, frame_end, cache_dir, normalization_kwargs,
                            clip_length, trial_indices, index_entries)
            return

        # New CSV-based structure
        if split_csv_path is None:
            raise ValueError("split_csv_path must be provided for new data structure")
        if stats_json_path is None:
            raise ValueError("stats_json_path must be provided for new data structure")

        self.split_csv_path = split_csv_path
        self.split_name = split_name
        self.stats_json_path = stats_json_path
        self.processed_root = processed_root
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.clip_length = int(clip_length) if clip_length is not None else 1

        # Load CSV and filter by split
        print(f"Loading split CSV from {split_csv_path}...")
        # Try tab-separated first, then comma-separated
        try:
            df = pd.read_csv(split_csv_path, sep='\t')
        except Exception:
            df = pd.read_csv(split_csv_path, sep=',')
        
        # Validate required columns
        required_columns = ['target_file', 'trial_dataset', 'shape', 'split']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain columns: {required_columns}. Missing: {missing_columns}")
        
        # Filter by split
        self.trials = df[df['split'] == split_name].copy().reset_index(drop=True)
        
        if len(self.trials) == 0:
            raise ValueError(f"No trials found for split '{split_name}' in CSV file")
        
        print(f"Found {len(self.trials)} trials for split '{split_name}'")

        # Load stats from JSON
        print(f"Loading stats from {stats_json_path}...")
        with open(stats_json_path, 'r') as f:
            stats_data = json.load(f)
        
        # Get path to H5 file containing mean and std
        stats_h5_path = stats_data.get('stats_h5_path')
        if stats_h5_path is None:
            raise ValueError(f"Stats JSON must contain 'stats_h5_path' pointing to H5 file with mean and std")
        
        # Handle relative paths if processed_root is provided
        if self.processed_root is not None and not Path(stats_h5_path).is_absolute():
            stats_h5_path = str(Path(self.processed_root) / stats_h5_path)
        
        # Load mean and std from H5 file
        print(f"Loading mean and std from {stats_h5_path}...")
        with h5py.File(stats_h5_path, 'r') as f:
            # Check for mean and std datasets
            if 'mean' not in f:
                raise ValueError(f"H5 file {stats_h5_path} must contain 'mean' dataset")
            if 'std' not in f:
                raise ValueError(f"H5 file {stats_h5_path} must contain 'std' dataset")
            
            # Load mean and std arrays
            mean_array = f['mean'][...]  # Shape should be (1, 1, H, W) or similar
            std_array = f['std'][...]    # Shape should be (1, 1, H, W) or similar
        
        # Convert to tensors
        mean_tensor = torch.from_numpy(mean_array).float()
        std_tensor = torch.from_numpy(std_array).float()
        
        # Validate that mean and std are arrays, not scalars
        if mean_tensor.ndim == 0 or mean_tensor.numel() == 1:
            raise ValueError(f"Mean must be a spatial array, not a scalar. Got shape: {mean_array.shape}. "
                           f"Expected shape like (1, 1, 100, 100) or (100, 100).")
        if std_tensor.ndim == 0 or std_tensor.numel() == 1:
            raise ValueError(f"Std must be a spatial array, not a scalar. Got shape: {std_array.shape}. "
                           f"Expected shape like (1, 1, 100, 100) or (100, 100).")
        
        # Get expected shape from JSON if available, otherwise infer
        mean_shape = stats_data.get('mean_shape', [1, 1, 100, 100])
        expected_height = mean_shape[2] if len(mean_shape) >= 3 else 100
        expected_width = mean_shape[3] if len(mean_shape) >= 4 else 100
        
        # Reshape to (1, 1, H, W) if needed
        if mean_tensor.ndim == 2:
            # If 2D (H, W), add channel and temporal dimensions
            self.mean = mean_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mean_tensor.ndim == 4:
            # If already 4D, ensure it's (1, 1, H, W)
            if mean_tensor.shape != (1, 1, expected_height, expected_width):
                self.mean = mean_tensor.reshape(1, 1, expected_height, expected_width)
            else:
                self.mean = mean_tensor
        else:
            # Reshape to target shape
            self.mean = mean_tensor.reshape(1, 1, expected_height, expected_width)
        
        if std_tensor.ndim == 2:
            # If 2D (H, W), add channel and temporal dimensions
            self.std = std_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif std_tensor.ndim == 4:
            # If already 4D, ensure it's (1, 1, H, W)
            if std_tensor.shape != (1, 1, expected_height, expected_width):
                self.std = std_tensor.reshape(1, 1, expected_height, expected_width)
            else:
                self.std = std_tensor
        else:
            # Reshape to target shape
            self.std = std_tensor.reshape(1, 1, expected_height, expected_width)
        
        self.epsilon = 1e-8  # Small value to avoid division by zero
        
        # Print stats summary
        mean_min = float(self.mean.min())
        mean_max = float(self.mean.max())
        std_min = float(self.std.min())
        std_max = float(self.std.max())
        print(f"Loaded normalization stats from H5 file:")
        print(f"  Mean range: [{mean_min:.4f}, {mean_max:.4f}], shape: {self.mean.shape}")
        print(f"  Std range: [{std_min:.4f}, {std_max:.4f}], shape: {self.std.shape}")

        # Build data structure: list of (row_index, clip_start_frame)
        self.data_structure: List[Tuple[int, int]] = []
        
        # Process each trial row
        for row_idx, row in self.trials.iterrows():
            # Parse shape from CSV: "(10000, 256)" -> n_pixels=10000, n_frames=256
            shape_str = row['shape']
            if isinstance(shape_str, str):
                # Parse "(10000, 256)" format
                shape_str = shape_str.strip('()')
                parts = shape_str.split(',')
                n_pixels = int(parts[0].strip())
                n_frames = int(parts[1].strip())
            else:
                # Assume it's already parsed or use defaults
                n_pixels = 10000
                n_frames = 256
            
            # Compute frame range
            start = max(0, self.frame_start)
            end = (self.frame_end if self.frame_end is not None else n_frames - 1)
            end = min(end, n_frames - 1)
            if end < start:
                start, end = 0, n_frames - 1
            effective_frames = end - start + 1
            
            # Generate clips for this trial
            if self.clip_length > 0 and self.clip_length <= effective_frames:
                num_clips = effective_frames // self.clip_length
                for clip_idx in range(num_clips):
                    clip_start_frame = start + (clip_idx * self.clip_length)
                    self.data_structure.append((row_idx, clip_start_frame))
            else:
                # Single clip for entire range
                self.data_structure.append((row_idx, start))

        self.total_samples = len(self.data_structure)
        print(f"Created {self.total_samples} samples from {len(self.trials)} trials")

    def _init_legacy(self, cfg, hdf5_path, normalize, normalization_type, baseline_frame,
                     frame_start, frame_end, cache_dir, normalization_kwargs,
                     clip_length, trial_indices, index_entries):
        """Initialize using legacy single HDF5 file structure."""
        # This is the old implementation - keeping it for backward compatibility
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
        self.index_entries = index_entries
        
        self.data_structure: list[Tuple[str, str, Optional[int], Optional[int]]] = []
        self.dataset_is_2d: Dict[Tuple[str, str], bool] = {}

        selected_triples = None
        if self.index_entries is not None and self.trial_indices is not None:
            selected_triples = set(self.index_entries[g] for g in self.trial_indices)

        total_frames = 256
        start = max(0, self.frame_start)
        end = (self.frame_end if self.frame_end is not None else total_frames - 1)
        end = min(end, total_frames - 1)
        if end < start:
            start, end = 0, total_frames - 1
        effective_frames = end - start + 1
        
        if self.clip_length > 0 and self.clip_length <= effective_frames:
            num_clips = effective_frames // self.clip_length
        else:
            num_clips = None

        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    dataset_shape = dataset.shape
                    dataset_ndim = len(dataset_shape)
                    
                    is_2d = (dataset_ndim == 2)
                    self.dataset_is_2d[(group_name, dataset_name)] = is_2d
                    
                    if is_2d:
                        if num_clips is not None:
                            clip_entries = [(group_name, dataset_name, None, start + (clip_idx * self.clip_length))
                                           for clip_idx in range(num_clips)]
                            self.data_structure.extend(clip_entries)
                    else:
                        if dataset_ndim != 3:
                            raise ValueError(f"Unsupported dataset dimensionality: {dataset_ndim}. "
                                           f"Expected 2D (pixels, frames) or 3D (pixels, frames, trials).")
                        
                        num_trials = dataset_shape[-1]
                        
                        if selected_triples is not None:
                            trials_to_process = [t for t in range(num_trials)
                                                 if (group_name, dataset_name, t) in selected_triples]
                        else:
                            trials_to_process = range(num_trials)
                            if self.trial_indices is not None:
                                trials_to_process = [t for t in range(num_trials) if t in self.trial_indices]
                        
                        for trial_index in trials_to_process:
                            if num_clips is not None:
                                clip_entries = [(group_name, dataset_name, trial_index, start + (clip_idx * self.clip_length))
                                               for clip_idx in range(num_clips)]
                                self.data_structure.extend(clip_entries)
                            else:
                                self.data_structure.append((group_name, dataset_name, trial_index, start))

        self.total_samples = len(self.data_structure)
        
        if self.normalize:
            self._setup_normalization()
        else:
            self.normalizer = None
            self.normalization_stats = None

    def _setup_normalization(self):
        """Setup normalization by computing statistics if needed (legacy mode only)"""
        print(f"Setting up {self.normalization_type} normalization...")
        
        from .normalization import get_normalizer
        
        self.normalizer = get_normalizer(
            normalization_type=self.normalization_type,
            baseline_frame=self.baseline_frame,
            cache_dir=self.cache_dir,
            **self.normalization_kwargs
        )
        
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
        if self.use_legacy:
            return self._getitem_legacy(idx)
        
        # New CSV-based structure
        row_idx, clip_start = self.data_structure[idx]
        row = self.trials.iloc[row_idx]
        
        # Get file path and dataset name
        target_file = row['target_file']
        trial_dataset = row['trial_dataset']
        
        # Handle relative paths if processed_root is provided
        if self.processed_root is not None and not Path(target_file).is_absolute():
            target_file = str(Path(self.processed_root) / target_file)
        
        # Parse shape from CSV
        shape_str = row['shape']
        if isinstance(shape_str, str):
            shape_str = shape_str.strip('()')
            parts = shape_str.split(',')
            n_pixels = int(parts[0].strip())
            n_frames = int(parts[1].strip())
        else:
            n_pixels = 10000
            n_frames = 256
        
        # Compute spatial dimensions from n_pixels
        height = width = int(math.sqrt(n_pixels))
        
        # Open H5 file and read trial dataset
        with h5py.File(target_file, 'r') as f:
            if trial_dataset not in f:
                raise ValueError(f"Dataset '{trial_dataset}' not found in file {target_file}")
            trial_data = f[trial_dataset][...]  # Shape: (n_pixels, n_frames)
        
        # Apply frame slicing
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
        
        # Reshape from (n_pixels, frames) to (height, width, frames)
        frames = data_slice.shape[1]
        reshaped_data = data_slice.reshape(height, width, frames)
        
        # Convert to tensor and permute to (1, frames, height, width)
        tensor_data = torch.from_numpy(reshaped_data).unsqueeze(0).permute(0, 3, 1, 2).float()
        
        # Apply z-score normalization using precomputed stats
        tensor_data = (tensor_data - self.mean) / (self.std + self.epsilon)
        
        mask_tensor = torch.zeros(1, frames, height, width, dtype=torch.float32)
        
        return {"video": tensor_data, "mask": mask_tensor, "start_frame": int(abs_start_frame), "end_frame": int(abs_end_frame)}
    
    def _getitem_legacy(self, idx: int):
        """Legacy __getitem__ for old single HDF5 structure."""
        group_name, dataset_name, trial_index, clip_start = self.data_structure[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[group_name][dataset_name]
            is_2d = self.dataset_is_2d.get((group_name, dataset_name), False)
            
            if is_2d:
                trial_data = dataset[:, :]
            else:
                if trial_index is None:
                    raise ValueError(f"Trial index is None for 3D dataset {group_name}/{dataset_name}")
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
                data_slice = trial_data[:, start:end + 1]
                abs_start_frame = start
                abs_end_frame = end
        
        height, width = 100, 100
        frames = data_slice.shape[1]
        reshaped_data = data_slice.reshape(height, width, frames)
        tensor_data = torch.from_numpy(reshaped_data).unsqueeze(0).permute(0, 3, 1, 2)
        
        if self.normalize and self.normalizer is not None:
            tensor_data = self.normalizer.normalize(tensor_data, self.normalization_stats)

        mask_tensor = torch.zeros(1, frames, height, width, dtype=torch.float32)

        return {"video": tensor_data, "mask": mask_tensor, "start_frame": int(abs_start_frame), "end_frame": int(abs_end_frame)}

