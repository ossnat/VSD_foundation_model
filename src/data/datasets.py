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
                 crop_frame: Optional[str] = None,  # None, "square", or "circle"
                 crop_radius: Optional[float] = None,  # 10-50, where 50 = full width/height
                 monkeys: Optional[List[str]] = None  # Optional subset of monkeys to include
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
            monkeys (List[str], optional): List of monkey IDs/names to include. If None,
                                           all monkeys in the split are used.
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
            crop_frame = cfg.get('crop_frame', crop_frame)
            crop_radius = cfg.get('crop_radius', crop_radius)
            # Optional monkey filtering: can be a single string or list
            cfg_monkeys = cfg.get('monkeys', monkeys)
            if isinstance(cfg_monkeys, str):
                monkeys = [cfg_monkeys]
            else:
                monkeys = cfg_monkeys
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
        self.crop_frame = crop_frame
        self.crop_radius = crop_radius
        self.monkeys = monkeys
        
        # Validate crop parameters
        if self.crop_frame is not None:
            if self.crop_frame not in ['square', 'circle']:
                raise ValueError(f"crop_frame must be None, 'square', or 'circle', got '{self.crop_frame}'")
            if self.crop_radius is None:
                raise ValueError("crop_radius must be provided when crop_frame is set")
            if not (10 <= self.crop_radius <= 50):
                raise ValueError(f"crop_radius must be between 10 and 50, got {self.crop_radius}")

        # Load CSV and filter by split
        print(f"Loading split CSV from {split_csv_path}...")
        
        # Try reading with standard settings first (most common case)
        try:
            df = pd.read_csv(split_csv_path, sep=',', encoding='utf-8-sig', skipinitialspace=True)
            print(f"Successfully read CSV with standard settings")
        except Exception as e1:
            print(f"Standard read failed: {e1}, trying alternatives...")
            # Try different encodings
            df = None
            last_error = e1
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(split_csv_path, sep=',', encoding=encoding, skipinitialspace=True)
                    print(f"Successfully read CSV with encoding='{encoding}'")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if df is None:
                # Last resort: try with python engine
                try:
                    df = pd.read_csv(split_csv_path, sep=',', encoding='utf-8-sig', engine='python', 
                                   skipinitialspace=True, on_bad_lines='skip')
                    print(f"Successfully read CSV with engine='python'")
                except Exception as e:
                    raise ValueError(f"Failed to read CSV file {split_csv_path}. Last error: {e}")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Debug: print what columns were found
        print(f"Found CSV columns ({len(df.columns)} total): {list(df.columns)}")
        print(f"CSV shape: {df.shape} (rows x columns)")
        
        # Check if we have the required columns
        required_columns = ['target_file', 'trial_dataset', 'shape', 'split']
        found_required = [col for col in required_columns if col in df.columns]
        print(f"Required columns found: {found_required} out of {required_columns}")
        
        if len(df) > 0:
            print(f"Sample first row (showing required columns):")
            for col in required_columns:
                if col in df.columns:
                    val = df.iloc[0][col]
                    # Truncate long values for display
                    if isinstance(val, str) and len(val) > 80:
                        val = val[:80] + "..."
                    print(f"  {col}: {val}")
        
        # Validate required columns
        required_columns = ['target_file', 'trial_dataset', 'shape', 'split']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain columns: {required_columns}. Missing: {missing_columns}. "
                           f"Found columns: {list(df.columns)}")
        
        # Filter by split
        self.trials = df[df['split'] == split_name].copy()

        # Optionally filter by a subset of monkeys
        if self.monkeys is not None:
            if 'monkey' not in self.trials.columns:
                raise ValueError("CSV does not contain 'monkey' column but 'monkeys' filter was provided.")
            before_count = len(self.trials)
            self.trials = self.trials[self.trials['monkey'].isin(self.monkeys)].copy()
            self.trials.reset_index(drop=True, inplace=True)
            after_count = len(self.trials)
            print(f"Filtered trials by monkeys {self.monkeys}: {before_count} -> {after_count}")
        
        if len(self.trials) == 0:
            if self.monkeys is not None:
                raise ValueError(
                    f"No trials found for split '{split_name}' and monkeys {self.monkeys} in CSV file"
                )
            else:
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

            # Determine last usable frame for this trial, taking shutter_off into account if available
            last_valid_frame = n_frames - 1
            if 'shutter_off' in self.trials.columns:
                shutter_val = row.get('shutter_off', None)
                if pd.notna(shutter_val):
                    try:
                        shutter_idx = int(shutter_val)
                        # Frames at and after shutter_off are unusable, so last usable is shutter_idx - 1
                        last_valid_frame = min(last_valid_frame, max(0, shutter_idx - 1))
                    except (TypeError, ValueError):
                        # If conversion fails, fall back to using all frames
                        pass

            # Compute frame range with global frame_start/frame_end but never beyond last_valid_frame
            start = max(0, self.frame_start)
            end = (self.frame_end if self.frame_end is not None else last_valid_frame)
            end = min(end, last_valid_frame)
            if end < start:
                # If global range is invalid for this trial, fall back to full valid range
                start, end = 0, last_valid_frame

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

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.total_samples

    def __getitem__(self, idx: int):
        # New CSV-based structure
        row_idx, clip_start = self.data_structure[idx]
        row = self.trials.iloc[row_idx]
        
        # Get file path and dataset name
        target_file = row['target_file']
        trial_dataset = row['trial_dataset']

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

        # Apply frame slicing, respecting shutter_off if present
        total_frames = trial_data.shape[1]

        # Determine last usable frame for this trial (consistent with constructor logic)
        last_valid_frame = total_frames - 1
        if 'shutter_off' in self.trials.columns:
            shutter_val = row.get('shutter_off', None)
            if pd.notna(shutter_val):
                try:
                    shutter_idx = int(shutter_val)
                    last_valid_frame = min(last_valid_frame, max(0, shutter_idx - 1))
                except (TypeError, ValueError):
                    pass

        start = max(0, self.frame_start)
        end = (self.frame_end if self.frame_end is not None else last_valid_frame)
        end = min(end, last_valid_frame)
        if end < start:
            start, end = 0, last_valid_frame
        
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
        
        # Store original dimensions for normalization stats cropping
        orig_height, orig_width = height, width
        
        # Track if we need to apply circular mask after normalization
        apply_circle_mask_after = False
        circle_mask = None
        radius_pixels = None
        
        # Apply frame cropping if specified (this may change tensor size)
        if self.crop_frame is not None:
            if self.crop_frame == 'circle':
                # For circle, we'll apply the mask after normalization
                # First, crop to square bounding box
                tensor_data, crop_height, crop_width = self._apply_crop(tensor_data, height, width)
                # Store mask info for later
                apply_circle_mask_after = True
                radius_pixels = (self.crop_radius / 50.0) * (orig_width / 2.0)
                # Create circular mask
                crop_h, crop_w = crop_height, crop_width
                center_y_crop, center_x_crop = crop_h // 2, crop_w // 2
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(crop_h, device=tensor_data.device, dtype=torch.float32),
                    torch.arange(crop_w, device=tensor_data.device, dtype=torch.float32),
                    indexing='ij'
                )
                distances = torch.sqrt((x_coords - center_x_crop) ** 2 + (y_coords - center_y_crop) ** 2)
                circle_mask = (distances <= radius_pixels).float().unsqueeze(0).unsqueeze(0)
            else:
                # For square, just crop normally
                tensor_data, crop_height, crop_width = self._apply_crop(tensor_data, height, width)
            # Update height/width for mask creation
            height, width = crop_height, crop_width
        else:
            crop_height, crop_width = height, width
        
        # Crop normalization stats to match cropped tensor size
        mean_cropped = self._crop_normalization_stats(self.mean, orig_height, orig_width, crop_height, crop_width)
        std_cropped = self._crop_normalization_stats(self.std, orig_height, orig_width, crop_height, crop_width)
        
        # Apply z-score normalization using precomputed stats (cropped to match)
        tensor_data = (tensor_data - mean_cropped) / (std_cropped + self.epsilon)
        
        # Apply circular mask after normalization (so corners stay at 0)
        if apply_circle_mask_after:
            tensor_data = tensor_data * circle_mask
        
        # Impute any NaN/Inf values before returning
        tensor_data = self._impute_non_finite(tensor_data)
        
        mask_tensor = torch.zeros(1, frames, crop_height, crop_width, dtype=torch.float32)
        
        return {
            "video": tensor_data,
            "mask": mask_tensor,
            "start_frame": int(abs_start_frame),
            "end_frame": int(abs_end_frame),
            "monkey": row["monkey"],
            "date": row["date"],
            "condition": row["condition"],
        }
    
    def _impute_non_finite(self, tensor_data: torch.Tensor) -> torch.Tensor:
        """
        Replace NaN/Inf values with the average of finite neighbors.
        Neighboring voxels are defined in time and spatial directions.
        """
        if torch.isfinite(tensor_data).all():
            return tensor_data
        
        data = tensor_data.clone()
        finite = torch.isfinite(data)
        
        neighbor_sum = torch.zeros_like(data)
        neighbor_count = torch.zeros_like(data, dtype=torch.int32)
        
        # Time neighbors (previous and next frame)
        neighbor_sum[:, 1:, :, :] += data[:, :-1, :, :] * finite[:, :-1, :, :]
        neighbor_count[:, 1:, :, :] += finite[:, :-1, :, :].to(torch.int32)
        neighbor_sum[:, :-1, :, :] += data[:, 1:, :, :] * finite[:, 1:, :, :]
        neighbor_count[:, :-1, :, :] += finite[:, 1:, :, :].to(torch.int32)
        
        # Spatial neighbors (up/down/left/right)
        neighbor_sum[:, :, 1:, :] += data[:, :, :-1, :] * finite[:, :, :-1, :]
        neighbor_count[:, :, 1:, :] += finite[:, :, :-1, :].to(torch.int32)
        neighbor_sum[:, :, :-1, :] += data[:, :, 1:, :] * finite[:, :, 1:, :]
        neighbor_count[:, :, :-1, :] += finite[:, :, 1:, :].to(torch.int32)
        neighbor_sum[:, :, :, 1:] += data[:, :, :, :-1] * finite[:, :, :, :-1]
        neighbor_count[:, :, :, 1:] += finite[:, :, :, :-1].to(torch.int32)
        neighbor_sum[:, :, :, :-1] += data[:, :, :, 1:] * finite[:, :, :, 1:]
        neighbor_count[:, :, :, :-1] += finite[:, :, :, 1:].to(torch.int32)
        
        neighbor_count_f = neighbor_count.to(data.dtype)
        avg_neighbors = neighbor_sum / neighbor_count_f.clamp(min=1.0)
        
        needs_impute = ~finite
        data = torch.where(needs_impute, avg_neighbors, data)
        
        # If no finite neighbors exist, fall back to 0.0
        no_neighbors = needs_impute & (neighbor_count == 0)
        if no_neighbors.any():
            data[no_neighbors] = 0.0
        
        if not torch.isfinite(data).all():
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    def _apply_crop(self, tensor_data: torch.Tensor, height: int, width: int):
        """
        Apply circular or square crop to the frame data.
        For square crops, actually crops the tensor to smaller size.
        For circular crops, uses square bounding box.
        
        Args:
            tensor_data: Tensor of shape (1, frames, height, width)
            height: Frame height
            width: Frame width
        
        Returns:
            tuple: (cropped_tensor, new_height, new_width)
        """
        if self.crop_frame is None:
            return tensor_data, height, width
        
        # Calculate actual radius in pixels
        # crop_radius 50 = full width/height (100 pixels)
        radius_pixels = (self.crop_radius / 50.0) * (width / 2.0)
        
        # Calculate crop region (centered)
        center_y, center_x = height // 2, width // 2
        
        if self.crop_frame == 'square':
            # Square crop: actually crop the tensor to smaller size
            crop_size = int(2 * radius_pixels)
            # Ensure crop_size is even and within bounds
            crop_size = min(crop_size, height, width)
            if crop_size % 2 != 0:
                crop_size -= 1
            
            # Calculate crop boundaries (centered)
            y_start = center_y - crop_size // 2
            y_end = y_start + crop_size
            x_start = center_x - crop_size // 2
            x_end = x_start + crop_size
            
            # Ensure boundaries are within tensor
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            
            # Crop the tensor: (1, frames, height, width) -> (1, frames, crop_h, crop_w)
            cropped = tensor_data[:, :, y_start:y_end, x_start:x_end]
            new_height = y_end - y_start
            new_width = x_end - x_start
            
            return cropped, new_height, new_width
            
        elif self.crop_frame == 'circle':
            # For circle, use square bounding box (circumscribed square)
            # This gives us a rectangular tensor while approximating the circle
            crop_size = int(2 * radius_pixels)
            crop_size = min(crop_size, height, width)
            if crop_size % 2 != 0:
                crop_size -= 1
            
            y_start = center_y - crop_size // 2
            y_end = y_start + crop_size
            x_start = center_x - crop_size // 2
            x_end = x_start + crop_size
            
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            
            # Crop to square bounding box
            cropped = tensor_data[:, :, y_start:y_end, x_start:x_end]
            
            # Apply circular mask to zero out corners
            crop_h, crop_w = y_end - y_start, x_end - x_start
            center_y_crop, center_x_crop = crop_h // 2, crop_w // 2
            
            y_coords, x_coords = torch.meshgrid(
                torch.arange(crop_h, device=tensor_data.device, dtype=torch.float32),
                torch.arange(crop_w, device=tensor_data.device, dtype=torch.float32),
                indexing='ij'
            )
            
            distances = torch.sqrt((x_coords - center_x_crop) ** 2 + (y_coords - center_y_crop) ** 2)
            circle_mask = (distances <= radius_pixels).float().unsqueeze(0).unsqueeze(0)
            
            cropped = cropped * circle_mask
            new_height = crop_h
            new_width = crop_w
            
            return cropped, new_height, new_width
        else:
            raise ValueError(f"Unknown crop_frame type: {self.crop_frame}")
    
    def _crop_normalization_stats(self, stats_tensor: torch.Tensor, orig_h: int, orig_w: int, 
                                   crop_h: int, crop_w: int) -> torch.Tensor:
        """
        Crop normalization statistics to match cropped tensor size.
        
        Args:
            stats_tensor: Normalization stats tensor of shape (1, 1, orig_h, orig_w)
            orig_h, orig_w: Original height and width
            crop_h, crop_w: Cropped height and width
        
        Returns:
            Cropped stats tensor of shape (1, 1, crop_h, crop_w)
        """
        if crop_h == orig_h and crop_w == orig_w:
            return stats_tensor
        
        # Calculate crop region (centered, same as in _apply_crop)
        center_y, center_x = orig_h // 2, orig_w // 2
        y_start = center_y - crop_h // 2
        y_end = y_start + crop_h
        x_start = center_x - crop_w // 2
        x_end = x_start + crop_w
        
        # Ensure boundaries are within tensor
        y_start = max(0, y_start)
        y_end = min(orig_h, y_end)
        x_start = max(0, x_start)
        x_end = min(orig_w, x_end)
        
        # Crop the stats tensor
        cropped_stats = stats_tensor[:, :, y_start:y_end, x_start:x_end]
        
        return cropped_stats
    
