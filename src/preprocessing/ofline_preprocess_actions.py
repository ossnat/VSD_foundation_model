import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm


def create_circular_mask(height=100, width=100, margin_factor=0.995):
    """
    Create a circular binary mask for 100x100 frame

    Args:
        height: Frame height (default 100)
        width: Frame width (default 100)
        margin_factor: How much of max radius to use (0.85 = 85%)

    Returns:
        mask_flat: Flattened circular mask (10000,) with 1s inside circle, 0s outside
    """
    # Find center of frame
    center_x, center_y = width // 2, height // 2

    # Calculate largest radius that fits with margin
    radius = int(min(center_x, center_y, width - center_x, height - center_y) * margin_factor)

    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]

    # Create circular mask: 1 inside circle, 0 outside
    mask_2d = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= radius ** 2

    # Flatten mask to match your data format
    mask_flat = mask_2d.flatten().astype(np.float32)

    return mask_flat

def apply_circular_mask_to_trial(trial_data, mask_flat):
    """
    Apply circular mask to a single trial

    Args:
        trial_data: Shape (10000, 256, 8) - one trial data
        mask_flat: Shape (10000,) - flattened circular mask

    Returns:
        masked_trial: Shape (10000, 256, 8) - masked trial data
    """
    # Apply mask to each frame (multiply each of 256*8=2048 frames by the mask)
    masked_trial = trial_data * mask_flat[:, np.newaxis, np.newaxis]

    return masked_trial

def mask_all_trials(datasets, mask_flat=None):
    """
    Apply circular mask to multiple datasets from HDF5 groups

    Args:
        datasets: List of numpy arrays or dict of datasets
                 Each dataset shape: (10000, num_frames, num_trials)
                 e.g., [(10000, 256, 8), (10000, 200, 12), ...]
        mask_flat: Precomputed mask, if None will create new one

    Returns:
        masked_datasets: Same structure as input, with circular mask applied to all
    """
    if mask_flat is None:
        mask_flat = create_circular_mask()

    # Handle different input types
    if isinstance(datasets, dict):
        # If input is dictionary (e.g., from HDF5 groups)
        masked_datasets = {}
        for group_name, dataset in datasets.items():
            print(f"Masking dataset '{group_name}' with shape {dataset.shape}")
            masked_datasets[group_name] = apply_circular_mask_to_dataset(dataset, mask_flat)

    elif isinstance(datasets, list):
        # If input is list of datasets
        masked_datasets = []
        for i, dataset in enumerate(datasets):
            print(f"Masking dataset {i} with shape {dataset.shape}")
            masked_dataset = apply_circular_mask_to_dataset(dataset, mask_flat)
            masked_datasets.append(masked_dataset)

    else:
        # Single dataset
        print(f"Masking single dataset with shape {datasets.shape}")
        masked_datasets = apply_circular_mask_to_dataset(datasets, mask_flat)

    return masked_datasets

def apply_circular_mask_to_dataset(dataset, mask_flat):
    """
    Apply circular mask to a single dataset

    Args:
        dataset: Shape (10000, num_frames, num_trials)
        mask_flat: Shape (10000,) - flattened circular mask

    Returns:
        masked_dataset: Same shape as input, with circular mask applied
    """
    if len(dataset.shape) != 3:
        raise ValueError(f"Expected 3D dataset (10000, num_frames, num_trials), got shape {dataset.shape}")

    if dataset.shape[0] != 10000:
        raise ValueError(f"Expected first dimension to be 10000 (pixels), got {dataset.shape[0]}")

    if np.isnan(dataset).any():
      print("Warning: NaNs found in dataset, will be replaced by zeros")
      dataset = np.nan_to_num(dataset, nan=0.0)

    # Apply mask using broadcasting: (10000, num_frames, num_trials) * (10000, 1, 1)
    mask_broadcast = mask_flat[:, np.newaxis, np.newaxis]
    masked_dataset = dataset * mask_broadcast

    return masked_dataset

def mask_all_trials(data, mask_flat=None):
    """
    Apply circular mask to all trials

    Args:
        data: Shape (num_trials, 10000, 256, 8) or (10000, 256, 8) for single trial
        mask_flat: Precomputed mask, if None will create new one

    Returns:
        masked_data: Same shape as input, with circular mask applied
    """
    if mask_flat is None:
        mask_flat = create_circular_mask()

    if len(data.shape) == 4:  # Multiple trials
        masked_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            masked_data[i] = apply_circular_mask_to_trial(data[i], mask_flat)
    elif len(data.shape) == 3:  # Single trial
        masked_data = apply_circular_mask_to_trial(data, mask_flat)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    return masked_data

def process_hdf5_with_circular_mask(input_hdf5_path, output_hdf5_path, mask_flat=None, overwrite=False):
    """
    Process entire HDF5 file: apply circular mask to all suitable datasets and save to new file

    Args:
        input_hdf5_path: Path to input HDF5 file
        output_hdf5_path: Path to output HDF5 file
        mask_flat: Precomputed mask, if None will create new one
        overwrite: Whether to overwrite existing output file

    Returns:
        processing_stats: Dict with statistics about processed datasets
    """
    # Setup
    input_path = Path(input_hdf5_path)
    output_path = Path(output_hdf5_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use overwrite=True to replace.")

    if mask_flat is None:
        mask_flat = create_circular_mask()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    processing_stats = {
        'processed_datasets': [],
        'skipped_datasets': [],
        'total_datasets_found': 0,
        'datasets_processed': 0,
        'mask_coverage': np.sum(mask_flat) / len(mask_flat)
    }

    logger.info(f"Processing HDF5 file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Mask coverage: {processing_stats['mask_coverage']*100:.1f}%")

    with h5py.File(input_path, 'r') as input_file, h5py.File(output_path, 'w') as output_file:

        # Copy global attributes
        for attr_name, attr_value in input_file.attrs.items():
            output_file.attrs[attr_name] = attr_value

        # Add processing metadata
        output_file.attrs['circular_mask_applied'] = True
        output_file.attrs['mask_coverage_percent'] = processing_stats['mask_coverage'] * 100
        output_file.attrs['processing_source'] = str(input_path)

        def process_item(name, obj):
            """Recursively process HDF5 groups and datasets"""

            if isinstance(obj, h5py.Group):
                # Create corresponding group in output file
                output_group = output_file.create_group(name)

                # Copy group attributes
                for attr_name, attr_value in obj.attrs.items():
                    output_group.attrs[attr_name] = attr_value

                logger.info(f"Created group: {name}")

            elif isinstance(obj, h5py.Dataset):
                processing_stats['total_datasets_found'] += 1
                dataset_info = {
                    'name': name,
                    'original_shape': obj.shape,
                    'dtype': obj.dtype,
                    'size_mb': obj.nbytes / (1024**2)
                }

                # Check if dataset is suitable for masking (3D with 10000 pixels)
                if len(obj.shape) == 3 and obj.shape[0] == 10000:
                    logger.info(f"Processing dataset: {name} with shape {obj.shape}")

                    # Load data
                    data = obj[:]

                    # Apply circular mask
                    masked_data = apply_circular_mask_to_dataset(data, mask_flat)

                    # Create dataset in output file with same properties
                    output_dataset = output_file.create_dataset(
                        name,
                        data=masked_data,
                        dtype=obj.dtype,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True  # Can improve compression
                    )

                    # Copy dataset attributes
                    for attr_name, attr_value in obj.attrs.items():
                        output_dataset.attrs[attr_name] = attr_value

                    # Add masking metadata
                    output_dataset.attrs['circular_mask_applied'] = True
                    output_dataset.attrs['original_nonzero_values'] = int(np.count_nonzero(data))
                    output_dataset.attrs['masked_nonzero_values'] = int(np.count_nonzero(masked_data))

                    dataset_info['status'] = 'processed'
                    dataset_info['nonzero_original'] = int(np.count_nonzero(data))
                    dataset_info['nonzero_masked'] = int(np.count_nonzero(masked_data))
                    processing_stats['processed_datasets'].append(dataset_info)
                    processing_stats['datasets_processed'] += 1

                    logger.info(f"  ✓ Processed: {name}")

                else:
                    # Dataset not suitable for masking, copy as-is
                    logger.info(f"Copying dataset unchanged: {name} (shape: {obj.shape})")

                    # Copy dataset as-is
                    output_file.create_dataset(
                        name,
                        data=obj[:],
                        dtype=obj.dtype,
                        compression='gzip' if obj.size > 1000 else None
                    )

                    # Copy attributes
                    for attr_name, attr_value in obj.attrs.items():
                        output_file[name].attrs[attr_name] = attr_value

                    dataset_info['status'] = 'copied_unchanged'
                    processing_stats['skipped_datasets'].append(dataset_info)

        # Process all items in the file
        print("Scanning and processing HDF5 structure...")
        input_file.visititems(process_item)

    # Final statistics
    logger.info(f"Processing complete!")
    logger.info(f"  - Total datasets found: {processing_stats['total_datasets_found']}")
    logger.info(f"  - Datasets processed with mask: {processing_stats['datasets_processed']}")
    logger.info(f"  - Datasets copied unchanged: {len(processing_stats['skipped_datasets'])}")
    logger.info(f"  - Output file size: {output_path.stat().st_size / (1024**2):.1f} MB")

    return processing_stats
