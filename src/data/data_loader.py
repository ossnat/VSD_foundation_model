"""
Data loader utility functions for creating DataLoaders from datasets.
"""
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from .dataset_factory import create_dataset


def load_dataset(cfg: Dict[str, Any], 
                 split: str = "train",
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 shuffle: Optional[bool] = None) -> DataLoader:
    """
    Create a DataLoader from a configuration dictionary.
    
    Args:
        cfg: Configuration dictionary containing dataset parameters
        split: Split name ('train', 'val', or 'test')
        batch_size: Batch size (overrides cfg['batch_size'] if provided)
        num_workers: Number of workers (overrides cfg['num_workers'] if provided)
        shuffle: Whether to shuffle (overrides cfg['shuffle'] if provided)
    
    Returns:
        DataLoader instance
    """
    # Determine dataset name from config
    dataset_name = cfg.get('dataset_name', 'vsd_video')
    
    # Extract dataset parameters
    dataset_kwargs = {}
    
    # New CSV-based structure
    split_csv_path = cfg.get('split_csv_path')
    stats_json_path = cfg.get('stats_json_path')
    processed_root = cfg.get('processed_root')

    # If split_csv_path contains split name, extract it
    # Otherwise, assume split CSV is in a split folder
    if split_csv_path is None:
        # Try to construct from split folder
        split_folder = cfg.get('split_folder')
        if split_folder:
            import os
            split_csv_path = os.path.join(split_folder, f'split_{split}.csv')
            stats_json_path = os.path.join(split_folder, f'stats_{split}.json')

    dataset_kwargs = {
        'split_csv_path': split_csv_path,
        'split_name': split,
        'stats_json_path': stats_json_path,
        'processed_root': processed_root,
        'frame_start': cfg.get('frame_start', 1),
        'frame_end': cfg.get('frame_end', None),
        'clip_length': cfg.get('clip_length', 1),
        'crop_frame': cfg.get('crop_frame'),
        'crop_radius': cfg.get('crop_radius'),
        'monkeys': cfg.get('monkeys'),
        'preload_into_ram': cfg.get('preload_into_ram', False),
    }

    # Remove None values so dataset can use its defaults (keep preload_into_ram since it has a default)
    dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}
    # Debug: track monkeys from config
    print(f"[data_loader] monkeys from config (cfg.get('monkeys')): {cfg.get('monkeys')!r}")
    print(f"[data_loader] monkeys passed to dataset (after dropping None): {dataset_kwargs.get('monkeys')!r}")
    
    # Add any additional dataset-specific parameters
    if dataset_name == 'vsd_mae':
        dataset_kwargs['mask_ratio'] = cfg.get('mask_ratio', 0.75)
        dataset_kwargs['patch_size'] = cfg.get('patch_size', (4, 16, 16))
    elif dataset_name == 'vsd_dino':
        dataset_kwargs['n_local_crops'] = cfg.get('n_local_crops', 6)
        dataset_kwargs['global_crop_scale'] = cfg.get('global_crop_scale', (0.4, 1.0))
        dataset_kwargs['local_crop_scale'] = cfg.get('local_crop_scale', (0.05, 0.4))
    
    # Create dataset
    dataset = create_dataset(dataset_name, **dataset_kwargs)
    
    # Get DataLoader parameters
    batch_size = batch_size or cfg.get('batch_size', 4)
    num_workers = num_workers if num_workers is not None else cfg.get('num_workers', 0)
    shuffle = shuffle if shuffle is not None else cfg.get('shuffle', split == 'train')
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg.get('pin_memory', False)
    )
    
    return loader

