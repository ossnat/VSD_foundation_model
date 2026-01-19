"""
Utility functions for MAE 2D training and data processing.

This module contains shared helper functions used by both training scripts
and test files for MAE 2D model training.
"""
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import load_dataset
from src.models.backbone.mae_backbone_2d import MAEResNet18Backbone
from src.models.heads.mae_decoder_2d import MAEDecoder2D
from src.models.systems.mae_system import MAESystem


def load_config(config_path):
    """Load and validate configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Resolve any relative data paths based on the directory that contains the project
    # (parent.parent.parent from the config file, since Data/ is a sibling of the project root)
    base_dir = config_path.resolve().parent.parent.parent
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if value is None:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            full_path = (base_dir / value_path).resolve()
            cfg[key] = str(full_path)

    # Check if using new CSV structure
    if 'split_csv_path' not in cfg or 'stats_json_path' not in cfg:
        raise ValueError("Config must contain 'split_csv_path' and 'stats_json_path' for new structure")
    
    # Verify paths exist
    split_csv_path = Path(cfg['split_csv_path'])
    stats_json_path = Path(cfg['stats_json_path'])
    
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")
    if not stats_json_path.exists():
        raise FileNotFoundError(f"Stats JSON not found: {stats_json_path}")
    
    print(f"  ✓ Config loaded successfully")
    print(f"  ✓ Split CSV found: {split_csv_path}")
    print(f"  ✓ Stats JSON found: {stats_json_path}")
    
    return cfg


def create_limited_dataset(dataset, max_samples=200):
    """Create a subset of the dataset with at most max_samples."""
    actual_samples = min(len(dataset), max_samples)
    indices = list(range(actual_samples))
    return Subset(dataset, indices)


def create_datasets(cfg, max_train=None, max_val=None, max_test=None):
    """
    Create train/val/test datasets with optional sample limits.
    
    Args:
        cfg: Configuration dictionary
        max_train: Maximum number of training samples (None for all)
        max_val: Maximum number of validation samples (None for all)
        max_test: Maximum number of test samples (None for all)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print(f"\nCreating datasets...")
    
    # Create full datasets first
    train_dataset_full = load_dataset(cfg, split="train", batch_size=1, num_workers=0, shuffle=False).dataset
    val_dataset_full = load_dataset(cfg, split="val", batch_size=1, num_workers=0, shuffle=False).dataset
    test_dataset_full = load_dataset(cfg, split="test", batch_size=1, num_workers=0, shuffle=False).dataset
    
    print(f"  Full dataset sizes: train={len(train_dataset_full)}, val={len(val_dataset_full)}, test={len(test_dataset_full)}")
    
    # Apply limits if specified
    if max_train is not None:
        train_dataset = create_limited_dataset(train_dataset_full, max_samples=max_train)
    else:
        train_dataset = train_dataset_full
    
    if max_val is not None:
        val_dataset = create_limited_dataset(val_dataset_full, max_samples=max_val)
    else:
        val_dataset = val_dataset_full
    
    if max_test is not None:
        test_dataset = create_limited_dataset(test_dataset_full, max_samples=max_test)
    else:
        test_dataset = test_dataset_full
    
    print(f"  Final dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=0):
    """Create DataLoaders for train/val/test datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"  ✓ DataLoaders created with batch_size={batch_size}, num_workers={num_workers}")
    return train_loader, val_loader, test_loader


def extract_single_frame(batch, frame_idx=0):
    """
    Extract a single frame from video batch for 2D MAE training.
    
    Args:
        batch: Dict with 'video' key, shape (B, C, T, H, W) or (B, C, H, W)
        frame_idx: Which frame to extract (default: first frame)
    
    Returns:
        Dict with 'video' as single frame (B, C, H, W)
    """
    video = batch['video']
    
    # Handle different input shapes
    if len(video.shape) == 5:  # (B, C, T, H, W)
        # Extract single frame
        video_frame = video[:, :, frame_idx, :, :]  # (B, C, H, W)
    elif len(video.shape) == 4:  # (B, C, H, W) - already single frame
        video_frame = video
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")
    
    return {'video': video_frame}


def create_masked_batch(batch, mask_ratio=0.75, patch_size=16):
    """
    Create masked batch for MAE training from single frame.
    
    Args:
        batch: Dict with 'video' key, shape (B, C, H, W)
        mask_ratio: Fraction of patches to mask
        patch_size: Size of patches for masking
    
    Returns:
        Dict with 'video_masked', 'video_target', 'mask'
    """
    video = batch['video']  # (B, C, H, W)
    B, C, H, W = video.shape
    
    # Create random mask
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    num_masked = int(num_patches * mask_ratio)
    
    # Randomly select patches to mask
    mask_indices = torch.randperm(num_patches, device=video.device)[:num_masked]
    mask_flat = torch.zeros(num_patches, device=video.device, dtype=torch.bool)
    mask_flat[mask_indices] = True
    mask_patches = mask_flat.reshape(num_patches_h, num_patches_w)
    
    # Expand mask to full resolution
    mask = F.interpolate(
        mask_patches.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode='nearest'
    )  # (1, 1, H, W)
    mask = mask.expand(B, 1, H, W)  # (B, 1, H, W)
    
    # Apply mask to video
    video_masked = video * (1 - mask)  # Masked regions become 0
    
    return {
        'video_masked': video_masked,
        'video_target': video,
        'mask': mask
    }


def build_mae_2d_model(cfg, device):
    """
    Build MAE 2D model (ResNet18 encoder + decoder).
    
    Args:
        cfg: Configuration dictionary
        device: torch device to place model on
    
    Returns:
        MAESystem model
    """
    print(f"\nBuilding MAE 2D model...")
    
    in_channels = cfg.get("channels", 1)
    pretrained = cfg.get("pretrained", True)
    hidden_dim = cfg.get("hidden_dim", 256)
    
    # Build encoder
    encoder = MAEResNet18Backbone(pretrained=pretrained, in_channels=in_channels)
    
    # Build decoder
    decoder = MAEDecoder2D(
        in_channels=encoder.feature_dim,
        out_channels=in_channels,
        hidden_dim=hidden_dim
    )
    
    # Build config for MAESystem
    mae_config = {
        "training": {
            "lr": cfg.get("lr", 1e-4),
            "weight_decay": cfg.get("weight_decay", 0.05)
        },
        "loss": {
            "normalize": cfg.get("normalize_loss", True)
        }
    }
    
    # Create model
    model = MAESystem(encoder=encoder, decoder=decoder, config=mae_config).to(device)
    
    print(f"  ✓ Model created successfully")
    print(f"    Model type: {type(model).__name__}")
    print(f"    Encoder: {type(encoder).__name__}")
    print(f"    Decoder: {type(decoder).__name__}")
    
    return model

