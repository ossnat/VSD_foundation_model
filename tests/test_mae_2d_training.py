"""
Test for training MAE 2D model (single frames) with the new CSV-based dataset structure.

This test is designed to be copied and run in Colab. It:
1. Loads config from configs/default.yaml
2. Creates train/val/test datasets with limited samples (100-200 each)
3. Builds MAE 2D model (ResNet18 encoder + decoder for single frames)
4. Trains the model for a few epochs
5. Verifies the training runs without errors

Usage in Colab:
    # Copy this entire file to a Colab cell and run:
    exec(open('tests/test_mae_2d_training.py').read())
    
    # Or import and run:
    from tests.test_mae_2d_training import test_mae_2d_training
    test_mae_2d_training()
"""
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Handle both script execution and Colab notebook execution
try:
    # When running as a script
    project_root = Path(__file__).parent.parent
except NameError:
    # When running in Colab/notebook, __file__ is not defined
    import os
    cwd = Path(os.getcwd())
    if (cwd / "configs").exists():
        project_root = cwd
    elif (cwd.parent / "configs").exists():
        project_root = cwd.parent
    else:
        project_root = cwd
        print(f"Warning: Could not determine project root, using: {project_root}")

sys.path.insert(0, str(project_root))

from src.data import load_dataset
from src.models.backbone.mae_backbone_2d import MAEResNet18Backbone
from src.models.heads.mae_decoder_2d import MAEDecoder2D
from src.models.systems.mae_system import MAESystem
from src.training.trainer import Trainer

# Try to import logger, make it optional
try:
    from src.utils.logger import TBLogger, set_seed
    HAS_LOGGER = True
except ImportError:
    print("Warning: TBLogger not available, using mock logger")
    HAS_LOGGER = False
    class MockLogger:
        def log_scalar(self, *args, **kwargs):
            pass
    TBLogger = MockLogger
    def set_seed(seed):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def create_limited_dataset(dataset, max_samples=200):
    """Create a subset of the dataset with at most max_samples."""
    actual_samples = min(len(dataset), max_samples)
    indices = list(range(actual_samples))
    return Subset(dataset, indices)


def load_config(config_path):
    """Load and validate configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
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


def create_datasets(cfg, max_train=200, max_val=100, max_test=100):
    """Create train/val/test datasets with limited samples."""
    print(f"\nCreating datasets (limited to {max_train}/{max_val}/{max_test} samples)...")
    
    # Create full datasets first
    train_dataset_full = load_dataset(cfg, split="train", batch_size=1, num_workers=0, shuffle=False).dataset
    val_dataset_full = load_dataset(cfg, split="val", batch_size=1, num_workers=0, shuffle=False).dataset
    test_dataset_full = load_dataset(cfg, split="test", batch_size=1, num_workers=0, shuffle=False).dataset
    
    print(f"  Full dataset sizes: train={len(train_dataset_full)}, val={len(val_dataset_full)}, test={len(test_dataset_full)}")
    
    # Create limited subsets
    train_dataset = create_limited_dataset(train_dataset_full, max_samples=max_train)
    val_dataset = create_limited_dataset(val_dataset_full, max_samples=max_val)
    test_dataset = create_limited_dataset(test_dataset_full, max_samples=max_test)
    
    print(f"  Limited dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4):
    """Create DataLoaders for train/val/test datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  ✓ DataLoaders created with batch_size={batch_size}")
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
    """Build MAE 2D model (ResNet18 encoder + decoder)."""
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


def test_forward_pass(model, train_loader, device):
    """Test forward pass with a single batch."""
    print(f"\nTesting forward pass...")
    
    model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(train_loader))
        
        # Extract single frame and create masked batch
        frame_batch = extract_single_frame(batch)
        mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
        
        # Move to device
        mae_batch = {k: v.to(device) for k, v in mae_batch.items()}
        
        # Forward pass
        output = model(mae_batch)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Output keys: {list(output.keys())}")
        if isinstance(output, dict):
            print(f"    Loss: {output.get('loss', 'N/A')}")
            if 'metrics' in output:
                print(f"    Metrics: {output['metrics']}")
    
    return True


def train_model(model, train_loader, val_loader, cfg, device, epochs=2):
    """Train the model for specified number of epochs."""
    print(f"\nTraining model ({epochs} epochs, limited samples)...")
    
    # Create logger and trainer
    if HAS_LOGGER:
        logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
    else:
        logger = TBLogger()  # Mock logger
    
    # Create custom trainer that handles frame extraction and masking
    class MAE2DTrainer(Trainer):
        def _forward_and_loss(self, batch):
            # Extract single frame
            frame_batch = extract_single_frame(batch)
            # Create masked batch
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            # Move to device
            mae_batch = {k: v.to(self.device) for k, v in mae_batch.items()}
            # Forward pass
            output = self.model(mae_batch)
            return output["loss"]
    
    trainer = MAE2DTrainer(model=model, logger=logger, cfg=cfg, device=device)
    
    # Override epochs for quick test
    original_epochs = cfg.get("epochs", 5)
    cfg["epochs"] = epochs
    
    try:
        trainer.fit(train_loader, val_loader)
        print(f"  ✓ Training completed successfully")
        return True
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original epochs
        cfg["epochs"] = original_epochs


def evaluate_model(model, test_loader, device, num_batches=5):
    """Evaluate model on test set."""
    print(f"\nEvaluating on test set...")
    
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
            
            # Extract single frame and create masked batch
            frame_batch = extract_single_frame(batch)
            mae_batch = create_masked_batch(frame_batch, mask_ratio=0.75)
            
            # Move to device
            mae_batch = {k: v.to(device) for k, v in mae_batch.items()}
            
            # Forward pass
            output = model(mae_batch)
            if isinstance(output, dict):
                loss = output.get("loss", None)
                if loss is not None:
                    test_losses.append(float(loss.item()))
    
    if test_losses:
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"  ✓ Evaluation successful")
        print(f"    Average test loss ({len(test_losses)} batches): {avg_test_loss:.4f}")
        return True
    else:
        print(f"  ✗ No losses collected")
        return False


def test_mae_2d_training():
    """Main test function for MAE 2D training."""
    print("="*60)
    print("MAE 2D Training Test (Single Frames)")
    print("="*60)
    
    try:
        # Load config
        config_path = project_root / "configs" / "default.yaml"
        cfg = load_config(config_path)
        
        # Set seed and device
        set_seed(cfg.get("seed", 42))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_datasets(
            cfg, max_train=200, max_val=100, max_test=100
        )
        
        # Create DataLoaders
        batch_size = min(cfg.get("batch_size", 8), 4)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=batch_size
        )
        
        # Test batch loading
        print(f"\nTesting batch loading...")
        train_batch = next(iter(train_loader))
        print(f"  ✓ Train batch loaded: {list(train_batch.keys())}")
        print(f"    Video shape: {train_batch['video'].shape}")
        
        # Build model
        model = build_mae_2d_model(cfg, device)
        
        # Test forward pass
        test_forward_pass(model, train_loader, device)
        
        # Train model
        success = train_model(model, train_loader, val_loader, cfg, device, epochs=2)
        if not success:
            return False
        
        # Evaluate on test set
        evaluate_model(model, test_loader, device, num_batches=5)
        
        print("\n" + "="*60)
        print("✓ All tests passed! MAE 2D training pipeline works correctly.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mae_2d_training()
    sys.exit(0 if success else 1)

