"""
Sanity test for training with the new CSV-based dataset structure.

This test:
1. Loads config from configs/default.yaml
2. Creates train/val/test datasets with limited samples (100-200 each)
3. Trains a model for a few epochs
4. Verifies the training runs without errors

Usage:
    python tests/test_training_sanity.py
"""
import os
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to path
# Handle both script execution and Colab notebook execution
try:
    # When running as a script
    project_root = Path(__file__).parent.parent
except NameError:
    # When running in Colab/notebook, __file__ is not defined
    # Try to find project root from current working directory
    import os
    cwd = Path(os.getcwd())
    # Look for configs directory to identify project root
    if (cwd / "configs").exists():
        project_root = cwd
    elif (cwd.parent / "configs").exists():
        project_root = cwd.parent
    else:
        # Fallback: assume we're in the project root
        project_root = cwd
        print(f"Warning: Could not determine project root, using: {project_root}")

sys.path.insert(0, str(project_root))

from src.data import load_dataset
from src.models import build_ssl_model
from src.training.trainer import Trainer

# Try to import logger, make it optional
try:
    from src.utils.logger import TBLogger, set_seed
    HAS_LOGGER = True
except ImportError:
    print("Warning: TBLogger not available, using mock logger")
    HAS_LOGGER = False
    # Create a simple mock logger
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


def test_training_sanity():
    """Test that training works with the new CSV-based dataset structure."""
    print("="*60)
    print("Training Sanity Test")
    print("="*60)
    
    # Load config
    config_path = project_root / "configs" / "default.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    print(f"\n[1] Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check if using new CSV structure
    if 'split_csv_path' not in cfg or 'stats_json_path' not in cfg:
        print("ERROR: Config must contain 'split_csv_path' and 'stats_json_path' for new structure")
        print("Current config keys:", list(cfg.keys()))
        return False
    
    # Verify paths exist
    split_csv_path = Path(cfg['split_csv_path'])
    stats_json_path = Path(cfg['stats_json_path'])
    
    if not split_csv_path.exists():
        print(f"\nERROR: Split CSV not found: {split_csv_path}")
        print(f"\nTo run this test, you need to:")
        print(f"  1. Update configs/default.yaml with correct paths to your data")
        print(f"  2. Or mount/access your Google Drive with the data")
        print(f"\nThe test expects:")
        print(f"  - split_csv_path: {split_csv_path}")
        print(f"  - stats_json_path: {stats_json_path}")
        return False
    
    if not stats_json_path.exists():
        print(f"\nERROR: Stats JSON not found: {stats_json_path}")
        print(f"\nTo run this test, you need to:")
        print(f"  1. Update configs/default.yaml with correct paths to your data")
        print(f"  2. Or mount/access your Google Drive with the data")
        print(f"\nThe test expects:")
        print(f"  - split_csv_path: {split_csv_path}")
        print(f"  - stats_json_path: {stats_json_path}")
        return False
    
    print(f"  ✓ Config loaded successfully")
    print(f"  ✓ Split CSV found: {split_csv_path}")
    print(f"  ✓ Stats JSON found: {stats_json_path}")
    
    # Set seed
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] Using device: {device}")
    
    # Create datasets with limited samples
    print(f"\n[3] Creating datasets (limited to 200 samples each)...")
    
    # Create full datasets first
    train_dataset_full = load_dataset(cfg, split="train", batch_size=1, num_workers=0, shuffle=False).dataset
    val_dataset_full = load_dataset(cfg, split="val", batch_size=1, num_workers=0, shuffle=False).dataset
    test_dataset_full = load_dataset(cfg, split="test", batch_size=1, num_workers=0, shuffle=False).dataset
    
    print(f"  Full dataset sizes: train={len(train_dataset_full)}, val={len(val_dataset_full)}, test={len(test_dataset_full)}")
    
    # Create limited subsets
    train_dataset = create_limited_dataset(train_dataset_full, max_samples=200)
    val_dataset = create_limited_dataset(val_dataset_full, max_samples=100)
    test_dataset = create_limited_dataset(test_dataset_full, max_samples=100)
    
    print(f"  Limited dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create DataLoaders
    batch_size = min(cfg.get("batch_size", 8), 4)  # Use smaller batch for test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing
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
    
    # Test loading a batch
    print(f"\n[4] Testing batch loading...")
    try:
        train_batch = next(iter(train_loader))
        print(f"  ✓ Train batch loaded: {list(train_batch.keys())}")
        if 'video' in train_batch:
            print(f"    Video shape: {train_batch['video'].shape}")
        elif 'video_masked' in train_batch:
            print(f"    Video masked shape: {train_batch['video_masked'].shape}")
            print(f"    Video target shape: {train_batch['video_target'].shape}")
        
        val_batch = next(iter(val_loader))
        print(f"  ✓ Val batch loaded: {list(val_batch.keys())}")
    except Exception as e:
        print(f"  ✗ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Build model
    print(f"\n[5] Building model...")
    try:
        model = build_ssl_model(cfg).to(device)
        print(f"  ✓ Model created successfully")
        print(f"    Model type: {type(model).__name__}")
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False
    
    # Test forward pass
    print(f"\n[6] Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # Move batch to device
            test_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in train_batch.items()}
            output = model(test_batch)
            print(f"  ✓ Forward pass successful")
            if isinstance(output, dict):
                print(f"    Output keys: {list(output.keys())}")
                print(f"    Loss: {output.get('loss', 'N/A')}")
            elif torch.is_tensor(output):
                print(f"    Output is tensor, shape: {output.shape}")
            else:
                print(f"    Output type: {type(output)}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train for a few epochs (very short)
    print(f"\n[7] Training model (2 epochs, limited samples)...")
    try:
        # Override epochs for quick test
        original_epochs = cfg.get("epochs", 5)
        cfg["epochs"] = 2
        
        # Create logger and trainer
        if HAS_LOGGER:
            logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
        else:
            logger = TBLogger()  # Mock logger
        trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)
        
        # Train
        trainer.fit(train_loader, val_loader)
        
        print(f"  ✓ Training completed successfully")
        
        # Restore original epochs
        cfg["epochs"] = original_epochs
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test evaluation on test set
    print(f"\n[8] Testing evaluation on test set...")
    try:
        model.eval()
        test_losses = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 5:  # Only test first 5 batches
                    break
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                output = model(batch)
                if isinstance(output, dict):
                    loss = output.get("loss", None)
                    if loss is not None:
                        test_losses.append(float(loss.item()))
                elif torch.is_tensor(output):
                    test_losses.append(float(output.item()))
        
        avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
        print(f"  ✓ Evaluation successful")
        print(f"    Average test loss ({len(test_losses)} batches): {avg_test_loss:.4f}")
        
    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed! Training pipeline works correctly.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_training_sanity()
    sys.exit(0 if success else 1)

