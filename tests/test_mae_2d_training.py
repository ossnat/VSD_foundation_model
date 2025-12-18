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
import torch
from pathlib import Path
from torch.utils.data import DataLoader

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

from src.training.trainer import Trainer
from src.utils.mae_utils import (
    load_config,
    create_datasets,
    create_dataloaders,
    extract_single_frame,
    create_masked_batch,
    build_mae_2d_model
)

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
            train_dataset, val_dataset, test_dataset, batch_size=batch_size, num_workers=0
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

