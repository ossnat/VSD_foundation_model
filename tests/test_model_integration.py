"""
Integration test: End-to-end test of data loading and MAE 2D model training.
Combines dataset functionality from test_vsd_dataset with model testing from test_models_unitest.
"""
import os
import sys
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.vsd_masked_dataset import VsdMaskedDataset
from src.models.backbone.mae_backbone_2d import MAEResNet18Backbone
from src.models.heads.mae_decoder_2d import MAEDecoder2D
from src.models.systems.mae_system import MAESystem


def _create_minimal_vsd_hdf5(hdf5_path: Path, *,
                              height: int = 100,
                              width: int = 100,
                              frames: int = 16,
                              trials: int = 5) -> None:
    """Create a minimal HDF5 file for testing (from test_vsd_dataset pattern)."""
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(hdf5_path), "w") as f:
        grp = f.create_group("groupA")
        # dataset shape expected by code: (pixels, frames, trials)
        pixels = height * width
        data = np.random.randn(pixels, frames, trials).astype(np.float32)
        grp.create_dataset("dataset1", data=data)


def test_mae_2d_integration():
    """
    End-to-end test: Load data from HDF5, create masked dataset, train MAE 2D model.
    """
    print("############################")
    
    # Step 1: Create temporary HDF5 file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h5_path = tmp_path / "vsd_mae_test.hdf5"
        _create_minimal_vsd_hdf5(h5_path, frames=20, trials=5)
        print(f"[OK] Created test HDF5 file: {h5_path}")
        
        # Step 2: Create masked dataset for MAE 2D (single frames)
        dataset = VsdMaskedDataset(
            hdf5_path=str(h5_path),
            clip_length=1,  # Single frames for 2D MAE
            mask_ratio=0.75,
            patch_size=(1, 16, 16)  # 2D patches
        )
        print(f"[OK] Created VsdMaskedDataset: {len(dataset)} samples")
        
        # Step 3: Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        print(f"[OK] Created DataLoader with batch_size=4")
        
        # Step 4: Create MAE 2D model components (from test_models_unitest pattern)
        config = {
            "training": {"lr": 1e-4, "weight_decay": 0.05},
            "loss": {"normalize": True}
        }
        
        encoder = MAEResNet18Backbone(pretrained=False)
        decoder = MAEDecoder2D(in_channels=512, out_channels=1, hidden_dim=256)
        system = MAESystem(encoder, decoder, config)
        print(f"[OK] Created MAE 2D System")
        
        # Step 5: Test forward pass with real data
        batch = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  video_masked: {batch['video_masked'].shape}")
        print(f"  video_target: {batch['video_target'].shape}")
        print(f"  mask: {batch['mask'].shape}")
        
        result = system.forward(batch)
        loss = result["loss"]
        metrics = result["metrics"]
        
        print(f"\n[OK] Forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics: {metrics}")
        
        # Step 6: Test training step (backward + optimizer)
        optimizer = system.get_optimizer()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[OK] Training step successful")
        
        # Step 7: Test multiple batches (mini training loop)
        print(f"\nRunning mini training loop (3 batches)...")
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Limit to 3 batches for speed
                break
            
            result = system.forward(batch)
            loss = result["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"  Batch {i+1}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / min(3, len(dataloader))
        print(f"[OK] Average loss over 3 batches: {avg_loss:.4f}")
        
        print("\n" + "="*60)
        print("[OK] MAE 2D Integration Test Passed!")
        print("="*60 + "\n")


def test_mae_2d_integration_real_data():
    """
    End-to-end test: Load data from existing test_vsd_data.hdf5, create masked dataset, train MAE 2D model.
    Same as test_mae_2d_integration() but uses the real data file instead of random data.
    """
    print("\n" + "="*60)
    print("MAE 2D Integration Test: Real Data → Model → Training")
    print("="*60)
    
    # Step 1: Use existing HDF5 file
    h5_path = project_root / "test_vsd_data.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Expected HDF5 file not found: {h5_path}")
    print(f"[OK] Using existing HDF5 file: {h5_path}")
    
    # Step 2: Create masked dataset for MAE 2D (single frames)
    dataset = VsdMaskedDataset(
        hdf5_path=str(h5_path),
        clip_length=1,  # Single frames for 2D MAE
        mask_ratio=0.75,
        patch_size=(1, 16, 16)  # 2D patches
    )
    print(f"[OK] Created VsdMaskedDataset: {len(dataset)} samples")
    
    # Step 3: Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=0
    )
    print(f"[OK] Created DataLoader with batch_size=4")
    
    # Step 4: Create MAE 2D model components (from test_models_unitest pattern)
    config = {
        "training": {"lr": 1e-4, "weight_decay": 0.05},
        "loss": {"normalize": True}
    }
    
    encoder = MAEResNet18Backbone(pretrained=False)
    decoder = MAEDecoder2D(in_channels=512, out_channels=1, hidden_dim=256)
    system = MAESystem(encoder, decoder, config)
    print(f"[OK] Created MAE 2D System")
    
    # Step 5: Test forward pass with real data
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  video_masked: {batch['video_masked'].shape}")
    print(f"  video_target: {batch['video_target'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    
    result = system.forward(batch)
    loss = result["loss"]
    metrics = result["metrics"]
    
    print(f"\n[OK] Forward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")
    
    # Step 6: Test training step (backward + optimizer)
    optimizer = system.get_optimizer()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[OK] Training step successful")
    
    # Step 7: Test multiple batches (mini training loop)
    print(f"\nRunning mini training loop (3 batches)...")
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        if i >= 12:  # Limit to 3 batches for speed
            break
        
        result = system.forward(batch)
        loss = result["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Batch {i+1}: loss = {loss.item():.4f}")
    
    avg_loss = total_loss / min(3, len(dataloader))
    print(f"[OK] Average loss over 3 batches: {avg_loss:.4f}")
    
    print("\n" + "="*60)
    print("[OK] MAE 2D Integration Test with Real Data Passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_mae_2d_integration()
        test_mae_2d_integration_real_data()
    except Exception as e:
        print(f"\n[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

