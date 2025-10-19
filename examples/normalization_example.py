#!/usr/bin/env python3
"""
Example script showing how to use the new normalization system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_dataset
import torch

def main():
    print("🎬 VSD Dataset Normalization Examples")
    print("=" * 50)
    
    # Example 1: Baseline Z-score normalization
    print("\n1. Baseline Z-score Normalization:")
    cfg_zscore = {
        "dataset": "vsd",
        "vsd_hdf5_path": Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),  # Update this path
        "normalize": True,
        "normalization_type": "baseline_zscore",
        "baseline_frame": 20,
        "frame_start": 1,
        "frame_end": 100,
        "cache_dir": "cache",
        "normalization_kwargs": {"epsilon": 1e-8}
    }
    
    try:
        loader_zscore = load_dataset(cfg_zscore, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader_zscore))
        print(f"✓ Z-score normalized batch shape: {batch['video'].shape}")
        print(f"  Video range: [{batch['video'].min():.4f}, {batch['video'].max():.4f}]")
    except Exception as e:
        print(f"❌ Error with Z-score normalization: {e}")
    
    # Example 2: Baseline Robust normalization
    print("\n2. Baseline Robust Normalization:")
    cfg_robust = {
        "dataset": "vsd",
        "vsd_hdf5_path": Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),  # Update this path
        "normalize": True,
        "normalization_type": "baseline_robust",
        "baseline_frame": 20,
        "frame_start": 0,
        "frame_end": 100,
        "cache_dir": "cache",
        "normalization_kwargs": {"epsilon": 1e-8}
    }
    
    try:
        loader_robust = load_dataset(cfg_robust, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader_robust))
        print(f"✓ Robust normalized batch shape: {batch['video'].shape}")
        print(f"  Video range: [{batch['video'].min():.4f}, {batch['video'].max():.4f}]")
    except Exception as e:
        print(f"❌ Error with robust normalization: {e}")
    
    # Example 3: No normalization
    print("\n3. No Normalization (Original):")
    cfg_no_norm = {
        "dataset": "vsd",
        "vsd_hdf5_path": Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),  # Update this path
        "normalize": False,
        "frame_start": 0,
        "frame_end": 100
    }
    
    try:
        loader_no_norm = load_dataset(cfg_no_norm, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader_no_norm))
        print(f"✓ No normalization batch shape: {batch['video'].shape}")
        print(f"  Video range: [{batch['video'].min():.4f}, {batch['video'].max():.4f}]")
    except Exception as e:
        print(f"❌ Error without normalization: {e}")
    
    # Example 4: Custom frame slicing
    print("\n4. Custom Frame Slicing:")
    cfg_custom = {
        "dataset": "vsd",
        "vsd_hdf5_path": Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),  # Update this path
        "normalize": True,
        "normalization_type": "baseline_zscore",
        "baseline_frame": 30,
        "frame_start": 10,
        "frame_end": 50,
        "cache_dir": "cache"
    }
    
    try:
        loader_custom = load_dataset(cfg_custom, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader_custom))
        print(f"✓ Custom slicing batch shape: {batch['video'].shape}")
        print(f"  Frames: {batch['video'].shape[2]} (sliced from frames 10-50)")
    except Exception as e:
        print(f"❌ Error with custom slicing: {e}")

def demonstrate_direct_dataset_usage():
    """Demonstrate direct dataset usage without DataLoader"""
    print("\n" + "=" * 50)
    print("📊 Direct Dataset Usage Examples")
    print("=" * 50)
    
    from src.data.datasets import VsdVideoDataset
    
    # Example: Direct dataset creation with normalization
    print("\n1. Direct Dataset Creation with Baseline Z-score:")
    
    try:
        dataset = VsdVideoDataset(
            hdf5_path=Path(r"G:\My Drive\HDF5_DATA_AFTER_PREPROCESSING2\vsd_video_data.hdf5"),  # Update this path
            normalize=True,
            normalization_type="baseline_zscore",
            baseline_frame=20,
            frame_start=0,
            frame_end=100,
            cache_dir="cache"
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Get a sample
        sample = dataset[0]
        print(f"  Sample shape: {sample['video'].shape}")
        print(f"  Video range: [{sample['video'].min():.4f}, {sample['video'].max():.4f}]")
        
    except Exception as e:
        print(f"❌ Error with direct dataset creation: {e}")

if __name__ == "__main__":
    main()
    demonstrate_direct_dataset_usage()
    
    print("\n Normalization examples completed!")
    print("\n Usage Notes:")
    print("- First run will be slower as it computes normalization statistics")
    print("- Subsequent runs will use cached statistics for faster loading")
    print("- Cache files are stored in the specified cache_dir")
