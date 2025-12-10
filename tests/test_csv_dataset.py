"""
Test script for the new CSV-based dataset structure.

This script tests the refactored VsdVideoDataset with the new file hierarchy:
- /main_data_dir/monkeyname/*.h5 files
- /main_data_dir/splits/split_v1_seed17_strat_monkey.csv
- /main_data_dir/splits/stats_v1_seed17_strat_monkey.json

Usage:
    # Test with your actual data (update paths below)
    python test_csv_dataset.py --data_dir /path/to/main_data_dir

    # Test with mock data (creates temporary test files)
    python test_csv_dataset.py --mock
"""
import os
import sys
import argparse
import tempfile
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import VsdVideoDataset
from src.data.data_loader import load_dataset


def create_mock_data(main_data_dir: Path):
    """Create mock data matching the expected directory structure."""
    print("Creating mock data structure...")
    
    # Create directory structure
    splits_dir = main_data_dir / "splits"
    monkey_dir = main_data_dir / "tolkin"
    splits_dir.mkdir(parents=True, exist_ok=True)
    monkey_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few mock H5 files
    n_trials = 5
    n_pixels = 10000  # 100x100
    n_frames = 256
    
    h5_files = []
    for i in range(n_trials):
        h5_path = monkey_dir / f"session_test_trial_{i:03d}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Create trial dataset with shape (n_pixels, n_frames)
            data = np.random.randn(n_pixels, n_frames).astype(np.float32)
            dataset_name = f"trial_{i:06d}"
            f.create_dataset(dataset_name, data=data)
        h5_files.append((str(h5_path), dataset_name))
        print(f"  Created {h5_path} with dataset {dataset_name}")
    
    # Create CSV file
    csv_data = []
    for i, (h5_path, dataset_name) in enumerate(h5_files):
        # Split: first 3 trials -> train, next 1 -> val, last 1 -> test
        if i < 3:
            split = "train"
        elif i < 4:
            split = "val"
        else:
            split = "test"
        
        csv_data.append({
            'trial_global_id': i,
            'monkey': 'tolkin',
            'date': f'2024010{i}',
            'condition': 'condAN2',
            'source_file': f'/fake/source_{i}.mat',
            'target_file': h5_path,
            'trial_index_in_condition': i,
            'shape': f'({n_pixels}, {n_frames})',
            'trial_dataset': dataset_name,
            'split': split
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = splits_dir / "split_v1_seed17_strat_monkey.csv"
    df.to_csv(csv_path, sep='\t', index=False)
    print(f"  Created {csv_path} with {len(df)} trials")
    
    # Create stats JSON
    # Compute actual stats from the data
    all_data = []
    for h5_path, dataset_name in h5_files[:3]:  # Only train split
        with h5py.File(h5_path, 'r') as f:
            data = f[dataset_name][:]
            all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=1)  # Concatenate along frames
    mean_val = float(np.mean(all_data))
    std_val = float(np.std(all_data))
    
    stats_json = {
        "created": "2025-01-01T00:00:00.000000",
        "split_name": "v1",
        "random_seed": 17,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "stratify_level": "monkey",
        "stats": {
            "mean": mean_val,
            "std": std_val,
            "n_values": int(np.prod(all_data.shape)),
            "n_trials_used": 3,
            "subsample_frac": 1.0
        }
    }
    
    json_path = splits_dir / "stats_v1_seed17_strat_monkey.json"
    with open(json_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"  Created {json_path} with mean={mean_val:.4f}, std={std_val:.4f}")
    
    return csv_path, json_path, main_data_dir


def test_dataset_loading(split_csv_path: Path, stats_json_path: Path, processed_root: Path = None):
    """Test loading the dataset with the new CSV structure."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    # Test train split
    print("\n[1] Testing train split...")
    train_dataset = VsdVideoDataset(
        split_csv_path=str(split_csv_path),
        split_name="train",
        stats_json_path=str(stats_json_path),
        processed_root=str(processed_root) if processed_root else None,
        frame_start=1,
        frame_end=100,
        clip_length=16
    )
    
    print(f"  ✓ Train dataset created: {len(train_dataset)} samples")
    assert len(train_dataset) > 0, "Train dataset should have samples"
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"  ✓ Sample loaded successfully")
    print(f"    - Keys: {list(sample.keys())}")
    print(f"    - Video shape: {sample['video'].shape}")
    print(f"    - Mask shape: {sample['mask'].shape}")
    print(f"    - Start frame: {sample['start_frame']}, End frame: {sample['end_frame']}")
    
    # Verify shapes
    video = sample['video']
    assert video.shape[0] == 1, "Channel dimension should be 1"
    assert video.shape[1] == 16, "Temporal dimension should match clip_length"
    assert video.shape[2] == 100 and video.shape[3] == 100, "Spatial dimensions should be 100x100"
    
    # Verify normalization was applied (data should be roughly centered)
    video_mean = float(video.mean())
    video_std = float(video.std())
    print(f"    - Video stats after normalization: mean={video_mean:.4f}, std={video_std:.4f}")
    
    # Test val split
    print("\n[2] Testing val split...")
    val_dataset = VsdVideoDataset(
        split_csv_path=str(split_csv_path),
        split_name="val",
        stats_json_path=str(stats_json_path),
        processed_root=str(processed_root) if processed_root else None,
        frame_start=1,
        frame_end=100,
        clip_length=16
    )
    print(f"  ✓ Val dataset created: {len(val_dataset)} samples")
    
    # Test test split
    print("\n[3] Testing test split...")
    test_dataset = VsdVideoDataset(
        split_csv_path=str(split_csv_path),
        split_name="test",
        stats_json_path=str(stats_json_path),
        processed_root=str(processed_root) if processed_root else None,
        frame_start=1,
        frame_end=100,
        clip_length=16
    )
    print(f"  ✓ Test dataset created: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def test_dataloader(split_csv_path: Path, stats_json_path: Path, processed_root: Path = None):
    """Test creating DataLoaders from the dataset."""
    print("\n" + "="*60)
    print("Testing DataLoader Creation")
    print("="*60)
    
    # Create config
    cfg = {
        'split_csv_path': str(split_csv_path),
        'stats_json_path': str(stats_json_path),
        'processed_root': str(processed_root) if processed_root else None,
        'frame_start': 1,
        'frame_end': 100,
        'clip_length': 16,
        'batch_size': 2,
        'num_workers': 0,
        'shuffle': True
    }
    
    # Test train loader
    print("\n[1] Creating train DataLoader...")
    train_loader = load_dataset(cfg, split="train", batch_size=2, num_workers=0, shuffle=True)
    print(f"  ✓ Train DataLoader created")
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"  ✓ Batch loaded successfully")
    print(f"    - Batch video shape: {batch['video'].shape}")
    print(f"    - Batch mask shape: {batch['mask'].shape}")
    
    assert batch['video'].shape[0] == 2, "Batch size should be 2"
    assert batch['video'].shape[1] == 1, "Channel dimension should be 1"
    assert batch['video'].shape[2] == 16, "Temporal dimension should match clip_length"
    
    # Test val loader
    print("\n[2] Creating val DataLoader...")
    val_loader = load_dataset(cfg, split="val", batch_size=2, num_workers=0, shuffle=False)
    print(f"  ✓ Val DataLoader created")
    
    val_batch = next(iter(val_loader))
    print(f"  ✓ Val batch loaded successfully")
    print(f"    - Batch video shape: {val_batch['video'].shape}")


def test_edge_cases(split_csv_path: Path, stats_json_path: Path, processed_root: Path = None):
    """Test edge cases."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Test with different clip lengths
    print("\n[1] Testing different clip lengths...")
    for clip_len in [1, 8, 32]:
        ds = VsdVideoDataset(
            split_csv_path=str(split_csv_path),
            split_name="train",
            stats_json_path=str(stats_json_path),
            processed_root=str(processed_root) if processed_root else None,
            frame_start=1,
            frame_end=100,
            clip_length=clip_len
        )
        sample = ds[0]
        assert sample['video'].shape[1] == clip_len, f"Clip length should be {clip_len}"
        print(f"  ✓ clip_length={clip_len}: {len(ds)} samples, video shape={sample['video'].shape}")
    
    # Test with different frame ranges
    print("\n[2] Testing different frame ranges...")
    ds = VsdVideoDataset(
        split_csv_path=str(split_csv_path),
        split_name="train",
        stats_json_path=str(stats_json_path),
        processed_root=str(processed_root) if processed_root else None,
        frame_start=20,
        frame_end=80,
        clip_length=10
    )
    sample = ds[0]
    print(f"  ✓ frame_start=20, frame_end=80: {len(ds)} samples, start_frame={sample['start_frame']}")


def main():
    parser = argparse.ArgumentParser(description="Test CSV-based dataset structure")
    parser.add_argument("--data_dir", type=str, help="Path to main data directory")
    parser.add_argument("--mock", action="store_true", help="Create and use mock data")
    args = parser.parse_args()
    
    if args.mock:
        # Create mock data in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            main_data_dir = Path(tmpdir) / "main_data_dir"
            csv_path, json_path, processed_root = create_mock_data(main_data_dir)
            
            print("\n" + "="*60)
            print("Running Tests with Mock Data")
            print("="*60)
            
            # Run tests
            train_ds, val_ds, test_ds = test_dataset_loading(csv_path, json_path, processed_root)
            test_dataloader(csv_path, json_path, processed_root)
            test_edge_cases(csv_path, json_path, processed_root)
            
            print("\n" + "="*60)
            print("✓ All tests passed!")
            print("="*60)
    
    elif args.data_dir:
        # Use actual data
        main_data_dir = Path(args.data_dir)
        splits_dir = main_data_dir / "splits"
        
        csv_path = splits_dir / "split_v1_seed17_strat_monkey.csv"
        json_path = splits_dir / "stats_v1_seed17_strat_monkey.json"
        
        if not csv_path.exists():
            print(f"ERROR: CSV file not found: {csv_path}")
            return
        
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}")
            return
        
        print("\n" + "="*60)
        print(f"Running Tests with Actual Data from {main_data_dir}")
        print("="*60)
        
        # Run tests
        train_ds, val_ds, test_ds = test_dataset_loading(csv_path, json_path, main_data_dir)
        test_dataloader(csv_path, json_path, main_data_dir)
        test_edge_cases(csv_path, json_path, main_data_dir)
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
    
    else:
        print("Please provide either --data_dir <path> or --mock flag")
        print("\nExample usage:")
        print("  python test_csv_dataset.py --mock")
        print("  python test_csv_dataset.py --data_dir /path/to/main_data_dir")


if __name__ == "__main__":
    main()

