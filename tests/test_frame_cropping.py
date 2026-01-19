"""
Test script to verify frame cropping functionality (circle and square).

This test:
1. Creates mock data with known patterns
2. Tests cropping with circle and square
3. Visualizes before/after cropping
4. Verifies crop is centered and has correct radius

Usage:
    python tests/test_frame_cropping.py
"""
import os
import sys
import tempfile
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import VsdVideoDataset


def create_mock_data_for_crop_test(temp_dir: Path):
    """Create mock data for testing cropping."""
    # Create directory structure
    monkey_dir = temp_dir / "tolkin"
    splits_dir = temp_dir / "splits"
    monkey_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test H5 file with a pattern that's easy to see
    h5_path = monkey_dir / "test_session.h5"
    with h5py.File(h5_path, 'w') as f:
        # Create data with a clear pattern: gradient from center
        # Shape: (10000 pixels, 256 frames)
        height, width = 100, 100
        n_frames = 256
        
        # Create pattern: radial gradient (brighter in center)
        y_coords, x_coords = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        # Normalize to 0-1, then scale
        pattern = 1.0 - (distances / max_dist)
        
        # Create data: pattern repeated across frames with some variation
        data = np.zeros((height * width, n_frames))
        for frame_idx in range(n_frames):
            frame_data = pattern.flatten() * (0.5 + 0.5 * np.sin(frame_idx / 10.0))
            data[:, frame_idx] = frame_data
        
        f.create_dataset('trial_000000', data=data)
    
    # Create CSV file
    csv_path = splits_dir / "test_split.csv"
    df = pd.DataFrame({
        'trial_global_id': [1],
        'monkey': ['tolkin'],
        'date': ['test'],
        'condition': ['test'],
        'source_file': ['test.mat'],
        'target_file': [str(h5_path)],
        'trial_index_in_condition': [0],
        'shape': ['(10000, 256)'],
        'trial_dataset': ['trial_000000'],
        'split': ['train']
    })
    df.to_csv(csv_path, index=False)
    
    # Create stats JSON and H5
    stats_json_path = splits_dir / "test_stats.json"
    stats_h5_path = splits_dir / "test_stats.h5"
    
    # Create mean and std arrays (spatial, not scalar)
    mean_array = np.ones((100, 100)) * 0.5  # (H, W)
    std_array = np.ones((100, 100)) * 0.1   # (H, W)
    
    with h5py.File(stats_h5_path, 'w') as f:
        f.create_dataset('mean', data=mean_array)
        f.create_dataset('std', data=std_array)
    
    stats_data = {
        'stats_h5_path': str(stats_h5_path),
        'mean_shape': [1, 1, 100, 100]
    }
    
    with open(stats_json_path, 'w') as f:
        json.dump(stats_data, f)
    
    return csv_path, stats_json_path, temp_dir


def visualize_crop_comparison(dataset_no_crop, dataset_with_crop, crop_type, crop_radius, save_dir="test_crop_visualizations"):
    """Visualize before/after cropping."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a sample from each dataset
    sample_no_crop = dataset_no_crop[0]
    sample_with_crop = dataset_with_crop[0]
    
    # Extract first frame
    video_no_crop = sample_no_crop['video'][0, 0].numpy()  # (H, W)
    video_with_crop = sample_with_crop['video'][0, 0].numpy()  # (H, W)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Before crop (full frame)
    im1 = axes[0, 0].imshow(video_no_crop, cmap='hot', vmin=video_no_crop.min(), vmax=video_no_crop.max())
    axes[0, 0].set_title("Before Crop (Full Frame)")
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # After crop
    im2 = axes[0, 1].imshow(video_with_crop, cmap='hot', vmin=video_with_crop.min(), vmax=video_with_crop.max())
    crop_size_str = f"{video_with_crop.shape[0]}x{video_with_crop.shape[1]}"
    axes[0, 1].set_title(f"After {crop_type.capitalize()} Crop (radius={crop_radius}, size={crop_size_str})")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference (to see what was cropped)
    # For square crops, we need to extract the same region from original
    if crop_type == 'square':
        orig_center_y, orig_center_x = video_no_crop.shape[0] // 2, video_no_crop.shape[1] // 2
        y_start = orig_center_y - video_with_crop.shape[0] // 2
        y_end = y_start + video_with_crop.shape[0]
        x_start = orig_center_x - video_with_crop.shape[1] // 2
        x_end = x_start + video_with_crop.shape[1]
        video_no_crop_region = video_no_crop[y_start:y_end, x_start:x_end]
        difference = video_no_crop_region - video_with_crop
    else:
        # For circle, pad the cropped video to match original size for visualization
        pad_h = (video_no_crop.shape[0] - video_with_crop.shape[0]) // 2
        pad_w = (video_no_crop.shape[1] - video_with_crop.shape[1]) // 2
        video_with_crop_padded = np.pad(video_with_crop, 
                                        ((pad_h, pad_h), (pad_w, pad_w)), 
                                        mode='constant', constant_values=0)
        difference = video_no_crop - video_with_crop_padded
    
    im3 = axes[1, 0].imshow(difference, cmap='RdBu_r', vmin=-abs(difference).max(), vmax=abs(difference).max())
    axes[1, 0].set_title("Difference (Before - After)")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Show crop region overlay
    axes[1, 1].imshow(video_no_crop, cmap='gray', alpha=0.5)
    # Draw crop boundary
    height, width = video_no_crop.shape
    center_y, center_x = height // 2, width // 2
    radius_pixels = (crop_radius / 50.0) * (width / 2.0)
    
    if crop_type == 'circle':
        circle = plt.Circle((center_x, center_y), radius_pixels, fill=False, color='red', linewidth=2)
        axes[1, 1].add_patch(circle)
    elif crop_type == 'square':
        square = plt.Rectangle(
            (center_x - radius_pixels, center_y - radius_pixels),
            2 * radius_pixels, 2 * radius_pixels,
            fill=False, color='red', linewidth=2
        )
        axes[1, 1].add_patch(square)
    
    axes[1, 1].set_title(f"Crop Region Overlay ({crop_type}, radius={crop_radius})")
    axes[1, 1].axis('off')
    axes[1, 1].set_xlim(0, width)
    axes[1, 1].set_ylim(height, 0)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"crop_comparison_{crop_type}_radius{crop_radius}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {save_path}")


def test_crop_statistics(dataset_no_crop, dataset_with_crop, crop_type, crop_radius):
    """Test that cropping works correctly by checking statistics."""
    print(f"\n  Testing {crop_type} crop with radius {crop_radius}...")
    
    # Get samples
    sample_no_crop = dataset_no_crop[0]
    sample_with_crop = dataset_with_crop[0]
    
    video_no_crop = sample_no_crop['video'][0, 0].numpy()  # (H, W)
    video_with_crop = sample_with_crop['video'][0, 0].numpy()  # (H, W)
    
    # Calculate expected crop region
    height, width = video_no_crop.shape
    center_y, center_x = height // 2, width // 2
    radius_pixels = (crop_radius / 50.0) * (width / 2.0)
    
    # Create mask for crop region
    y_coords, x_coords = np.ogrid[:height, :width]
    if crop_type == 'circle':
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        crop_mask = distances <= radius_pixels
    elif crop_type == 'square':
        dx = np.abs(x_coords - center_x)
        dy = np.abs(y_coords - center_y)
        crop_mask = (dx <= radius_pixels) & (dy <= radius_pixels)
    
    # For square crops, the tensor is actually cropped, so we need to check differently
    if crop_type == 'square':
        # For square, tensor should be smaller
        expected_size = int(2 * radius_pixels)
        if video_with_crop.shape[0] != expected_size or video_with_crop.shape[1] != expected_size:
            print(f"    WARNING: Square crop should result in {expected_size}x{expected_size} tensor, "
                  f"got {video_with_crop.shape}")
            return False
        print(f"    ✓ Square crop: tensor size is {video_with_crop.shape[0]}x{video_with_crop.shape[1]}")
    elif crop_type == 'circle':
        # For circle, we use square bounding box, so check that corners are zeroed
        # The tensor should be square (bounding box)
        expected_size = int(2 * radius_pixels)
        if video_with_crop.shape[0] != expected_size or video_with_crop.shape[1] != expected_size:
            print(f"    WARNING: Circle crop should use {expected_size}x{expected_size} bounding box, "
                  f"got {video_with_crop.shape}")
            return False
        
        # Check that corners (outside circle) are zero
        crop_h, crop_w = video_with_crop.shape
        center_y_crop, center_x_crop = crop_h // 2, crop_w // 2
        y_coords, x_coords = np.ogrid[:crop_h, :crop_w]
        distances = np.sqrt((x_coords - center_x_crop) ** 2 + (y_coords - center_y_crop) ** 2)
        outside_circle = distances > radius_pixels
        pixels_outside = video_with_crop[outside_circle]
        max_outside = np.abs(pixels_outside).max()
        
        print(f"    Pixels outside circle (corners): {outside_circle.sum()}")
        print(f"    Max absolute value outside circle: {max_outside:.6f}")
        
        if max_outside > 1e-5:  # Allow small numerical errors
            print(f"    WARNING: Pixels outside circle should be zero, but max is {max_outside}")
            return False
    
    # Check that pixels inside crop are preserved (allowing for normalization differences)
    # For square crops, we need to extract the corresponding region from the original
    if crop_type == 'square':
        # Extract the same region from original
        orig_center_y, orig_center_x = video_no_crop.shape[0] // 2, video_no_crop.shape[1] // 2
        y_start = orig_center_y - video_with_crop.shape[0] // 2
        y_end = y_start + video_with_crop.shape[0]
        x_start = orig_center_x - video_with_crop.shape[1] // 2
        x_end = x_start + video_with_crop.shape[1]
        pixels_inside_no_crop = video_no_crop[y_start:y_end, x_start:x_end]
        pixels_inside_with_crop = video_with_crop
    else:  # circle
        inside_crop = crop_mask
        pixels_inside_no_crop = video_no_crop[inside_crop]
        # For circle, extract same region from cropped tensor
        crop_h, crop_w = video_with_crop.shape
        center_y_crop, center_x_crop = crop_h // 2, crop_w // 2
        y_coords, x_coords = np.ogrid[:crop_h, :crop_w]
        distances = np.sqrt((x_coords - center_x_crop) ** 2 + (y_coords - center_y_crop) ** 2)
        inside_circle = distances <= radius_pixels
        pixels_inside_with_crop = video_with_crop[inside_circle]
    
    # After normalization, values will be different, but the pattern should be similar
    # Check correlation instead of exact match
    if len(pixels_inside_no_crop) > 0 and len(pixels_inside_with_crop) > 0:
        # Flatten for correlation
        flat_no_crop = pixels_inside_no_crop.flatten()
        flat_with_crop = pixels_inside_with_crop.flatten()
        
        # Make sure they're the same length
        min_len = min(len(flat_no_crop), len(flat_with_crop))
        flat_no_crop = flat_no_crop[:min_len]
        flat_with_crop = flat_with_crop[:min_len]
        
        correlation = np.corrcoef(flat_no_crop, flat_with_crop)[0, 1]
        print(f"    Correlation inside crop region: {correlation:.4f}")
        
        if correlation < 0.7:  # Lower threshold due to normalization
            print(f"    WARNING: Low correlation inside crop region")
            return False
    
    print(f"    ✓ Crop test passed")
    return True


def test_frame_cropping():
    """Main test function for frame cropping."""
    print("="*60)
    print("Frame Cropping Test")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        print("\n[1] Creating mock data...")
        csv_path, stats_json_path, processed_root = create_mock_data_for_crop_test(temp_path)
        print(f"  ✓ Mock data created")
        
        # Test configurations
        test_configs = [
            {'crop_frame': None, 'crop_radius': None, 'name': 'no_crop'},
            {'crop_frame': 'circle', 'crop_radius': 35, 'name': 'circle_35'},
            {'crop_frame': 'square', 'crop_radius': 40, 'name': 'square_40'},
            {'crop_frame': 'circle', 'crop_radius': 50, 'name': 'circle_50_full'},
        ]
        
        datasets = {}
        
        # Create datasets for each config
        print("\n[2] Creating datasets with different crop settings...")
        for config in test_configs:
            cfg = {
                'split_csv_path': str(csv_path),
                'split_name': 'train',
                'stats_json_path': str(stats_json_path),
                'processed_root': str(processed_root),
                'frame_start': 1,
                'frame_end': 10,  # Use only first 10 frames for quick test
                'clip_length': 1,
                'crop_frame': config['crop_frame'],
                'crop_radius': config['crop_radius'],
            }
            
            try:
                dataset = VsdVideoDataset(cfg=cfg)
                datasets[config['name']] = dataset
                print(f"  ✓ Created dataset: {config['name']}")
            except Exception as e:
                print(f"  ✗ Failed to create dataset {config['name']}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Test statistics
        print("\n[3] Testing crop statistics...")
        no_crop_dataset = datasets['no_crop']
        
        for config in test_configs[1:]:  # Skip no_crop
            if config['name'] in datasets:
                success = test_crop_statistics(
                    no_crop_dataset,
                    datasets[config['name']],
                    config['crop_frame'],
                    config['crop_radius']
                )
                if not success:
                    print(f"  ✗ Test failed for {config['name']}")
                    return False
        
        # Visualize
        print("\n[4] Creating visualizations...")
        os.makedirs("test_crop_visualizations", exist_ok=True)
        
        for config in test_configs[1:]:  # Skip no_crop
            if config['name'] in datasets:
                visualize_crop_comparison(
                    no_crop_dataset,
                    datasets[config['name']],
                    config['crop_frame'],
                    config['crop_radius']
                )
        
        print("\n" + "="*60)
        print("✓ All crop tests passed!")
        print("="*60)
        print(f"\nVisualizations saved to: test_crop_visualizations/")
        return True


if __name__ == "__main__":
    success = test_frame_cropping()
    sys.exit(0 if success else 1)

