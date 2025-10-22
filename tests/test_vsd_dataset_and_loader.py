import os
import sys
from pathlib import Path

# Add the project root to Python path to allow importing src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Allow running tests directly with: python -m pytest tests/test_vsd_dataset_and_loader.py -v --tb=line
if __name__ == "__main__":
    import subprocess
    import pytest
    
    # Run pytest with the same arguments
    result = pytest.main([__file__, "-v", "--tb=line"])
    sys.exit(result)

import h5py
import numpy as np
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from src.data.datasets import VsdVideoDataset
from src.data.vsd_multi_crop_dataset import VsdMultiCropDataset
from src.data.vsd_masked_dataset import VsdMaskedDataset
from src.data.data_loader import load_dataset


def _create_minimal_vsd_hdf5(hdf5_path: Path, *,
                              height: int = 100,
                              width: int = 100,
                              frames: int = 16,
                              trials: int = 3) -> None:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(hdf5_path), "w") as f:
        grp = f.create_group("groupA")
        # dataset shape expected by code: (pixels, frames, trials)
        pixels = height * width
        data = np.random.randn(pixels, frames, trials).astype(np.float32)
        grp.create_dataset("dataset1", data=data)


def test_vsd_video_dataset_shapes(tmp_path: Path):
    h5_path = tmp_path / "vsd_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=5)

    # Test with default clip_length=1 (single frames)
    ds = VsdVideoDataset(hdf5_path=str(h5_path))

    # Length equals number of frames across all trials (5 trials * 16 frames = 80)
    assert len(ds) == 5 * 16

    sample = ds[0]
    assert isinstance(sample, dict)
    assert "video" in sample and "mask" in sample

    video: torch.Tensor = sample["video"]
    mask: torch.Tensor = sample["mask"]

    # Expected shapes per implementation (single frame)
    assert video.ndim == 4  # (C, T, H, W)
    assert video.shape[0] == 1  # channels
    assert video.shape[1] == 1  # frames (single frame)
    assert video.shape[2] == 100 and video.shape[3] == 100

    assert mask.shape == (1, 1, 100, 100)
    assert video.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.uint8)


def test_load_dataset_and_visualize(tmp_path: Path):
    h5_path = tmp_path / "vsd_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=24, trials=4)

    cfg = {
        "dataset": "vsd",
        "vsd_hdf5_path": str(h5_path),
        "clip_length": 1,  # Explicitly set to single frames
    }

    loader = load_dataset(cfg, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))

    # Validate batch structure
    assert isinstance(batch, dict)
    assert "video" in batch and "mask" in batch

    video: torch.Tensor = batch["video"]  # (N, C, T, H, W) when collated
    mask: torch.Tensor = batch["mask"]

    # Shape checks (single frames)
    assert video.shape[0] == 2  # batch size
    assert video.shape[1] == 1  # channels
    assert video.shape[2] == 1  # frames (single frame)
    assert video.shape[3] == 100 and video.shape[4] == 100
    assert mask.shape == (2, 1, 1, 100, 100)

    # Minimal visualization to verify configuration
    # Take one sample and render a single frame (since clip_length=1)
    sample_video = video[0]
    if sample_video.shape[0] == 1:
        frame_to_plot = sample_video.squeeze(0).squeeze(0)  # (H, W) - single frame
    else:
        frame_to_plot = sample_video.mean(dim=0).squeeze(0)  # combine channels if present

    # Create a simple single frame visualization
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    vmin, vmax = -0.003, 0.003
    
    ax.imshow(frame_to_plot.cpu().numpy(), cmap="hot", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title("Single Frame", fontsize=10)

    out_png = tmp_path / "vsd_sample_grid.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    assert out_png.exists() and out_png.stat().st_size > 0


def test_vsd_video_dataset_clip_length(tmp_path: Path):
    """Test that VsdVideoDataset returns clips of the correct length."""
    h5_path = tmp_path / "vsd_clip_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=20, trials=3)

    # Test single frame (clip_length=1)
    ds_single = VsdVideoDataset(hdf5_path=str(h5_path), clip_length=1)
    assert len(ds_single) == 3 * 20  # 3 trials * 20 frames per trial
    
    sample_single = ds_single[0]
    video_single = sample_single["video"]
    assert video_single.shape[1] == 1  # 1 frame
    assert video_single.shape[0] == 1  # 1 channel
    assert video_single.shape[2] == 100 and video_single.shape[3] == 100  # spatial dimensions

    # Test video clips (clip_length=5)
    ds_clips = VsdVideoDataset(hdf5_path=str(h5_path), clip_length=5)
    assert len(ds_clips) == 3 * 4  # 3 trials * 4 clips per trial (20 frames // 5 = 4 clips)
    
    sample_clip = ds_clips[0]
    video_clip = sample_clip["video"]
    assert video_clip.shape[1] == 5  # 5 frames
    assert video_clip.shape[0] == 1  # 1 channel
    assert video_clip.shape[2] == 100 and video_clip.shape[3] == 100  # spatial dimensions

    # Test longer clips (clip_length=8)
    ds_long = VsdVideoDataset(hdf5_path=str(h5_path), clip_length=8)
    assert len(ds_long) == 3 * 2  # 3 trials * 2 clips per trial (20 frames // 8 = 2 clips)
    
    sample_long = ds_long[0]
    video_long = sample_long["video"]
    assert video_long.shape[1] == 8  # 8 frames
    assert video_long.shape[0] == 1  # 1 channel
    assert video_long.shape[2] == 100 and video_long.shape[3] == 100  # spatial dimensions

    # Test that clips don't overlap by checking frame indices
    # For clip_length=5, we should get clips starting at frames 0, 5, 10, 15
    expected_start_frames = [0, 5, 10, 15]
    for i in range(4):
        sample = ds_clips[i]
        assert sample["start_frame"] == expected_start_frames[i]
        assert sample["end_frame"] == expected_start_frames[i] + 4  # 5 frames: 0-4, 5-9, 10-14, 15-19

    # Test with frame slicing
    ds_sliced = VsdVideoDataset(hdf5_path=str(h5_path), clip_length=3, frame_start=2, frame_end=17)
    # Available frames: 2-17 (16 frames), with clip_length=3: 16//3 = 5 clips per trial
    assert len(ds_sliced) == 3 * 5  # 3 trials * 5 clips per trial
    
    sample_sliced = ds_sliced[0]
    video_sliced = sample_sliced["video"]
    assert video_sliced.shape[1] == 3  # 3 frames
    assert sample_sliced["start_frame"] == 2  # First clip starts at frame 2
    assert sample_sliced["end_frame"] == 4   # First clip ends at frame 4


def test_vsd_multi_crop_dataset_single_frames(tmp_path: Path):
    """Test VsdMultiCropDataset with single frames (clip_length=1) for DINO training."""
    h5_path = tmp_path / "vsd_multi_crop_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=20, trials=3)

    # Test with single frames (clip_length=1)
    ds = VsdMultiCropDataset(
        hdf5_path=str(h5_path), 
        clip_length=1,
        n_local_crops=4,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4)
    )
    
    # Should have same number of samples as base dataset
    assert len(ds) == 3 * 20  # 3 trials * 20 frames
    
    # Get a sample (should return list of crops)
    crops = ds[0]
    assert isinstance(crops, list)
    
    # Should have 2 global crops + 4 local crops = 6 total crops
    assert len(crops) == 2 + 4  # 2 global + 4 local
    
    # Check that all crops are tensors
    for i, crop in enumerate(crops):
        assert isinstance(crop, torch.Tensor)
        assert crop.ndim == 4  # (C, T, H, W)
        assert crop.shape[0] == 1  # single channel
        
        if i < 2:  # Global crops
            # Global crops should be larger
            assert crop.shape[1] == 1  # T=1 for single frames
            assert crop.shape[2] == 100 and crop.shape[3] == 100  # Full spatial size
        else:  # Local crops
            # Local crops should be smaller
            assert crop.shape[1] == 1  # T=1 for single frames
            assert crop.shape[2] == 50 and crop.shape[3] == 50  # Smaller spatial size


def test_vsd_multi_crop_dataset_video_clips(tmp_path: Path):
    """Test VsdMultiCropDataset with video clips (clip_length>1) for DINO training."""
    h5_path = tmp_path / "vsd_multi_crop_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=20, trials=3)

    # Test with video clips (clip_length=5)
    ds = VsdMultiCropDataset(
        hdf5_path=str(h5_path), 
        clip_length=5,
        n_local_crops=3,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4)
    )
    
    # Should have clips: 3 trials * 4 clips per trial = 12 samples
    assert len(ds) == 3 * 4  # 3 trials * 4 clips per trial
    
    # Get a sample (should return list of crops)
    crops = ds[0]
    assert isinstance(crops, list)
    
    # Should have 2 global crops + 3 local crops = 5 total crops
    assert len(crops) == 2 + 3  # 2 global + 3 local
    
    # Check that all crops are tensors
    for i, crop in enumerate(crops):
        assert isinstance(crop, torch.Tensor)
        assert crop.ndim == 4  # (C, T, H, W)
        assert crop.shape[0] == 1  # single channel
        
        if i < 2:  # Global crops
            # Global crops should be larger
            assert crop.shape[1] == 5  # T=5 for video clips
            assert crop.shape[2] == 100 and crop.shape[3] == 100  # Full spatial size
        else:  # Local crops
            # Local crops should be smaller
            assert crop.shape[1] == 2  # T=2 for local crops (clip_length // 2)
            assert crop.shape[2] == 50 and crop.shape[3] == 50  # Smaller spatial size


def test_vsd_multi_crop_dataset_crop_diversity(tmp_path: Path):
    """Test that VsdMultiCropDataset produces diverse crops (different random crops)."""
    h5_path = tmp_path / "vsd_multi_crop_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=10, trials=1)

    ds = VsdMultiCropDataset(
        hdf5_path=str(h5_path), 
        clip_length=1,
        n_local_crops=2,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4)
    )
    
    # Get multiple samples and check they produce different crops
    crops1 = ds[0]
    crops2 = ds[1]
    
    # Should have same number of crops
    assert len(crops1) == len(crops2)
    
    # Crops should be different (due to random cropping)
    # Check that at least some crops are different
    different_crops = 0
    for crop1, crop2 in zip(crops1, crops2):
        if not torch.allclose(crop1, crop2, atol=1e-6):
            different_crops += 1
    
    # At least some crops should be different due to random cropping
    assert different_crops > 0, "All crops are identical - random cropping may not be working"


def test_vsd_multi_crop_dataset_edge_cases(tmp_path: Path):
    """Test VsdMultiCropDataset with edge cases (very small clips, single frame)."""
    h5_path = tmp_path / "vsd_multi_crop_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=3, trials=1)  # Very small dataset

    # Test with clip_length=3 (all frames in one clip)
    ds = VsdMultiCropDataset(
        hdf5_path=str(h5_path), 
        clip_length=3,
        n_local_crops=2,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4)
    )
    
    # Should have 1 sample (1 trial * 1 clip)
    assert len(ds) == 1
    
    # Get the sample
    crops = ds[0]
    assert isinstance(crops, list)
    assert len(crops) == 2 + 2  # 2 global + 2 local
    
    # All crops should be valid tensors
    for crop in crops:
        assert isinstance(crop, torch.Tensor)
        assert crop.ndim == 4  # (C, T, H, W)
        assert crop.shape[0] == 1  # single channel
        assert crop.shape[1] > 0  # at least 1 temporal dimension
        assert crop.shape[2] > 0 and crop.shape[3] > 0  # valid spatial dimensions


def test_vsd_masked_dataset_2d_images(tmp_path: Path):
    """Test VsdMaskedDataset with 2D images (clip_length=1) for masked autoencoding."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=20, trials=3)

    # Test with single frames (clip_length=1) - 2D images
    ds = VsdMaskedDataset(
        hdf5_path=str(h5_path), 
        clip_length=1,
        mask_ratio=0.75,
        patch_size=(1, 16, 16)  # pT=1 for images
    )
    
    # Should have same number of samples as base dataset
    assert len(ds) == 3 * 20  # 3 trials * 20 frames
    
    # Get a sample
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "video_masked" in sample
    assert "video_target" in sample
    assert "mask" in sample
    
    video_masked = sample["video_masked"]
    video_target = sample["video_target"]
    mask = sample["mask"]
    
    # Check tensor shapes (video is trimmed to fit patches)
    # With patch_size=(1, 16, 16), video is trimmed to 96x96 (6*16 = 96)
    assert video_masked.shape == (1, 1, 96, 96)  # (C, T, H, W) - trimmed to fit patches
    assert video_target.shape == (1, 1, 96, 96)   # (C, T, H, W) - trimmed to fit patches
    assert mask.shape == (1, 1, 6, 6)  # (C, T, nPT, nPH) where nPT=1, nPH=96//16=6
    
    # Check that mask is binary
    assert torch.all((mask == 0) | (mask == 1))
    
    # Check that masked video has some zeros (due to masking)
    assert torch.any(video_masked == 0)
    
    # Check that target video is not all zeros (should be original data)
    assert not torch.all(video_target == 0)


def test_vsd_masked_dataset_3d_video_clips(tmp_path: Path):
    """Test VsdMaskedDataset with 3D video clips (clip_length>1) for masked autoencoding."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=20, trials=3)

    # Test with video clips (clip_length=8) - 3D video
    ds = VsdMaskedDataset(
        hdf5_path=str(h5_path), 
        clip_length=8,
        mask_ratio=0.5,
        patch_size=(4, 16, 16)  # pT=4 for video clips
    )
    
    # Should have clips: 3 trials * 2 clips per trial = 6 samples
    assert len(ds) == 3 * 2  # 3 trials * 2 clips per trial
    
    # Get a sample
    sample = ds[0]
    assert isinstance(sample, dict)
    
    video_masked = sample["video_masked"]
    video_target = sample["video_target"]
    mask = sample["mask"]
    
    # Check tensor shapes for video clips (video is trimmed to fit patches)
    # With patch_size=(4, 16, 16), video is trimmed to 8x96x96 (2*4=8, 6*16=96)
    assert video_masked.shape == (1, 8, 96, 96)  # (C, T, H, W) - trimmed to fit patches
    assert video_target.shape == (1, 8, 96, 96)  # (C, T, H, W) - trimmed to fit patches
    assert mask.shape == (1, 2, 6, 6)  # (C, nPT, nPH, nPW) where nPT=8//4=2, nPH=96//16=6
    
    # Check that mask is binary
    assert torch.all((mask == 0) | (mask == 1))
    
    # Check that masked video has some zeros (due to masking)
    assert torch.any(video_masked == 0)
    
    # Check that target video is not all zeros (should be original data)
    assert not torch.all(video_target == 0)


def test_vsd_masked_dataset_mask_ratio_control(tmp_path: Path):
    """Test that mask_ratio controls the number of masked patches."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=1)

    # Test different mask ratios
    mask_ratios = [0.25, 0.5, 0.75]
    masked_counts = []
    
    for mask_ratio in mask_ratios:
        ds = VsdMaskedDataset(
            hdf5_path=str(h5_path), 
            clip_length=1,
            mask_ratio=mask_ratio,
            patch_size=(1, 16, 16)
        )
        
        # Get multiple samples and count masked patches
        total_masked = 0
        total_patches = 0
        
        for i in range(min(5, len(ds))):  # Test first 5 samples
            sample = ds[i]
            mask = sample["mask"]
            
            # Count masked patches (0) vs visible patches (1)
            masked_patches = torch.sum(mask == 0).item()
            visible_patches = torch.sum(mask == 1).item()
            
            total_masked += masked_patches
            total_patches += masked_patches + visible_patches
        
        if total_patches > 0:
            actual_ratio = total_masked / total_patches
            masked_counts.append(actual_ratio)
    
    # Check that higher mask_ratio leads to more masked patches
    if len(masked_counts) >= 2:
        assert masked_counts[0] < masked_counts[1] < masked_counts[2], \
            f"Mask ratios should be increasing: {masked_counts}"


def test_vsd_masked_dataset_patch_size_control(tmp_path: Path):
    """Test that patch_size controls the volume of each patch."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=1)

    # Test different patch sizes
    patch_sizes = [(1, 8, 8), (1, 16, 16), (1, 32, 32)]
    
    for patch_size in patch_sizes:
        ds = VsdMaskedDataset(
            hdf5_path=str(h5_path), 
            clip_length=1,
            mask_ratio=0.5,
            patch_size=patch_size
        )
        
        # Get a sample
        sample = ds[0]
        mask = sample["mask"]
        
        # Check that mask shape corresponds to patch size
        expected_nPH = 100 // patch_size[1]  # 100 // patch_height
        expected_nPW = 100 // patch_size[2]  # 100 // patch_width
        
        assert mask.shape == (1, 1, expected_nPH, expected_nPW), \
            f"Expected mask shape (1, 1, {expected_nPH}, {expected_nPW}), got {mask.shape}"


def test_vsd_masked_dataset_video_clip_patch_size(tmp_path: Path):
    """Test patch_size with video clips (pT > 1)."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=1)

    # Test with video clips and temporal patches
    ds = VsdMaskedDataset(
        hdf5_path=str(h5_path), 
        clip_length=8,
        mask_ratio=0.5,
        patch_size=(2, 16, 16)  # pT=2 for temporal patches
    )
    
    # Get a sample
    sample = ds[0]
    mask = sample["mask"]
    
    # Check that mask shape includes temporal dimension
    expected_nPT = 8 // 2  # clip_length // pT
    expected_nPH = 100 // 16  # height // patch_height
    expected_nPW = 100 // 16  # width // patch_width
    
    assert mask.shape == (1, expected_nPT, expected_nPH, expected_nPW), \
        f"Expected mask shape (1, {expected_nPT}, {expected_nPH}, {expected_nPW}), got {mask.shape}"


def test_vsd_masked_dataset_return_dict_structure(tmp_path: Path):
    """Test that the dataset returns the correct dictionary structure."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=1)

    ds = VsdMaskedDataset(
        hdf5_path=str(h5_path), 
        clip_length=4,
        mask_ratio=0.6,
        patch_size=(2, 16, 16)
    )
    
    # Get a sample
    sample = ds[0]
    
    # Check dictionary structure
    expected_keys = {"video_masked", "video_target", "mask"}
    assert set(sample.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(sample.keys())}"
    
    # Check that all values are tensors
    for key, value in sample.items():
        assert isinstance(value, torch.Tensor), f"{key} should be a tensor, got {type(value)}"
    
    # Check tensor properties
    video_masked = sample["video_masked"]
    video_target = sample["video_target"]
    mask = sample["mask"]
    
    # All should have the same spatial dimensions
    assert video_masked.shape[2:] == video_target.shape[2:]  # H, W should match
    assert video_masked.shape[1] == video_target.shape[1]    # T should match
    
    # Mask should be binary
    assert torch.all((mask == 0) | (mask == 1)), "Mask should contain only 0s and 1s"
    
    # Masked video should have zeros where mask is 0
    # (This is a basic check - the actual masking logic is more complex)


def test_vsd_masked_dataset_edge_cases(tmp_path: Path):
    """Test VsdMaskedDataset with edge cases (very small clips, extreme ratios)."""
    h5_path = tmp_path / "vsd_masked_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=8, trials=1)

    # Test with very small clip and small patch size
    ds = VsdMaskedDataset(
        hdf5_path=str(h5_path), 
        clip_length=4,
        mask_ratio=0.9,  # Very high masking
        patch_size=(1, 8, 8)  # Small patches
    )
    
    # Should have 2 samples (1 trial * 2 clips: 8 frames // 4 clip_length = 2 clips)
    assert len(ds) == 2
    
    # Get the sample
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "video_masked" in sample
    assert "video_target" in sample
    assert "mask" in sample
    
    # All should be valid tensors
    for key, value in sample.items():
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == 1  # single channel
        assert value.shape[1] > 0   # at least 1 temporal dimension
        assert value.shape[2] > 0 and value.shape[3] > 0  # valid spatial dimensions


