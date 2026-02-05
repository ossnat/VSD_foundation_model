import torch
import numpy as np
import random
from .datasets import VsdVideoDataset

class VsdMaskedDataset(VsdVideoDataset):
    """
    Extends VsdVideoDataset to yield masked input + target video pairs
    for masked autoencoding in 2D (images) and 3D (video clips).
    """
    
    def __init__(self, *args, mask_ratio=0.75, patch_size=(4,16,16), **kwargs):
        """
        Args:
            mask_ratio (float): Fraction of patches to mask out.
            patch_size (tuple): Patch size (T, H, W) for masking.
        """
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size  # (T, H, W)
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        video = sample["video"]  # (1, T, H, W)
        video_np = video.squeeze(0).numpy()  # (T, H, W)
        
        # Compute mask blocks
        mask, masked_video, target = self.apply_mask(video_np)
        
        # Convert tensors back with channel dim
        masked_tensor = torch.from_numpy(masked_video).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # binary mask
        
        return {
            "video_masked": masked_tensor,      # Input to model, has masked patches zeroed
            "video_target": target_tensor,      # Ground truth for masked patches
            "mask": mask_tensor,                # Mask indicator: 1=keep, 0=mask patch
            "monkey": sample["monkey"],
            "date": sample["date"],
            "condition": sample["condition"],
        }
    
    def apply_mask(self, video: np.ndarray):
        """
        Generates a binary mask, masked video input, and target
        Args:
            video: ndarray (T, H, W)
        Returns:
            mask: ndarray with shape (num_patches_T, num_patches_H, num_patches_W), 1=visible patch
            masked_video: video with masked patches zeroed
            target: original video (unchanged)
        """
        T, H, W = video.shape
        pT, pH, pW = self.patch_size
        
        # Number of patches per dimension
        nPT = T // pT
        nPH = H // pH
        nPW = W // pW
        
        # Trim video to full patches only (if needed)
        video_trimmed = video[: nPT * pT, : nPH * pH, : nPW * pW]
        
        # Initialize mask: 1=keep, 0=mask
        total_patches = nPT * nPH * nPW
        num_masked = int(self.mask_ratio * total_patches)
        
        mask_flat = np.ones(total_patches, dtype=np.float32)
        mask_flat[:num_masked] = 0  # mask out first num_masked patches
        np.random.shuffle(mask_flat)
        mask = mask_flat.reshape(nPT, nPH, nPW)
        
        # Create masked video by zeroing masked patches
        masked_video = video_trimmed.copy()
        
        for t_idx in range(nPT):
            for h_idx in range(nPH):
                for w_idx in range(nPW):
                    if mask[t_idx, h_idx, w_idx] == 0:
                        # Zero-out this patch
                        t_start, h_start, w_start = (t_idx * pT, h_idx * pH, w_idx * pW)
                        masked_video[t_start:t_start+pT, h_start:h_start+pH, w_start:w_start+pW] = 0
        
        return mask, masked_video, video_trimmed
