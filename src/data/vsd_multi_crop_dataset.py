import torch
from typing import List, Tuple
import random
import numpy as np
from .datasets import VsdVideoDataset

class VsdMultiCropDataset(VsdVideoDataset):
    """
    Extends VsdVideoDataset to yield DINO-style multi-crop views.
    Supports both 2D image frames (clip_length=1) and 3D video clips (clip_length>1).
    """
    
    def __init__(self, *args, n_local_crops: int = 6, global_crop_scale: Tuple[float,float]=(0.4,1.0),
                 local_crop_scale: Tuple[float,float]=(0.05,0.4), **kwargs):
        super().__init__(*args, **kwargs)  # Initialize base class
        
        self.n_local_crops = n_local_crops
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        
        self.clip_length = getattr(self, 'clip_length', 1)
        
        # Define patch sizes for resizing global/local crops
        # Can be adjusted or taken from config
        self.global_crop_size = (self.clip_length, 100, 100)  # Temporal, H, W
        self.local_crop_size = (max(1, self.clip_length // 2), 50, 50)
    
    def __getitem__(self, idx):
        # Use base class to get video clip tensor
        sample = super().__getitem__(idx)  # Returns dict with 'video', 'mask', 'start_frame', 'end_frame'
        
        video = sample['video']  # shape (1, T, 100, 100)
        
        # Remove channel dim for augmentation functions
        video_np = video.squeeze(0).numpy()  # shape (T, H, W)
        
        crops = []
        # Generate 2 global crops
        for _ in range(2):
            crop = self.random_crop_resize(
                video_np, self.global_crop_scale,
                self.global_crop_size
            )
            crop_tensor = torch.from_numpy(crop).unsqueeze(0)  # Add channel back: (1, T, H, W)
            crops.append(crop_tensor)
        
        # Generate local crops
        for _ in range(self.n_local_crops):
            crop = self.random_crop_resize(
                video_np, self.local_crop_scale,
                self.local_crop_size
            )
            crop_tensor = torch.from_numpy(crop).unsqueeze(0)  # Add channel back: (1, T, H, W)
            crops.append(crop_tensor)
        
        return {
            "crops": crops,
            "monkey": sample["monkey"],
            "date": sample["date"],
            "condition": sample["condition"],
        }
    
    def random_crop_resize(self, video: np.ndarray, scale_range: Tuple[float,float], target_size: Tuple[int,int,int]) -> np.ndarray:
        """
        Randomly crops and resizes a (T,H,W) video clip
        Args:
            video: np.ndarray (T,H,W)
            scale_range: (min_scale, max_scale) fraction of spatial crop size
            target_size: (T_out, H_out, W_out)
        Returns:
            cropped and resized video: np.ndarray (T_out, H_out, W_out)
        """
        T, H, W = video.shape
        
        # Temporal crop size is full length or clip_length
        t_out, h_out, w_out = target_size
        
        # Spatial crop scaling
        scale = random.uniform(*scale_range)
        crop_h = min(H, max(1, int(H * scale)))
        crop_w = min(W, max(1, int(W * scale)))
        
        # Temporal crop size
        crop_t = min(T, t_out)
        
        # Random starting indices
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        t0 = random.randint(0, T - crop_t) if T > crop_t else 0
        
        # Crop
        cropped = video[t0:t0+crop_t, top:top+crop_h, left:left+crop_w]
        
        # Resize spatially and temporally using trilinear interpolation
        # Convert to torch tensor for interpolation
        crop_tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0).float()  # (1,1,T,H,W)
        resized = torch.nn.functional.interpolate(
            crop_tensor, size=target_size, mode='trilinear', align_corners=False
        ).squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions: (T,H,W)
        
        return resized
