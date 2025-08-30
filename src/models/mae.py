# ==================================
# File: src/models/mae.py
# ==================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class Patchify3D(nn.Module):
    def __init__(self, patch_size=(4,8,8)):
        super().__init__()
        self.pt, self.ph, self.pw = patch_size

    def forward(self, x):
        # x: (B,C,T,H,W)
        B,C,T,H,W = x.shape
        assert T % self.pt == 0 and H % self.ph == 0 and W % self.pw == 0, "Input must be divisible by patch size"
        x = rearrange(x, 'b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)', pt=self.pt, ph=self.ph, pw=self.pw)
        return x  # (B, N, P)

    def unpatchify(self, patches, out_shape):
        # patches: (B, N, P), out_shape=(B,C,T,H,W)
        B,C,T,H,W = out_shape
        t = T//self.pt; h = H//self.ph; w = W//self.pw
        x = rearrange(patches, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                      t=t,h=h,w=w, c=C, pt=self.pt, ph=self.ph, pw=self.pw)
        return x


def random_masking(x, mask_ratio: float):
    # x: (B, N, P)
    B, N, P = x.shape
    if mask_ratio <= 0.0:
        ids_restore = torch.arange(N, device=x.device).repeat(B,1)
        mask = torch.zeros((B, N), device=x.device)
        return x, mask, ids_restore
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,x.shape[2]))

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class LightweightDecoder(nn.Module):
    def __init__(self, patch_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, patch_dim),
        )
    def forward(self, x):
        return self.net(x)


class VideoMAE(nn.Module):
    """Minimal masked autoencoder wrapper around a 3D CNN encoder.
    - Patchify input
    - Randomly mask patches
    - Encode visible patches only via 3D CNN on the *full* video (simplification)
      and predict all patches via a light MLP decoder in patch space.
    This keeps the code compact for a smoke-tested baseline.
    """
    def __init__(self, encoder: nn.Module, patch_size=(4,8,8), mask_ratio=0.5, in_channels=1):
        super().__init__()
        self.encoder = encoder
        self.patch = Patchify3D(patch_size)
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels
        P = in_channels * patch_size[0]*patch_size[1]*patch_size[2]
        self.decoder = LightweightDecoder(patch_dim=P, embed_dim=encoder.proj.out_channels)

    def forward(self, x):
        # x: (B,C,T,H,W)
        B,C,T,H,W = x.shape
        patches = self.patch(x)                # (B, N, P)
        x_masked, mask, ids_restore = random_masking(patches, self.mask_ratio)

        # Encode full video (simple choice for baseline) and pool to embedding per patch position
        feat_map, pooled = self.encoder(x)    # feat_map: (B, D, T', H', W')
        # For a minimal baseline, predict each patch independently via decoder
        # (More advanced versions would align feat_map to patches.)
        pred_visible = self.decoder(x_masked)             # (B, N_keep, P)

        # Reconstruct full sequence by inserting mask tokens (zeros) then inverse-shuffling
        N = patches.shape[1]
        N_keep = pred_visible.shape[1]
        mask_tokens = torch.zeros((B, N - N_keep, pred_visible.shape[2]), device=x.device)
        pred_all = torch.cat([pred_visible, mask_tokens], dim=1)
        # Unshuffle
        pred_all = torch.gather(pred_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,pred_all.shape[2]))

        # Loss on masked patches only
        loss = ((pred_all - patches)**2)
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum()*patches.shape[-1] + 1e-6)

        return {"loss": loss, "recon_patches": pred_all, "target_patches": patches, "mask": mask,
                "feat_map": feat_map, "pooled": pooled,
                "shape": (B,C,T,H,W)}

    def reconstruct_video(self, pred_patches, shape_tuple):
        return self.patch.unpatchify(pred_patches, shape_tuple)
