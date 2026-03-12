# src/models/heads/mae_decoder_2d_lstm.py
"""
Decoder for 2D+LSTM video MAE: takes (B, 512, T, H', W') and runs 2D decoder per frame -> (B, 1, T, H, W).
"""
import torch
import torch.nn as nn
from .mae_decoder_2d import MAEDecoder2D


class Video2DLSTMDecoder(nn.Module):
    """
    Decode feature video (B, 512, T, H', W') by running 2D decoder on each frame.
    """
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.decoder_2d = MAEDecoder2D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: (B, 512, T, H', W')
            target_size: (T, H, W) or (H, W). If (T, H, W) we output (B, 1, T, H, W).
        Returns:
            (B, 1, T, H, W) reconstructed video
        """
        B, C, T, Hp, Wp = x.shape
        # (B*T, 512, H', W')
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, Hp, Wp)
        if target_size is not None and len(target_size) == 3:
            # (T, H, W) -> pass (H, W) to 2D decoder per frame
            _, H, W = target_size
            frame_size = (H, W)
        elif target_size is not None and len(target_size) == 2:
            frame_size = target_size
        else:
            frame_size = None
        # (B*T, 1, H, W)
        out = self.decoder_2d(x_flat, target_size=frame_size)
        # (B, 1, T, H, W)
        out = out.view(B, T, 1, out.shape[2], out.shape[3]).permute(0, 2, 1, 3, 4).contiguous()
        return out
