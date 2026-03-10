# src/models/backbone/mae_backbone_2d_lstm.py
"""
2D encoder + LSTM for video MAE (memory-efficient alternative to full 3D CNN).
Encodes each frame with a shared 2D ResNet, pools to a vector per frame,
then runs an LSTM over time. Output is (B, 512, T, H', W') for the decoder.
"""
import torch
import torch.nn as nn
from .mae_backbone_2d import MAEResNet18Backbone


class Video2DLSTMEncoder(nn.Module):
    """
    Encode video (B, C, T, H, W) by: 2D ResNet per frame -> pool -> LSTM -> expand to (B, 512, T, H', W').
    """
    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 1,
        lstm_hidden: int = 256,
        input_height: int = 100,
        input_width: int = 100,
    ):
        super().__init__()
        self.backbone_2d = MAEResNet18Backbone(pretrained=pretrained, in_channels=in_channels)
        self.feature_dim = 512
        # ResNet18 downsamples by 32
        self.encoder_h = max(1, input_height // 32)
        self.encoder_w = max(1, input_width // 32)
        self.flatten_size = self.feature_dim * self.encoder_h * self.encoder_w

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_hidden = lstm_hidden
        self.to_spatial = nn.Linear(lstm_hidden, self.flatten_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) masked video
        Returns:
            (B, 512, T, H', W') feature video for decoder
        """
        B, C, T, H, W = x.shape
        # (B*T, C, H, W)
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        # (B*T, 512, H', W')
        feats = self.backbone_2d(x_flat)
        # Pool to (B*T, 512)
        feats = feats.mean(dim=(2, 3))
        # (B, T, 512)
        feats = feats.view(B, T, -1)
        # LSTM -> (B, T, hidden)
        out, _ = self.lstm(feats)
        # (B, T, flatten_size) -> (B, 512, T, H', W')
        out = self.to_spatial(out)
        out = out.view(B, T, self.feature_dim, self.encoder_h, self.encoder_w)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out
