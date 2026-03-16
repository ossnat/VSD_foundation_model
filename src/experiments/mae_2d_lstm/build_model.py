from typing import Dict, Any

import torch
from torch import nn

from src.models import build_ssl_model


def build_mae_2d_lstm_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build the MAE 2D+LSTM model from a flat config dict and move it to device.

    Assumes cfg['model'] == 'mae_2d_lstm' and that src.models.build_ssl_model
    is wired to construct Video2DLSTMEncoder + Video2DLSTMDecoder + MAESystem.
    """
    model = build_ssl_model(cfg).to(device)
    return model

