import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    Projection head used in DINO for both 2D and 3D inputs.

    Takes flattened feature vectors from any backbone (e.g., ResNet18 or R3D-18)
    and maps them into a high-dimensional, normalized space suitable for
    self-distillation.

    Works with both 2D (images) and 3D (videos) since input shape = (B, D).
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 out_dim: int = 65536,
                 use_bn: bool = False):
        super().__init__()

        # ---- 1. Base MLP:  in_dim → hidden_dim → bottleneck_dim ----
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        # ---- 2. Weight-normalized output projection ----
        #   Produces logits for DINO self-distillation
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # DINO trick: fix the weight_g parameter to 1
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone feature tensor of shape (B, D)
        Returns:
            Normalized logits of shape (B, out_dim)
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
