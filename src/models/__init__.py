# Legacy imports - commented out to avoid import errors
# from .cnn3d import Simple3DEncoder
# from .mae import VideoMAE

# MAE 2D imports
from .backbone.mae_backbone_2d import MAEResNet18Backbone
from .heads.mae_decoder_2d import MAEDecoder2D
from .systems.mae_system import MAESystem


def build_ssl_model(cfg):
    """
    Build SSL model based on configuration.
    
    Supports:
    - model="mae_2d": MAE with 2D ResNet18 encoder (single frames)
    - model="cnn3d": 3D CNN encoder (legacy, requires old imports)
    """
    model_type = cfg.get("model", "mae_2d")
    task = cfg.get("task", "mae")
    in_channels = cfg.get("channels", 1)
    
    # MAE 2D model (single frames)
    if model_type == "mae_2d":
        # Build encoder
        encoder = MAEResNet18Backbone(
            pretrained=cfg.get("pretrained", True),
            in_channels=in_channels
        )
        
        # Build decoder
        hidden_dim = cfg.get("hidden_dim", 256)
        decoder = MAEDecoder2D(
            in_channels=encoder.feature_dim,
            out_channels=in_channels,
            hidden_dim=hidden_dim
        )
        
        # Build config for MAESystem
        mae_config = {
            "training": {
                "lr": cfg.get("lr", 1e-4),
                "weight_decay": cfg.get("weight_decay", 0.05)
            },
            "loss": {
                "normalize": cfg.get("normalize_loss", True)
            }
        }
        
        return MAESystem(encoder=encoder, decoder=decoder, config=mae_config)
    
    # Legacy 3D CNN model (requires old imports)
    elif model_type == "cnn3d":
        try:
            from .old_version.cnn3d import Simple3DEncoder
            from .old_version.mae import VideoMAE
        except ImportError:
            raise ImportError("Legacy 3D CNN model requires old_version imports. Use model='mae_2d' instead.")
        
        embed_dim = cfg.get("embed_dim", 256)
        encoder = Simple3DEncoder(in_channels=in_channels, dim=embed_dim)
        
        if task == "mae":
            return VideoMAE(
                encoder=encoder,
                patch_size=tuple(cfg.get("patch_size", [4,8,8])),
                mask_ratio=float(cfg.get("mask_ratio", 0.5)),
                in_channels=in_channels
            )
        elif task == "recon":
            return VideoMAE(
                encoder=encoder,
                patch_size=tuple(cfg.get("patch_size", [4,8,8])),
                mask_ratio=0.0,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'mae_2d', 'cnn3d'")