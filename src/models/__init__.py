from .cnn3d import Simple3DEncoder
from .mae import VideoMAE




def build_ssl_model(cfg):
    encoder_name = cfg.get("model", "cnn3d")
    embed_dim = cfg.get("embed_dim", 256)
    # Build encoder
    if encoder_name == "cnn3d":
        encoder = Simple3DEncoder(in_channels=cfg.get("channels", 1), dim=embed_dim)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


    # Wrap in SSL task (MAE or plain recon)
    task = cfg.get("task", "mae")
    if task == "mae":
        return VideoMAE(encoder=encoder,
                        patch_size=tuple(cfg.get("patch_size", [4,8,8])),
                        mask_ratio=float(cfg.get("mask_ratio", 0.5)),
                        in_channels=cfg.get("channels", 1))
    elif task == "recon":
        return VideoMAE(encoder=encoder,
                        patch_size=tuple(cfg.get("patch_size", [4,8,8])),
                        mask_ratio=0.0,
                        in_channels=cfg.get("channels", 1))
    else:
        raise ValueError(f"Unknown task: {task}")