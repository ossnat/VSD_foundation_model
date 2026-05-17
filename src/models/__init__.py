# Legacy imports - commented out to avoid import errors
# from .cnn3d import Simple3DEncoder
# from .mae import VideoMAE

# MAE 2D imports
from .backbone.mae_backbone_2d import MAEResNet18Backbone
from .backbone.mae_backbone_2d_CNN import MAEShallowCNNBackbone
from .heads.mae_decoder_2d import MAEDecoder2D
from .backbone.mae_backbone_2d_lstm import Video2DLSTMEncoder
from .heads.mae_decoder_2d_lstm import Video2DLSTMDecoder
# MAE 3D imports
from .backbone.mae_backbone_3d import MAER3D18Backbone
from .heads.mae_decoder_3d import MAEDecoder3D
from .systems.mae_system import MAESystem
from .systems.linear_mae_system import LinearMAESystem

# ---------------------------------------------------------------------------
# Backbone registry — maps config name -> class.
# Every backbone must expose a `feature_dim` attribute.
# To add a new backbone: import it above and add a key here.
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    "resnet18": MAEResNet18Backbone,
    "MAEShallowCNNBackbone": MAEShallowCNNBackbone,
    "MAER3D18Backbone": MAER3D18Backbone,
}


def _infer_linear_mae_spatial_hw(cfg) -> tuple:
    """
    Spatial (H, W) after patch alignment (same trimming as VsdMaskedDataset.apply_mask).

    Resolution order:
      1) cfg['linear_spatial_hw'] = [H, W] if set (should be patch-aligned for your data)
      2) stats_json mean_shape[2:4] if the file exists
      3) default (100, 100)

    Then trim to multiples of patch_size (H, W) from cfg (default [1, 8, 8]).
    """
    import json
    from pathlib import Path

    hw = cfg.get("linear_spatial_hw")
    if hw is not None and isinstance(hw, (list, tuple)) and len(hw) >= 2:
        h0, w0 = int(hw[0]), int(hw[1])
    else:
        h0, w0 = 100, 100
        stats_path = cfg.get("stats_json_path")
        if stats_path:
            p = Path(stats_path)
            if p.is_file():
                try:
                    with open(p, "r") as f:
                        sd = json.load(f)
                    ms = sd.get("mean_shape")
                    if isinstance(ms, (list, tuple)) and len(ms) >= 4:
                        h0, w0 = int(ms[2]), int(ms[3])
                except Exception:
                    pass

    patch = cfg.get("patch_size", [1, 8, 8])
    p_h = int(patch[1]) if len(patch) > 1 else 8
    p_w = int(patch[2]) if len(patch) > 2 else 8
    h_trim = (h0 // p_h) * p_h
    w_trim = (w0 // p_w) * p_w
    if h_trim <= 0 or w_trim <= 0:
        raise ValueError(
            f"Invalid trimmed spatial size ({h_trim}, {w_trim}) from base ({h0}, {w0}) "
            f"and patch ({p_h}, {p_w})."
        )
    return h_trim, w_trim


def build_ssl_model(cfg):
    """
    Build SSL model from a **flat** config dict.

    Expected flat keys (model_configs/*.yaml)::

        model: mae_2d             # "mae_2d" | "cnn3d"
        backbone: resnet18        # any key in BACKBONE_REGISTRY
        pretrained: false
        channels: 1
        hidden_dim: 256
        normalize_loss: true

    Returns:
        nn.Module (e.g. MAESystem)
    """
    import inspect

    model_type = cfg.get("model", "mae_2d")
    task = cfg.get("task", "mae")
    in_channels = cfg.get("channels", 1)

    # ------------------------------------------------------------------
    # MAE 2D model (single frames)
    # ------------------------------------------------------------------
    if model_type == "mae_2d":
        # --- Backbone ---
        backbone_name = cfg.get("backbone", "resnet18")
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(BACKBONE_REGISTRY.keys())}"
            )

        backbone_cls = BACKBONE_REGISTRY[backbone_name]

        # Only pass `pretrained` to backbones whose __init__ accepts it
        sig = inspect.signature(backbone_cls.__init__)
        encoder_kwargs = {"in_channels": in_channels}
        if "pretrained" in sig.parameters:
            encoder_kwargs["pretrained"] = cfg.get("pretrained", False)

        encoder = backbone_cls(**encoder_kwargs)

        # --- Decoder ---
        decoder = MAEDecoder2D(
            in_channels=encoder.feature_dim,
            out_channels=in_channels,
            hidden_dim=cfg.get("hidden_dim", 256),
        )

        # --- MAESystem config (nested dict expected by MAESystem) ---
        mae_config = {
            "loss": {
                "normalize": cfg.get("normalize_loss", True),
                "crop_loss": cfg.get("crop_loss", None),
                "crop_loss_radius": cfg.get("crop_loss_radius", 30),
                "loss_type": cfg.get("loss_type", "mse"),
                "alpha": cfg.get("alpha", 0.84),
                "ssim_window_size": cfg.get("ssim_window_size", 11),
                "ssim_sigma": cfg.get("ssim_sigma", 1.5),
            },
            "training": {
                "lr": cfg.get("lr", 1e-4),
                "weight_decay": cfg.get("weight_decay", 0.05),
            },
        }

        return MAESystem(encoder=encoder, decoder=decoder, config=mae_config)

    # ------------------------------------------------------------------
    # MAE 3D model (video clips: B, C, T, H, W)
    # ------------------------------------------------------------------
    elif model_type == "mae_3d":
        backbone_name = cfg.get("backbone", "MAER3D18Backbone")
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(BACKBONE_REGISTRY.keys())}"
            )
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        sig = inspect.signature(backbone_cls.__init__)
        encoder_kwargs = {"in_channels": in_channels}
        if "pretrained" in sig.parameters:
            encoder_kwargs["pretrained"] = cfg.get("pretrained", False)
        encoder = backbone_cls(**encoder_kwargs)

        decoder = MAEDecoder3D(
            in_channels=encoder.feature_dim,
            out_channels=in_channels,
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        mae_config = {
            "loss": {
                "normalize": cfg.get("normalize_loss", True),
                "loss_type": cfg.get("loss_type", "mse"),
                "alpha": cfg.get("alpha", 0.84),
                "ssim_window_size": cfg.get("ssim_window_size", 11),
                "ssim_sigma": cfg.get("ssim_sigma", 1.5),
            },
        }
        return MAESystem(encoder=encoder, decoder=decoder, config=mae_config)

    # ------------------------------------------------------------------
    # MAE 2D + LSTM (video clips: 2D backbone per frame + LSTM over time)
    # ------------------------------------------------------------------
    elif model_type == "mae_2d_lstm":
        encoder = Video2DLSTMEncoder(
            pretrained=cfg.get("pretrained", False),
            in_channels=in_channels,
            lstm_hidden=cfg.get("lstm_hidden", 256),
            input_height=cfg.get("input_height", 100),
            input_width=cfg.get("input_width", 100),
        )
        decoder = Video2DLSTMDecoder(
            in_channels=encoder.feature_dim,
            out_channels=in_channels,
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        mae_config = {
            "loss": {
                "normalize": cfg.get("normalize_loss", True),
                "crop_loss": cfg.get("crop_loss", None),
                "crop_loss_radius": cfg.get("crop_loss_radius", 30),
                "loss_type": cfg.get("loss_type", "mse"),
                "alpha": cfg.get("alpha", 0.84),
                "ssim_window_size": cfg.get("ssim_window_size", 11),
                "ssim_sigma": cfg.get("ssim_sigma", 1.5),
            },
            "training": {
                "lr": cfg.get("lr", 1e-4),
                "weight_decay": cfg.get("weight_decay", 0.05),
            },
        }
        return MAESystem(encoder=encoder, decoder=decoder, config=mae_config)

    # ------------------------------------------------------------------
    # Linear MAE 2D baseline (global linear map; ridge via weight_decay, optional L1 on W)
    # ------------------------------------------------------------------
    elif model_type == "linear_mae_2d":
        h, w = _infer_linear_mae_spatial_hw(cfg)
        print(f"[build_ssl_model] linear_mae_2d spatial (patch-trimmed): {h} x {w}")

        mae_config = {
            "loss": {
                "normalize": cfg.get("normalize_loss", True),
                "crop_loss": cfg.get("crop_loss", None),
                "crop_loss_radius": cfg.get("crop_loss_radius", 30),
                "loss_type": cfg.get("loss_type", "mse"),
                "alpha": cfg.get("alpha", 0.84),
                "ssim_window_size": cfg.get("ssim_window_size", 11),
                "ssim_sigma": cfg.get("ssim_sigma", 1.5),
                "linear_l1_penalty": float(cfg.get("linear_l1_penalty", 0.0)),
            },
            "training": {
                "lr": cfg.get("lr", 1e-3),
                "weight_decay": cfg.get("weight_decay", 1e-2),
            },
        }
        return LinearMAESystem(
            num_channels=in_channels,
            height=h,
            width=w,
            config=mae_config,
        )

    # ------------------------------------------------------------------
    # Legacy 3D CNN model
    # ------------------------------------------------------------------
    elif model_type == "cnn3d":
        try:
            from .old_version.cnn3d import Simple3DEncoder
            from .old_version.mae import VideoMAE
        except ImportError:
            raise ImportError(
                "Legacy 3D CNN model requires old_version imports. "
                "Use model='mae_2d' instead."
            )

        embed_dim = cfg.get("embed_dim", 256)
        encoder = Simple3DEncoder(in_channels=in_channels, dim=embed_dim)

        if task == "mae":
            return VideoMAE(
                encoder=encoder,
                patch_size=tuple(cfg.get("patch_size", [4, 8, 8])),
                mask_ratio=float(cfg.get("mask_ratio", 0.5)),
                in_channels=in_channels,
            )
        elif task == "recon":
            return VideoMAE(
                encoder=encoder,
                patch_size=tuple(cfg.get("patch_size", [4, 8, 8])),
                mask_ratio=0.0,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported: "
            f"'mae_2d', 'mae_2d_lstm', 'mae_3d', 'linear_mae_2d', 'cnn3d'"
        )
