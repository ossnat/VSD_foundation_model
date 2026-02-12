# Legacy imports - commented out to avoid import errors
# from .cnn3d import Simple3DEncoder
# from .mae import VideoMAE

# MAE 2D imports
from .backbone.mae_backbone_2d import MAEResNet18Backbone
from .backbone.mae_backbone_2d_CNN import MAEShallowCNNBackbone
from .heads.mae_decoder_2d import MAEDecoder2D
from .systems.mae_system import MAESystem

# ---------------------------------------------------------------------------
# Backbone registry — maps config name -> class.
# Every backbone must expose a `feature_dim` attribute.
# To add a new backbone: import it above and add a key here.
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    "resnet18": MAEResNet18Backbone,
    "MAEShallowCNNBackbone": MAEShallowCNNBackbone,
}


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
            },
        }

        return MAESystem(encoder=encoder, decoder=decoder, config=mae_config)

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
            f"Unknown model type: {model_type}. Supported: 'mae_2d', 'cnn3d'"
        )
