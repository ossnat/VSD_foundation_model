import yaml
from src.data.datasets import build_dataset
from src.models import build_ssl_model


def test_build_and_forward():
    cfg = yaml.safe_load(open("configs/default.yaml","r"))
    ds = build_dataset(cfg, split="train")
    sample = ds[0]
    vid = sample["video"]  # (1,T,H,W)
    vid = vid.unsqueeze(0)  # (B=1,C,T,H,W)
    model = build_ssl_model(cfg)
    out = model(vid)
    assert "loss" in out and out["recon_patches"].shape == out["target_patches"].shape
