import os
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.data import *
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed




def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve any relative data paths based on the directory that contains the project
    # (parent.parent.parent from the config file, since Data/ is a sibling of the project root)
    base_dir = cfg_path.resolve().parent.parent.parent
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if value is None:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            full_path = (base_dir / value_path).resolve()
            cfg[key] = str(full_path)


    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Datasets/Dataloaders
    train_loader = load_dataset(cfg, split="train", 
                               batch_size=cfg["batch_size"], 
                               num_workers=cfg["num_workers"], 
                               shuffle=True)
    
    val_loader = load_dataset(cfg, split="val", 
                             batch_size=cfg["batch_size"], 
                             num_workers=cfg["num_workers"], 
                             shuffle=False)


    # Model (SSL task wrapper e.g., MAE)
    model = build_ssl_model(cfg).to(device)


    # Logger & Trainer
    logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)


    # Train
    trainer.fit(train_loader, val_loader)


    # Save final encoder checkpoint
    os.makedirs(cfg.get("ckpt_dir", "checkpoints"), exist_ok=True)
    enc_path = os.path.join(cfg.get("ckpt_dir", "checkpoints"), "encoder_final.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}")

    # Optional: evaluate reconstruction quality over time on the test set (MSE per clip start frame).
    # Uncomment the block below to run temporal evaluation after training. This also saves a plot
    # and a JSON file to results_dir/temporal_eval/ (or ckpt_dir/temporal_eval/ if results_dir not set).
    # test_loader = load_dataset(cfg, split="test", batch_size=cfg.get("batch_size", 256), num_workers=cfg.get("num_workers", 4), shuffle=False)
    # temporal_metrics = trainer.evaluate_metrics_over_time(test_loader, split_name="test")
    # # Optional: pass save_dir="path/to/results" to control where the plot and JSON are saved.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
