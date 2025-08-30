import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.datasets import build_dataset
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed




def main(cfg_path: str):
with open(cfg_path, "r") as f:
cfg = yaml.safe_load(f)


set_seed(cfg.get("seed", 42))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Datasets/Dataloaders
train_ds = build_dataset(cfg, split="train")
val_ds = build_dataset(cfg, split="val")


train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
num_workers=cfg["num_workers"], pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
num_workers=cfg["num_workers"], pin_memory=True)


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
