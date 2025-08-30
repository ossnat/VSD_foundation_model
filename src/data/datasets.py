import math
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Literal




class DummyVideoDataset(Dataset):
"""Generates synthetic grayscale video clips for smoke testing.
Produces simple moving Gaussian blobs + noise.
Returns dict with keys: video (B,C,T,H,W), mask (T,H,W) for MAE.
"""
def __init__(self, num_videos: int, frames: int, size: Tuple[int, int], split: Literal["train","val"],
    channels: int = 1, seed: int = 123):
    self.num_videos = num_videos if split == "train" else max(16, num_videos // 10)
    self.frames = frames
    self.h, self.w = size
    self.channels = channels
    self.rng = np.random.RandomState(seed + (0 if split == "train" else 1))


def __len__(self):
    return self.num_videos


def __getitem__(self, idx):
    T, H, W = self.frames, self.h, self.w
    vid = np.zeros((T, H, W), dtype=np.float32)
    # random walk center
    x, y = self.rng.randint(W//4, 3*W//4), self.rng.randint(H//4, 3*H//4)
    vx, vy = self.rng.randn()*0.8, self.rng.randn()*0.8
    sigma = self.rng.uniform(2.0, 4.0)
    for t in range(T):
        x = (x + vx)
        y = (y + vy)
        x = max(0, min(W-1, x))
        y = max(0, min(H-1, y))
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        blob = np.exp(-(((xx - x)**2 + (yy - y)**2) / (2*sigma**2)))
        vid[t] = blob
    # normalize and add noise
    vid = (vid - vid.mean()) / (vid.std() + 1e-6)
    vid += self.rng.randn(T, H, W).astype(np.float32) * 0.1
    vid = vid[None, ...] # C=1
    sample = {"video": torch.from_numpy(vid), "mask": None}