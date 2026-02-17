# ==================================
# File: src/utils/logger.py
# ==================================

import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.w = SummaryWriter(log_dir)
    def log_scalar(self, name, value, step):
        self.w.add_scalar(name, value, step)

    def flush(self):
        self.w.flush()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
