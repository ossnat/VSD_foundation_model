# src/models/systems/base_system.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSystem(nn.Module, ABC):
    """
    Abstract base class for all training systems.
    Defines common interface for forward pass and teacher updates.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, batch):
        """
        Forward pass that processes a batch and returns loss and metrics.
        
        Args:
            batch: Input batch (dict or tensor)
        
        Returns:
            Dict with at least:
                - "loss": Scalar loss tensor
                - "metrics": Dict of metric values
        """
        pass
    
    def update_teacher(self):
        """
        Optional method to update teacher network (e.g., EMA update).
        Default is no-op for systems without teacher networks.
        """
        pass
