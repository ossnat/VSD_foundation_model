from abc import ABC, abstractmethod

class BaseSystem(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def forward(self, batch):
        pass

    def update_teacher(self):
        pass  # Optional, overridden by some child classes
