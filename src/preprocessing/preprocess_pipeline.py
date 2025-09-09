# preprocessing/pipeline.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class PreprocessingStep(ABC):
    """Base class for all preprocessing operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return step parameters for logging/reproducibility"""
        pass


class PreprocessingPipeline:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.steps = self._build_pipeline()

    def _build_pipeline(self) -> List[PreprocessingStep]:
        steps = []
        for step_config in self.config['pipeline']['steps']:
            step_class = self._get_step_class(step_config['name'])
            steps.append(step_class(step_config['params']))
        return steps

    def process_offline(self, data: np.ndarray) -> np.ndarray:
        """Apply offline preprocessing steps"""
        result = data
        for step in self.steps:
            if step.config.get('offline', False):
                result = step.process(result)
        return result

    def process_online(self, data: np.ndarray) -> np.ndarray:
        """Apply online preprocessing steps"""
        result = data
        for step in self.steps:
            if not step.config.get('offline', True):  # Default to online
                result = step.process(result)
        return result

