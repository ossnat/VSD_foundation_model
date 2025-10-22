from typing import Dict, Any

from .datasets import VsdVideoDataset
from .vsd_multi_crop_dataset import VsdMultiCropDataset
from .vsd_masked_dataset import VsdMaskedDataset

# Registry of available dataset classes, keyed by string ID
DATASET_REGISTRY = {
    "vsd_video": VsdVideoDataset,
    "vsd_dino": VsdMultiCropDataset,
    "vsd_mae": VsdMaskedDataset, 
}

def create_dataset(dataset_name: str, **kwargs) -> object:
    """
    Factory function to instantiate datasets.
    Args:
        dataset_name (str): Key in DATASET_REGISTRY
        kwargs: Arguments to pass to dataset constructor
    Returns:
        Dataset instance (inherits torch.utils.data.Dataset)
    Raises:
        ValueError: If dataset_name not registered
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not in registry. Available: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_instance = dataset_cls(**kwargs)
    return dataset_instance
