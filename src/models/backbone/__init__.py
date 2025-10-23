"""
Backbone modules for various architectures.
"""

# from .r3d_18_wrapper import R3D18Backbone
from .resnet_wrapper import ResNetWrapper
from .vit_wrapper import ViTWrapper

__all__ = [
    'R3D18Backbone',
    'ResNetWrapper', 
    'ViTWrapper'
]

