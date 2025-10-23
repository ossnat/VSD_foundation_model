"""
Head modules for various tasks.
"""

from .dino_head import DINOHead, MultiCropWrapper, DINOProjectionHead
# from .mae_decoder import MAEDecoder, DecoderBlock

__all__ = [
    'DINOHead',
    'MultiCropWrapper',
    'DINOProjectionHead',
    'MAEDecoder',
    'DecoderBlock'
]

