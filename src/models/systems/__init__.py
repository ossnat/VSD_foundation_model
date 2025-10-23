"""
System modules for self-supervised learning frameworks.
"""

from .base_system import BaseSystem
from .dino_system import DINOSystem


__all__ = [
    'BaseSystem',
    'DINOSystem',
    'MAESystem'
]

