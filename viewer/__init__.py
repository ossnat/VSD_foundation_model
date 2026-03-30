"""
HDF5 frame viewer (standalone tool, not part of core `src/` package).
"""

from .h5_viewer import interactive_viewer, main

__all__ = ["interactive_viewer", "main"]
