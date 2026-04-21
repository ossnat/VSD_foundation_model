#!/usr/bin/env python3
"""
Interactive HDF5 frame viewer.

Usage (from repo root):
  PYTHONPATH=. python scripts/h5_viewer.py
  PYTHONPATH=. python scripts/h5_viewer.py --folder /path/to/dir --file myfile.h5 --dataset my/dataset --start 0 --end 5 --save

All logic lives in viewer.h5_viewer (top-level package, not under src/).
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viewer.h5_viewer import main  # noqa: E402

if __name__ == "__main__":
    main()
