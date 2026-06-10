#!/usr/bin/env python3
"""
Project sanity check — run on local machine, Colab, or remote Unix.

  python scripts/sanity_check.py              # env + unit tests (no data files)
  python scripts/sanity_check.py --with-data  # also run integration tests needing H5

Exits 0 if all selected checks pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_env_check(device: str) -> int:
    print("=== Environment check ===\n")
    cmd = [sys.executable, str(_ROOT / "scripts" / "check_env.py"), "--device", device]
    return subprocess.call(cmd, cwd=str(_ROOT))


def _run_pytest(with_data: bool) -> int:
    print("\n=== Pytest ===\n")
    tests = [
        "tests/test_data_paths.py",
        "tests/test_lr_schedule.py",
        "tests/test_smoke.py",
        "tests/test_models_unitest.py",
    ]
    if with_data:
        tests.extend(
            [
                "tests/test_model_integration.py",
                "tests/test_training_sanity.py",
            ]
        )
    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "--tb=short"]
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(_ROOT)}
    return subprocess.call(cmd, cwd=str(_ROOT), env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="VSD foundation model sanity check")
    p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for check_env forward pass",
    )
    p.add_argument(
        "--with-data",
        action="store_true",
        help="Include tests that require local H5 / split CSV data",
    )
    p.add_argument("--skip-env", action="store_true", help="Skip check_env.py")
    p.add_argument("--skip-pytest", action="store_true", help="Skip pytest")
    args = p.parse_args()

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    rc = 0
    if not args.skip_env:
        rc = _run_env_check(args.device)
        if rc != 0:
            sys.exit(rc)

    if not args.skip_pytest:
        rc = _run_pytest(args.with_data)
        if rc != 0:
            sys.exit(rc)

    print("\n=== Sanity check passed ===")


if __name__ == "__main__":
    main()
