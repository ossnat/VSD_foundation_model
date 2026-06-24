"""Tests for H5-metadata index rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from data_pp_and_split.src.index_build import rebuild_index_dataframe, rows_from_h5, validate_index


def _write_demo_h5(path: Path) -> None:
    """Two conditions, three physical trials — metadata order matches H5 keys."""
    meta = [
        {
            "trial_global_id": 100,
            "monkey": "demo",
            "date": "010101a",
            "condition": "condAN1",
            "source_file": "/fake/condsAN.mat",
            "target_file": str(path),
            "trial_index_in_condition": 0,
            "shape": [100, 10],
        },
        {
            "trial_global_id": 101,
            "monkey": "demo",
            "date": "010101a",
            "condition": "condAN1",
            "source_file": "/fake/condsAN.mat",
            "target_file": str(path),
            "trial_index_in_condition": 1,
            "shape": [100, 10],
        },
        {
            "trial_global_id": 102,
            "monkey": "demo",
            "date": "010101a",
            "condition": "condAN2",
            "source_file": "/fake/condsAN.mat",
            "target_file": str(path),
            "trial_index_in_condition": 0,
            "shape": [100, 10],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i in range(3):
            f.create_dataset(f"trial_{i:06d}", data=np.zeros((100, 10), dtype=np.float32))
        f.attrs["trial_metadata_json"] = json.dumps(meta)


def test_rows_from_h5_maps_sequential_trials(tmp_path: Path) -> None:
    h5 = tmp_path / "session_010101a_condsAN.h5"
    _write_demo_h5(h5)
    rows = rows_from_h5(h5)
    assert len(rows) == 3
    assert rows[2]["trial_dataset"] == "trial_000002"
    assert rows[2]["condition"] == "condAN2"
    assert rows[2]["trial_index_in_condition"] == 0


def test_validate_index_rejects_duplicate_physical_trial() -> None:
    df = pd.DataFrame(
        [
            {
                "trial_global_id": 1,
                "monkey": "m",
                "date": "d",
                "condition": "condAN1",
                "target_file": "Data/x.h5",
                "trial_dataset": "trial_000000",
                "shape": "(100, 10)",
                "trial_index_in_condition": 0,
            },
            {
                "trial_global_id": 2,
                "monkey": "m",
                "date": "d",
                "condition": "condAN2",
                "target_file": "Data/x.h5",
                "trial_dataset": "trial_000000",
                "shape": "(100, 10)",
                "trial_index_in_condition": 0,
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate physical trials"):
        validate_index(df)


def test_rebuild_index_dataframe(tmp_path: Path) -> None:
    h5 = tmp_path / "session_010101a_condsAN.h5"
    _write_demo_h5(h5)
    df = rebuild_index_dataframe([str(h5)])
    assert len(df) == 3
    validate_index(df)
