"""Layout: workspace/Data is sibling of workspace/VSD_foundation_model."""

from pathlib import Path

from src.utils.data_paths import data_dir, resolve_data_path, workspace_root


def test_sibling_data_layout():
    project_root = Path(__file__).resolve().parent.parent
    ws = workspace_root(project_root)
    assert (ws / "VSD_foundation_model").resolve() == project_root.resolve()
    assert data_dir(project_root) == (ws / "Data").resolve()

    cfg_path = "Data/FoundationData/ProcessedData/splits/foo.csv"
    resolved = resolve_data_path(project_root, cfg_path)
    assert resolved == (ws / "Data" / "FoundationData" / "ProcessedData" / "splits" / "foo.csv").resolve()
    assert str(resolved).startswith(str(ws))
