import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG_IMPORT = re.compile(
    r"(?m)^\s*(?:from\s+config\s+import|import\s+config(?:\s|$))"
)


def _notebook_source(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_colab_training_uses_central_clipcap_configuration():
    source = _notebook_source(
        "notebook/clipcap/clipcap_training_colab.ipynb"
    )

    assert "clipcap_training_config" not in source
    assert "from src.config.clipcap_config import" in source
    assert "SUBSETS_TO_TRAIN = CLIPCAP_TRAIN_SUBSETS" in source
    assert "BASE_CONFIG = create_clipcap_fixed_epoch_config(" in source
    assert "output_root=CHECKPOINT_ROOT" in source
    assert 'CHECKPOINT_ROOT = DRIVE_ROOT / "experiments_fixed_epoch"' in source
    assert 'result["official_checkpoint"]' in source
    assert 'result["final_epoch"] == BASE_CONFIG.max_epochs' in source

    copied_assignments = (
        "batch_size=32",
        "learning_rate=2e-5",
        "weight_decay=0.01",
        "max_epochs=50",
        "early_stopping_patience=5",
        "max_grad_norm=1.0",
        "clip_length=10",
        "prefix_length=10",
        "num_layers=4",
        "num_heads=8",
        "dropout=0.1",
        "max_epochs=7",
        "training_policy=\"fixed_epoch\"",
    )
    for assignment in copied_assignments:
        assert assignment not in source


def test_production_notebooks_do_not_copy_clipcap_hyperparameters():
    notebook_paths = (
        "notebook/preprocessing/clipcap_and_loader.ipynb",
        "notebook/mapping_network/clip_projection.ipynb",
        "notebook/mapping_network/prefix_encoder.ipynb",
        "notebook/mapping_network/Transformer_Mapper.ipynb",
    )
    copied_assignment = re.compile(
        r"(?:batch_size|clip_length|prefix_length|num_layers|num_heads|nhead)"
        r"\s*(?::\s*int\s*)?=\s*(?:32|10|8|4)(?:\s|,|$)"
    )

    for relative_path in notebook_paths:
        source = _notebook_source(relative_path)
        assert copied_assignment.search(source) is None, relative_path


def test_non_zerocap_notebooks_do_not_depend_on_deleted_config_module():
    notebook_root = PROJECT_ROOT / "notebook"
    checked_notebooks = 0

    for path in notebook_root.rglob("*.ipynb"):
        if "zerocap" in {part.lower() for part in path.parts}:
            continue
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        source = _notebook_source(relative_path)
        assert LEGACY_CONFIG_IMPORT.search(source) is None, relative_path
        assert '(candidate / "config.py")' not in source, relative_path
        checked_notebooks += 1

    assert checked_notebooks > 0


def test_deleted_config_file_is_not_referenced_outside_zerocap():
    assert not (PROJECT_ROOT / "config.py").exists()

    legacy_reference = re.compile(r"(?<![A-Za-z0-9_])config\.py")
    for path in (PROJECT_ROOT / "docs").rglob("*.md"):
        if "zerocap" in path.name.lower():
            continue
        content = path.read_text(encoding="utf-8")
        assert legacy_reference.search(content) is None, path.name


def test_removed_project_root_dependency_is_not_required():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "pyprojroot" not in requirements.splitlines()
