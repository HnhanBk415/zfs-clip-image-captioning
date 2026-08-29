"""Tests for the ZeroCap fixed-test evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebook/zerocap/zerocap_fixed_test_evaluation_colab.ipynb")


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source() -> str:
    notebook = _load_notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_zerocap_evaluation_notebook_code_cells_compile_and_are_clean():
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_zerocap_evaluation_notebook_exports_clipcap_shaped_report():
    source = _code_source()

    assert "google.colab" in source
    assert "BRANCH = 'refactor/huuthien/evaluation'" in source
    assert "RUN_TAG = 'zerocap_fixed_test_round_001_report'" in source
    assert "DRIVE_ROOT / 'zerocap' / 'predictions.json'" in source
    assert "PROJECT_ROOT / 'outputs' / 'zerocap' / 'predictions.json'" in source
    assert "fixed_test_round_001.json" in source
    assert "REFERENCES_PATH = SPLIT_DIR / 'test.json'" in source
    assert "len(prediction_records) != 104" in source
    assert "load_feature_cache" in source
    assert "src.common.caption_metrics" in source
    assert "src.zerocap" not in source
    assert "REPORT_OUTPUT_BASE / 'fixed_test_round_001' / RUN_TAG" in source
    assert "REPORT_CAPTIONS_DIR" in source
    assert "REPORT_METRICS_DIR" in source
    assert "REPORT_PREDICTIONS_DIR" in source
    assert "comparison.json" in source
    assert "per_image_comparison.csv" in source
    assert "results_table.csv" in source
    assert "run_config.json" in source
    assert "metric_artifact" in source
    for metric in ("CIDEr", "BLEU-4", "CLIPScore", "RefCLIPScore"):
        assert metric in source
    assert "shutil.make_archive" in source
