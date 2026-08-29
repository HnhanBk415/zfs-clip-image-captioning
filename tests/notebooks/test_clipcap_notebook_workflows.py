"""Tests for end-to-end ClipCap notebook workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CLIPCAP_NOTEBOOK_DIRECTORY = Path("notebook/clipcap")
GENERATION_NOTEBOOK = (
    CLIPCAP_NOTEBOOK_DIRECTORY / "inference" / "clipcap_caption_generation.ipynb"
)
EVALUATION_NOTEBOOK = Path("notebook/evaluation/clipcap_evaluation.ipynb")
END_TO_END_NOTEBOOK = (
    CLIPCAP_NOTEBOOK_DIRECTORY
    / "inference"
    / "clipcap_validation_end_to_end_colab.ipynb"
)
FIXED_TEST_END_TO_END_NOTEBOOK = (
    CLIPCAP_NOTEBOOK_DIRECTORY
    / "inference"
    / "clipcap_fixed_test_end_to_end_colab.ipynb"
)
FINAL_TEST_MANIFEST = Path("data/flickr8k/splits/fixed_test_round_001.json")
VALIDATION_CHUNK_DIRECTORY = Path("data/flickr8k/splits/val_chunks")


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_source(path: Path) -> str:
    notebook = _load_notebook(path)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize(
    "path",
    [
        GENERATION_NOTEBOOK,
        EVALUATION_NOTEBOOK,
        END_TO_END_NOTEBOOK,
        FIXED_TEST_END_TO_END_NOTEBOOK,
    ],
)
def test_notebook_code_cells_compile_and_are_clean(path: Path):
    notebook = _load_notebook(path)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_generation_notebook_selects_split_and_only_runs_inference():
    source = _code_source(GENERATION_NOTEBOOK)

    assert "CLIPCAP_DEFAULT_INFERENCE_CONFIG" in source
    assert "CLIPCAP_CHECKPOINT_ROOT" in source
    assert "ClipCapInferenceConfig(" in source
    assert "ZFS_CLIP_SPLIT_NAME" in source
    assert "ZFS_CLIP_ALLOW_TEST" in source
    assert "ZFS_CLIP_INFERENCE_MANIFEST_PATH" in source
    assert "isinstance(manifest, list)" in source
    assert "SPLIT_NAME not in {'val', 'test'}" in source
    assert "ALLOW_TEST_INFERENCE" in source
    assert "subprocess.run(inference_command" in source
    assert "run_config.json" in source
    assert "predictions.jsonl" in source
    assert "'checkpoint': 'final.pt'" in source
    assert "'training_policy': 'fixed_epoch'" in source
    assert "record.get('checkpoint') != 'final.pt'" in source
    for option in (
        "--manifest",
        "--feature-cache",
        "--checkpoint-root",
        "--max-new-tokens",
        "--num-beams",
        "--num-return-sequences",
        "--length-penalty",
    ):
        assert option in source


def test_evaluation_notebook_consumes_predictions_and_writes_metrics():
    source = _code_source(EVALUATION_NOTEBOOK)

    assert "ZFS_CLIP_SPLIT_NAME" in source
    assert "ZFS_CLIP_ALLOW_TEST" in source
    assert "ZFS_CLIP_INFERENCE_MANIFEST_PATH" in source
    assert "ZFS_CLIP_REFERENCES_PATH" in source
    assert "load_manifest_ids" in source
    assert "all_references[image_id] for image_id in inference_ids" in source
    assert "sha256_file(INFERENCE_MANIFEST_PATH)" in source
    assert "run_config.json" in source
    assert "predictions.jsonl" in source
    assert "evaluate_coco_metrics" in source
    assert "evaluate_clipscore_from_cache" in source
    assert "from src.common.caption_metrics import evaluate_clipscore_from_cache" in source
    assert "RefCLIPScore" in source
    assert "A photo depicts" in source
    assert "summary.json" in source
    assert "per_image_scores.csv" in source
    assert "subprocess.run(inference_command" not in source


def test_colab_notebook_runs_validation_end_to_end_for_all_subsets():
    source = _code_source(END_TO_END_NOTEBOOK)

    assert "google.colab" in source
    assert "https://github.com/HnhanBk415/zfs-clip-image-captioning.git" in source
    assert "BRANCH = 'refactor/huuthien/Model'" in source
    assert "PROJECT_ROOT = Path('/content/zfs-clip-image-captioning')" in source
    assert "DATA_CACHE_DIR = DRIVE_ROOT / 'data_cache'" in source
    assert "DATA_CACHE_DIR / 'features' / 'clip_features.pt'" in source
    assert "CHECKPOINT_ROOT = DRIVE_ROOT / 'experiments_fixed_epoch'" in source
    assert "ZFS_CLIP_SPLIT_NAME': 'val'" in source
    assert "ZFS_CLIP_ALLOW_TEST': '0'" in source
    assert "ZFS_CLIP_INFERENCE_MANIFEST_PATH" in source
    assert "ZFS_CLIP_REFERENCES_PATH" in source
    assert "SPLIT_DIR / f'{SPLIT_NAME}.json'" in _code_source(GENERATION_NOTEBOOK)
    assert "VALIDATION_CHUNK: int | None = 1" in source
    assert "CLIPCAP_DEFAULT_INFERENCE_CONFIG" in source
    assert "SUBSETS = CLIPCAP_TRAIN_SUBSETS" in source
    assert "1 <= VALIDATION_CHUNK <= 9" in source
    assert "MANIFEST_PATH = FULL_VALIDATION_PATH" in source
    assert "REFERENCES_PATH = FULL_VALIDATION_PATH" in source
    assert "EFFECTIVE_RUN_TAG" in source
    assert "else f'{RUN_TAG}_val_chunk_{VALIDATION_CHUNK:03d}'" in source
    assert "'ZFS_CLIP_REFERENCES_PATH': str(REFERENCES_PATH)" in source
    assert "clipcap_caption_generation.ipynb" in source
    assert "clipcap_evaluation.ipynb" in source
    assert "summary.json" in source


def test_colab_notebook_runs_fixed_test_and_exports_report_artifacts():
    source = _code_source(FIXED_TEST_END_TO_END_NOTEBOOK)

    assert "google.colab" in source
    assert "https://github.com/HnhanBk415/zfs-clip-image-captioning.git" in source
    assert "BRANCH = 'refactor/huuthien/evaluation'" in source
    assert "PROJECT_ROOT = Path('/content/zfs-clip-image-captioning')" in source
    assert "DATA_CACHE_PATH" not in source
    assert "data_cache' / 'features' / 'clip_features.pt" in source
    assert "DRIVE_ROOT / 'experiments_fixed_epoch'" in source
    assert "fixed_test_round_001.json" in source
    assert "REFERENCES_PATH = SPLIT_DIR / 'test.json'" in source
    assert "len(fixed_test_ids) != 104" in source
    assert "EXPECTED_SUBSETS" in source
    for subset_name in (
        "train_1pct",
        "train_5pct",
        "train_10pct",
        "train_25pct",
        "train_100pct",
    ):
        assert subset_name in source
    assert "validate_artifact_directory" in source
    assert "load_feature_cache" in source
    assert "src.clipcap.inference.run_inference" in source
    assert "'--allow-test'" in source
    assert "'--no-resume'" not in source
    assert "src.common.caption_metrics" in source
    assert "captions.json" in source
    assert "results_by_experiment" in source
    assert "report_metric_artifact" in source
    assert "REPORT_METRICS_DIR / f'{subset_name}.json'" in source
    assert "prediction_paths[subset_name].parent / 'metrics.json'" not in source
    assert "comparison.json" in source
    assert "per_image_comparison.csv" in source
    assert "results_table.csv" in source
    assert "'train_1pct': 1" in source
    assert "'train_5pct': 5" in source
    assert "'train_10pct': 10" in source
    assert "'train_25pct': 25" in source
    assert "'train_100pct': 100" in source
    assert "results_table['training_data_pct']" in source
    assert "sort_values(\n    'training_data_pct'" in source
    assert "RefCLIPScore" in source
    assert "shutil.make_archive" in source


def test_final_test_round_is_a_clean_subset_of_canonical_test():
    selected_ids = json.loads(FINAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    split_directory = Path("data/flickr8k/splits")
    train = json.loads((split_directory / "train.json").read_text(encoding="utf-8"))
    validation = json.loads((split_directory / "val.json").read_text(encoding="utf-8"))
    test = json.loads((split_directory / "test.json").read_text(encoding="utf-8"))

    assert isinstance(selected_ids, list)
    assert len(selected_ids) == 104
    assert len(selected_ids) == len(set(selected_ids))
    assert set(selected_ids).isdisjoint(train)
    assert set(selected_ids).isdisjoint(validation)
    assert set(selected_ids).issubset(test)
    assert all(len(test[image_id]) == 5 for image_id in selected_ids)


def test_validation_chunks_partition_the_full_validation_split():
    validation = json.loads(
        Path("data/flickr8k/splits/val.json").read_text(encoding="utf-8")
    )
    chunk_paths = sorted(VALIDATION_CHUNK_DIRECTORY.glob("val_chunk_*.json"))
    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in chunk_paths]
    flattened = [image_id for chunk in chunks for image_id in chunk]

    assert [path.name for path in chunk_paths] == [
        f"val_chunk_{index:03d}.json" for index in range(1, 10)
    ]
    assert [len(chunk) for chunk in chunks] == [100] * 8 + [7]
    assert len(flattened) == len(set(flattened)) == 807
    assert flattened == list(validation)
    assert all(len(validation[image_id]) == 5 for image_id in flattened)
