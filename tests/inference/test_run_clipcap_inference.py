"""Tests for the ClipCap inference entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.clipcap.inference.run_inference as run_inference
from src.config.clipcap_config import (
    CLIPCAP_CHECKPOINT_ROOT,
    CLIPCAP_DEFAULT_INFERENCE_CONFIG,
    ClipCapInferenceConfig,
)


@pytest.fixture
def inference_paths(tmp_path: Path) -> dict[str, Path]:
    split_dir = tmp_path / "splits"
    chunk_dir = split_dir / "val_chunks"
    chunk_dir.mkdir(parents=True)
    (split_dir / "val.json").write_text(
        json.dumps({"a.jpg": ["reference"] * 5}),
        encoding="utf-8",
    )
    (chunk_dir / "val_chunk_001.json").write_text(
        json.dumps(["a.jpg"]),
        encoding="utf-8",
    )
    (split_dir / "fixed_test_round_001.json").write_text(
        json.dumps(["b.jpg"]),
        encoding="utf-8",
    )
    feature_cache = tmp_path / "features.pt"
    feature_cache.write_bytes(b"cache placeholder")
    return {
        "split_dir": split_dir,
        "feature_cache": feature_cache,
        "checkpoint_root": tmp_path / "checkpoints",
        "output_base": tmp_path / "outputs",
        "missing_image_dir": tmp_path / "missing-images",
    }


def _base_arguments(paths: dict[str, Path]) -> list[str]:
    return [
        "--run-tag",
        "baseline_v1",
        "--split-dir",
        str(paths["split_dir"]),
        "--feature-cache",
        str(paths["feature_cache"]),
        "--image-dir",
        str(paths["missing_image_dir"]),
        "--checkpoint-root",
        str(paths["checkpoint_root"]),
        "--output-base",
        str(paths["output_base"]),
    ]


def _value_after(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_inference_defaults_to_dedicated_checkpoint_root():
    args = run_inference._parse_args(
        ["--dataset", "val", "--run-tag", "baseline_v1"]
    )

    assert args.checkpoint_root == CLIPCAP_CHECKPOINT_ROOT


def test_validation_full_resolves_to_val_manifest(
    inference_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[list[str]] = []
    monkeypatch.setattr(run_inference, "inference_main", captured.append)

    run_inference.main(
        ["--dataset", "val", *_base_arguments(inference_paths)]
    )

    arguments = captured[0]
    assert _value_after(arguments, "--manifest") == str(
        inference_paths["split_dir"] / "val.json"
    )
    assert _value_after(arguments, "--output-dir") == str(
        inference_paths["output_base"] / "val" / "baseline_v1"
    )
    assert _value_after(arguments, "--image-batch-size") == str(
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.image_batch_size
    )
    assert _value_after(arguments, "--max-new-tokens") == str(
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.max_new_tokens
    )
    assert _value_after(arguments, "--num-beams") == str(
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_beams
    )
    assert _value_after(arguments, "--num-return-sequences") == str(
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_return_sequences
    )
    assert _value_after(arguments, "--length-penalty") == str(
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.length_penalty
    )
    assert "--early-stopping" in arguments


def test_validation_chunk_gets_an_isolated_run_tag(
    inference_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[list[str]] = []
    monkeypatch.setattr(run_inference, "inference_main", captured.append)

    run_inference.main(
        [
            "--dataset",
            "val",
            "--validation-chunk",
            "1",
            *_base_arguments(inference_paths),
        ]
    )

    arguments = captured[0]
    assert _value_after(arguments, "--manifest").endswith(
        "val_chunks/val_chunk_001.json"
    ) or _value_after(arguments, "--manifest").endswith(
        "val_chunks\\val_chunk_001.json"
    )
    assert _value_after(arguments, "--output-dir") == str(
        inference_paths["output_base"]
        / "val"
        / "baseline_v1_val_chunk_001"
    )


def test_final_test_requires_explicit_unlock(inference_paths: dict[str, Path]):
    with pytest.raises(RuntimeError, match="--allow-test"):
        run_inference.main(
            [
                "--dataset",
                "fixed_test_round_001",
                *_base_arguments(inference_paths),
            ]
        )


def test_final_test_uses_only_the_fixed_id_manifest(
    inference_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[list[str]] = []
    monkeypatch.setattr(run_inference, "inference_main", captured.append)

    run_inference.main(
        [
            "--dataset",
            "fixed_test_round_001",
            "--allow-test",
            *_base_arguments(inference_paths),
        ]
    )

    arguments = captured[0]
    assert _value_after(arguments, "--manifest") == str(
        inference_paths["split_dir"] / "fixed_test_round_001.json"
    )
    assert _value_after(arguments, "--output-dir") == str(
        inference_paths["output_base"] / "test" / "baseline_v1"
    )
    assert "--image-dir" not in arguments


def test_inference_cli_overrides_central_defaults(
    inference_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[list[str]] = []
    monkeypatch.setattr(run_inference, "inference_main", captured.append)

    run_inference.main(
        [
            "--dataset",
            "val",
            "--image-batch-size",
            "8",
            "--max-new-tokens",
            "20",
            "--num-beams",
            "6",
            "--num-return-sequences",
            "4",
            "--length-penalty",
            "0.8",
            "--no-early-stopping",
            *_base_arguments(inference_paths),
        ]
    )

    arguments = captured[0]
    assert _value_after(arguments, "--image-batch-size") == "8"
    assert _value_after(arguments, "--max-new-tokens") == "20"
    assert _value_after(arguments, "--num-beams") == "6"
    assert _value_after(arguments, "--num-return-sequences") == "4"
    assert _value_after(arguments, "--length-penalty") == "0.8"
    assert "--no-early-stopping" in arguments


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_beams": 1}, "num_beams"),
        ({"num_beams": 3, "num_return_sequences": 4}, "num_return_sequences"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"length_penalty": 0.0}, "length_penalty"),
        ({"length_penalty": float("nan")}, "length_penalty"),
    ],
)
def test_inference_config_rejects_invalid_values(overrides, message):
    values = CLIPCAP_DEFAULT_INFERENCE_CONFIG.to_dict()
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        ClipCapInferenceConfig(**values)
