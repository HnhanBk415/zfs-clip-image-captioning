"""Tests for shared caption evaluation metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import src.common.caption_metrics as caption_metrics


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metric_dependency_is_declared():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "pycocoevalcap" in requirements


def test_common_loaders_support_blind_manifest_and_prediction_formats(
    tmp_path: Path,
):
    manifest_path = tmp_path / "ids.json"
    references_path = tmp_path / "references.json"
    jsonl_path = tmp_path / "clipcap.jsonl"
    json_path = tmp_path / "zerocap.json"
    _write_json(manifest_path, ["a.jpg", "b.jpg"])
    _write_json(
        references_path,
        {
            "a.jpg": ["a one", "a two", "a three", "a four", "a five"],
            "b.jpg": ["b one", "b two", "b three", "b four", "b five"],
        },
    )
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "a.jpg", "caption": "prediction a"}),
                json.dumps({"image_id": "b.jpg", "caption": "prediction b"}),
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        json_path,
        {"a.jpg": "prediction a", "b.jpg": {"caption": "prediction b"}},
    )

    assert caption_metrics.load_manifest_ids(manifest_path) == ["a.jpg", "b.jpg"]
    assert len(caption_metrics.load_references(references_path)["a.jpg"]) == 5
    assert caption_metrics.load_predictions(jsonl_path) == {
        "a.jpg": "prediction a",
        "b.jpg": "prediction b",
    }
    assert caption_metrics.load_predictions(json_path) == {
        "a.jpg": "prediction a",
        "b.jpg": "prediction b",
    }


def test_prediction_coverage_rejects_missing_or_extra_images():
    with pytest.raises(ValueError, match="missing=1, extra=1"):
        caption_metrics.validate_prediction_coverage(
            ["a.jpg", "b.jpg"],
            {"a.jpg": "caption", "c.jpg": "caption"},
            "experiment",
        )


def test_run_caption_evaluation_writes_common_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path = tmp_path / "ids.json"
    references_path = tmp_path / "references.json"
    prediction_path = tmp_path / "predictions.jsonl"
    run_config_path = tmp_path / "run_config.json"
    feature_cache_path = tmp_path / "features.pt"
    output_dir = tmp_path / "metrics"
    image_ids = ["a.jpg", "b.jpg"]
    _write_json(manifest_path, image_ids)
    _write_json(
        references_path,
        {
            image_id: [f"{image_id} reference {index}" for index in range(5)]
            for image_id in image_ids
        },
    )
    prediction_path.write_text(
        "\n".join(
            json.dumps({"image_id": image_id, "caption": f"caption {image_id}"})
            for image_id in image_ids
        ),
        encoding="utf-8",
    )
    _write_json(
        run_config_path,
        {
            "manifest_sha256": caption_metrics.sha256_file(manifest_path),
            "num_images": 2,
            "prompt": None,
        },
    )

    def fake_coco(references, predictions):
        assert set(references) == set(predictions) == set(image_ids)
        return {
            "CIDEr": 50.0,
            "BLEU-4": 25.0,
            "per_image": {
                image_id: {"CIDEr": 50.0, "BLEU-4": 25.0}
                for image_id in image_ids
            },
        }

    def fake_clipscore(
        selected_ids,
        predictions_by_experiment,
        feature_cache_path,
        model_name,
        batch_size,
        device_name,
    ):
        del feature_cache_path, model_name, batch_size, device_name
        assert selected_ids == image_ids
        return {
            label: {
                "CLIPScore": 30.0,
                "per_image": {image_id: 30.0 for image_id in image_ids},
            }
            for label in predictions_by_experiment
        }

    monkeypatch.setattr(caption_metrics, "evaluate_coco_metrics", fake_coco)
    monkeypatch.setattr(
        caption_metrics,
        "evaluate_clipscore_from_cache",
        fake_clipscore,
    )

    artifacts = caption_metrics.run_caption_evaluation(
        inference_manifest_path=manifest_path,
        references_path=references_path,
        prediction_paths={"clipcap": prediction_path},
        feature_cache_path=feature_cache_path,
        output_dir=output_dir,
        run_config_path=run_config_path,
        expected_run_config={"prompt": None},
        artifact_metadata={"split": "val"},
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    with artifacts.per_image_path.open("r", encoding="utf-8", newline="") as file:
        per_image_rows = list(csv.DictReader(file))
    assert summary["metadata"] == {"split": "val"}
    assert summary["coverage"]["clipcap"]["num_predictions"] == 2
    assert summary["results"][0]["CIDEr"] == 50.0
    assert summary["results"][0]["BLEU-4"] == 25.0
    assert summary["results"][0]["CLIPScore"] == 30.0
    assert len(per_image_rows) == 2
