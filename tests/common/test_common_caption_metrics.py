"""Tests for shared caption evaluation metrics."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

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


def test_coco_metrics_uses_checked_tokenizer_and_standard_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_tokenize(captions):
        calls.append(captions)
        return captions

    class FakeBleu:
        def __init__(self, order):
            assert order == 4

        def compute_score(self, ground_truth, results, verbose):
            del ground_truth, results
            assert verbose == 0
            return [0.1, 0.2, 0.3, 0.4], [[0.1], [0.2], [0.3], [0.4]]

    class FakeCider:
        def compute_score(self, ground_truth, results):
            del ground_truth, results
            return 0.5, [0.5]

    module_names = [
        "pycocoevalcap",
        "pycocoevalcap.bleu",
        "pycocoevalcap.bleu.bleu",
        "pycocoevalcap.cider",
        "pycocoevalcap.cider.cider",
        "pycocoevalcap.tokenizer",
        "pycocoevalcap.tokenizer.ptbtokenizer",
    ]
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["pycocoevalcap.bleu.bleu"].Bleu = FakeBleu
    modules["pycocoevalcap.cider.cider"].Cider = FakeCider
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(caption_metrics.shutil, "which", lambda command: "java")
    monkeypatch.setattr(caption_metrics, "_tokenize_coco_captions", fake_tokenize)

    result = caption_metrics.evaluate_coco_metrics(
        {"image.jpg": ["reference"]},
        {"image.jpg": "prediction"},
    )

    assert result["CIDEr"] == pytest.approx(50.0)
    assert result["BLEU-4"] == pytest.approx(40.0)
    assert calls == [{"image.jpg": ["reference"]}, {"image.jpg": ["prediction"]}]


def _mock_ptb_output(monkeypatch, stdout, stderr=""):
    def run(command, **kwargs):
        assert command[-3:] == [
            "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase"
        ]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["timeout"] == 120
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    monkeypatch.setattr(caption_metrics.shutil, "which", lambda _: "java")
    monkeypatch.setattr(caption_metrics.subprocess, "run", run)


def test_ptb_preserves_image_and_reference_order_with_container_warnings(monkeypatch):
    warning_lines = "".join(
        f"[0.001s][warning][os,container] Cgroup {kind} controller path "
        "at '/sys/fs/cgroup' seems to have moved to '/..', "
        "detected limits won't be accurate\n"
        for kind in ("memory", "cpu")
    )
    output = "a dog .\nan animal !\na swimmer .\n"
    captions = {"z.jpg": ["A dog.", "An animal!"], "a.jpg": ["A swimmer."]}
    _mock_ptb_output(monkeypatch, output)
    clean = caption_metrics._tokenize_coco_captions(captions)
    _mock_ptb_output(monkeypatch, warning_lines + output, "PTBTokenizer tokenized 9 tokens")
    with pytest.warns(RuntimeWarning, match="Ignored 2 JVM"):
        checked = caption_metrics._tokenize_coco_captions(captions)
    assert checked == clean == {"z.jpg": ["a dog", "an animal"], "a.jpg": ["a swimmer"]}


@pytest.mark.parametrize("output", [
    "unexpected Java log\na dog\na swimmer\n",
    "a dog\n",
    "",
    "a dog\na swimmer\n\n",
])
def test_ptb_rejects_extra_or_missing_output_lines(monkeypatch, output):
    _mock_ptb_output(monkeypatch, output)
    with pytest.raises(RuntimeError, match="line count mismatch"):
        caption_metrics._tokenize_coco_captions({"a": ["a dog"], "b": ["a swimmer"]})


def test_ptb_normalizes_caption_line_breaks_without_shifting_ids(monkeypatch):
    _mock_ptb_output(monkeypatch, "a dog\na swimmer\n")
    original_run = caption_metrics.subprocess.run

    def run(command, **kwargs):
        assert kwargs["input"] == "A dog\nA swimmer\n"
        return original_run(command, **kwargs)

    monkeypatch.setattr(caption_metrics.subprocess, "run", run)
    assert caption_metrics._tokenize_coco_captions(
        {"a": ["A\r\ndog"], "b": ["A\nswimmer"]}
    ) == {"a": ["a dog"], "b": ["a swimmer"]}


@pytest.mark.parametrize("error, message", [
    (subprocess.CalledProcessError(1, ["java"], stderr="Java failed"), "exit 1.*Java failed"),
    (subprocess.TimeoutExpired(["java"], 120), "timed out"),
])
def test_ptb_rejects_java_failure(monkeypatch, error, message):
    monkeypatch.setattr(caption_metrics.shutil, "which", lambda _: "java")

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(caption_metrics.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=message):
        caption_metrics._tokenize_coco_captions({"a": ["a dog"]})


@pytest.mark.skipif(shutil.which("java") is None, reason="Java required for Stanford integration")
def test_real_stanford_tokenization_and_perfect_reference_bleu():
    refs = {
        "b.jpg": ["A girl swims in a blue pool."],
        "a.jpg": ["A brown dog runs through the grass."],
    }
    tokens = caption_metrics._tokenize_coco_captions(refs)
    assert tokens == {
        "b.jpg": ["a girl swims in a blue pool"],
        "a.jpg": ["a brown dog runs through the grass"],
    }
    result = caption_metrics.evaluate_coco_metrics(refs, {i: c[0] for i, c in refs.items()})
    assert result["BLEU-4"] == pytest.approx(100.0)
    assert list(result["per_image"]) == ["a.jpg", "b.jpg"]
    assert all(row["BLEU-4"] == pytest.approx(100.0) for row in result["per_image"].values())


@pytest.mark.skipif(shutil.which("java") is None, reason="Java required for Stanford integration")
def test_real_scores_unchanged_when_java_emits_two_container_warnings(monkeypatch):
    refs = {
        "b.jpg": ["A girl swims in a blue pool."] * 5,
        "a.jpg": ["A brown dog runs through the grass."] * 5,
        "c.jpg": ["A man rides a red bicycle."] * 5,
    }
    predictions = {i: captions[0] for i, captions in refs.items()}
    clean = caption_metrics.evaluate_coco_metrics(refs, predictions)
    original_run = caption_metrics.subprocess.run

    def run_with_warnings(*args, **kwargs):
        completed = original_run(*args, **kwargs)
        completed.stdout = (
            "[0.001s][warning][os,container] Cgroup memory controller path moved\n"
            "[0.001s][warning][os,container] Cgroup cpu controller path moved\n"
            + completed.stdout
        )
        return completed

    monkeypatch.setattr(caption_metrics.subprocess, "run", run_with_warnings)
    with pytest.warns(RuntimeWarning, match="Ignored 2 JVM"):
        checked = caption_metrics.evaluate_coco_metrics(refs, predictions)
    assert checked == clean


def test_clipscore_uses_published_prefix_weight_and_refclipscore(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_texts: list[str] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "openai/clip-vit-base-patch32"
            return cls()

        def __call__(self, *, text, **kwargs):
            del kwargs
            observed_texts.extend(text)
            batch_size = len(text)
            return {
                "input_ids": torch.ones((batch_size, 1), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 1), dtype=torch.long),
            }

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "openai/clip-vit-base-patch32"
            return cls()

        def to(self, device):
            del device
            return self

        def eval(self):
            return self

        def parameters(self):
            return ()

        def get_text_features(self, input_ids, attention_mask):
            del attention_mask
            return torch.tensor([[0.6, 0.8]]).repeat(input_ids.size(0), 1)

    monkeypatch.setattr(
        caption_metrics,
        "_load_feature_cache",
        lambda *args, **kwargs: {"image.jpg": torch.tensor([1.0, 0.0])},
    )
    monkeypatch.setattr(caption_metrics, "CLIPProcessor", FakeProcessor)
    monkeypatch.setattr(caption_metrics, "CLIPModel", FakeModel)

    result = caption_metrics.evaluate_clipscore_from_cache(
        ["image.jpg"],
        {"image.jpg": ["a dog running"] * 5},
        {"clipcap": {"image.jpg": "a dog running"}},
        "unused.pt",
        device_name="cpu",
    )

    assert observed_texts == ["A photo depicts a dog running"] * 6
    assert result["clipcap"]["CLIPScore"] == pytest.approx(1.5)
    assert result["clipcap"]["RefCLIPScore"] == pytest.approx(1.2)
    assert result["clipcap"]["per_image"]["image.jpg"] == pytest.approx(
        {"CLIPScore": 1.5, "RefCLIPScore": 1.2}
    )


def test_run_caption_evaluation_writes_common_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path = tmp_path / "ids.json"
    references_path = tmp_path / "references.json"
    clipcap_dir = tmp_path / "clipcap"
    zerocap_dir = tmp_path / "zerocap"
    clipcap_dir.mkdir()
    zerocap_dir.mkdir()
    prediction_path = clipcap_dir / "captions.jsonl"
    zerocap_prediction_path = zerocap_dir / "captions.json"
    run_config_path = tmp_path / "run_config.json"
    feature_cache_path = tmp_path / "features.pt"
    output_dir = tmp_path / "model_comparison"
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
        zerocap_prediction_path,
        {image_id: f"zerocap caption {image_id}" for image_id in image_ids},
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
        references,
        predictions_by_experiment,
        feature_cache_path,
        model_name,
        batch_size,
        device_name,
    ):
        del feature_cache_path, model_name, batch_size, device_name
        assert selected_ids == image_ids
        assert set(references) == set(image_ids)
        return {
            label: {
                "CLIPScore": 30.0,
                "RefCLIPScore": 31.0,
                "per_image": {
                    image_id: {"CLIPScore": 30.0, "RefCLIPScore": 31.0}
                    for image_id in image_ids
                },
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
        prediction_paths={
            "clipcap": prediction_path,
            "zerocap": zerocap_prediction_path,
        },
        feature_cache_path=feature_cache_path,
        output_dir=output_dir,
        run_config_path=run_config_path,
        expected_run_config={"prompt": None},
        artifact_metadata={"split": "val"},
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    clipcap_metrics = json.loads(
        artifacts.model_metric_paths["clipcap"].read_text(encoding="utf-8")
    )
    zerocap_metrics = json.loads(
        artifacts.model_metric_paths["zerocap"].read_text(encoding="utf-8")
    )
    with artifacts.per_image_path.open("r", encoding="utf-8", newline="") as file:
        per_image_rows = list(csv.DictReader(file))
    assert summary["metadata"] == {"split": "val"}
    assert artifacts.summary_path.name == "comparison.json"
    assert artifacts.per_image_path.name == "per_image_comparison.csv"
    assert artifacts.model_metric_paths["clipcap"] == clipcap_dir / "metrics.json"
    assert artifacts.model_metric_paths["zerocap"] == zerocap_dir / "metrics.json"
    assert clipcap_metrics["experiment"] == "clipcap"
    assert clipcap_metrics["metrics"]["CIDEr"] == 50.0
    assert clipcap_metrics["coverage"]["num_predictions"] == 2
    assert zerocap_metrics["experiment"] == "zerocap"
    assert zerocap_metrics["metrics"]["BLEU-4"] == 25.0
    assert summary["model_metric_files"]["clipcap"] == str(
        (clipcap_dir / "metrics.json").resolve()
    )
    assert summary["coverage"]["clipcap"]["num_predictions"] == 2
    assert summary["results"][0]["CIDEr"] == 50.0
    assert summary["results"][0]["BLEU-4"] == 25.0
    assert summary["results"][0]["CLIPScore"] == 30.0
    assert summary["results"][0]["RefCLIPScore"] == 31.0
    assert summary["supplementary_metrics"] == ["CLIPScore", "RefCLIPScore"]
    assert clipcap_metrics["metrics"]["RefCLIPScore"] == 31.0
    assert per_image_rows[0]["RefCLIPScore"] == "31.0"
    assert len(per_image_rows) == 4
