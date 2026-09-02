from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from src.config.common_config import CLIP_MODEL_NAME


CLIPSCORE_TEXT_PREFIX = "A photo depicts"
CLIPSCORE_WEIGHT = 2.5


@dataclass(frozen=True)
class CaptionMetricArtifacts:
    summary_path: Path
    per_image_path: Path
    model_metric_paths: Mapping[str, Path]
    results: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    selected_path = Path(path)
    digest = hashlib.sha256()
    with selected_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: str | Path) -> Any:
    selected_path = Path(path)
    with selected_path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=_reject_duplicate_keys)


def load_manifest_ids(path: str | Path) -> list[str]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"Inference manifest not found: {selected_path}")
    payload = load_json_strict(selected_path)
    if isinstance(payload, dict):
        image_ids = list(payload)
    elif isinstance(payload, list):
        image_ids = payload
    else:
        raise TypeError("Inference manifest must be a JSON object or list")
    if not image_ids or not all(
        isinstance(image_id, str) and image_id.strip() for image_id in image_ids
    ):
        raise ValueError("Inference manifest contains an invalid image ID")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Inference manifest contains duplicate image IDs")
    return image_ids


def load_references(
    path: str | Path,
    expected_per_image: int = 5,
) -> dict[str, list[str]]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"Reference manifest not found: {selected_path}")
    payload = load_json_strict(selected_path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError("Reference manifest must be a non-empty JSON object")
    references: dict[str, list[str]] = {}
    for image_id, captions in payload.items():
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("Reference image IDs must be non-empty strings")
        if not isinstance(captions, list) or len(captions) != expected_per_image:
            raise ValueError(
                f"{image_id} must have exactly {expected_per_image} references"
            )
        cleaned: list[str] = []
        for caption in captions:
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{image_id} contains an invalid reference")
            cleaned.append(caption.strip())
        references[image_id] = cleaned
    return references


def _add_prediction(
    predictions: dict[str, str],
    image_id: Any,
    caption: Any,
    location: str,
) -> None:
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError(f"Invalid image_id at {location}")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"Invalid caption at {location}")
    if image_id in predictions:
        raise ValueError(f"Duplicate prediction for {image_id}")
    predictions[image_id] = caption.strip()


def load_predictions(path: str | Path) -> dict[str, str]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {selected_path}")
    predictions: dict[str, str] = {}
    if selected_path.suffix.lower() == ".jsonl":
        with selected_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at {selected_path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Prediction record must be an object at "
                        f"{selected_path}:{line_number}"
                    )
                _add_prediction(
                    predictions,
                    record.get("image_id"),
                    record.get("caption"),
                    f"{selected_path}:{line_number}",
                )
    else:
        payload = load_json_strict(selected_path)
        if isinstance(payload, dict):
            for image_id, value in payload.items():
                caption = value.get("caption") if isinstance(value, dict) else value
                _add_prediction(predictions, image_id, caption, str(selected_path))
        elif isinstance(payload, list):
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Prediction record must be an object at index {index}"
                    )
                _add_prediction(
                    predictions,
                    record.get("image_id"),
                    record.get("caption"),
                    f"{selected_path}:{index}",
                )
        else:
            raise TypeError("Prediction JSON must be an object or list")
    if not predictions:
        raise ValueError(f"Prediction file is empty: {selected_path}")
    return predictions


def validate_prediction_coverage(
    expected_image_ids: Sequence[str],
    predictions: Mapping[str, str],
    label: str,
) -> None:
    expected = set(expected_image_ids)
    actual = set(predictions)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{label}: missing={len(missing)}, extra={len(extra)}"
        )


def evaluate_coco_metrics(
    references: Mapping[str, Sequence[str]],
    predictions: Mapping[str, str],
) -> dict[str, Any]:
    if shutil.which("java") is None:
        raise RuntimeError("CIDEr/BLEU-4 evaluation requires Java in PATH")
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    image_ids = sorted(references)
    raw_ground_truth = {
        image_id: [{"caption": caption} for caption in references[image_id]]
        for image_id in image_ids
    }
    raw_results = {
        image_id: [{"caption": predictions[image_id]}]
        for image_id in image_ids
    }
    tokenizer = PTBTokenizer()
    ground_truth = tokenizer.tokenize(raw_ground_truth)
    results = tokenizer.tokenize(raw_results)
    bleu_score, bleu_per_image = Bleu(4).compute_score(
        ground_truth,
        results,
        verbose=0,
    )
    cider_score, cider_per_image = Cider().compute_score(ground_truth, results)
    per_image = {
        image_id: {
            "CIDEr": float(cider_value) * 100.0,
            "BLEU-4": float(bleu_value) * 100.0,
        }
        for image_id, cider_value, bleu_value in zip(
            image_ids,
            cider_per_image,
            bleu_per_image[3],
        )
    }
    return {
        "CIDEr": float(cider_score) * 100.0,
        "BLEU-4": float(bleu_score[3]) * 100.0,
        "per_image": per_image,
    }


def _load_feature_cache(
    path: str | Path,
    expected_image_ids: Sequence[str],
    clip_model_name: str,
) -> dict[str, Tensor]:
    selected_path = Path(path)
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid feature cache: {selected_path}")
    if payload.get("clip_model") != clip_model_name:
        raise ValueError("Feature cache was created with a different CLIP model")
    image_ids = payload.get("image_ids")
    features = payload.get("features")
    if not isinstance(image_ids, list) or not isinstance(features, Tensor):
        raise TypeError("Feature cache must contain image_ids and features")
    if features.ndim != 2 or features.size(0) != len(image_ids):
        raise ValueError("Feature cache IDs and feature rows do not align")
    indexed = {image_id: feature for image_id, feature in zip(image_ids, features)}
    missing = [image_id for image_id in expected_image_ids if image_id not in indexed]
    if missing:
        raise KeyError(f"Feature cache is missing {len(missing)} images")
    return {image_id: indexed[image_id] for image_id in expected_image_ids}


def _unwrap_features(value: Any) -> Tensor:
    return value.pooler_output if hasattr(value, "pooler_output") else value


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def evaluate_clipscore_from_cache(
    image_ids: Sequence[str],
    references: Mapping[str, Sequence[str]],
    predictions_by_experiment: Mapping[str, Mapping[str, str]],
    feature_cache_path: str | Path,
    model_name: str = CLIP_MODEL_NAME,
    batch_size: int = 32,
    device_name: str = "auto",
) -> dict[str, dict[str, Any]]:
    device = _select_device(device_name)
    cached = _load_feature_cache(feature_cache_path, image_ids, model_name)
    image_features = torch.stack([cached[image_id] for image_id in image_ids]).float()
    image_features = functional.normalize(image_features, dim=-1)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    def encode_texts(texts: Sequence[str]) -> Tensor:
        encoded_batches: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            prompted_texts = [
                f"{CLIPSCORE_TEXT_PREFIX} {text}"
                for text in texts[start : start + batch_size]
            ]
            text_inputs = processor(
                text=prompted_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_features = _unwrap_features(
                model.get_text_features(
                    input_ids=text_inputs["input_ids"].to(device),
                    attention_mask=text_inputs["attention_mask"].to(device),
                )
            )
            encoded_batches.append(
                functional.normalize(text_features.float(), dim=-1).cpu()
            )
        return torch.cat(encoded_batches, dim=0)

    results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        flattened_references = [
            reference
            for image_id in image_ids
            for reference in references[image_id]
        ]
        encoded_references = encode_texts(flattened_references)
        reference_features: dict[str, Tensor] = {}
        offset = 0
        for image_id in image_ids:
            next_offset = offset + len(references[image_id])
            reference_features[image_id] = encoded_references[offset:next_offset]
            offset = next_offset

        for label, predictions in predictions_by_experiment.items():
            per_image_scores: dict[str, dict[str, float]] = {}
            for start in range(0, len(image_ids), batch_size):
                batch_ids = image_ids[start : start + batch_size]
                captions = [predictions[image_id] for image_id in batch_ids]
                text_features = encode_texts(captions).to(device)
                batch_image_features = image_features[
                    start : start + len(batch_ids)
                ].to(device)
                clip_scores = (
                    CLIPSCORE_WEIGHT
                    * (batch_image_features * text_features).sum(dim=-1)
                ).clamp(min=0.0)
                reference_scores = torch.stack(
                    [
                        (
                            reference_features[image_id].to(device)
                            @ text_feature
                        ).max()
                        for image_id, text_feature in zip(batch_ids, text_features)
                    ]
                ).clamp(min=0.0)
                denominators = clip_scores + reference_scores
                refclip_scores = torch.where(
                    denominators > 0,
                    2.0 * clip_scores * reference_scores / denominators,
                    torch.zeros_like(denominators),
                )
                for image_id, clip_score, refclip_score in zip(
                    batch_ids,
                    clip_scores.cpu().tolist(),
                    refclip_scores.cpu().tolist(),
                ):
                    per_image_scores[image_id] = {
                        "CLIPScore": float(clip_score),
                        "RefCLIPScore": float(refclip_score),
                    }
            results[label] = {
                "CLIPScore": sum(
                    scores["CLIPScore"] for scores in per_image_scores.values()
                )
                / len(per_image_scores),
                "RefCLIPScore": sum(
                    scores["RefCLIPScore"] for scores in per_image_scores.values()
                )
                / len(per_image_scores),
                "per_image": per_image_scores,
            }

    del model
    del processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def validate_run_config(
    run_config_path: str | Path,
    inference_manifest_path: str | Path,
    expected_num_images: int,
    expected_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_path = Path(run_config_path)
    payload = load_json_strict(selected_path)
    if not isinstance(payload, dict):
        raise TypeError("run_config.json must contain a JSON object")
    required = {
        "manifest_sha256": sha256_file(inference_manifest_path),
        "num_images": expected_num_images,
        **dict(expected_values or {}),
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"run_config mismatch at {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    return payload


def _model_metrics_path(prediction_path: Path) -> Path:
    if prediction_path.stem == "captions":
        return prediction_path.with_name("metrics.json")
    return prediction_path.with_name(f"{prediction_path.stem}_metrics.json")


def run_caption_evaluation(
    *,
    inference_manifest_path: str | Path,
    references_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    feature_cache_path: str | Path,
    output_dir: str | Path,
    clip_model_name: str = CLIP_MODEL_NAME,
    expected_references_per_image: int = 5,
    clip_batch_size: int = 32,
    device_name: str = "auto",
    run_config_path: str | Path | None = None,
    expected_run_config: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    experiment_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> CaptionMetricArtifacts:
    if not prediction_paths:
        raise ValueError("At least one prediction file is required")
    inference_manifest = Path(inference_manifest_path)
    reference_manifest = Path(references_path)
    selected_output_dir = Path(output_dir)
    image_ids = load_manifest_ids(inference_manifest)
    all_references = load_references(
        reference_manifest,
        expected_references_per_image,
    )
    missing_references = set(image_ids) - set(all_references)
    if missing_references:
        raise ValueError(
            f"Missing references for {len(missing_references)} inference images"
        )
    references = {image_id: all_references[image_id] for image_id in image_ids}

    run_config: dict[str, Any] | None = None
    if run_config_path is not None:
        run_config = validate_run_config(
            run_config_path,
            inference_manifest,
            len(image_ids),
            expected_run_config,
        )

    selected_prediction_paths = {
        label: Path(path) for label, path in prediction_paths.items()
    }
    predictions_by_experiment: dict[str, dict[str, str]] = {}
    coverage: dict[str, dict[str, int]] = {}
    for label, path in selected_prediction_paths.items():
        if not label.strip():
            raise ValueError("Experiment labels must be non-empty")
        predictions = load_predictions(path)
        validate_prediction_coverage(image_ids, predictions, label)
        predictions_by_experiment[label] = predictions
        coverage[label] = {
            "num_images": len(image_ids),
            "num_predictions": len(predictions),
            "num_reference_captions": sum(
                len(items) for items in references.values()
            ),
        }

    coco_results = {
        label: evaluate_coco_metrics(references, predictions)
        for label, predictions in predictions_by_experiment.items()
    }
    clip_results = evaluate_clipscore_from_cache(
        image_ids,
        references,
        predictions_by_experiment,
        feature_cache_path,
        clip_model_name,
        clip_batch_size,
        device_name,
    )

    experiment_details = experiment_metadata or {}
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for label, predictions in predictions_by_experiment.items():
        summary_rows.append(
            {
                "experiment": label,
                **dict(experiment_details.get(label, {})),
                "CIDEr": coco_results[label]["CIDEr"],
                "BLEU-4": coco_results[label]["BLEU-4"],
                "CLIPScore": clip_results[label]["CLIPScore"],
                "RefCLIPScore": clip_results[label]["RefCLIPScore"],
            }
        )
        for image_id in sorted(references):
            per_image_rows.append(
                {
                    "experiment": label,
                    "image_id": image_id,
                    "prediction": predictions[image_id],
                    "CIDEr": coco_results[label]["per_image"][image_id]["CIDEr"],
                    "BLEU-4": coco_results[label]["per_image"][image_id]["BLEU-4"],
                    "CLIPScore": clip_results[label]["per_image"][image_id][
                        "CLIPScore"
                    ],
                    "RefCLIPScore": clip_results[label]["per_image"][image_id][
                        "RefCLIPScore"
                    ],
                }
            )
    summary_rows.sort(key=lambda row: (-row["CIDEr"], -row["BLEU-4"]))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    selected_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = selected_output_dir / "comparison.json"
    per_image_path = selected_output_dir / "per_image_comparison.csv"
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    metric_scales = {
        "CIDEr": "raw CIDEr multiplied by 100; not a percentage",
        "BLEU-4": "corpus BLEU-4 multiplied by 100",
        "CLIPScore": (
            "2.5 * max(cosine_similarity(image, "
            "'A photo depicts ' + caption), 0)"
        ),
        "RefCLIPScore": (
            "harmonic mean of CLIPScore and max non-negative cosine similarity "
            "between candidate and references"
        ),
    }
    inference_manifest_sha256 = sha256_file(inference_manifest)
    references_sha256 = sha256_file(reference_manifest)
    prediction_sha256 = {
        label: sha256_file(path)
        for label, path in selected_prediction_paths.items()
    }
    model_metric_paths: dict[str, Path] = {}
    for row in summary_rows:
        label = row["experiment"]
        prediction_path = selected_prediction_paths[label]
        model_metric_path = _model_metrics_path(prediction_path)
        model_metric_path.parent.mkdir(parents=True, exist_ok=True)
        model_metric_artifact = {
            "generated_at_utc": generated_at_utc,
            "experiment": label,
            "metric_scales": metric_scales,
            "metrics": {
                "CIDEr": row["CIDEr"],
                "BLEU-4": row["BLEU-4"],
                "CLIPScore": row["CLIPScore"],
                "RefCLIPScore": row["RefCLIPScore"],
            },
            "inference_manifest_path": str(inference_manifest.resolve()),
            "inference_manifest_sha256": inference_manifest_sha256,
            "references_path": str(reference_manifest.resolve()),
            "references_sha256": references_sha256,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": prediction_sha256[label],
            "coverage": coverage[label],
            "metadata": dict(experiment_details.get(label, {})),
        }
        with model_metric_path.open("w", encoding="utf-8") as file:
            json.dump(model_metric_artifact, file, ensure_ascii=False, indent=2)
        model_metric_paths[label] = model_metric_path
    artifact: dict[str, Any] = {
        "generated_at_utc": generated_at_utc,
        "ranking_policy": ["CIDEr", "BLEU-4"],
        "supplementary_metrics": ["CLIPScore", "RefCLIPScore"],
        "metric_scales": metric_scales,
        "inference_manifest_path": str(inference_manifest.resolve()),
        "inference_manifest_sha256": inference_manifest_sha256,
        "references_path": str(reference_manifest.resolve()),
        "references_sha256": references_sha256,
        "feature_cache_path": str(Path(feature_cache_path).resolve()),
        "coverage": coverage,
        "results": summary_rows,
        "prediction_files": {
            label: {
                "path": str(path.resolve()),
                "sha256": prediction_sha256[label],
            }
            for label, path in selected_prediction_paths.items()
        },
        "model_metric_files": {
            label: str(path.resolve())
            for label, path in model_metric_paths.items()
        },
        "metadata": dict(artifact_metadata or {}),
    }
    if run_config_path is not None:
        selected_run_config_path = Path(run_config_path)
        artifact["run_config"] = run_config
        artifact["run_config_path"] = str(selected_run_config_path.resolve())
        artifact["run_config_sha256"] = sha256_file(selected_run_config_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)
    with per_image_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "image_id",
                "prediction",
                "CIDEr",
                "BLEU-4",
                "CLIPScore",
                "RefCLIPScore",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)
    return CaptionMetricArtifacts(
        summary_path=summary_path,
        per_image_path=per_image_path,
        model_metric_paths=model_metric_paths,
        results=tuple(summary_rows),
    )


def _parse_prediction_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label.strip(), Path(path.strip())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image captions with CIDEr, BLEU-4, CLIPScore, and "
            "RefCLIPScore."
        )
    )
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=_parse_prediction_argument,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--clip-model", default=CLIP_MODEL_NAME)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--references-per-image", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    prediction_paths = dict(args.prediction)
    if len(prediction_paths) != len(args.prediction):
        raise ValueError("Prediction labels must be unique")
    artifacts = run_caption_evaluation(
        inference_manifest_path=args.inference_manifest,
        references_path=args.references,
        prediction_paths=prediction_paths,
        feature_cache_path=args.feature_cache,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model,
        expected_references_per_image=args.references_per_image,
        clip_batch_size=args.clip_batch_size,
        device_name=args.device,
        run_config_path=args.run_config,
    )
    print("Evaluation ranking")
    for row in artifacts.results:
        print(
            f"{row['rank']:>4}  {row['experiment']:<28} "
            f"CIDEr={row['CIDEr']:.4f} "
            f"BLEU-4={row['BLEU-4']:.4f} "
            f"CLIPScore={row['CLIPScore']:.4f} "
            f"RefCLIPScore={row['RefCLIPScore']:.4f}"
        )
    for label, path in artifacts.model_metric_paths.items():
        print(f"{label} metrics: {path}")
    print(f"Comparison: {artifacts.summary_path}")
    print(f"Per-image comparison: {artifacts.per_image_path}")


if __name__ == "__main__":
    main()
