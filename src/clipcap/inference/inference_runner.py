from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel

from src.clipcap.models.clipcap_model import ClipCaptionModel
from src.config.clipcap_config import CLIPCAP_DEFAULT_INFERENCE_CONFIG

from .checkpoints import FINAL_CHECKPOINT_NAME, build_mapper_from_checkpoint
from .decoding import generate_caption_from_feature
from .records import EvaluationItem, FixedEpochCheckpoint, GenerationResult


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def prepare_run_config(
    payload: Mapping[str, Any],
    path: Path,
    resume: bool,
) -> None:
    if resume and path.is_file():
        existing = _load_json_object(path)
        if existing != dict(payload):
            raise ValueError(
                "Existing run_config.json does not match this evaluation. "
                "Use another --output-dir or pass --no-resume."
            )
        return
    _atomic_json_save(payload, path)


def _load_completed_image_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from error
            image_id = record.get("image_id")
            if not isinstance(image_id, str):
                raise TypeError(f"Missing image_id at {path}:{line_number}")
            completed.add(image_id)
    return completed


def generation_result_to_record(
    item: EvaluationItem,
    result: GenerationResult,
    subset_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "image_id": item.image_id,
        "subset_name": subset_name,
        "seed": seed,
        "checkpoint": FINAL_CHECKPOINT_NAME,
        "caption": result.caption,
        "selected_beam_rank": result.selected_beam_rank,
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }


def run_evaluation(
    items: Sequence[EvaluationItem],
    features: Mapping[str, Tensor],
    checkpoints: Sequence[FixedEpochCheckpoint],
    processor: CLIPProcessor,
    encoder: CLIPModel,
    gpt2: GPT2LMHeadModel,
    tokenizer: Any,
    device: torch.device,
    output_dir: str | Path,
    max_new_tokens: int = CLIPCAP_DEFAULT_INFERENCE_CONFIG.max_new_tokens,
    num_beams: int = CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_beams,
    num_return_sequences: int = (
        CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_return_sequences
    ),
    length_penalty: float = CLIPCAP_DEFAULT_INFERENCE_CONFIG.length_penalty,
    early_stopping: bool = CLIPCAP_DEFAULT_INFERENCE_CONFIG.early_stopping,
    resume: bool = True,
) -> dict[str, Path]:
    selected_output_dir = Path(output_dir)
    output_paths: dict[str, Path] = {}
    for artifact in checkpoints:
        mapper = build_mapper_from_checkpoint(artifact, gpt2, encoder, device)
        model = ClipCaptionModel(mapper=mapper, gpt2=gpt2).to(device)
        model.eval()
        predictions_path = (
            selected_output_dir
            / artifact.subset_name
            / f"seed_{artifact.seed}"
            / "predictions.jsonl"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        completed = _load_completed_image_ids(predictions_path) if resume else set()
        mode = "a" if resume else "w"
        with predictions_path.open(mode, encoding="utf-8") as file:
            for item in items:
                if item.image_id in completed:
                    continue
                if item.image_id not in features:
                    raise KeyError(f"Missing CLIP feature for {item.image_id}")
                result = generate_caption_from_feature(
                    features[item.image_id],
                    processor,
                    encoder,
                    model,
                    tokenizer,
                    device,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                )
                record = generation_result_to_record(
                    item,
                    result,
                    artifact.subset_name,
                    artifact.seed,
                )
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
        output_paths[artifact.subset_name] = predictions_path
        del model
        del mapper
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output_paths
