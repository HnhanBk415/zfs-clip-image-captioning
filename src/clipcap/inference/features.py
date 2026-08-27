from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from src.config.clipcap_config import CLIP_MODEL_NAME

from .records import EvaluationItem


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, path)


def extract_feature_tensor(outputs: Any) -> Tensor:
    return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs


def load_evaluation_manifest(path: str | Path) -> tuple[EvaluationItem, ...]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    items: list[EvaluationItem] = []
    if isinstance(payload, dict):
        for image_id, references in payload.items():
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("Manifest image IDs must be non-empty strings")
            if not isinstance(references, list) or not all(
                isinstance(reference, str) for reference in references
            ):
                raise TypeError(f"References for {image_id} must be a list of strings")
            items.append(EvaluationItem(image_id, tuple(references)))
    elif isinstance(payload, list):
        for image_id in payload:
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("Manifest entries must be non-empty image ID strings")
            items.append(EvaluationItem(image_id, ()))
    else:
        raise TypeError("Manifest must be a JSON object or a list of image IDs")

    if not items:
        raise ValueError("Evaluation manifest is empty")
    image_ids = [item.image_id for item in items]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Evaluation manifest contains duplicate image IDs")
    return tuple(items)


def load_rgb_image(path: str | Path) -> Image.Image:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"Image not found: {selected_path}")
    with Image.open(selected_path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def encode_image_with_clip(
    image: Image.Image,
    processor: CLIPProcessor,
    encoder: CLIPModel,
    device: torch.device,
) -> Tensor:
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed.get("pixel_values")
    if pixel_values is None:
        raise KeyError("CLIPProcessor did not return pixel_values")
    features = extract_feature_tensor(
        encoder.get_image_features(pixel_values=pixel_values.to(device))
    )
    if features.ndim != 2 or features.size(0) != 1:
        raise ValueError("CLIP image feature must have shape [1, clip_dim]")
    if not torch.isfinite(features).all():
        raise ValueError("CLIP image feature contains NaN or Inf")
    return features


@torch.inference_mode()
def encode_evaluation_images(
    items: Sequence[EvaluationItem],
    image_dir: str | Path,
    processor: CLIPProcessor,
    encoder: CLIPModel,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Tensor]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = Path(image_dir)
    features: dict[str, Tensor] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        images = [load_rgb_image(root / item.image_id) for item in batch]
        processed = processor(images=images, return_tensors="pt")
        pixel_values = processed.get("pixel_values")
        if pixel_values is None:
            raise KeyError("CLIPProcessor did not return pixel_values")
        batch_features = extract_feature_tensor(
            encoder.get_image_features(pixel_values=pixel_values.to(device))
        )
        if batch_features.ndim != 2 or batch_features.size(0) != len(batch):
            raise ValueError("CLIP batch feature shape does not match the image batch")
        if not torch.isfinite(batch_features).all():
            raise ValueError("CLIP image features contain NaN or Inf")
        for item, feature in zip(batch, batch_features.detach().cpu()):
            features[item.image_id] = feature
    return features


def load_feature_cache(
    path: str | Path,
    expected_image_ids: Sequence[str],
) -> dict[str, Tensor]:
    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid feature cache: {cache_path}")
    if payload.get("clip_model") != CLIP_MODEL_NAME:
        raise ValueError("Feature cache was created with a different CLIP model")
    image_ids = payload.get("image_ids")
    feature_tensor = payload.get("features")
    if not isinstance(image_ids, list) or not isinstance(feature_tensor, Tensor):
        raise TypeError("Feature cache must contain image_ids and a feature tensor")
    if feature_tensor.ndim != 2 or feature_tensor.size(0) != len(image_ids):
        raise ValueError("Feature cache image IDs and feature rows do not align")
    indexed = {image_id: feature for image_id, feature in zip(image_ids, feature_tensor)}
    missing = [image_id for image_id in expected_image_ids if image_id not in indexed]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Feature cache is missing {len(missing)} images: {preview}")
    return {image_id: indexed[image_id] for image_id in expected_image_ids}


def save_feature_cache(features: Mapping[str, Tensor], path: str | Path) -> None:
    image_ids = list(features)
    if not image_ids:
        raise ValueError("Cannot save an empty feature cache")
    feature_tensor = torch.stack(
        [features[image_id].detach().cpu().float() for image_id in image_ids]
    )
    _atomic_torch_save(
        {
            "image_ids": image_ids,
            "features": feature_tensor,
            "clip_model": CLIP_MODEL_NAME,
            "feature_dim": int(feature_tensor.size(1)),
        },
        Path(path),
    )
