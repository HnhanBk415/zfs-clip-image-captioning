"""Extract and cache Flickr8k image features with pretrained CLIP.

This module contains the reusable artifact-producing logic from
``notebook/preprocessing/clip_feature_extraction.ipynb``. Importing it does
not download a dataset or model and does not write any files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from src.config.common_config import (
    KAGGLE_DATASET_HANDLE,
    SPLIT_DIR,
)
from src.config.clipcap_config import (
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    FEATURE_DIR,
    setup_clipcap_directories,
)
import kagglehub
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


REQUIRED_SPLITS = ("train", "val", "test")


def load_json(path: str | Path) -> Any:
    """Load one UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split_image_ids(split_dir: str | Path) -> list[str]:
    """Load disjoint split files and return all image IDs in sorted order."""
    split_path = Path(split_dir)
    split_files = {
        split_name: split_path / f"{split_name}.json"
        for split_name in REQUIRED_SPLITS
    }
    missing_files = [
        path.name for path in split_files.values() if not path.is_file()
    ]
    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(f"Missing required split files: {missing}")

    split_ids: dict[str, set[str]] = {}
    for split_name, path in split_files.items():
        split_data = load_json(path)
        if not isinstance(split_data, dict):
            raise TypeError(f"{path.name} must contain a JSON object")
        split_ids[split_name] = set(split_data.keys())

    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    if not train_ids.isdisjoint(val_ids):
        raise ValueError("Train and validation splits share image IDs")
    if not train_ids.isdisjoint(test_ids):
        raise ValueError("Train and test splits share image IDs")
    if not val_ids.isdisjoint(test_ids):
        raise ValueError("Validation and test splits share image IDs")

    return sorted(train_ids | val_ids | test_ids)


def load_image_batch(
    image_ids: Sequence[str],
    images_dir: str | Path,
) -> list[Image.Image]:
    """Load one image batch and convert every image to RGB."""
    image_directory = Path(images_dir)
    images: list[Image.Image] = []

    for image_id in image_ids:
        image_path = image_directory / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    return images


def extract_clip_features(
    split_dir: str | Path,
    images_dir: str | Path,
    feature_dir: str | Path,
    model_name: str,
    batch_size: int,
    device: str | None = None,
    processor: Any | None = None,
    clip_model: Any | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Extract CLIP features and save the notebook-compatible cache file."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    image_ids = load_split_image_ids(split_dir)
    if not image_ids:
        raise ValueError("No image IDs were found in the split files")

    image_directory = Path(images_dir)
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {image_directory}")

    if processor is None or clip_model is None:
        if processor is None:
            processor = CLIPProcessor.from_pretrained(model_name)
        if clip_model is None:
            clip_model = CLIPModel.from_pretrained(model_name)

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    clip_model = clip_model.to(selected_device)
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    clip_model.eval()

    feature_batches: list[Any] = []
    processed_image_ids: list[str] = []

    for start_index in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start_index : start_index + batch_size]
        images = load_image_batch(batch_ids, image_directory)
        inputs = processor(images=images, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise KeyError("CLIP processor output does not contain pixel_values")
        pixel_values = inputs["pixel_values"].to(selected_device)

        with torch.no_grad():
            outputs = clip_model.get_image_features(pixel_values=pixel_values)
            features = _get_feature_tensor(outputs)

        features = features.detach().cpu()
        feature_batches.append(features)
        processed_image_ids.extend(batch_ids)

        if show_progress:
            processed = min(start_index + batch_size, len(image_ids))
            print(f"Processed {processed}/{len(image_ids)} images")

    all_features = torch.cat(feature_batches, dim=0)
    expected_feature_dim = int(clip_model.config.projection_dim)
    _validate_features(
        all_features,
        processed_image_ids,
        expected_feature_dim,
    )

    output_directory = Path(feature_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "clip_features.pt"
    feature_data = {
        "image_ids": processed_image_ids,
        "features": all_features,
        "clip_model": model_name,
        "feature_dim": all_features.shape[1],
    }
    torch.save(feature_data, output_path)

    if show_progress:
        print(f"Feature shape: {all_features.shape}")
        print(f"Saved CLIP features to: {output_path}")

    return feature_data


def run_clip_feature_extraction(
    dataset_path: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run feature extraction with paths and settings from project config."""
    setup_clipcap_directories

    if dataset_path is None:
        resolved_dataset_path = Path(
            kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        )
    else:
        resolved_dataset_path = Path(dataset_path)

    return extract_clip_features(
        split_dir=SPLIT_DIR,
        images_dir=resolved_dataset_path / "Images",
        feature_dir=FEATURE_DIR,
        model_name=CLIP_MODEL_NAME,
        batch_size=CLIP_BATCH_SIZE,
        show_progress=show_progress,
    )


def _get_feature_tensor(outputs: Any) -> Any:
    """Support CLIP versions returning either model output or a tensor."""
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    return outputs


def _validate_features(
    features: Any,
    image_ids: Sequence[str],
    expected_feature_dim: int,
) -> None:
    if features.shape[0] != len(image_ids):
        raise ValueError("Feature count does not match image ID count")
    if features.shape[1] != expected_feature_dim:
        raise ValueError(
            "Unexpected feature dimension: "
            f"{features.shape[1]} != {expected_feature_dim}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("CLIP features contain NaN or Inf values")


def main() -> None:
    """Command-line entry point for the complete CLIP extraction stage."""
    run_clip_feature_extraction()


if __name__ == "__main__":
    main()
