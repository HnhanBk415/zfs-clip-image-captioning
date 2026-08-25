from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config.clipcap_config import (
    CLIPCAP_TRAIN_BATCH_SIZE,
    CLIPCAP_TRAIN_NUM_WORKERS,
    CLIPCAP_TRAIN_SEED,
    FEATURE_DIR,
    TOKENIZED_DIR,
)


def _load_torch_data(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class ClipCapDataset(Dataset):
    def __init__(
        self,
        tokenized_path: str | Path,
        clip_feature_data: dict[str, Any],
        *,
        show_alignment: bool = True,
    ) -> None:
        self.tokenized_data = _load_torch_data(tokenized_path)
        self.image_ids = self.tokenized_data["image_ids"]
        self.input_ids = self.tokenized_data["input_ids"]
        self.attention_mask = self.tokenized_data["attention_mask"]

        self.clip_features = clip_feature_data["features"]
        self.img_id_to_idx = {
            img_id: idx
            for idx, img_id in enumerate(clip_feature_data["image_ids"])
        }

        token_image_ids = set(self.image_ids)
        clip_image_ids = set(clip_feature_data["image_ids"])

        if len(clip_feature_data["image_ids"]) != len(clip_image_ids):
            raise ValueError("Duplicate image_id in CLIP features")

        missing_clip = token_image_ids - clip_image_ids
        if missing_clip:
            raise ValueError(
                f"Missing CLIP features for {len(missing_clip)} images. "
                f"Examples: {list(missing_clip)[:5]}"
            )

        if show_alignment:
            print(
                "Alignment PASS: "
                f"{len(token_image_ids)} images <-> "
                f"{len(self.image_ids)} caption samples"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_id = self.image_ids[idx]
        feature_idx = self.img_id_to_idx[img_id]
        image_embed = self.clip_features[feature_idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        return {
            "image_embed": image_embed,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def create_dataloaders(
    batch_size: int = CLIPCAP_TRAIN_BATCH_SIZE,
    train_filename: str = "train.pt",
    *,
    num_workers: int = CLIPCAP_TRAIN_NUM_WORKERS,
    pin_memory: bool = False,
    seed: int = CLIPCAP_TRAIN_SEED,
    include_test: bool = True,
) -> dict[str, DataLoader]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    clip_feature_path = FEATURE_DIR / "clip_features.pt"
    clip_feature_data = _load_torch_data(clip_feature_path)

    train_path = TOKENIZED_DIR / train_filename
    train_dataset = ClipCapDataset(train_path, clip_feature_data)
    val_dataset = ClipCapDataset(TOKENIZED_DIR / "val.pt", clip_feature_data)

    train_ids = set(train_dataset.image_ids)
    val_ids = set(val_dataset.image_ids)
    if not train_ids.isdisjoint(val_ids):
        raise ValueError("Data leakage: Train and validation share images")

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        ),
        "val": DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        ),
    }

    if include_test:
        test_dataset = ClipCapDataset(
            TOKENIZED_DIR / "test.pt",
            clip_feature_data,
        )
        test_ids = set(test_dataset.image_ids)
        if not train_ids.isdisjoint(test_ids):
            raise ValueError("Data leakage: Train and test share images")
        if not val_ids.isdisjoint(test_ids):
            raise ValueError("Data leakage: Validation and test share images")
        dataloaders["test"] = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_options,
        )

    return dataloaders
