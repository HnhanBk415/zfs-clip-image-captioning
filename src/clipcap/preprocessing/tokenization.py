#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flickr8k GPT-2 Tokenization Pipeline

Responsibilities
----------------
1. Load frozen split manifests (train, val, test) and nested subsets.
2. Validate integrity of loaded splits.
3. Initialize GPT-2 tokenizer with EOS as PAD token.
4. Tokenize captions with deterministic EOS appending and fixed-length padding.
5. Verify tokenized output invariants (shapes, token bounds, EOS presence).
6. Export tokenized tensors (.pt) for PyTorch training and evaluation.

EDA/Exploration intentionally remains in the notebook:
    notebook/gpt2_tokenization.ipynb
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from src.config.common_config import (
    SPLIT_DIR,
    SUBSET_DIR,
    SUBSET_NAMES,
)
from src.config.clipcap_config import (
    GPT2_MAX_LENGTH,
    GPT2_MODEL_NAME,
    TOKENIZED_DIR,
    setup_clipcap_directories,
)


def load_json(path: Path) -> Dict[str, List[str]]:
    """Load JSON file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split(split_name: str, data: Dict[str, Any]) -> None:
    """Validate structure and caption content in a single pass."""
    assert isinstance(data, dict), f"{split_name} must be a dictionary."

    invalid_lists, invalid_types, empty_caps = [], [], []

    for img_id, caps in data.items():
        if not isinstance(caps, list):
            invalid_lists.append(img_id)
            continue
        for cap in caps:
            if not isinstance(cap, str):
                invalid_types.append((img_id, cap))
            elif not cap.strip():
                empty_caps.append((img_id, cap))

    assert not invalid_lists, f"{split_name} contains non-list captions: {len(invalid_lists)}"
    assert not invalid_types, f"{split_name} contains non-string captions: {len(invalid_types)}"
    assert not empty_caps, f"{split_name} contains empty captions: {len(empty_caps)}"


def flatten_split(data: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Flatten image-caption dictionary into individual sample records."""
    return [
        {"image_id": img_id, "caption_idx": idx, "caption": cap}
        for img_id, captions in data.items()
        for idx, cap in enumerate(captions)
    ]


def setup_tokenizer(model_name: str):
    """Load pretrained tokenizer and set pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_samples_batch(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Batch tokenization appending EOS and padding to max_length.
    Generates input_ids and attention_mask.

    Training labels are intentionally created by the ClipCap model only when
    loss computation is requested.
    """
    # Gắn sẵn EOS vào chuỗi để tận dụng C-Rust tokenizer tốc độ cao
    raw_texts = [s["caption"] + tokenizer.eos_token for s in samples]

    encoded = tokenizer(
        raw_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Đảm bảo token thực cuối cùng luôn là EOS nếu câu bị cắt (truncation)
    real_lengths = attention_mask.sum(dim=1)
    last_real_idx = real_lengths - 1
    row_idx = torch.arange(input_ids.shape[0])
    input_ids[row_idx, last_real_idx] = tokenizer.eos_token_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def build_tokenized_data(
    samples: List[Dict[str, Any]],
    encoded: Dict[str, torch.Tensor],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Pack token tensors and metadata into standard storage dict."""
    return {
        "image_ids": [sample["image_id"] for sample in samples],
        "caption_indices": [sample["caption_idx"] for sample in samples],
        "captions": [sample["caption"] for sample in samples],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "tokenizer_name": GPT2_MODEL_NAME,
        "max_length": max_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def verify_tokenized_invariants(
    split_name: str,
    data: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> None:
    """Run strict verification on dimensions, token bounds, and EOS positions."""
    num_samples = len(data["image_ids"])
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    assert "labels" not in data, (
        f"{split_name}: labels must be created by the model when computing loss"
    )
    assert len(data["caption_indices"]) == num_samples, f"{split_name}: caption_indices mismatch"
    assert len(data["captions"]) == num_samples, f"{split_name}: captions mismatch"
    assert input_ids.shape == (num_samples, max_length), f"{split_name}: input_ids shape mismatch"
    assert attention_mask.shape == (num_samples, max_length), f"{split_name}: attention_mask shape mismatch"

    real_lengths = attention_mask.sum(dim=1)
    last_real_indices = real_lengths - 1
    row_indices = torch.arange(input_ids.shape[0])
    last_real_tokens = input_ids[row_indices, last_real_indices]

    assert torch.all(
        last_real_tokens == tokenizer.eos_token_id
    ), f"{split_name}: Real sequence must terminate with EOS token."


def verify_nested_subsets(subset_tokenized: Dict[str, Dict[str, Any]]) -> None:
    """Verify strictly nested image hierarchy across generated subsets."""
    previous_ids = set()
    for subset_name in SUBSET_NAMES:
        current_ids = set(subset_tokenized[subset_name]["image_ids"])
        assert previous_ids.issubset(current_ids), f"Nested subset violation at {subset_name}"
        previous_ids = current_ids


def main():
    setup_clipcap_directories()
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLICKR8K GPT-2 TOKENIZATION PIPELINE")
    print("=" * 60)

    print("\n[1/4] Loading and validating split files...")
    splits = {
        "train": load_json(SPLIT_DIR / "train.json"),
        "val": load_json(SPLIT_DIR / "val.json"),
        "test": load_json(SPLIT_DIR / "test.json"),
    }

    for name, data in splits.items():
        validate_split(name, data)
        print(f"{name.capitalize():5}: {len(data):,} images | {sum(len(v) for v in data.values()):,} captions")

    print("\n[2/4] Initializing Tokenizer...")
    tokenizer = setup_tokenizer(GPT2_MODEL_NAME)
    print(f"Loaded: {GPT2_MODEL_NAME} | Max Length: {GPT2_MAX_LENGTH} | Vocab Size: {len(tokenizer):,}")

    print("\n[3/4] Tokenizing standard splits...")
    tokenized_splits = {}
    for name, data in splits.items():
        samples = flatten_split(data)
        encoded = tokenize_samples_batch(samples, tokenizer, GPT2_MAX_LENGTH)
        packed = build_tokenized_data(samples, encoded, tokenizer, GPT2_MAX_LENGTH)
        verify_tokenized_invariants(name, packed, tokenizer, GPT2_MAX_LENGTH)

        save_path = TOKENIZED_DIR / f"{name}.pt"
        torch.save(packed, save_path)
        tokenized_splits[name] = packed
        print(f"Saved {name}.pt | Tensor shape: {packed['input_ids'].shape}")

    print("\n[4/4] Tokenizing nested subsets...")
    subset_tokenized = {}
    for subset_name in SUBSET_NAMES:
        subset_path = SUBSET_DIR / f"{subset_name}.json"
        subset_data = load_json(subset_path)
        subset_samples = flatten_split(subset_data)
        subset_encoded = tokenize_samples_batch(subset_samples, tokenizer, GPT2_MAX_LENGTH)
        packed_subset = build_tokenized_data(subset_samples, subset_encoded, tokenizer, GPT2_MAX_LENGTH)
        verify_tokenized_invariants(subset_name, packed_subset, tokenizer, GPT2_MAX_LENGTH)

        save_path = TOKENIZED_DIR / f"{subset_name}.pt"
        torch.save(packed_subset, save_path)
        subset_tokenized[subset_name] = packed_subset
        print(f"Saved {subset_name}.pt | Samples: {len(subset_samples):,}")

    verify_nested_subsets(subset_tokenized)

    print("\n" + "=" * 60)
    print("ALL TOKENIZATION INVARIANTS PASSED")
    print(f"Outputs written to: {TOKENIZED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
