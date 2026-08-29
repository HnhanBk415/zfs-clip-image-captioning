"""Tests for caption tokenization."""

import torch
import pytest

from src.clipcap.preprocessing.tokenization import (
    build_tokenized_data,
    tokenize_samples_batch,
    validate_split,
    verify_tokenized_invariants,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, raw_texts, **kwargs):
        assert raw_texts == ["first<eos>", "second<eos>"]
        assert kwargs["padding"] == "max_length"
        assert kwargs["return_tensors"] == "pt"
        return {
            "input_ids": torch.tensor(
                [[11, 12, 0, 0], [21, 22, 23, 0]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 0]],
                dtype=torch.long,
            ),
        }


def test_tokenization_does_not_create_labels():
    samples = [
        {"image_id": "image-1", "caption_idx": 0, "caption": "first"},
        {"image_id": "image-2", "caption_idx": 0, "caption": "second"},
    ]
    tokenizer = FakeTokenizer()

    encoded = tokenize_samples_batch(
        samples,
        tokenizer,
        max_length=4,
    )
    packed = build_tokenized_data(
        samples,
        encoded,
        tokenizer,
        max_length=4,
    )

    assert set(encoded) == {"input_ids", "attention_mask"}
    assert "labels" not in packed
    assert encoded["input_ids"][0, 1].item() == tokenizer.eos_token_id
    assert encoded["input_ids"][1, 2].item() == tokenizer.eos_token_id


def test_split_validation_raises_explicit_exceptions():
    with pytest.raises(TypeError, match="must be a dictionary"):
        validate_split("train", [])

    with pytest.raises(TypeError, match="non-list captions"):
        validate_split("train", {"image-1": "not-a-list"})

    with pytest.raises(ValueError, match="empty captions"):
        validate_split("train", {"image-1": ["  "]})


def test_tokenized_validation_rejects_precomputed_labels():
    tokenizer = FakeTokenizer()
    data = {
        "image_ids": ["image-1"],
        "caption_indices": [0],
        "captions": ["caption"],
        "input_ids": torch.tensor([[11, 99, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        "labels": torch.tensor([[11, 99, 0, 0]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="labels must be created"):
        verify_tokenized_invariants(
            "train",
            data,
            tokenizer,
            max_length=4,
        )
