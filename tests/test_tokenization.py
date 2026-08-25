import torch

from src.clipcap.preprocessing.tokenization import (
    build_tokenized_data,
    tokenize_samples_batch,
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
