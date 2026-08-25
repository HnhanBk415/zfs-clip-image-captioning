import torch
from transformers import AutoConfig

from src.config.clipcap_config import GPT2_MODEL_NAME
from src.clipcap.preprocessing.clipcap_dataset import create_dataloaders
from src.clipcap.models.mapping_network.transformer_mapper import (
    TransformerMapper,
)


CLIP_LENGTH = 10
PREFIX_LENGTH = 10


def test_preprocessing_output_shapes():
    loaders = create_dataloaders(
        batch_size=4,
        train_filename="train_100pct.pt"
    )

    batch = next(iter(loaders["train"]))

    assert set(batch) == {
        "image_embed",
        "input_ids",
        "attention_mask",
    }

    assert batch["image_embed"].ndim == 2
    assert batch["image_embed"].shape[0] == 4

    assert (
        batch["input_ids"].shape
        == batch["attention_mask"].shape
    )

    assert torch.isfinite(
        batch["image_embed"]
    ).all()


def test_preprocessing_to_mapper():
    loaders = create_dataloaders(
        batch_size=4,
        train_filename="train_100pct.pt"
    )

    batch = next(iter(loaders["train"]))

    clip_dim = int(batch["image_embed"].shape[1])
    language_model_config = AutoConfig.from_pretrained(
        GPT2_MODEL_NAME
    )
    embedding_dim = int(language_model_config.hidden_size)

    mapper = TransformerMapper(
        clip_dim = clip_dim,
        embedding_dim = embedding_dim,
        clip_length = CLIP_LENGTH,
        prefix_length = PREFIX_LENGTH,
        num_layers = 4,
        num_heads = 8,
    )

    prefix = mapper(
        batch["image_embed"]
    )

    assert prefix.shape == (
        4,
        PREFIX_LENGTH,
        embedding_dim,
    )
