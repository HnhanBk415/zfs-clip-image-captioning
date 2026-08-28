import torch
import pytest
from transformers import AutoConfig

import src.common.preprocessing.data_preparation as data_preparation

from src.config.clipcap_config import (
    CLIPCAP_MAPPER_CLIP_LENGTH,
    CLIPCAP_MAPPER_NUM_HEADS,
    CLIPCAP_MAPPER_NUM_LAYERS,
    CLIPCAP_MAPPER_PREFIX_LENGTH,
    GPT2_MODEL_NAME,
)
from src.clipcap.preprocessing.clipcap_dataset import create_dataloaders
from src.clipcap.models.mapping_network.transformer_mapper import (
    TransformerMapper,
)


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
        clip_length = CLIPCAP_MAPPER_CLIP_LENGTH,
        prefix_length = CLIPCAP_MAPPER_PREFIX_LENGTH,
        num_layers = CLIPCAP_MAPPER_NUM_LAYERS,
        num_heads = CLIPCAP_MAPPER_NUM_HEADS,
    )

    prefix = mapper(
        batch["image_embed"]
    )

    assert prefix.shape == (
        4,
        CLIPCAP_MAPPER_PREFIX_LENGTH,
        embedding_dim,
    )


def test_data_preparation_main_calls_directory_setup(monkeypatch):
    setup_calls = []

    def stop_after_setup():
        raise RuntimeError("stop after setup")

    monkeypatch.setattr(
        data_preparation,
        "setup_common_directories",
        lambda: setup_calls.append(True),
    )
    monkeypatch.setattr(
        data_preparation,
        "load_raw_dataset",
        stop_after_setup,
    )

    with pytest.raises(RuntimeError, match="stop after setup"):
        data_preparation.main()

    assert setup_calls == [True]
