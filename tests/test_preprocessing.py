import torch

from src.preprocessing.clipcap_dataset import create_dataloaders
from src.mapping_network.transformer_mapper import TransformerMapper


def test_preprocessing_output_shapes():
    loaders = create_dataloaders(
        batch_size=4,
        train_filename="train_100pct.pt"
    )

    batch = next(iter(loaders["train"]))

    assert batch["image_embed"].shape == (4, 512)

    assert (
        batch["input_ids"].shape
        == batch["attention_mask"].shape
    )

    assert (
        batch["input_ids"].shape
        == batch["labels"].shape
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

    mapper = TransformerMapper(
        clip_dim=512,
        embedding_dim=768,
        clip_length=10,
        prefix_length=10,
        num_layers=4,
        num_heads=8,
    )

    prefix = mapper(
        batch["image_embed"]
    )

    assert prefix.shape == (
        4,
        10,
        768
    )