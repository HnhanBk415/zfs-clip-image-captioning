"""Tests for the integrated ClipCap model."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.clipcap.models.clipcap_model import (
    ClipCaptionModel,
    build_extended_labels,
    extend_attention_mask,
)
from src.clipcap.models.mapping_network import TransformerMapper


BATCH_SIZE = 2
CLIP_DIM = 32
EMBEDDING_DIM = 48
CLIP_LENGTH = 3
PREFIX_LENGTH = 5
CAPTION_LENGTH = 6
VOCAB_SIZE = 128


def create_batch():
    return {
        "image_embed": torch.randn(BATCH_SIZE, CLIP_DIM),
        "input_ids": torch.tensor(
            [[11, 12, 13, 14, 0, 0], [21, 22, 23, 24, 25, 0]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
            dtype=torch.long,
        ),
    }


def create_model(*, freeze_gpt2=False):
    mapper = TransformerMapper(
        clip_dim=CLIP_DIM,
        embedding_dim=EMBEDDING_DIM,
        clip_length=CLIP_LENGTH,
        prefix_length=PREFIX_LENGTH,
        num_layers=2,
        num_heads=6,
        feedforward_dim=96,
        dropout=0.0,
    )
    gpt2 = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=PREFIX_LENGTH + CAPTION_LENGTH + 4,
            n_ctx=PREFIX_LENGTH + CAPTION_LENGTH + 4,
            n_embd=EMBEDDING_DIM,
            n_layer=2,
            n_head=6,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    gpt2.loss_type = "ForCausalLM"
    if freeze_gpt2:
        for parameter in gpt2.parameters():
            parameter.requires_grad = False
    return ClipCaptionModel(mapper=mapper, gpt2=gpt2)


def test_extended_attention_mask_and_labels():
    batch = create_batch()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    original_input_ids = input_ids.clone()

    extended_mask = extend_attention_mask(attention_mask, PREFIX_LENGTH)
    extended_labels = build_extended_labels(
        input_ids,
        attention_mask,
        PREFIX_LENGTH,
    )

    assert extended_mask.shape == (BATCH_SIZE, PREFIX_LENGTH + CAPTION_LENGTH)
    assert extended_labels.shape == extended_mask.shape
    assert torch.all(extended_mask[:, :PREFIX_LENGTH] == 1)
    assert torch.all(extended_labels[:, :PREFIX_LENGTH] == -100)
    assert torch.all(
        extended_labels[:, PREFIX_LENGTH:][attention_mask == 0] == -100
    )
    assert torch.equal(input_ids, original_input_ids)


def test_model_creates_labels_only_when_requested():
    model = create_model().eval()
    batch = create_batch()

    with_labels = model(
        **batch,
        labels=batch["input_ids"],
    )
    without_labels = model(**batch)

    assert with_labels.loss is not None
    assert torch.isfinite(with_labels.loss)
    assert without_labels.loss is None
    assert with_labels.logits.shape == (
        BATCH_SIZE,
        PREFIX_LENGTH + CAPTION_LENGTH,
        VOCAB_SIZE,
    )
    assert without_labels.logits.shape == with_labels.logits.shape


def test_frozen_gpt2_propagates_gradient_to_mapper():
    model = create_model(freeze_gpt2=True).train()
    batch = create_batch()
    model.zero_grad(set_to_none=True)

    outputs = model(
        **batch,
        labels=batch["input_ids"],
    )
    outputs.loss.backward()

    mapper_gradients = [
        parameter.grad
        for parameter in model.mapper.parameters()
        if parameter.requires_grad
    ]
    gpt2_gradients = [parameter.grad for parameter in model.gpt2.parameters()]

    assert mapper_gradients
    assert all(gradient is not None for gradient in mapper_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in mapper_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in mapper_gradients)
    assert all(gradient is None for gradient in gpt2_gradients)


@pytest.mark.parametrize(
    "call",
    (
        lambda model, batch: model(
            batch["image_embed"],
            batch["input_ids"],
            batch["attention_mask"][:, :-1],
        ),
        lambda model, batch: model(
            batch["image_embed"][:1],
            batch["input_ids"],
            batch["attention_mask"],
        ),
        lambda model, batch: model(
            batch["image_embed"],
            batch["input_ids"].float(),
            batch["attention_mask"],
        ),
    ),
)
def test_model_rejects_invalid_inputs(call):
    with pytest.raises((TypeError, ValueError)):
        call(create_model(), create_batch())
