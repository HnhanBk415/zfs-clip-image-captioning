"""Tests for ClipCap mapping networks."""

import pytest
import torch

from src.clipcap.models.mapping_network.prefix_encoder import (
    PrefixTransformerEncoder,
)
from src.clipcap.models.mapping_network.transformer_mapper import (
    TransformerMapper,
)


TEST_CLIP_DIM = 32
TEST_EMBEDDING_DIM = 48
TEST_CLIP_LENGTH = 3
TEST_PREFIX_LENGTH = 5


def create_mapper(device="cpu", **overrides):
    configuration = {
        "clip_dim": TEST_CLIP_DIM,
        "embedding_dim": TEST_EMBEDDING_DIM,
        "clip_length": TEST_CLIP_LENGTH,
        "prefix_length": TEST_PREFIX_LENGTH,
        "num_layers": 2,
        "num_heads": 6,
        "dropout": 0.1,
    }
    configuration.update(overrides)
    return TransformerMapper(**configuration).to(device)


@pytest.mark.parametrize("batch_size", (1, 4, 32))
def test_mapper_shape_for_multiple_batch_sizes(batch_size):
    mapper = create_mapper().eval()
    clip_features = torch.randn(batch_size, TEST_CLIP_DIM)

    with torch.no_grad():
        prefix_embeddings = mapper(clip_features)

    assert prefix_embeddings.shape == (
        batch_size,
        TEST_PREFIX_LENGTH,
        TEST_EMBEDDING_DIM,
    )
    assert torch.isfinite(prefix_embeddings).all()


def test_eval_output_is_deterministic():
    mapper = create_mapper().eval()
    clip_features = torch.randn(4, TEST_CLIP_DIM)

    with torch.no_grad():
        first_output = mapper(clip_features)
        second_output = mapper(clip_features)

    assert torch.equal(first_output, second_output)


def test_gradient_flow():
    mapper = create_mapper().train()
    mapper.zero_grad(set_to_none=True)

    clip_features = torch.randn(4, TEST_CLIP_DIM)
    prefix_embeddings = mapper(clip_features)
    loss = (prefix_embeddings * torch.randn_like(prefix_embeddings)).mean()
    loss.backward()

    projection_grad = mapper.clip_projection.projection.weight.grad
    prefix_grad = mapper.transformer.prefix_const.grad
    transformer_grads = [
        parameter.grad
        for parameter in mapper.transformer.transformer_encoder.parameters()
        if parameter.requires_grad
    ]

    for gradient in (projection_grad, prefix_grad, *transformer_grads):
        assert gradient is not None
        assert torch.isfinite(gradient).all()

    assert projection_grad.abs().sum() > 0
    assert prefix_grad.abs().sum() > 0
    assert any(gradient.abs().sum() > 0 for gradient in transformer_grads)


def test_parameter_count_matches_module_parameters():
    mapper = create_mapper()
    stats = mapper.count_parameters()
    expected_total = sum(
        parameter.numel()
        for parameter in mapper.parameters()
        if parameter.requires_grad
    )

    assert stats["projection_parameters"] > 0
    assert stats["transformer_parameters"] > 0
    assert stats["total_parameters"] == expected_total


def test_invalid_clip_dimension_rejected():
    mapper = create_mapper()

    with pytest.raises(ValueError, match="Unexpected CLIP feature dimension"):
        mapper(torch.randn(4, TEST_CLIP_DIM - 1))


@pytest.mark.parametrize(
    "overrides",
    (
        {"clip_dim": 0},
        {"embedding_dim": 0},
        {"clip_length": 0},
        {"prefix_length": 0},
        {"num_layers": 0},
        {"num_heads": 0},
        {"embedding_dim": 47, "num_heads": 6},
        {"feedforward_dim": 0},
        {"feedforward_dim": -1},
        {"feedforward_dim": True},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": "invalid"},
    ),
)
def test_invalid_mapper_configuration_rejected(overrides):
    with pytest.raises(ValueError):
        create_mapper(**overrides)


def test_prefix_encoder_validates_public_interface():
    with pytest.raises(ValueError):
        PrefixTransformerEncoder(
            prefix_length=0,
            d_model=TEST_EMBEDDING_DIM,
            nhead=6,
            num_layers=2,
        )

    encoder = PrefixTransformerEncoder(
        prefix_length=TEST_PREFIX_LENGTH,
        d_model=TEST_EMBEDDING_DIM,
        nhead=6,
        num_layers=2,
    )
    with pytest.raises(ValueError, match="image_tokens must have shape"):
        encoder(torch.randn(4, TEST_EMBEDDING_DIM))
    with pytest.raises(ValueError, match="Unexpected image-token dimension"):
        encoder(torch.randn(4, TEST_CLIP_LENGTH, TEST_EMBEDDING_DIM - 1))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_execution():
    mapper = create_mapper(device="cuda")
    clip_features = torch.randn(4, TEST_CLIP_DIM, device="cuda")
    prefix_embeddings = mapper(clip_features)

    assert prefix_embeddings.shape == (
        4,
        TEST_PREFIX_LENGTH,
        TEST_EMBEDDING_DIM,
    )
    assert prefix_embeddings.device.type == "cuda"
