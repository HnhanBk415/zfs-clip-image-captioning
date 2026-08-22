import pytest
import torch

from src.mapping_network.transformer_mapper import TransformerMapper


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_mapper():
    return TransformerMapper(
        clip_dim=512,
        embedding_dim=768,
        clip_length=10,
        prefix_length=10,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    ).to(DEVICE)


def test_mapper_shape():
    mapper = create_mapper().eval()

    x = torch.randn(
        4,
        512,
        device=DEVICE
    )

    with torch.no_grad():
        y = mapper(x)

    assert y.shape == (4, 10, 768)


def test_multiple_batch_sizes():
    mapper = create_mapper().eval()

    with torch.no_grad():

        for batch_size in (1, 4, 32):

            x = torch.randn(
                batch_size,
                512,
                device=DEVICE
            )

            y = mapper(x)

            assert y.shape == (
                batch_size,
                10,
                768
            )


def test_output_is_finite():
    mapper = create_mapper().eval()

    x = torch.randn(
        4,
        512,
        device=DEVICE
    )

    with torch.no_grad():
        y = mapper(x)

    assert torch.isfinite(y).all()


def test_gradient_flow():
    mapper = create_mapper().train()

    mapper.zero_grad(
        set_to_none=True
    )

    x = torch.randn(
        4,
        512,
        device=DEVICE
    )

    y = mapper(x)

    probe = torch.randn_like(y)

    loss = (
        y * probe
    ).mean()

    loss.backward()

    projection_grad = (
        mapper
        .clip_projection
        .projection
        .weight
        .grad
    )

    prefix_grad = (
        mapper
        .transformer
        .prefix_const
        .grad
    )

    transformer_has_grad = any(
        p.grad is not None
        for p in (
            mapper
            .transformer
            .transformer_encoder
            .parameters()
        )
        if p.requires_grad
    )

    assert projection_grad is not None
    assert torch.isfinite(
        projection_grad
    ).all()

    assert prefix_grad is not None
    assert torch.isfinite(
        prefix_grad
    ).all()

    assert transformer_has_grad


def test_parameter_count():
    mapper = create_mapper()

    stats = mapper.count_parameters()

    assert (
        stats["projection_parameters"]
        > 0
    )

    assert (
        stats["transformer_parameters"]
        > 0
    )

    assert stats["total_parameters"] == (
        stats["projection_parameters"]
        + stats["transformer_parameters"]
    )


def test_invalid_clip_dimension_rejected():
    mapper = create_mapper()

    x = torch.randn(
        4,
        511,
        device=DEVICE
    )

    with pytest.raises(ValueError):
        mapper(x)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available"
)
def test_cuda_execution():
    mapper = create_mapper()

    x = torch.randn(
        4,
        512,
        device="cuda"
    )

    y = mapper(x)

    assert y.is_cuda
    assert y.device.type == "cuda"