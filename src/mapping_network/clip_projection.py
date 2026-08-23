from __future__ import annotations

from torch import Tensor, nn


class ClipProjection(nn.Module):
    """Project one global CLIP feature into a sequence of image tokens."""

    def __init__(
        self,
        clip_dim: int,
        embedding_dim: int,
        clip_length: int,
    ) -> None:
        super().__init__()

        dimensions = {
            "clip_dim": clip_dim,
            "embedding_dim": embedding_dim,
            "clip_length": clip_length,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.clip_dim = clip_dim
        self.embedding_dim = embedding_dim
        self.clip_length = clip_length
        self.projection = nn.Linear(
            in_features=clip_dim,
            out_features=clip_length * embedding_dim,
        )

    def forward(self, clip_features: Tensor) -> Tensor:
        if clip_features.ndim != 2:
            raise ValueError(
                "clip_features must have shape [batch_size, clip_dim]"
            )
        if clip_features.shape[1] != self.clip_dim:
            raise ValueError(
                "Unexpected CLIP feature dimension: "
                f"{clip_features.shape[1]} != {self.clip_dim}"
            )

        batch_size = clip_features.shape[0]
        projected_features = self.projection(clip_features)
        return projected_features.reshape(
            batch_size,
            self.clip_length,
            self.embedding_dim,
        )
