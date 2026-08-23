from __future__ import annotations

import torch
from torch import Tensor, nn


class PrefixTransformerEncoder(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        positive_int_params = {
            "prefix_length": prefix_length,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
        }
        for name, value in positive_int_params.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if feedforward_dim is None:
            feedforward_dim = 4 * d_model
        elif (
            isinstance(feedforward_dim, bool)
            or not isinstance(feedforward_dim, int)
            or feedforward_dim < 1
        ):
            raise ValueError("feedforward_dim must be a positive integer")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be in the interval [0, 1)")

        self.prefix_length = prefix_length
        self.d_model = d_model
        self.feedforward_dim = feedforward_dim

        # Learnable prefix queries
        self.prefix_const = nn.Parameter(
            torch.empty(prefix_length, d_model)
        )
        nn.init.normal_(self.prefix_const, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, image_tokens: Tensor) -> Tensor:
        if image_tokens.ndim != 3:
            raise ValueError(
                "image_tokens must have shape "
                "[batch_size, clip_length, d_model]"
            )
        if image_tokens.shape[2] != self.d_model:
            raise ValueError(
                "Unexpected image-token dimension: "
                f"{image_tokens.shape[2]} != {self.d_model}"
            )

        batch_size = image_tokens.shape[0]

        prefix_queries = self.prefix_const.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        concat_sequence = torch.cat([image_tokens, prefix_queries], dim=1)
        return self.transformer_encoder(concat_sequence)
