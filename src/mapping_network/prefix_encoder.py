import torch
import torch.nn as nn

import torch
import torch.nn as nn


class PrefixTransformerEncoder(nn.Module):
    def __init__(
        self,
        prefix_length=10,
        d_model=768,
        nhead=8,
        num_layers=4,
        feedforward_dim=None,
        dropout=0.1
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.d_model = d_model

        # Nếu không truyền thì mặc định 4 * d_model
        if feedforward_dim is None:
            feedforward_dim = d_model * 4

        # Learnable prefix queries
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, image_tokens):
        # image_tokens: [B, 10, 768]
        batch_size = image_tokens.shape[0]

        # [10,768] -> [B,10,768]
        prefix_queries = (
            self.prefix_const
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # [B,10,768] + [B,10,768]
        # -> [B,20,768]
        concat_sequence = torch.cat(
            [image_tokens, prefix_queries],
            dim=1
        )

        encoded_sequence = self.transformer_encoder(
            concat_sequence
        )

        return encoded_sequence
        
    def forward(self, image_tokens): # Tensor[B, 10, 768] -> Tensor[B, 20, 768]
        batch_size = image_tokens.shape[0]

        # Expand prefix queries [10, 768] -> [B, 10, 768]
        prefix_queries = self.prefix_const.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Image Tokens ([B, 10, 768]) + Prefix Queries ([B, 10, 768]) -> [B, 20, 768]
        concat_sequence = torch.cat([image_tokens, prefix_queries], dim=1)
        encoded_sequence = self.transformer_encoder(concat_sequence)
        return encoded_sequence