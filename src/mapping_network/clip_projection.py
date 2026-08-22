# CLIP Linear Projection to Image Tokens
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoConfig

import config as project_config

def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "config.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project config.py")


PROJECT_ROOT = find_project_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

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