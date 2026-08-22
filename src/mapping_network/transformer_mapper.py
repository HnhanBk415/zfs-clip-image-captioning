from torch import Tensor, nn

from .clip_projection import ClipProjection
from .prefix_encoder import PrefixTransformerEncoder

class TransformerMapper(nn.Module):
    """
    Gộp toàn bộ pipeline:
      CLIP Features [B, clip_dim]
        -> ClipProjection -> Image Tokens [B, clip_length, embedding_dim]
        -> Transformer Encoder -> Encoded Sequence [B, clip_length + prefix_length, embedding_dim]
        -> Slicing K token cuối -> Soft Prefix [B, prefix_length, embedding_dim]
    """

    def __init__(
        self,
        clip_dim: int = 512,
        embedding_dim: int = 768,
        clip_length: int = 10,
        prefix_length: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- 1. Validation ---
        self._validate_configuration(
            clip_dim=clip_dim,
            embedding_dim=embedding_dim,
            clip_length=clip_length,
            prefix_length=prefix_length,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.clip_dim = clip_dim
        self.embedding_dim = embedding_dim
        self.clip_length = clip_length
        self.prefix_length = prefix_length
        self.feedforward_dim = feedforward_dim or (4 * embedding_dim)
        self.dropout = dropout

        # --- 2. Khởi tạo submodule ---
        self.clip_projection = ClipProjection(
            clip_dim=self.clip_dim,
            embedding_dim=self.embedding_dim,
            clip_length=self.clip_length,
        )

        self.transformer = PrefixTransformerEncoder(
            prefix_length=self.prefix_length,
            d_model=self.embedding_dim,
            nhead=num_heads,
            num_layers=num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )

    @staticmethod
    def _validate_configuration(
        clip_dim: int,
        embedding_dim: int,
        clip_length: int,
        prefix_length: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        positive_int_params = {
            "clip_dim": clip_dim,
            "embedding_dim": embedding_dim,
            "clip_length": clip_length,
            "prefix_length": prefix_length,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }
        for name, val in positive_int_params.items():
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"Tham số '{name}' phải là số nguyên > 0, nhận được: {val}")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) phải chia hết cho num_heads ({num_heads})"
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout phải nằm trong khoảng [0.0, 1.0), nhận được: {dropout}"
            )

    def forward(self, clip_features: Tensor) -> Tensor:
        # Bước 1: Project & reshape thành Image Tokens [B, clip_length, embedding_dim]
        image_tokens = self.clip_projection(clip_features)

        # Bước 2: Transformer Encoder [B, clip_length + prefix_length, embedding_dim]
        encoded_sequence = self.transformer(image_tokens)

        # Bước 3: Lấy đúng K prefix token cuối (không hard-code số 10)
        prefix_embeddings = encoded_sequence[:, self.clip_length :, :]
        return prefix_embeddings

    def count_parameters(self) -> dict[str, int]:
        """Thống kê chi tiết số lượng trainable parameters trong Mapper."""
        proj_params = sum(p.numel() for p in self.clip_projection.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total = proj_params + transformer_params
        return {
            "projection_parameters": proj_params,
            "transformer_parameters": transformer_params,
            "total_parameters": total,
        }