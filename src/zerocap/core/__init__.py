"""Public ZeroCap captioning facade and shared data types."""

from .captioner import ZeroCapCaptioner
from .types import GenerationResult

__all__ = ["GenerationResult", "ZeroCapCaptioner"]
