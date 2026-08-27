from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BeamCandidate:
    beam_rank: int
    caption: str
    token_ids: tuple[int, ...]
    beam_score: float
    clip_score: float | None = None


@dataclass(frozen=True)
class GenerationResult:
    caption: str
    selected_beam_rank: int
    candidates: tuple[BeamCandidate, ...]


@dataclass(frozen=True)
class FixedEpochCheckpoint:
    subset_name: str
    seed: int
    directory: Path
    config: dict[str, Any]
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class EvaluationItem:
    image_id: str
    references: tuple[str, ...]
