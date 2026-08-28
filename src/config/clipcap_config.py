from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.common_config import (
    DATA_ROOT,
    PROJECT_ROOT,
    SEED,
    SUBSET_NAMES,
)


# Paths
FEATURE_DIR = DATA_ROOT / "features"
TOKENIZED_DIR = DATA_ROOT / "tokenized"
CLIPCAP_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "clipcap"

# Pretrained models and preprocessing
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 32
GPT2_MODEL_NAME = "openai-community/gpt2"
GPT2_MAX_LENGTH = 48

# Mapper architecture
CLIPCAP_MAPPER_CLIP_LENGTH = 10
CLIPCAP_MAPPER_PREFIX_LENGTH = 10
CLIPCAP_MAPPER_NUM_LAYERS = 4
CLIPCAP_MAPPER_NUM_HEADS = 8
CLIPCAP_MAPPER_FEEDFORWARD_DIM: int | None = None
CLIPCAP_MAPPER_FEEDFORWARD_MULTIPLIER = 4
CLIPCAP_MAPPER_DROPOUT = 0.1
CLIPCAP_PREFIX_INIT_MEAN = 0.0
CLIPCAP_PREFIX_INIT_STD = 0.02

# Subset experiments and training
CLIPCAP_TRAIN_SUBSETS = tuple(SUBSET_NAMES)
CLIPCAP_DEFAULT_TRAIN_SUBSET = CLIPCAP_TRAIN_SUBSETS[0]
CLIPCAP_TRAIN_SEED = SEED
CLIPCAP_TRAIN_BATCH_SIZE = 32
CLIPCAP_TRAIN_NUM_WORKERS = 2
CLIPCAP_LEARNING_RATE = 2e-5
CLIPCAP_WEIGHT_DECAY = 0.01
CLIPCAP_ADAM_BETAS = (0.9, 0.999)
CLIPCAP_ADAM_EPS = 1e-8
CLIPCAP_MAX_EPOCHS = 50
CLIPCAP_EARLY_STOPPING_PATIENCE = 5
CLIPCAP_EARLY_STOPPING_MIN_DELTA = 0.0
CLIPCAP_MAX_GRAD_NORM = 1.0
CLIPCAP_EARLY_STOPPING_POLICY = "early_stopping"
CLIPCAP_FIXED_EPOCH_POLICY = "fixed_epoch"
CLIPCAP_TRAINING_POLICIES = (
    CLIPCAP_EARLY_STOPPING_POLICY,
    CLIPCAP_FIXED_EPOCH_POLICY,
)
CLIPCAP_FIXED_EPOCHS = 7


@dataclass(frozen=True)
class ClipCapTrainingConfig:
    """Configuration snapshot for one independent ClipCap experiment."""

    subset_name: str = CLIPCAP_DEFAULT_TRAIN_SUBSET
    seed: int = CLIPCAP_TRAIN_SEED
    batch_size: int = CLIPCAP_TRAIN_BATCH_SIZE
    num_workers: int = CLIPCAP_TRAIN_NUM_WORKERS
    learning_rate: float = CLIPCAP_LEARNING_RATE
    weight_decay: float = CLIPCAP_WEIGHT_DECAY
    adam_betas: tuple[float, float] = CLIPCAP_ADAM_BETAS
    adam_eps: float = CLIPCAP_ADAM_EPS
    max_epochs: int = CLIPCAP_MAX_EPOCHS
    early_stopping_patience: int = CLIPCAP_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = CLIPCAP_EARLY_STOPPING_MIN_DELTA
    max_grad_norm: float = CLIPCAP_MAX_GRAD_NORM
    training_policy: str = CLIPCAP_EARLY_STOPPING_POLICY
    clip_length: int = CLIPCAP_MAPPER_CLIP_LENGTH
    prefix_length: int = CLIPCAP_MAPPER_PREFIX_LENGTH
    num_layers: int = CLIPCAP_MAPPER_NUM_LAYERS
    num_heads: int = CLIPCAP_MAPPER_NUM_HEADS
    feedforward_dim: int | None = CLIPCAP_MAPPER_FEEDFORWARD_DIM
    dropout: float = CLIPCAP_MAPPER_DROPOUT
    output_root: Path = CLIPCAP_OUTPUT_ROOT

    def __post_init__(self) -> None:
        if self.subset_name not in CLIPCAP_TRAIN_SUBSETS:
            supported = ", ".join(CLIPCAP_TRAIN_SUBSETS)
            raise ValueError(
                f"Unsupported subset_name '{self.subset_name}'. "
                f"Expected one of: {supported}"
            )
        if self.training_policy not in CLIPCAP_TRAINING_POLICIES:
            supported = ", ".join(CLIPCAP_TRAINING_POLICIES)
            raise ValueError(
                f"Unsupported training_policy '{self.training_policy}'. "
                f"Expected one of: {supported}"
            )

        positive_integers = {
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "clip_length": self.clip_length,
            "prefix_length": self.prefix_length,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        positive_floats = {
            "learning_rate": self.learning_rate,
            "adam_eps": self.adam_eps,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive_floats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive number")
            if value <= 0:
                raise ValueError(f"{name} must be a positive number")

        non_negative_floats = {
            "weight_decay": self.weight_decay,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }
        for name, value in non_negative_floats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a non-negative number")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative number")

        if not isinstance(self.adam_betas, tuple) or len(self.adam_betas) != 2:
            raise ValueError("adam_betas must contain exactly two values")
        for beta in self.adam_betas:
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not 0.0 <= beta < 1.0
            ):
                raise ValueError("adam_betas values must be in the interval [0, 1)")

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in the interval [0, 1)")
        if self.feedforward_dim is not None and (
            isinstance(self.feedforward_dim, bool)
            or not isinstance(self.feedforward_dim, int)
            or self.feedforward_dim < 1
        ):
            raise ValueError("feedforward_dim must be a positive integer or None")

        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def train_filename(self) -> str:
        return f"{self.subset_name}.pt"

    @property
    def experiment_dir(self) -> Path:
        return self.output_root / self.subset_name / f"seed_{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["adam_betas"] = list(self.adam_betas)
        data["output_root"] = str(self.output_root)
        return data


def create_clipcap_fixed_epoch_config(
    *,
    output_root: str | Path = CLIPCAP_OUTPUT_ROOT,
) -> ClipCapTrainingConfig:
    """Create the centrally configured fixed-epoch experiment snapshot."""
    return ClipCapTrainingConfig(
        max_epochs=CLIPCAP_FIXED_EPOCHS,
        training_policy=CLIPCAP_FIXED_EPOCH_POLICY,
        output_root=Path(output_root),
    )


def setup_clipcap_directories() -> None:
    directories = (
        FEATURE_DIR,
        TOKENIZED_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
