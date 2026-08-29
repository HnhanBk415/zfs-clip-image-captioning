from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import CLIPModel, GPT2LMHeadModel

from src.clipcap.models.mapping_network import TransformerMapper
from src.config.clipcap_config import CLIPCAP_FIXED_EPOCH_POLICY

from .records import FixedEpochCheckpoint


CHECKPOINT_VERSION = 1
FINAL_CHECKPOINT_NAME = "final.pt"
REQUIRED_ARTIFACTS = (
    "best.pt",
    "latest.pt",
    FINAL_CHECKPOINT_NAME,
    "config.json",
    "history.json",
    "result.json",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def validate_artifact_directory(directory: str | Path) -> dict[str, Path]:
    selected_directory = Path(directory)
    artifacts = {
        name: selected_directory / name
        for name in REQUIRED_ARTIFACTS
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing ClipCap artifacts in {selected_directory}: {', '.join(missing)}"
        )
    return artifacts


def load_fixed_epoch_checkpoint(
    directory: str | Path,
    subset_name: str,
    seed: int,
) -> FixedEpochCheckpoint:
    artifacts = validate_artifact_directory(directory)
    saved_config = _load_json_object(artifacts["config.json"])
    checkpoint = torch.load(
        artifacts[FINAL_CHECKPOINT_NAME],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint payload: {artifacts[FINAL_CHECKPOINT_NAME]}")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Only checkpoint_version=1 is supported")
    if checkpoint.get("checkpoint_type") != "final":
        raise ValueError("Fixed-epoch evaluation requires final.pt")
    if checkpoint.get("config") != saved_config:
        raise ValueError("config.json does not match the config stored in final.pt")
    if saved_config.get("subset_name") != subset_name:
        raise ValueError(
            f"Expected subset {subset_name}, got {saved_config.get('subset_name')}"
        )
    if saved_config.get("seed") != seed:
        raise ValueError(f"Expected seed {seed}, got {saved_config.get('seed')}")
    if saved_config.get("training_policy") != CLIPCAP_FIXED_EPOCH_POLICY:
        raise ValueError("Checkpoint was not trained with the fixed_epoch policy")

    state = checkpoint.get("state")
    if not isinstance(state, dict):
        raise TypeError("final.pt does not contain a valid training state")
    if state.get("last_epoch") != saved_config.get("max_epochs"):
        raise ValueError("final.pt is not from the configured final epoch")
    mapper_state = checkpoint.get("mapper_state_dict")
    if not isinstance(mapper_state, dict) or not mapper_state:
        raise TypeError("final.pt does not contain a valid mapper_state_dict")
    if not checkpoint.get("gpt2_model_name"):
        raise KeyError("final.pt does not contain gpt2_model_name")

    return FixedEpochCheckpoint(
        subset_name=subset_name,
        seed=seed,
        directory=Path(directory),
        config=saved_config,
        checkpoint=checkpoint,
    )


def resolve_fixed_epoch_checkpoints(
    checkpoint_root: str | Path,
    subset_names: Sequence[str],
    seed: int,
) -> tuple[FixedEpochCheckpoint, ...]:
    root = Path(checkpoint_root)
    resolved = tuple(
        load_fixed_epoch_checkpoint(
            root / subset_name / f"seed_{seed}",
            subset_name,
            seed,
        )
        for subset_name in subset_names
    )
    gpt2_names = {
        item.checkpoint["gpt2_model_name"]
        for item in resolved
    }
    if len(gpt2_names) != 1:
        raise ValueError("All subset checkpoints must use the same GPT-2 model")
    return resolved


def build_mapper_from_checkpoint(
    artifact: FixedEpochCheckpoint,
    gpt2: GPT2LMHeadModel,
    clip_encoder: CLIPModel,
    device: torch.device,
) -> TransformerMapper:
    mapper_state = artifact.checkpoint["mapper_state_dict"]
    projection_key = "clip_projection.projection.weight"
    if projection_key not in mapper_state:
        raise KeyError(f"Mapper state is missing {projection_key}")

    clip_dim = int(mapper_state[projection_key].shape[1])
    clip_projection_dim = int(clip_encoder.config.projection_dim)
    if clip_projection_dim != clip_dim:
        raise ValueError(
            f"CLIP output dim {clip_projection_dim} does not match checkpoint dim {clip_dim}"
        )
    config = artifact.config
    mapper = TransformerMapper(
        clip_dim=clip_dim,
        embedding_dim=int(gpt2.get_input_embeddings().embedding_dim),
        clip_length=int(config["clip_length"]),
        prefix_length=int(config["prefix_length"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        feedforward_dim=config.get("feedforward_dim"),
        dropout=float(config["dropout"]),
    ).to(device)
    mapper.load_state_dict(mapper_state, strict=True)
    mapper.eval()
    return mapper
