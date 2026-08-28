from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

from src.clipcap.models.clipcap_model import ClipCaptionModel
from src.clipcap.models.mapping_network import TransformerMapper
from src.clipcap.preprocessing.clipcap_dataset import create_dataloaders
from src.config.clipcap_config import (
    CLIPCAP_EARLY_STOPPING_POLICY,
    CLIPCAP_FIXED_EPOCH_POLICY,
    CLIPCAP_TRAIN_SUBSETS,
    GPT2_MODEL_NAME,
    ClipCapTrainingConfig,
)


CHECKPOINT_VERSION = 1
SUMMARY_FIELDS = (
    "subset_name",
    "seed",
    "training_policy",
    "train_images",
    "train_samples",
    "val_images",
    "val_samples",
    "best_epoch",
    "last_epoch",
    "optimizer_steps",
    "best_val_loss",
    "elapsed_seconds",
    "stopped_early",
    "best_checkpoint",
    "final_epoch",
    "final_checkpoint",
    "official_checkpoint",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clipcap_model(
    config: ClipCapTrainingConfig,
    clip_dim: int,
) -> ClipCaptionModel:
    """Create a fresh mapper and a frozen pretrained GPT-2."""
    gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)
    embedding_dim = int(gpt2.get_input_embeddings().embedding_dim)
    mapper = TransformerMapper(
        clip_dim=clip_dim,
        embedding_dim=embedding_dim,
        clip_length=config.clip_length,
        prefix_length=config.prefix_length,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    )
    for parameter in gpt2.parameters():
        parameter.requires_grad = False
    gpt2.eval()
    return ClipCaptionModel(mapper=mapper, gpt2=gpt2)


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    non_blocking = device.type == "cuda"
    return {
        name: tensor.to(device, non_blocking=non_blocking)
        for name, tensor in batch.items()
    }


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def train_one_epoch(
    model: ClipCaptionModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    max_grad_norm: float,
    *,
    description: str | None = None,
    show_progress: bool = True,
) -> float:
    """Train the mapper for one complete pass over a subset."""
    model.train()
    model.gpt2.eval()
    trainable_parameters = _trainable_parameters(model)
    total_loss = 0.0
    total_samples = 0

    batches: Iterable[Mapping[str, Tensor]] = dataloader
    if show_progress:
        batches = tqdm(dataloader, desc=description or "Train", leave=False)

    for batch in batches:
        batch_on_device = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            image_embed=batch_on_device["image_embed"],
            input_ids=batch_on_device["input_ids"],
            attention_mask=batch_on_device["attention_mask"],
            labels=batch_on_device["input_ids"],
        )
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Training loss is missing, NaN, or Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()

        batch_size = int(batch_on_device["input_ids"].size(0))
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader is empty")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: ClipCaptionModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    description: str | None = None,
    show_progress: bool = True,
) -> float:
    """Compute mean validation loss without changing model weights."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    batches: Iterable[Mapping[str, Tensor]] = dataloader
    if show_progress:
        batches = tqdm(dataloader, desc=description or "Validation", leave=False)

    for batch in batches:
        batch_on_device = _move_batch(batch, device)
        outputs = model(
            image_embed=batch_on_device["image_embed"],
            input_ids=batch_on_device["input_ids"],
            attention_mask=batch_on_device["attention_mask"],
            labels=batch_on_device["input_ids"],
        )
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is missing, NaN, or Inf")

        batch_size = int(batch_on_device["input_ids"].size(0))
        total_loss += float(loss) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader is empty")
    return total_loss / total_samples


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _latest_checkpoint_payload(
    model: ClipCaptionModel,
    optimizer: Optimizer,
    config: ClipCapTrainingConfig,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "latest",
        "mapper_state_dict": model.mapper.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gpt2_model_name": GPT2_MODEL_NAME,
        "config": config.to_dict(),
        "state": dict(state),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _best_checkpoint_payload(
    model: ClipCaptionModel,
    config: ClipCapTrainingConfig,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "best",
        "mapper_state_dict": model.mapper.state_dict(),
        "gpt2_model_name": GPT2_MODEL_NAME,
        "config": config.to_dict(),
        "state": dict(state),
    }


def _final_checkpoint_payload(
    model: ClipCaptionModel,
    config: ClipCapTrainingConfig,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "final",
        "mapper_state_dict": model.mapper.state_dict(),
        "gpt2_model_name": GPT2_MODEL_NAME,
        "config": config.to_dict(),
        "state": dict(state),
    }


def _normalize_saved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Treat checkpoints created before training policies as early-stopping."""
    normalized = dict(config)
    normalized.setdefault("training_policy", CLIPCAP_EARLY_STOPPING_POLICY)
    return normalized


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version in {path}")
    return checkpoint


def load_mapper_checkpoint(
    model: ClipCaptionModel,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore mapper weights from a best, latest, or final checkpoint."""
    selected_device = torch.device(device)
    checkpoint = _load_checkpoint(Path(checkpoint_path), selected_device)
    model.mapper.load_state_dict(checkpoint["mapper_state_dict"])
    return checkpoint


def _restore_latest_checkpoint(
    path: Path,
    model: ClipCaptionModel,
    optimizer: Optimizer,
    config: ClipCapTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(path, device)
    if checkpoint.get("checkpoint_type") != "latest":
        raise ValueError(f"Expected a latest checkpoint: {path}")
    saved_config = _normalize_saved_config(checkpoint.get("config", {}))
    if saved_config != config.to_dict():
        raise ValueError(
            "Checkpoint configuration differs from the current experiment. "
            "Use a different output_root or restore the original configuration."
        )

    model.mapper.load_state_dict(checkpoint["mapper_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    cuda_random_state = checkpoint.get("cuda_random_state")
    if cuda_random_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_random_state)
    return dict(checkpoint["state"])


def fit(
    model: ClipCaptionModel,
    dataloaders: Mapping[str, DataLoader],
    config: ClipCapTrainingConfig,
    device: torch.device,
    *,
    resume: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train one subset under its configured stopping policy."""
    if "train" not in dataloaders or "val" not in dataloaders:
        raise KeyError("dataloaders must contain train and val loaders")

    experiment_dir = config.experiment_dir
    latest_path = experiment_dir / "latest.pt"
    best_path = experiment_dir / "best.pt"
    final_path = experiment_dir / "final.pt"
    history_path = experiment_dir / "history.json"
    config_path = experiment_dir / "config.json"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    trainable_parameters = _trainable_parameters(model)
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_eps,
    )
    state: dict[str, Any] = {
        "last_epoch": 0,
        "best_epoch": 0,
        "best_val_loss": math.inf,
        "epochs_without_improvement": 0,
        "optimizer_steps": 0,
        "elapsed_seconds": 0.0,
        "stopped_early": False,
        "history": [],
    }

    if resume and latest_path.is_file():
        state = _restore_latest_checkpoint(
            latest_path,
            model,
            optimizer,
            config,
            device,
        )
        print(
            f"Resumed {config.subset_name} from epoch "
            f"{state['last_epoch']}"
        )

    _atomic_json_save(config.to_dict(), config_path)
    start_epoch = int(state["last_epoch"]) + 1
    stopped_early = (
        config.training_policy == CLIPCAP_EARLY_STOPPING_POLICY
        and bool(state.get("stopped_early", False))
    )
    previous_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
    started_at = time.perf_counter()
    epoch_numbers: Iterable[int]
    if stopped_early:
        epoch_numbers = ()
    else:
        epoch_numbers = range(start_epoch, config.max_epochs + 1)

    for epoch in epoch_numbers:
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            config.max_grad_norm,
            description=f"{config.subset_name} train {epoch}/{config.max_epochs}",
            show_progress=show_progress,
        )
        val_loss = evaluate(
            model,
            dataloaders["val"],
            device,
            description=f"{config.subset_name} val {epoch}/{config.max_epochs}",
            show_progress=show_progress,
        )

        state["last_epoch"] = epoch
        state["optimizer_steps"] += len(dataloaders["train"])
        state["elapsed_seconds"] = (
            previous_elapsed_seconds + time.perf_counter() - started_at
        )
        state["history"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "optimizer_steps": state["optimizer_steps"],
            }
        )

        improved = (
            val_loss
            < float(state["best_val_loss"])
            - config.early_stopping_min_delta
        )
        if improved:
            state["best_val_loss"] = val_loss
            state["best_epoch"] = epoch
            state["epochs_without_improvement"] = 0
            _atomic_torch_save(
                _best_checkpoint_payload(model, config, state),
                best_path,
            )
        else:
            state["epochs_without_improvement"] += 1

        stopped_early = (
            config.training_policy == CLIPCAP_EARLY_STOPPING_POLICY
            and state["epochs_without_improvement"]
            >= config.early_stopping_patience
        )
        state["stopped_early"] = stopped_early

        _atomic_torch_save(
            _latest_checkpoint_payload(model, optimizer, config, state),
            latest_path,
        )
        _atomic_json_save({"history": state["history"]}, history_path)

        print(
            f"[{config.subset_name}] epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_epoch={state['best_epoch']}"
        )

        if stopped_early:
            print(
                f"Early stopping {config.subset_name} after epoch {epoch}"
            )
            break

    elapsed_seconds = previous_elapsed_seconds + time.perf_counter() - started_at
    state["elapsed_seconds"] = elapsed_seconds
    final_checkpoint: str | None = None
    final_epoch: int | None = None
    if config.training_policy == CLIPCAP_FIXED_EPOCH_POLICY:
        if int(state["last_epoch"]) != config.max_epochs:
            raise RuntimeError(
                "Fixed-epoch training did not reach the configured final epoch"
            )
        _atomic_torch_save(
            _final_checkpoint_payload(model, config, state),
            final_path,
        )
        final_checkpoint = str(final_path)
        final_epoch = int(state["last_epoch"])

    train_dataset = dataloaders["train"].dataset
    val_dataset = dataloaders["val"].dataset
    official_checkpoint = (
        final_checkpoint
        if config.training_policy == CLIPCAP_FIXED_EPOCH_POLICY
        else str(best_path)
    )
    result = {
        "subset_name": config.subset_name,
        "seed": config.seed,
        "training_policy": config.training_policy,
        "train_images": len(set(train_dataset.image_ids)),
        "train_samples": len(train_dataset),
        "val_images": len(set(val_dataset.image_ids)),
        "val_samples": len(val_dataset),
        "best_epoch": int(state["best_epoch"]),
        "last_epoch": int(state["last_epoch"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "best_val_loss": float(state["best_val_loss"]),
        "elapsed_seconds": elapsed_seconds,
        "stopped_early": stopped_early,
        "best_checkpoint": str(best_path),
        "final_epoch": final_epoch,
        "final_checkpoint": final_checkpoint,
        "official_checkpoint": official_checkpoint,
        "history": state["history"],
    }
    _atomic_json_save(result, experiment_dir / "result.json")
    return result


def _feature_dimension(dataloader: DataLoader) -> int:
    dataset = dataloader.dataset
    features = getattr(dataset, "clip_features", None)
    if features is None or features.ndim != 2:
        raise ValueError("Training dataset does not expose 2D CLIP features")
    return int(features.shape[1])


def run_subset_experiment(
    config: ClipCapTrainingConfig,
    *,
    device: str | torch.device | None = None,
    resume: bool = True,
    skip_completed: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build fresh components and run one independent subset experiment."""
    result_path = config.experiment_dir / "result.json"
    if skip_completed and result_path.is_file():
        config_path = config.experiment_dir / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            saved_config = json.load(file)
        if _normalize_saved_config(saved_config) != config.to_dict():
            raise ValueError(
                "Completed experiment configuration differs from the current "
                "configuration. Use a different output_root."
            )
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        print(f"Skipped completed experiment: {config.subset_name}")
        return result

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_everything(config.seed)
    dataloaders = create_dataloaders(
        batch_size=config.batch_size,
        train_filename=config.train_filename,
        num_workers=config.num_workers,
        pin_memory=selected_device.type == "cuda",
        seed=config.seed,
        include_test=False,
    )
    clip_dim = _feature_dimension(dataloaders["train"])
    model = build_clipcap_model(config, clip_dim)
    result = fit(
        model,
        dataloaders,
        config,
        selected_device,
        resume=resume,
        show_progress=show_progress,
    )
    del model
    del dataloaders
    gc.collect()
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _write_summary(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            training_policy = result.get(
                "training_policy",
                CLIPCAP_EARLY_STOPPING_POLICY,
            )
            fallbacks = {
                "training_policy": training_policy,
                "final_epoch": None,
                "final_checkpoint": None,
                "official_checkpoint": result.get("best_checkpoint"),
            }
            writer.writerow(
                {
                    name: result.get(name, fallbacks.get(name))
                    for name in SUMMARY_FIELDS
                }
            )
    os.replace(temporary_path, path)


def run_subset_experiments(
    base_config: ClipCapTrainingConfig,
    subset_names: Sequence[str] = CLIPCAP_TRAIN_SUBSETS,
    *,
    device: str | torch.device | None = None,
    resume: bool = True,
    skip_completed: bool = True,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Run all requested subsets independently and update a shared summary."""
    results: list[dict[str, Any]] = []
    for subset_name in subset_names:
        config = replace(base_config, subset_name=subset_name)
        print(f"Starting independent experiment: {subset_name}")
        result = run_subset_experiment(
            config,
            device=device,
            resume=resume,
            skip_completed=skip_completed,
            show_progress=show_progress,
        )
        results.append(result)
        _write_summary(
            results,
            base_config.output_root / "experiment_summary.csv",
        )
    return results
