"""Tests for ClipCap training behavior."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel

import src.clipcap.training.trainer as training_module
from src.clipcap.models.clipcap_model import ClipCaptionModel
from src.clipcap.models.mapping_network import TransformerMapper
from src.clipcap.training.trainer import (
    _normalize_saved_config,
    evaluate,
    fit,
    train_one_epoch,
)
from src.config.clipcap_config import (
    CLIPCAP_FIXED_EPOCH_POLICY,
    CLIPCAP_FIXED_EPOCHS,
    ClipCapTrainingConfig,
    create_clipcap_fixed_epoch_config,
)


class TinyCaptionDataset(Dataset):
    def __init__(self) -> None:
        self.image_ids = ["image-1", "image-1", "image-2", "image-2"]
        self.clip_features = torch.randn(2, 8)
        self.input_ids = torch.tensor(
            [
                [3, 4, 5, 2, 0],
                [6, 7, 8, 2, 0],
                [9, 10, 11, 2, 0],
                [12, 13, 14, 2, 0],
            ],
            dtype=torch.long,
        )
        self.attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0]] * 4,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        feature_index = 0 if index < 2 else 1
        return {
            "image_embed": self.clip_features[feature_index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def create_tiny_model() -> ClipCaptionModel:
    mapper = TransformerMapper(
        clip_dim=8,
        embedding_dim=24,
        clip_length=2,
        prefix_length=2,
        num_layers=1,
        num_heads=4,
        feedforward_dim=48,
        dropout=0.0,
    )
    gpt2 = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=24,
            n_layer=1,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    gpt2.loss_type = "ForCausalLM"
    for parameter in gpt2.parameters():
        parameter.requires_grad = False
    return ClipCaptionModel(mapper=mapper, gpt2=gpt2)


def create_loaders() -> dict[str, DataLoader]:
    dataset = TinyCaptionDataset()
    return {
        "train": DataLoader(dataset, batch_size=2, shuffle=False),
        "val": DataLoader(dataset, batch_size=2, shuffle=False),
    }


def test_train_one_epoch_updates_only_mapper():
    model = create_tiny_model()
    loaders = create_loaders()
    optimizer = torch.optim.AdamW(model.mapper.parameters(), lr=1e-3)
    mapper_before = model.mapper.clip_projection.projection.weight.detach().clone()
    gpt2_before = model.gpt2.get_input_embeddings().weight.detach().clone()

    train_loss = train_one_epoch(
        model,
        loaders["train"],
        optimizer,
        torch.device("cpu"),
        max_grad_norm=1.0,
        show_progress=False,
    )
    val_loss = evaluate(
        model,
        loaders["val"],
        torch.device("cpu"),
        show_progress=False,
    )

    assert train_loss > 0
    assert val_loss > 0
    assert not torch.equal(
        mapper_before,
        model.mapper.clip_projection.projection.weight,
    )
    assert torch.equal(gpt2_before, model.gpt2.get_input_embeddings().weight)
    assert all(parameter.grad is None for parameter in model.gpt2.parameters())


def test_fit_saves_and_resumes_subset_checkpoint(tmp_path: Path):
    config = ClipCapTrainingConfig(
        subset_name="train_1pct",
        batch_size=2,
        num_workers=0,
        learning_rate=1e-3,
        max_epochs=2,
        early_stopping_patience=2,
        clip_length=2,
        prefix_length=2,
        num_layers=1,
        num_heads=4,
        feedforward_dim=48,
        dropout=0.0,
        output_root=tmp_path,
    )
    first_result = fit(
        create_tiny_model(),
        create_loaders(),
        config,
        torch.device("cpu"),
        resume=False,
        show_progress=False,
    )

    assert (config.experiment_dir / "best.pt").is_file()
    assert (config.experiment_dir / "latest.pt").is_file()
    assert (config.experiment_dir / "config.json").is_file()
    assert (config.experiment_dir / "history.json").is_file()
    assert (config.experiment_dir / "result.json").is_file()
    assert len(first_result["history"]) == 2

    resumed_result = fit(
        create_tiny_model(),
        create_loaders(),
        config,
        torch.device("cpu"),
        resume=True,
        show_progress=False,
    )

    assert resumed_result["last_epoch"] == 2
    assert resumed_result["optimizer_steps"] == first_result["optimizer_steps"]
    assert resumed_result["history"] == first_result["history"]


def test_fit_stops_after_validation_patience(tmp_path: Path, monkeypatch):
    config = ClipCapTrainingConfig(
        subset_name="train_1pct",
        batch_size=2,
        num_workers=0,
        learning_rate=1e-3,
        max_epochs=10,
        early_stopping_patience=2,
        clip_length=2,
        prefix_length=2,
        num_layers=1,
        num_heads=4,
        feedforward_dim=48,
        dropout=0.0,
        output_root=tmp_path,
    )
    validation_losses = iter((1.0, 1.1, 1.2))
    monkeypatch.setattr(
        training_module,
        "train_one_epoch",
        lambda *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        training_module,
        "evaluate",
        lambda *args, **kwargs: next(validation_losses),
    )

    result = fit(
        create_tiny_model(),
        create_loaders(),
        config,
        torch.device("cpu"),
        resume=False,
        show_progress=False,
    )

    assert result["stopped_early"] is True
    assert result["last_epoch"] == 3
    assert result["best_epoch"] == 1


def test_fixed_epoch_config_uses_central_policy_and_epoch_count(tmp_path: Path):
    config = create_clipcap_fixed_epoch_config(output_root=tmp_path)

    assert config.training_policy == CLIPCAP_FIXED_EPOCH_POLICY
    assert config.max_epochs == CLIPCAP_FIXED_EPOCHS == 7
    assert config.output_root == tmp_path


def test_legacy_config_defaults_to_early_stopping_policy(tmp_path: Path):
    config = ClipCapTrainingConfig(output_root=tmp_path)
    legacy_config = config.to_dict()
    legacy_config.pop("training_policy")

    assert _normalize_saved_config(legacy_config) == config.to_dict()


def test_fit_fixed_epoch_ignores_patience_and_saves_final_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    config = ClipCapTrainingConfig(
        subset_name="train_1pct",
        batch_size=2,
        num_workers=0,
        learning_rate=1e-3,
        max_epochs=3,
        early_stopping_patience=1,
        training_policy=CLIPCAP_FIXED_EPOCH_POLICY,
        clip_length=2,
        prefix_length=2,
        num_layers=1,
        num_heads=4,
        feedforward_dim=48,
        dropout=0.0,
        output_root=tmp_path,
    )
    validation_losses = iter((1.0, 1.1, 1.2))
    monkeypatch.setattr(
        training_module,
        "train_one_epoch",
        lambda *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        training_module,
        "evaluate",
        lambda *args, **kwargs: next(validation_losses),
    )

    result = fit(
        create_tiny_model(),
        create_loaders(),
        config,
        torch.device("cpu"),
        resume=False,
        show_progress=False,
    )

    final_path = config.experiment_dir / "final.pt"
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    assert result["training_policy"] == CLIPCAP_FIXED_EPOCH_POLICY
    assert result["last_epoch"] == 3
    assert result["final_epoch"] == 3
    assert result["stopped_early"] is False
    assert result["official_checkpoint"] == str(final_path)
    assert len(result["history"]) == 3
    assert checkpoint["checkpoint_type"] == "final"
    assert checkpoint["state"]["last_epoch"] == 3

    resumed_result = fit(
        create_tiny_model(),
        create_loaders(),
        config,
        torch.device("cpu"),
        resume=True,
        show_progress=False,
    )
    assert resumed_result["history"] == result["history"]
    assert resumed_result["optimizer_steps"] == result["optimizer_steps"]
