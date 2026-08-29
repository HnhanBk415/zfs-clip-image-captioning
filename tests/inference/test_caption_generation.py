"""Tests for caption generation and evaluation behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import src.clipcap.inference.inference_runner as evaluation_module
from src.clipcap.inference.checkpoints import (
    build_mapper_from_checkpoint,
    load_fixed_epoch_checkpoint,
)
from src.clipcap.inference.decoding import (
    beam_search_from_prefix,
    clip_rerank_candidates,
)
from src.clipcap.inference.inference_runner import run_evaluation
from src.clipcap.inference.features import load_evaluation_manifest
from src.clipcap.inference.records import (
    BeamCandidate,
    EvaluationItem,
    FixedEpochCheckpoint,
    GenerationResult,
)
from src.clipcap.models.mapping_network import TransformerMapper
from src.config.clipcap_config import CLIPCAP_FIXED_EPOCH_POLICY


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def batch_decode(
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        captions = []
        for sequence in sequences.detach().cpu().tolist():
            kept = [
                token_id
                for token_id in sequence
                if not skip_special_tokens or token_id != 0
            ]
            captions.append(" ".join(str(token_id) for token_id in kept))
        return captions


def _tiny_gpt2() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    return model.eval()


def _tiny_mapper() -> TransformerMapper:
    return TransformerMapper(
        clip_dim=4,
        embedding_dim=8,
        clip_length=2,
        prefix_length=2,
        num_layers=1,
        num_heads=1,
        feedforward_dim=16,
        dropout=0.0,
    )


def _fixed_epoch_config(subset_name: str, seed: int = 42) -> dict:
    return {
        "subset_name": subset_name,
        "seed": seed,
        "training_policy": CLIPCAP_FIXED_EPOCH_POLICY,
        "max_epochs": 7,
        "clip_length": 2,
        "prefix_length": 2,
        "num_layers": 1,
        "num_heads": 1,
        "feedforward_dim": 16,
        "dropout": 0.0,
    }


def _checkpoint_payload(subset_name: str, checkpoint_type: str = "final") -> dict:
    config = _fixed_epoch_config(subset_name)
    return {
        "checkpoint_version": 1,
        "checkpoint_type": checkpoint_type,
        "mapper_state_dict": _tiny_mapper().state_dict(),
        "gpt2_model_name": "tiny-gpt2",
        "config": config,
        "state": {
            "last_epoch": 7,
            "best_epoch": 6,
            "best_val_loss": 1.0,
        },
    }


def _write_artifacts(
    directory: Path,
    subset_name: str,
    checkpoint_type: str = "final",
) -> None:
    directory.mkdir(parents=True)
    payload = _checkpoint_payload(subset_name, checkpoint_type)
    for name in ("best.pt", "latest.pt", "final.pt"):
        torch.save(payload, directory / name)
    for name in ("history.json", "result.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps(payload["config"]),
        encoding="utf-8",
    )


def test_load_fixed_epoch_checkpoint_accepts_final_mapper(tmp_path: Path):
    directory = tmp_path / "train_1pct" / "seed_42"
    _write_artifacts(directory, "train_1pct")

    artifact = load_fixed_epoch_checkpoint(directory, "train_1pct", 42)
    loaded_mapper = build_mapper_from_checkpoint(
        artifact,
        _tiny_gpt2(),
        SimpleNamespace(config=SimpleNamespace(projection_dim=4)),
        torch.device("cpu"),
    )

    expected_state = artifact.checkpoint["mapper_state_dict"]
    for key, value in loaded_mapper.state_dict().items():
        assert torch.equal(value, expected_state[key])


def test_load_fixed_epoch_checkpoint_rejects_non_final_checkpoint(tmp_path: Path):
    directory = tmp_path / "train_1pct" / "seed_42"
    _write_artifacts(directory, "train_1pct", checkpoint_type="best")

    with pytest.raises(ValueError, match="requires final.pt"):
        load_fixed_epoch_checkpoint(directory, "train_1pct", 42)


def test_beam_search_returns_five_ranked_candidates():
    candidates = beam_search_from_prefix(
        _tiny_gpt2(),
        TinyTokenizer(),
        torch.zeros(1, 2, 8),
        max_new_tokens=4,
        num_beams=5,
        num_return_sequences=5,
    )

    assert len(candidates) == 5
    assert [candidate.beam_rank for candidate in candidates] == [1, 2, 3, 4, 5]
    assert all(torch.isfinite(torch.tensor(item.beam_score)) for item in candidates)


def test_clip_reranking_selects_highest_similarity():
    candidates = (
        BeamCandidate(1, "first", (1,), -0.1),
        BeamCandidate(2, "second", (2,), -0.2),
    )

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.tensor([[0], [1]]),
                "attention_mask": torch.ones(2, 1, dtype=torch.long),
            }

    class FakeEncoder:
        @staticmethod
        def get_text_features(**kwargs):
            del kwargs
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    result = clip_rerank_candidates(
        candidates,
        torch.tensor([[1.0, 0.0]]),
        FakeProcessor(),
        FakeEncoder(),
        torch.device("cpu"),
    )

    assert result.caption == "second"
    assert result.selected_beam_rank == 2
    assert result.candidates[1].clip_score > result.candidates[0].clip_score


def test_manifest_supports_references_or_image_id_list(tmp_path: Path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"a.jpg": ["reference one"], "b.jpg": []}),
        encoding="utf-8",
    )
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["c.jpg", "d.jpg"]), encoding="utf-8")

    mapping_items = load_evaluation_manifest(mapping_path)
    list_items = load_evaluation_manifest(list_path)

    assert mapping_items[0] == EvaluationItem("a.jpg", ("reference one",))
    assert [item.image_id for item in list_items] == ["c.jpg", "d.jpg"]


def test_prediction_record_never_contains_ground_truth_references():
    record = evaluation_module.generation_result_to_record(
        EvaluationItem("image.jpg", ("ground truth caption",)),
        GenerationResult(
            caption="generated caption",
            selected_beam_rank=1,
            candidates=(BeamCandidate(1, "generated caption", (1,), -0.1),),
        ),
        "train_1pct",
        42,
    )

    assert record["caption"] == "generated caption"
    assert "references" not in record


def test_resume_rejects_a_different_run_configuration(tmp_path: Path):
    path = tmp_path / "run_config.json"
    evaluation_module.prepare_run_config({"manifest": "first.json"}, path, True)

    with pytest.raises(ValueError, match="does not match"):
        evaluation_module.prepare_run_config(
            {"manifest": "second.json"},
            path,
            True,
        )


def test_runner_writes_every_subset_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    subset_names = (
        "train_1pct",
        "train_5pct",
        "train_10pct",
        "train_25pct",
        "train_100pct",
    )
    checkpoints = tuple(
        FixedEpochCheckpoint(
            subset_name=name,
            seed=42,
            directory=tmp_path / name,
            config=_fixed_epoch_config(name),
            checkpoint=_checkpoint_payload(name),
        )
        for name in subset_names
    )
    generated_calls: list[str] = []

    def fake_generate(*args, **kwargs):
        del args, kwargs
        generated_calls.append("called")
        candidate = BeamCandidate(1, "caption", (1,), -0.1, 0.5)
        return GenerationResult("caption", 1, (candidate,))

    monkeypatch.setattr(evaluation_module, "generate_caption_from_feature", fake_generate)
    items = (EvaluationItem("a.jpg", ("reference",)),)
    output_paths = run_evaluation(
        items=items,
        features={"a.jpg": torch.ones(4)},
        checkpoints=checkpoints,
        processor=object(),
        encoder=SimpleNamespace(config=SimpleNamespace(projection_dim=4)),
        gpt2=_tiny_gpt2(),
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "evaluation",
    )

    assert set(output_paths) == set(subset_names)
    assert len(generated_calls) == 5
    for path in output_paths.values():
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["checkpoint"] == "final.pt"
        assert record["caption"] == "caption"

    run_evaluation(
        items=items,
        features={"a.jpg": torch.ones(4)},
        checkpoints=checkpoints,
        processor=object(),
        encoder=SimpleNamespace(config=SimpleNamespace(projection_dim=4)),
        gpt2=_tiny_gpt2(),
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "evaluation",
        resume=True,
    )
    assert len(generated_calls) == 5
