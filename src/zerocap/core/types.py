"""ZeroCap result, beam, context, and diagnostic types."""

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


ContextDelta = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class GenerationResult:
    image_id: str
    caption: str
    generated_token_ids: List[int]
    num_generated_tokens: int
    stop_reason: str
    generation_time_sec: float
    end_to_end_time_sec: float
    final_clip_similarity: float
    gpt_model: str
    clip_model: str
    config_hash: str
    git_commit: str

    def to_dict(self):
        value = asdict(self)
        json.dumps(value)
        return value


@dataclass
class ContextStepResult:
    p_original: torch.Tensor
    p_guided_final: torch.Tensor
    context_delta: ContextDelta
    diagnostics: List[Dict[str, Any]]


@dataclass
class BeamState:
    token_ids: torch.Tensor
    generated_token_ids: Tuple[int, ...] = field(default_factory=tuple)
    accumulated_logprob: float = 0.0
    stopped: bool = False
    stop_reason: str = ""
    context_delta: Optional[ContextDelta] = None

    def normalized_score(self, length_penalty):
        length = max(1, len(self.generated_token_ids))
        return self.accumulated_logprob / (length ** length_penalty)


def probability_diagnostics(probabilities, tokenizer, top_n):
    detached = probabilities.detach().float().cpu()
    if detached.ndim != 2 or detached.shape[0] != 1:
        raise AssertionError("Diagnostic probability tensor must be [1, vocab].")
    values = detached[0]
    count = min(int(top_n), int(values.numel()))
    top_values, top_ids = torch.topk(values, k=count)
    entropy = float(
        -(values * torch.log(values.clamp_min(1e-30))).sum().item()
    )
    rows = []
    for rank, (token_id, probability) in enumerate(
        zip(top_ids.tolist(), top_values.tolist()),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token_text": tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "probability": float(probability),
                "log_probability": float(
                    math.log(max(float(probability), 1e-30))
                ),
            }
        )
    return {
        "sum": float(values.sum().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "entropy": entropy,
        "effective_vocabulary_size": float(math.exp(entropy)),
        "top_tokens": rows,
    }


def context_delta_diagnostics(delta):
    layer_rows = []
    squared_total = 0.0
    for layer_index, (delta_key, delta_value) in enumerate(delta):
        key_norm = float(delta_key.detach().float().norm().item())
        value_norm = float(delta_value.detach().float().norm().item())
        squared_total += key_norm ** 2 + value_norm ** 2
        layer_rows.append(
            {
                "layer": layer_index,
                "key_norm": key_norm,
                "value_norm": value_norm,
            }
        )
    return {
        "global_norm": float(math.sqrt(squared_total)),
        "layers": layer_rows,
    }


def compact_iteration_diagnostics(diagnostics, top_n, save_all_top_k):
    similarities = diagnostics["similarities"]
    candidate_count = len(similarities)
    top_n = min(int(top_n), candidate_count)

    def candidate_row(index):
        return {
            "top_k_index": int(index),
            "token_id": int(diagnostics["top_token_ids"][index]),
            "token_text": diagnostics["decoded_top_tokens"][index],
            "candidate_text": diagnostics["candidate_texts"][index],
            "text_prefix_match": bool(
                diagnostics["candidate_text_prefix_matches"][index]
            ),
            "guided_probability": float(
                diagnostics["top_guided_probabilities"][index]
            ),
            "clip_similarity": float(similarities[index]),
            "clip_target_probability": float(
                diagnostics["target_topk_probabilities"][index]
            ),
        }

    clip_ranked_indices = sorted(
        range(candidate_count),
        key=lambda index: similarities[index],
        reverse=True,
    )[:top_n]
    guided_ranked_indices = list(range(top_n))
    similarity_array = np.asarray(similarities, dtype=np.float64)
    compact = {
        key: diagnostics[key]
        for key in (
            "iteration",
            "evaluation_type",
            "is_best_so_far",
            "loss_clip",
            "loss_fluency",
            "loss_total",
            "p_guided_sum",
            "gradient_norm_min",
            "gradient_norm_max",
            "gradient_norm_mean",
            "target_sum",
            "delta_before",
            "delta_after",
        )
    }
    compact.update(
        {
            "current_text": diagnostics["current_text"],
            "candidate_count": candidate_count,
            "clip_similarity_stats": {
                "min": float(similarity_array.min()),
                "max": float(similarity_array.max()),
                "mean": float(similarity_array.mean()),
                "std": float(similarity_array.std()),
            },
            "top_by_clip_similarity": [
                candidate_row(index) for index in clip_ranked_indices
            ],
            "top_by_guided_probability": [
                candidate_row(index) for index in guided_ranked_indices
            ],
        }
    )
    if save_all_top_k:
        compact["all_top_k_candidates"] = [
            candidate_row(index) for index in range(candidate_count)
        ]
    return compact


class GenerationTrace:
    SCHEMA_VERSION = 2

    def __init__(self, config, image_id, image):
        self.config = config
        self.image_id = str(image_id)
        self.started_at = time.perf_counter()
        self.path = (
            Path(config.run_dir)
            / "diagnostics"
            / f"{Path(self.image_id).name}.json"
        )
        self.image_path = (
            self.path.parent
            / "images"
            / Path(self.image_id).name
        )
        self.payload = {
            "schema_version": self.SCHEMA_VERSION,
            "status": "running",
            "image_id": self.image_id,
            "config_hash": config.config_hash,
            "git_commit": config.git_commit,
            "run_mode": config.run_mode,
            "algorithm_revision": config.algorithm_revision,
            "image": {
                "mode": str(getattr(image, "mode", "unknown")),
                "size": [int(value) for value in getattr(image, "size", ())],
            },
            "trace_settings": {
                "top_n": config.diagnostic_top_n,
                "save_all_top_k": config.diagnostic_save_all_top_k,
            },
            "events": [],
        }
        self._save_input_image(image)

    def _save_input_image(self, image):
        if not self.config.save_diagnostic_images:
            self.payload["image"]["saved_path"] = None
            return
        try:
            self.image_path.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(
                self.image_path,
                format="JPEG",
                quality=95,
            )
            self.payload["image"]["saved_path"] = str(
                self.image_path.relative_to(Path(self.config.run_dir))
            )
        except Exception as error:
            self.payload["image"]["saved_path"] = None
            self.payload["image"]["save_error"] = (
                f"{type(error).__name__}: {error}"
            )

    def record(self, event_type, **payload):
        event = {
            "event": str(event_type),
            "elapsed_sec": float(time.perf_counter() - self.started_at),
            **payload,
        }
        json.dumps(event, ensure_ascii=False, allow_nan=False)
        self.payload["events"].append(event)

    def finish(self, result):
        self.payload["status"] = "success"
        self.payload["result"] = dict(result)
        self.payload["total_trace_elapsed_sec"] = float(
            time.perf_counter() - self.started_at
        )

    def fail(self, error):
        self.payload["status"] = "error"
        self.payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        self.payload["total_trace_elapsed_sec"] = float(
            time.perf_counter() - self.started_at
        )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                self.payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        os.replace(temporary, self.path)
        return self.path


def assert_probability_distribution(probabilities, label, tolerance=1e-4):
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise AssertionError(f"{label} must have shape [1, vocab].")
    if not torch.isfinite(probabilities).all():
        raise AssertionError(f"{label} contains NaN/Inf.")
    if (probabilities < 0).any():
        raise AssertionError(f"{label} contains negative probability.")
    total = probabilities.sum()
    if not torch.allclose(
        total,
        torch.ones_like(total),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise AssertionError(f"{label} sums to {total.item()}, expected 1.")


def parameter_snapshot(model):
    return {
        name: (id(parameter), parameter.data_ptr(), parameter._version)
        for name, parameter in model.named_parameters()
    }


def assert_parameter_snapshot_unchanged(model, before, label):
    after = parameter_snapshot(model)
    if before != after:
        raise AssertionError(f"{label} parameters changed during generation.")


def assert_model_parameter_grads_clear(model, label):
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise AssertionError(
            f"{label} unexpectedly accumulated gradients: {offenders[:5]}"
        )


def clone_context_delta(delta):
    if delta is None:
        return None
    return tuple(
        (
            delta_key.detach().clone(),
            delta_value.detach().clone(),
        )
        for delta_key, delta_value in delta
    )


def postprocess_caption(text, prompt):
    normalized = " ".join(str(text).strip().split())
    if normalized.startswith(prompt):
        normalized = normalized[len(prompt):].strip()
    normalized = normalized.replace(" .", ".")
    return normalized.strip()