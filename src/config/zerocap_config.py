"""Verified ZeroCap GPT-2 Base run configuration."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ZeroCapRunConfig:
    run_mode: str

    gpt_model: str = "openai-community/gpt2"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    model_dtype: str = "float32"
    algorithm_revision: str = "matched_backbone_base_v2_bos_caption_rerank"
    val_tune_candidate: str = ""
    val_tune_case_reason: str = ""

    prompt: str = "Image of a"
    prepend_bos_token: bool = True
    top_k: int = 512
    inner_iterations: int = 5
    beam_size: int = 5
    max_new_tokens: int = 15

    clip_temperature: float = 0.01
    clip_loss_scale: float = 1.0
    fluency_weight: float = 0.2

    step_size: float = 0.3
    grad_norm_factor: float = 0.9
    fusion_factor: float = 0.99

    reset_context_delta: bool = True
    repetition_penalty: float = 1.0
    min_new_tokens: int = 2

    stop_token: str = "."
    end_factor: float = 1.01
    forbidden_factor: float = 20.0

    gradient_eps: float = 1e-15
    probability_eps: float = 1e-12
    clip_candidate_chunk_size: int = 64
    clip_max_text_tokens: int = 77
    beam_length_penalty: float = 1.0
    final_rerank_mode: str = "clip_only"
    final_rerank_text_mode: str = "caption_only"
    final_rerank_clip_weight: float = 1.0
    final_rerank_beam_weight: float = 1.0
    seed: int = 42

    save_diagnostics: bool = False
    diagnostic_top_n: int = 20
    diagnostic_save_all_top_k: bool = False
    save_diagnostic_images: bool = False
    display_input_images: bool = False

    benchmark_warmup_images: int = 2
    benchmark_val_images: int = 5
    test_smoke_images: int = 5
    time_budget_hours: float = 4.0
    run_heavy_metrics: bool = False
    metric_timeout_sec: float = 120.0

    feature_cache_path: Optional[str] = None
    forbidden_tokens_url: str = (
        "https://raw.githubusercontent.com/"
        "YoadTew/zero-shot-image-to-text/main/forbidden_tokens.npy"
    )

    debug: bool = False
    git_commit: str = ""
    split_manifest_sha256: str = ""
    output_root: str = ""
    config_hash: str = ""
    run_dir: str = ""

    @classmethod
    def _with_identity(cls, draft):
        if draft.fluency_weight < 0:
            raise ValueError("fluency_weight must be non-negative.")
        if not 0 < draft.fusion_factor <= 1:
            raise ValueError("fusion_factor must be in (0, 1].")
        if draft.clip_temperature <= 0:
            raise ValueError("clip_temperature must be positive.")
        if draft.top_k <= 0 or draft.inner_iterations <= 0:
            raise ValueError("top_k and inner_iterations must be positive.")
        if draft.beam_size <= 0 or draft.max_new_tokens <= 0:
            raise ValueError("beam_size and max_new_tokens must be positive.")
        if draft.step_size <= 0 or draft.grad_norm_factor <= 0:
            raise ValueError("step_size and grad_norm_factor must be positive.")
        hash_payload = asdict(draft)
        for derived_key in ("output_root", "config_hash", "run_dir"):
            hash_payload.pop(derived_key)
        canonical = json.dumps(
            hash_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        config_hash = hashlib.sha256(canonical).hexdigest()[:12]
        run_dir = Path(draft.output_root) / (
            f"{draft.run_mode}_{config_hash}"
        )
        return replace(
            draft,
            config_hash=config_hash,
            run_dir=str(run_dir),
        )

    @staticmethod
    def val_tune_grid():
        return (
            (
                "A_baseline",
                {
                    "fluency_weight": 0.2,
                    "fusion_factor": 0.99,
                    "clip_temperature": 0.01,
                    "case_reason": (
                        "Official ZeroCap inference defaults with the "
                        "matched GPT-2 Base backbone."
                    ),
                },
            ),
            (
                "B_fluency_0_10",
                {
                    "fluency_weight": 0.1,
                    "fusion_factor": 0.99,
                    "clip_temperature": 0.01,
                    "case_reason": (
                        "Reduce the KL/fluency constraint so CLIP can steer "
                        "the context more strongly; grammar may weaken."
                    ),
                },
            ),
            (
                "C_fluency_0_05",
                {
                    "fluency_weight": 0.05,
                    "fusion_factor": 0.99,
                    "clip_temperature": 0.01,
                    "case_reason": (
                        "Stress-test stronger visual steering with a much "
                        "weaker GPT-2 KL constraint."
                    ),
                },
            ),
            (
                "D_clip_temperature_0_02",
                {
                    "fluency_weight": 0.2,
                    "fusion_factor": 0.99,
                    "clip_temperature": 0.02,
                    "case_reason": (
                        "Soften the CLIP target distribution to reduce "
                        "overreaction to small/noisy similarity gaps."
                    ),
                },
            ),
            (
                "E_fusion_0_95",
                {
                    "fluency_weight": 0.2,
                    "fusion_factor": 0.95,
                    "clip_temperature": 0.01,
                    "case_reason": (
                        "Give the untouched GPT-2 distribution more weight "
                        "only at final token fusion."
                    ),
                },
            ),
        )

    def for_val_tune_candidate(self, candidate_name, overrides):
        if self.run_mode != "val_tune":
            raise ValueError(
                "VAL-tune candidates require RUN_MODE='val_tune'."
            )
        allowed_overrides = {
            "fluency_weight",
            "fusion_factor",
            "clip_temperature",
            "case_reason",
        }
        unexpected = set(overrides) - allowed_overrides
        if unexpected:
            raise ValueError(
                f"Unsupported VAL-tune overrides: {sorted(unexpected)}"
            )
        if set(overrides) != allowed_overrides:
            raise ValueError(
                "Every candidate must set fluency_weight, fusion_factor, "
                "clip_temperature and case_reason."
            )
        candidate_root = Path(self.run_dir) / "candidates"
        draft = replace(
            self,
            val_tune_candidate=str(candidate_name),
            val_tune_case_reason=str(overrides["case_reason"]),
            fluency_weight=float(overrides["fluency_weight"]),
            fusion_factor=float(overrides["fusion_factor"]),
            clip_temperature=float(overrides["clip_temperature"]),
            save_diagnostics=True,
            save_diagnostic_images=False,
            display_input_images=False,
            debug=False,
            output_root=str(candidate_root),
            config_hash="",
            run_dir="",
        )
        return type(self)._with_identity(draft)

    @classmethod
    def for_mode(
        cls,
        run_mode,
        git_commit,
        split_manifest_sha256,
        output_root,
        time_budget_hours,
        benchmark_val_images,
        run_heavy_metrics,
    ):
        allowed_modes = {
            "smoke",
            "benchmark",
            "val_tune",
            "test_smoke",
            "final_test",
        }
        if run_mode not in allowed_modes:
            raise ValueError(
                f"RUN_MODE must be one of {sorted(allowed_modes)}, got {run_mode!r}."
            )
        if benchmark_val_images not in {5, 10}:
            raise ValueError("BENCHMARK_VAL_IMAGES must be 5 or 10.")

        draft = cls(
            run_mode=run_mode,
            git_commit=git_commit,
            split_manifest_sha256=split_manifest_sha256,
            output_root=str(output_root),
            time_budget_hours=float(time_budget_hours),
            benchmark_val_images=int(benchmark_val_images),
            run_heavy_metrics=bool(run_heavy_metrics),
        )
        allowed_rerank_modes = {"clip_only", "clip_beam"}
        if draft.final_rerank_mode not in allowed_rerank_modes:
            raise ValueError(
                "final_rerank_mode must be one of "
                f"{sorted(allowed_rerank_modes)}."
            )
        if (
            draft.final_rerank_clip_weight < 0
            or draft.final_rerank_beam_weight < 0
        ):
            raise ValueError("Final rerank weights must be non-negative.")
        if (
            draft.final_rerank_clip_weight == 0
            and draft.final_rerank_beam_weight == 0
        ):
            raise ValueError("At least one final rerank weight must be positive.")
        if (
            draft.final_rerank_mode == "clip_only"
            and draft.final_rerank_clip_weight == 0
        ):
            raise ValueError("clip_only reranking requires a positive CLIP weight.")
        allowed_rerank_text_modes = {"caption_only", "prompt_plus_caption"}
        if draft.final_rerank_text_mode not in allowed_rerank_text_modes:
            raise ValueError(
                "final_rerank_text_mode must be one of "
                f"{sorted(allowed_rerank_text_modes)}."
            )
        if not draft.prepend_bos_token:
            raise ValueError(
                "The baseline-faithful prototype requires prepend_bos_token=True."
            )
        if draft.metric_timeout_sec <= 0:
            raise ValueError("metric_timeout_sec must be positive.")

        if run_mode == "smoke":
            draft = replace(
                draft,
                top_k=5,
                inner_iterations=1,
                beam_size=1,
                max_new_tokens=3,
                clip_candidate_chunk_size=5,
                save_diagnostics=True,
                save_diagnostic_images=True,
                display_input_images=True,
                debug=True,
            )

        return cls._with_identity(draft)

    def public_dict(self):
        return asdict(self)


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
