"""Public end-to-end ZeroCap captioning interface."""

import gc
import time

import torch

from .clip_guidance import CLIPGuidance
from .context_optimizer import ContextOptimizer
from .decoding import ZeroCapDecoder
from .generator import ZeroCapGenerator
from .image_encoder import ImageEncoder
from .model_loader import ZeroCapModels
from .types import (
    GenerationResult,
    GenerationTrace,
    assert_model_parameter_grads_clear,
    assert_parameter_snapshot_unchanged,
    parameter_snapshot,
)


class ZeroCapCaptioner:
    def __init__(self, config, models=None):
        self.config = config
        if models is None:
            self.models = ZeroCapModels(config)
        else:
            compatibility_fields = (
                "gpt_model",
                "clip_model",
                "device",
                "model_dtype",
            )
            mismatches = {
                field_name: (
                    getattr(models.config, field_name),
                    getattr(config, field_name),
                )
                for field_name in compatibility_fields
                if getattr(models.config, field_name)
                != getattr(config, field_name)
            }
            if mismatches:
                raise ValueError(
                    f"Shared model bundle is incompatible: {mismatches}"
                )
            self.models = models
            self.models._assert_models()
        self.image_encoder = ImageEncoder(self.models, config)
        self.guidance = CLIPGuidance(self.models, config)
        self.context_optimizer = ContextOptimizer(
            self.models,
            self.guidance,
            config,
        )
        self.decoder = ZeroCapDecoder(self.models, config)
        self.generator = ZeroCapGenerator(
            self.models,
            self.guidance,
            self.context_optimizer,
            self.decoder,
            config,
        )
        self.assert_effective_config()

    def assert_effective_config(self):
        components = {
            "captioner": self,
            "image_encoder": self.image_encoder,
            "guidance": self.guidance,
            "context_optimizer": self.context_optimizer,
            "decoder": self.decoder,
            "generator": self.generator,
        }
        stale = [
            name
            for name, component in components.items()
            if component.config is not self.config
        ]
        if stale:
            raise AssertionError(
                f"Components hold a stale config object: {stale}"
            )
        return {
            "candidate": self.config.val_tune_candidate,
            "case_reason": self.config.val_tune_case_reason,
            "config_hash": self.config.config_hash,
            "fluency_weight": float(self.context_optimizer.config.fluency_weight),
            "fusion_factor": float(self.decoder.config.fusion_factor),
            "clip_temperature": float(self.guidance.config.clip_temperature),
            "prepend_bos_token": bool(self.generator.config.prepend_bos_token),
            "final_rerank_text_mode": (
                self.generator.config.final_rerank_text_mode
            ),
            "algorithm_revision": self.config.algorithm_revision,
            "all_component_configs_current": True,
        }

    def generate_caption(self, image, image_id):
        trace = (
            GenerationTrace(self.config, image_id, image)
            if self.config.save_diagnostics
            else None
        )
        image_feature = None
        payload = None

        try:
            gpt_snapshot = parameter_snapshot(self.models.gpt_model)
            clip_snapshot = parameter_snapshot(self.models.clip_model)
            assert_model_parameter_grads_clear(self.models.gpt_model, "GPT-2")
            assert_model_parameter_grads_clear(self.models.clip_model, "CLIP")

            torch.cuda.synchronize()
            end_to_end_start = time.perf_counter()
            image_encoding_start = time.perf_counter()
            image_feature = self.image_encoder.encode(
                image=image,
                image_id=image_id,
            )
            torch.cuda.synchronize()
            image_encoding_time_sec = (
                time.perf_counter() - image_encoding_start
            )

            if trace is not None:
                trace.record(
                    "image_encoded",
                    feature_shape=list(image_feature.shape),
                    feature_dtype=str(image_feature.dtype),
                    feature_device=str(image_feature.device),
                    feature_norm=float(image_feature.norm().item()),
                    feature_min=float(image_feature.min().item()),
                    feature_max=float(image_feature.max().item()),
                    image_encoding_time_sec=float(image_encoding_time_sec),
                )

            generation_start = time.perf_counter()
            payload = self.generator.generate(
                image_feature,
                trace=trace,
            )
            torch.cuda.synchronize()
            generation_time_sec = time.perf_counter() - generation_start
            end_to_end_time_sec = time.perf_counter() - end_to_end_start

            result = GenerationResult(
                image_id=str(image_id),
                caption=str(payload["caption"]),
                generated_token_ids=list(payload["generated_token_ids"]),
                num_generated_tokens=len(payload["generated_token_ids"]),
                stop_reason=str(payload["stop_reason"]),
                generation_time_sec=float(generation_time_sec),
                end_to_end_time_sec=float(end_to_end_time_sec),
                final_clip_similarity=float(payload["final_clip_similarity"]),
                gpt_model=self.config.gpt_model,
                clip_model=self.config.clip_model,
                config_hash=self.config.config_hash,
                git_commit=self.config.git_commit,
            )

            if not result.caption.strip():
                raise AssertionError("Caption is empty.")
            if result.num_generated_tokens != len(result.generated_token_ids):
                raise AssertionError("Generated token count mismatch.")
            if result.num_generated_tokens > self.config.max_new_tokens:
                raise AssertionError("Generated token count exceeds configured limit.")
            if result.stop_reason not in self.decoder.VALID_STOP_REASONS:
                raise AssertionError("Invalid generation stop reason.")

            assert_parameter_snapshot_unchanged(
                self.models.gpt_model,
                gpt_snapshot,
                "GPT-2",
            )
            assert_parameter_snapshot_unchanged(
                self.models.clip_model,
                clip_snapshot,
                "CLIP",
            )
            assert_model_parameter_grads_clear(self.models.gpt_model, "GPT-2")
            assert_model_parameter_grads_clear(self.models.clip_model, "CLIP")

            public_result = result.to_dict()
            if any(torch.is_tensor(value) for value in public_result.values()):
                raise AssertionError("Public result contains a GPU tensor.")

            if trace is not None:
                trace.record(
                    "generation_finished",
                    generation_time_sec=float(generation_time_sec),
                    end_to_end_time_sec=float(end_to_end_time_sec),
                    cuda_memory={
                        "allocated_mb": float(
                            torch.cuda.memory_allocated() / (1024 ** 2)
                        ),
                        "reserved_mb": float(
                            torch.cuda.memory_reserved() / (1024 ** 2)
                        ),
                        "peak_allocated_mb": float(
                            torch.cuda.max_memory_allocated() / (1024 ** 2)
                        ),
                    },
                )
                trace.finish(public_result)
                trace_path = trace.save()
                print("Diagnostic trace saved:", trace_path)

        except Exception as error:
            if trace is not None:
                trace.fail(error)
                trace_path = trace.save()
                print("FAILED diagnostic trace saved:", trace_path)
            if image_feature is not None:
                del image_feature
            if payload is not None:
                del payload
            gc.collect()
            raise

        del image_feature, payload, result
        gc.collect()
        return public_result
