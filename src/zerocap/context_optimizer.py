"""Inference-only optimization of GPT-2 past-key/value deltas."""

import math

import numpy as np
import torch

from .types import (
    ContextStepResult,
    assert_model_parameter_grads_clear,
    assert_probability_distribution,
    compact_iteration_diagnostics,
    context_delta_diagnostics,
    probability_diagnostics,
)


class ContextOptimizer:
    def __init__(self, models, guidance, config):
        self.models = models
        self.guidance = guidance
        self.config = config
        self.gpt = models.gpt_model

    def _legacy_cache(self, cache):
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        if not isinstance(cache, (tuple, list)):
            raise RuntimeError(
                "Pinned Transformers did not return a legacy-compatible cache."
            )
        legacy = []
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise AssertionError(
                    f"Invalid cache structure at layer {layer_index}."
                )
            key, value = layer
            if not torch.is_tensor(key) or not torch.is_tensor(value):
                raise AssertionError("Cache key/value must be tensors.")
            if key.ndim != 4 or value.ndim != 4:
                raise AssertionError("GPT-2 cache tensors must be four-dimensional.")
            if key.shape != value.shape:
                raise AssertionError("GPT-2 key/value shapes do not match.")
            legacy.append((key.detach(), value.detach()))

        expected_layers = int(self.gpt.config.n_layer)
        if len(legacy) != expected_layers:
            raise AssertionError(
                f"Cache has {len(legacy)} layers, expected {expected_layers}."
            )
        return tuple(legacy)

    @staticmethod
    def _cache_for_forward(legacy_cache):
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(
            tuple(legacy_cache)
        )

    def _new_delta(self, past_original, previous_delta=None):
        use_previous = (
            previous_delta is not None
            and not self.config.reset_context_delta
        )
        if use_previous and len(previous_delta) != len(past_original):
            raise AssertionError("Previous delta layer count mismatch.")

        delta = []
        for layer_index, (key, value) in enumerate(past_original):
            delta_key = torch.zeros_like(key)
            delta_value = torch.zeros_like(value)

            if use_previous:
                previous_key, previous_value = previous_delta[layer_index]
                for destination, source in (
                    (delta_key, previous_key),
                    (delta_value, previous_value),
                ):
                    if (
                        destination.device != source.device
                        or destination.dtype != source.dtype
                        or destination.shape[:-2] != source.shape[:-2]
                        or destination.shape[-1] != source.shape[-1]
                    ):
                        raise AssertionError(
                            "Cannot carry context delta across incompatible cache."
                        )
                    copied_length = min(
                        destination.shape[-2],
                        source.shape[-2],
                    )
                    destination[..., :copied_length, :].copy_(
                        source.detach()[..., :copied_length, :]
                    )

            delta_key.requires_grad_(True)
            delta_value.requires_grad_(True)
            delta.append((delta_key, delta_value))

        result = tuple(delta)
        self._assert_delta(past_original, result, require_grad=True)
        return result

    @staticmethod
    def _add_delta(past_original, delta):
        return tuple(
            (
                original_key + delta_key,
                original_value + delta_value,
            )
            for (
                (original_key, original_value),
                (delta_key, delta_value),
            ) in zip(past_original, delta)
        )

    @staticmethod
    def _assert_delta(past_original, delta, require_grad):
        if len(past_original) != len(delta):
            raise AssertionError("Context delta structure mismatch.")
        for layer_index, (
            (original_key, original_value),
            (delta_key, delta_value),
        ) in enumerate(zip(past_original, delta)):
            for original, candidate, name in (
                (original_key, delta_key, "key"),
                (original_value, delta_value, "value"),
            ):
                if candidate.shape != original.shape:
                    raise AssertionError(
                        f"Delta {name} shape mismatch at layer {layer_index}."
                    )
                if candidate.device != original.device:
                    raise AssertionError("Delta device mismatch.")
                if candidate.dtype != original.dtype:
                    raise AssertionError("Delta dtype mismatch.")
                if require_grad and not candidate.requires_grad:
                    raise AssertionError("Context delta must require gradients.")

    def _original_distribution_and_past(self, current_ids):
        if current_ids.ndim != 2 or current_ids.shape[0] != 1:
            raise AssertionError("current_ids must have shape [1, sequence].")
        if current_ids.shape[1] < 2:
            raise AssertionError(
                "Prompt/current sequence must contain at least two GPT-2 tokens."
            )

        with torch.no_grad():
            original_output = self.gpt(
                input_ids=current_ids,
                use_cache=False,
                return_dict=True,
            )
            p_original = torch.softmax(
                original_output.logits[:, -1, :],
                dim=-1,
            ).detach()

            prefix_ids = current_ids[:, :-1]
            prefix_output = self.gpt(
                input_ids=prefix_ids,
                use_cache=True,
                return_dict=True,
            )
            past_original = self._legacy_cache(
                prefix_output.past_key_values
            )

        assert_probability_distribution(p_original, "P_original")
        if p_original.requires_grad or p_original.grad_fn is not None:
            raise AssertionError("P_original must be fully detached.")
        return p_original, past_original, current_ids[:, -1:]

    def _assert_zero_delta_cache_equivalence(
        self,
        p_original,
        past_original,
        last_token,
    ):
        with torch.no_grad():
            cached_output = self.gpt(
                input_ids=last_token,
                past_key_values=self._cache_for_forward(past_original),
                use_cache=False,
                return_dict=True,
            )
            p_cached = torch.softmax(
                cached_output.logits[:, -1, :],
                dim=-1,
            ).detach()

        assert_probability_distribution(p_cached, "P_cached_zero_delta")
        absolute_error = (p_cached - p_original).abs()
        max_abs_error = float(absolute_error.max().item())
        mean_abs_error = float(absolute_error.mean().item())
        original_top_token_id = int(p_original.argmax(dim=-1).item())
        cached_top_token_id = int(p_cached.argmax(dim=-1).item())
        same_top_token = original_top_token_id == cached_top_token_id

        if not torch.allclose(
            p_cached,
            p_original,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise AssertionError(
                "Zero-delta cached GPT-2 distribution does not match "
                "the full-sequence P_original; stop before interpreting "
                "CLIP guidance results. "
                f"max_abs_error={max_abs_error:.6g}, "
                f"mean_abs_error={mean_abs_error:.6g}."
            )

        return {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "original_top_token_id": original_top_token_id,
            "cached_top_token_id": cached_top_token_id,
            "same_top_token": same_top_token,
            "p_original_sum": float(p_original.sum().item()),
            "p_cached_sum": float(p_cached.sum().item()),
        }

    def _evaluate_delta(
        self,
        current_ids,
        image_feature,
        p_original,
        past_original,
        last_token,
        delta,
    ):
        past_guided = self._add_delta(past_original, delta)
        guided_output = self.gpt(
            input_ids=last_token,
            past_key_values=self._cache_for_forward(past_guided),
            use_cache=False,
            return_dict=True,
        )
        p_guided = torch.softmax(
            guided_output.logits[:, -1, :],
            dim=-1,
        )
        assert_probability_distribution(p_guided, "P_guided")

        top_count = min(self.config.top_k, p_guided.shape[-1])
        top_probabilities, top_indices = torch.topk(
            p_guided,
            k=top_count,
            dim=-1,
        )
        target_vocab, clip_diagnostics = self.guidance.build_target(
            image_feature=image_feature,
            current_ids=current_ids,
            top_indices=top_indices,
            vocabulary_size=p_guided.shape[-1],
            target_dtype=p_guided.dtype,
        )
        if target_vocab.requires_grad or target_vocab.grad_fn is not None:
            raise AssertionError("CLIP target must be detached.")

        log_guided = torch.log(
            p_guided.clamp_min(self.config.probability_eps)
        )
        log_original = torch.log(
            p_original.clamp_min(self.config.probability_eps)
        )
        loss_clip = -(target_vocab * log_guided).sum()
        loss_fluency = (
            p_guided * (log_guided - log_original)
        ).sum()
        loss_total = (
            self.config.clip_loss_scale * loss_clip
            + self.config.fluency_weight * loss_fluency
        )

        for label, loss in (
            ("CLIP loss", loss_clip),
            ("fluency loss", loss_fluency),
            ("total loss", loss_total),
        ):
            if not torch.isfinite(loss):
                raise AssertionError(f"{label} is NaN/Inf.")

        return {
            "p_guided": p_guided,
            "top_probabilities": top_probabilities,
            "top_indices": top_indices,
            "target_vocab": target_vocab,
            "clip_diagnostics": clip_diagnostics,
            "loss_clip": loss_clip,
            "loss_fluency": loss_fluency,
            "loss_total": loss_total,
        }

    def optimize(
        self,
        current_ids,
        image_feature,
        previous_delta=None,
        trace=None,
        trace_context=None,
    ):
        p_original, past_original, last_token = (
            self._original_distribution_and_past(current_ids)
        )
        delta = self._new_delta(
            past_original,
            previous_delta=previous_delta,
        )
        diagnostics = []
        trace_context = dict(trace_context or {})

        cache_equivalence = None
        if (
            trace is not None
            and trace_context.get("step") == 1
            and trace_context.get("beam_index") == 0
        ):
            cache_equivalence = (
                self._assert_zero_delta_cache_equivalence(
                    p_original=p_original,
                    past_original=past_original,
                    last_token=last_token,
                )
            )
            trace.record(
                "zero_delta_cache_equivalence",
                context=trace_context,
                **cache_equivalence,
            )

        if trace is not None:
            first_key, first_value = past_original[0]
            trace.record(
                "context_initialized",
                context=trace_context,
                current_token_ids=[
                    int(value)
                    for value in current_ids.detach().cpu().reshape(-1).tolist()
                ],
                current_text=self.models.gpt_tokenizer.decode(
                    current_ids.detach().cpu().reshape(-1).tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                p_original=probability_diagnostics(
                    p_original,
                    self.models.gpt_tokenizer,
                    self.config.diagnostic_top_n,
                ),
                cache={
                    "layers": len(past_original),
                    "first_key_shape": list(first_key.shape),
                    "first_value_shape": list(first_value.shape),
                    "dtype": str(first_key.dtype),
                    "device": str(first_key.device),
                },
                initial_delta=context_delta_diagnostics(delta),
            )

        if self.config.debug:
            first_key, first_value = past_original[0]
            print("past_key_values layers:", len(past_original))
            print("past[0] key/value:", tuple(first_key.shape), tuple(first_value.shape))
            print("delta[0] key/value:", tuple(delta[0][0].shape), tuple(delta[0][1].shape))
            print("P_original sum:", float(p_original.sum().item()))
            if cache_equivalence is not None:
                print(
                    "zero-delta cache max/mean abs error:",
                    cache_equivalence["max_abs_error"],
                    cache_equivalence["mean_abs_error"],
                )

        best_loss = math.inf
        best_iteration = None

        for iteration in range(self.config.inner_iterations):
            delta_before = context_delta_diagnostics(delta)
            evaluation = self._evaluate_delta(
                current_ids=current_ids,
                image_feature=image_feature,
                p_original=p_original,
                past_original=past_original,
                last_token=last_token,
                delta=delta,
            )
            p_guided = evaluation["p_guided"]
            top_probabilities = evaluation["top_probabilities"]
            top_indices = evaluation["top_indices"]
            target_vocab = evaluation["target_vocab"]
            clip_diagnostics = evaluation["clip_diagnostics"]
            loss_clip = evaluation["loss_clip"]
            loss_fluency = evaluation["loss_fluency"]
            loss_total = evaluation["loss_total"]

            loss_value = float(loss_total.detach().item())
            is_best_so_far = loss_value < best_loss
            if is_best_so_far:
                best_loss = loss_value
                best_iteration = iteration

            loss_total.backward()

            updated_layers = []
            gradient_norms = []
            for delta_key, delta_value in delta:
                updated_pair = []
                for delta_tensor in (delta_key, delta_value):
                    gradient = delta_tensor.grad
                    if gradient is None:
                        raise AssertionError("Context delta has no gradient.")
                    if not torch.isfinite(gradient).all():
                        raise AssertionError("Context delta gradient is NaN/Inf.")
                    gradient_norm = gradient.norm()
                    normalized_gradient = gradient / (
                        (gradient_norm + self.config.gradient_eps)
                        ** self.config.grad_norm_factor
                    )
                    updated = (
                        delta_tensor
                        - self.config.step_size * normalized_gradient
                    ).detach()
                    updated.requires_grad_(True)
                    updated_pair.append(updated)
                    gradient_norms.append(float(gradient_norm.item()))
                updated_layers.append(tuple(updated_pair))
            delta = tuple(updated_layers)
            self._assert_delta(past_original, delta, require_grad=True)

            assert_model_parameter_grads_clear(
                self.models.gpt_model,
                "GPT-2",
            )
            assert_model_parameter_grads_clear(
                self.models.clip_model,
                "CLIP",
            )

            iteration_diagnostics = {
                "iteration": iteration,
                "evaluation_type": "pre_update",
                "is_best_so_far": is_best_so_far,
                "loss_clip": float(loss_clip.detach().item()),
                "loss_fluency": float(loss_fluency.detach().item()),
                "loss_total": loss_value,
                "p_guided_sum": float(p_guided.detach().sum().item()),
                "gradient_norm_min": min(gradient_norms),
                "gradient_norm_max": max(gradient_norms),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "top_token_ids": [
                    int(value)
                    for value in top_indices[0].detach().cpu().tolist()
                ],
                "top_guided_probabilities": [
                    float(value)
                    for value in top_probabilities[0].detach().cpu().tolist()
                ],
                "target_topk_probabilities": [
                    float(value)
                    for value in target_vocab[0, top_indices[0]].detach().cpu().tolist()
                ],
                "delta_before": delta_before,
                "delta_after": context_delta_diagnostics(delta),
                **clip_diagnostics,
            }
            diagnostics.append(iteration_diagnostics)

            if trace is not None:
                trace.record(
                    "inner_optimization_iteration",
                    context=trace_context,
                    diagnostics=compact_iteration_diagnostics(
                        iteration_diagnostics,
                        self.config.diagnostic_top_n,
                        self.config.diagnostic_save_all_top_k,
                    ),
                    p_guided=probability_diagnostics(
                        p_guided,
                        self.models.gpt_tokenizer,
                        self.config.diagnostic_top_n,
                    ),
                )

            if self.config.debug:
                print(
                    f"inner={iteration} "
                    f"clip={iteration_diagnostics['loss_clip']:.6f} "
                    f"fluency={iteration_diagnostics['loss_fluency']:.6f} "
                    f"total={iteration_diagnostics['loss_total']:.6f}"
                )
                print(
                    "Top-K decoded candidates:",
                    iteration_diagnostics["candidate_texts"],
                )
                print(
                    "CLIP similarities:",
                    iteration_diagnostics["similarities"],
                )
                print(
                    "P_guided sum:",
                    iteration_diagnostics["p_guided_sum"],
                )

        del (
            evaluation,
            p_guided,
            top_probabilities,
            top_indices,
            target_vocab,
            loss_clip,
            loss_fluency,
            loss_total,
        )

        post_update_delta = context_delta_diagnostics(delta)
        post_update_evaluation = self._evaluate_delta(
            current_ids=current_ids,
            image_feature=image_feature,
            p_original=p_original,
            past_original=past_original,
            last_token=last_token,
            delta=delta,
        )
        post_p_guided = post_update_evaluation["p_guided"]
        post_top_probabilities = post_update_evaluation[
            "top_probabilities"
        ]
        post_top_indices = post_update_evaluation["top_indices"]
        post_target_vocab = post_update_evaluation["target_vocab"]
        post_clip_diagnostics = post_update_evaluation[
            "clip_diagnostics"
        ]
        post_loss_clip = post_update_evaluation["loss_clip"]
        post_loss_fluency = post_update_evaluation["loss_fluency"]
        post_loss_total = post_update_evaluation["loss_total"]
        post_loss_value = float(post_loss_total.detach().item())
        post_is_best = post_loss_value < best_loss
        if post_is_best:
            best_loss = post_loss_value
            best_iteration = self.config.inner_iterations

        post_update_diagnostics = {
            "iteration": self.config.inner_iterations,
            "evaluation_type": "post_final_update",
            "is_best_so_far": post_is_best,
            "loss_clip": float(post_loss_clip.detach().item()),
            "loss_fluency": float(post_loss_fluency.detach().item()),
            "loss_total": post_loss_value,
            "p_guided_sum": float(post_p_guided.detach().sum().item()),
            "gradient_norm_min": None,
            "gradient_norm_max": None,
            "gradient_norm_mean": None,
            "top_token_ids": [
                int(value)
                for value in post_top_indices[0].detach().cpu().tolist()
            ],
            "top_guided_probabilities": [
                float(value)
                for value in post_top_probabilities[0].detach().cpu().tolist()
            ],
            "target_topk_probabilities": [
                float(value)
                for value in post_target_vocab[
                    0, post_top_indices[0]
                ].detach().cpu().tolist()
            ],
            "delta_before": post_update_delta,
            "delta_after": post_update_delta,
            **post_clip_diagnostics,
        }
        diagnostics.append(post_update_diagnostics)

        if trace is not None:
            trace.record(
                "inner_optimization_post_update_evaluation",
                context=trace_context,
                diagnostics=compact_iteration_diagnostics(
                    post_update_diagnostics,
                    self.config.diagnostic_top_n,
                    self.config.diagnostic_save_all_top_k,
                ),
                p_guided=probability_diagnostics(
                    post_p_guided,
                    self.models.gpt_tokenizer,
                    self.config.diagnostic_top_n,
                ),
            )

        if self.config.debug:
            print(
                f"post-update clip={post_update_diagnostics['loss_clip']:.6f} "
                f"fluency={post_update_diagnostics['loss_fluency']:.6f} "
                f"total={post_update_diagnostics['loss_total']:.6f}"
            )
            print("Diagnostic best delta evaluation:", best_iteration)
            print(
                "Generation uses post-final-update delta evaluation:",
                self.config.inner_iterations,
            )

        if best_iteration is None:
            raise AssertionError("No valid context delta evaluation was recorded.")
        detached_delta = tuple(
            (
                delta_key.detach(),
                delta_value.detach(),
            )
            for delta_key, delta_value in delta
        )
        self._assert_delta(
            past_original,
            detached_delta,
            require_grad=False,
        )
        for delta_key, delta_value in detached_delta:
            for delta_tensor in (delta_key, delta_value):
                if delta_tensor.requires_grad or delta_tensor.grad_fn is not None:
                    raise AssertionError("Final context delta retained a graph.")

        del (
            post_update_evaluation,
            post_p_guided,
            post_top_probabilities,
            post_top_indices,
            post_target_vocab,
            post_loss_clip,
            post_loss_fluency,
            post_loss_total,
        )
        past_final = self._add_delta(
            past_original,
            detached_delta,
        )
        with torch.no_grad():
            final_output = self.gpt(
                input_ids=last_token,
                past_key_values=self._cache_for_forward(past_final),
                use_cache=False,
                return_dict=True,
            )
            p_guided_final = torch.softmax(
                final_output.logits[:, -1, :],
                dim=-1,
            ).detach()
        assert_probability_distribution(
            p_guided_final,
            "P_guided_final",
        )

        if trace is not None:
            trace.record(
                "context_optimized",
                context=trace_context,
                best_delta_evaluation=int(best_iteration),
                best_loss_total=float(best_loss),
                selected_delta_evaluation=(
                    self.config.inner_iterations
                ),
                selected_loss_total=float(post_loss_value),
                selected_minus_best_loss=float(
                    post_loss_value - best_loss
                ),
                evaluated_delta_states=(
                    self.config.inner_iterations + 1
                ),
                p_guided_final=probability_diagnostics(
                    p_guided_final,
                    self.models.gpt_tokenizer,
                    self.config.diagnostic_top_n,
                ),
                final_delta=context_delta_diagnostics(detached_delta),
            )

        if self.config.debug:
            print("P_guided_final sum:", float(p_guided_final.sum().item()))
            print(
                "Diagnostic best delta evaluation/loss:",
                best_iteration,
                best_loss,
            )
            print(
                "Selected final-update evaluation/loss:",
                self.config.inner_iterations,
                post_loss_value,
            )

        return ContextStepResult(
            p_original=p_original,
            p_guided_final=p_guided_final,
            context_delta=detached_delta,
            diagnostics=diagnostics,
        )
