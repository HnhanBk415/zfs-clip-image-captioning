"""ZeroCap probability fusion and special-token policy."""

import math

import torch

from ..core.types import (
    assert_probability_distribution,
    postprocess_caption,
    probability_diagnostics,
)


class ZeroCapDecoder:
    VALID_STOP_REASONS = {
        "period",
        "eos",
        "eos_fallback",
        "max_tokens",
    }

    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.tokenizer = models.gpt_tokenizer
        self.eos_token_id = int(self.tokenizer.eos_token_id)

        stop_ids = set(
            self.tokenizer.encode(
                config.stop_token,
                add_special_tokens=False,
            )
        )
        stop_ids.update(
            self.tokenizer.encode(
                " " + config.stop_token,
                add_special_tokens=False,
            )
        )
        if not stop_ids:
            raise RuntimeError("stop_token did not map to any GPT-2 token.")
        self.stop_token_ids = {
            int(token_id) for token_id in stop_ids
        }

        forbidden = set(
            int(value)
            for value in models.forbidden_token_ids.detach().cpu().tolist()
        )
        forbidden.discard(self.eos_token_id)
        forbidden.difference_update(self.stop_token_ids)
        self.forbidden_token_ids = sorted(forbidden)

    def fuse(
        self,
        p_guided_constrained,
        p_original,
        hard_suppressed_token_ids=(),
    ):
        log_guided = torch.log(
            p_guided_constrained.clamp_min(self.config.probability_eps)
        )
        log_original = torch.log(
            p_original.clamp_min(self.config.probability_eps)
        )
        log_p_final = (
            self.config.fusion_factor * log_guided
            + (1.0 - self.config.fusion_factor) * log_original
        )
        if hard_suppressed_token_ids:
            hard_suppressed = torch.tensor(
                sorted(set(int(value) for value in hard_suppressed_token_ids)),
                dtype=torch.long,
                device=log_p_final.device,
            )
            log_p_final[:, hard_suppressed] = -torch.inf
        if not torch.isfinite(log_p_final).any():
            raise RuntimeError("Fusion suppressed every vocabulary token.")
        p_final = torch.softmax(log_p_final, dim=-1)
        assert_probability_distribution(p_final, "P_fused")
        return p_final

    def apply_guided_constraints(
        self,
        p_guided_final,
        generated_token_ids,
    ):
        before = probability_diagnostics(
            p_guided_final,
            self.tokenizer,
            self.config.diagnostic_top_n,
        )
        log_probabilities = torch.log(
            p_guided_final.clamp_min(self.config.probability_eps)
        ).clone()
        generated_count = len(generated_token_ids)
        repeated_token_ids = []
        early_stop_ids = []
        stop_boost = 0.0

        if self.forbidden_token_ids:
            forbidden_tensor = torch.tensor(
                self.forbidden_token_ids,
                dtype=torch.long,
                device=log_probabilities.device,
            )
            log_probabilities[:, forbidden_tensor] -= math.log(
                self.config.forbidden_factor
            )

        if (
            self.config.repetition_penalty > 1.0
            and generated_token_ids
        ):
            repeated = torch.tensor(
                sorted(set(generated_token_ids)),
                dtype=torch.long,
                device=log_probabilities.device,
            )
            log_probabilities[:, repeated] -= math.log(
                self.config.repetition_penalty
            )
            repeated_token_ids = [int(value) for value in repeated.tolist()]

        if generated_count < self.config.min_new_tokens:
            early_stop_ids = sorted(
                self.stop_token_ids | {self.eos_token_id}
            )
            log_probabilities[:, early_stop_ids] = -torch.inf
        else:
            boost_power = (
                generated_count
                - self.config.min_new_tokens
                + 1
            )
            stop_boost = boost_power * math.log(self.config.end_factor)
            for token_id in self.stop_token_ids:
                log_probabilities[:, token_id] += stop_boost

        if not torch.isfinite(log_probabilities).any():
            fallback = torch.zeros_like(p_guided_final)
            fallback[:, self.eos_token_id] = 1.0
            assert_probability_distribution(fallback, "P_final_eos_fallback")
            diagnostics = {
                "order": "guided_constraints_before_fusion",
                "used_fallback": True,
                "fallback_reason": "no_finite_constrained_logits",
                "generated_count": generated_count,
                "forbidden_token_count": len(self.forbidden_token_ids),
                "repeated_token_ids": repeated_token_ids,
                "early_stop_ids": early_stop_ids,
                "hard_suppressed_token_ids": early_stop_ids,
                "stop_boost_log": float(stop_boost),
                "before_guided_constraints": before,
                "after_guided_constraints": probability_diagnostics(
                    fallback, self.tokenizer, self.config.diagnostic_top_n
                ),
            }
            return fallback, True, diagnostics

        constrained = torch.softmax(log_probabilities, dim=-1)
        if not torch.isfinite(constrained).all() or constrained.sum() <= 0:
            fallback = torch.zeros_like(p_guided_final)
            fallback[:, self.eos_token_id] = 1.0
            assert_probability_distribution(fallback, "P_final_eos_fallback")
            diagnostics = {
                "order": "guided_constraints_before_fusion",
                "used_fallback": True,
                "fallback_reason": "invalid_constrained_distribution",
                "generated_count": generated_count,
                "forbidden_token_count": len(self.forbidden_token_ids),
                "repeated_token_ids": repeated_token_ids,
                "early_stop_ids": early_stop_ids,
                "hard_suppressed_token_ids": early_stop_ids,
                "stop_boost_log": float(stop_boost),
                "before_guided_constraints": before,
                "after_guided_constraints": probability_diagnostics(
                    fallback, self.tokenizer, self.config.diagnostic_top_n
                ),
            }
            return fallback, True, diagnostics

        assert_probability_distribution(
            constrained,
            "P_guided_constrained",
        )
        diagnostics = {
            "order": "guided_constraints_before_fusion",
            "used_fallback": False,
            "fallback_reason": "",
            "generated_count": generated_count,
            "forbidden_token_count": len(self.forbidden_token_ids),
            "forbidden_factor": float(self.config.forbidden_factor),
            "repetition_penalty": float(self.config.repetition_penalty),
            "repeated_token_ids": repeated_token_ids,
            "early_stop_ids": [int(value) for value in early_stop_ids],
            "hard_suppressed_token_ids": [
                int(value) for value in early_stop_ids
            ],
            "stop_boost_log": float(stop_boost),
            "before_guided_constraints": before,
            "after_guided_constraints": probability_diagnostics(
                constrained, self.tokenizer, self.config.diagnostic_top_n
            ),
        }
        return constrained, False, diagnostics

    def stop_reason_for_token(self, token_id, used_fallback):
        token_id = int(token_id)
        if used_fallback and token_id == self.eos_token_id:
            return "eos_fallback"
        if token_id in self.stop_token_ids:
            return "period"
        if token_id == self.eos_token_id:
            return "eos"
        return ""

    def decode_full(self, token_ids):
        flat_token_ids = (
            token_ids.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )

        return self.tokenizer.decode(
            flat_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def caption_from_full(self, full_text, prompt_text, generated_token_ids):
        if full_text.startswith(prompt_text):
            caption = full_text[len(prompt_text):]
        else:
            caption = self.tokenizer.decode(
                list(generated_token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        return postprocess_caption(caption, self.config.prompt)
