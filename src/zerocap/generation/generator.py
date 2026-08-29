"""ZeroCap beam generation with auditable CLIP final reranking."""

import math

import torch

from ..core.types import (
    BeamState,
    assert_probability_distribution,
    clone_context_delta,
    probability_diagnostics,
)


class ZeroCapGenerator:
    def __init__(
        self,
        models,
        guidance,
        context_optimizer,
        decoder,
        config,
    ):
        self.models = models
        self.guidance = guidance
        self.context_optimizer = context_optimizer
        self.decoder = decoder
        self.config = config

    def _prompt_ids(self):
        prompt_only_ids = self.models.gpt_tokenizer(
            self.config.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.models.device)
        if prompt_only_ids.shape[1] < 2:
            raise AssertionError(
                "Prompt must tokenize to at least two GPT-2 tokens."
            )
        if not self.config.prepend_bos_token:
            return prompt_only_ids
        bos_token_id = self.models.gpt_tokenizer.bos_token_id
        if bos_token_id is None:
            raise RuntimeError("GPT-2 tokenizer has no BOS token.")
        bos_ids = torch.full(
            (1, 1),
            int(bos_token_id),
            dtype=prompt_only_ids.dtype,
            device=prompt_only_ids.device,
        )
        encoded = torch.cat([bos_ids, prompt_only_ids], dim=1)
        if int(encoded[0, 0].item()) != int(bos_token_id):
            raise AssertionError("BOS token was not prepended to the prompt.")
        return encoded

    def generate(self, image_feature, trace=None):
        prompt_ids = self._prompt_ids()
        prompt_text = self.decoder.decode_full(prompt_ids)
        beams = [BeamState(token_ids=prompt_ids)]
        loop_iterations = 0

        if trace is not None:
            trace.record(
                "generation_started",
                prompt=self.config.prompt,
                prompt_text=prompt_text,
                prompt_token_ids=[
                    int(value)
                    for value in prompt_ids.detach().cpu().reshape(-1).tolist()
                ],
                profile={
                    "prepend_bos_token": self.config.prepend_bos_token,
                    "bos_token_id": (
                        int(self.models.gpt_tokenizer.bos_token_id)
                        if self.models.gpt_tokenizer.bos_token_id is not None
                        else None
                    ),
                    "top_k": self.config.top_k,
                    "inner_iterations": self.config.inner_iterations,
                    "beam_size": self.config.beam_size,
                    "max_new_tokens": self.config.max_new_tokens,
                    "fusion_factor": self.config.fusion_factor,
                    "fluency_weight": self.config.fluency_weight,
                    "final_rerank_mode": (
                        self.config.final_rerank_mode
                    ),
                    "final_rerank_text_mode": (
                        self.config.final_rerank_text_mode
                    ),
                    "final_rerank_clip_weight": (
                        self.config.final_rerank_clip_weight
                    ),
                    "final_rerank_beam_weight": (
                        self.config.final_rerank_beam_weight
                    ),
                },
            )

        for step_index in range(self.config.max_new_tokens):
            loop_iterations += 1
            candidates = []

            for beam_index, beam in enumerate(beams):
                if beam.stopped:
                    candidates.append(beam)
                    continue

                step_result = self.context_optimizer.optimize(
                    current_ids=beam.token_ids,
                    image_feature=image_feature,
                    previous_delta=beam.context_delta,
                    trace=trace,
                    trace_context={
                        "step": step_index + 1,
                        "beam_index": beam_index,
                        "beam_text": self.decoder.decode_full(beam.token_ids),
                        "generated_token_ids": [
                            int(value) for value in beam.generated_token_ids
                        ],
                    },
                )
                (
                    p_guided_constrained,
                    used_fallback,
                    constraint_diagnostics,
                ) = self.decoder.apply_guided_constraints(
                    step_result.p_guided_final,
                    beam.generated_token_ids,
                )
                if used_fallback:
                    p_final = p_guided_constrained
                else:
                    p_final = self.decoder.fuse(
                        p_guided_constrained,
                        step_result.p_original,
                        hard_suppressed_token_ids=(
                            constraint_diagnostics[
                                "hard_suppressed_token_ids"
                            ]
                        ),
                    )
                assert_probability_distribution(p_final, "P_final")

                positive_count = int(
                    torch.count_nonzero(p_final > 0).item()
                )
                expansion_count = min(
                    self.config.beam_size,
                    max(1, positive_count),
                )
                top_probabilities, top_ids = torch.topk(
                    p_final,
                    k=expansion_count,
                    dim=-1,
                )

                sibling_deltas = []
                expansion_rows = []
                for probability, token_id in zip(
                    top_probabilities[0],
                    top_ids[0],
                ):
                    token_value = int(token_id.item())
                    next_ids = torch.cat(
                        [beam.token_ids, token_id.view(1, 1)],
                        dim=1,
                    )
                    generated = (
                        *beam.generated_token_ids,
                        token_value,
                    )
                    stop_reason = self.decoder.stop_reason_for_token(
                        token_value,
                        used_fallback=used_fallback,
                    )
                    stopped = bool(stop_reason)

                    if (
                        step_index + 1 >= self.config.max_new_tokens
                        and not stopped
                    ):
                        stopped = True
                        stop_reason = "max_tokens"

                    child_delta = clone_context_delta(
                        step_result.context_delta
                    )
                    sibling_deltas.append(child_delta)
                    expansion_rows.append(
                        {
                            "token_id": token_value,
                            "token_text": self.models.gpt_tokenizer.decode(
                                [token_value],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            "probability": float(probability.item()),
                            "next_full_text": self.decoder.decode_full(next_ids),
                            "stopped": stopped,
                            "stop_reason": stop_reason,
                        }
                    )
                    candidates.append(
                        BeamState(
                            token_ids=next_ids,
                            generated_token_ids=generated,
                            accumulated_logprob=(
                                beam.accumulated_logprob
                                + math.log(
                                    max(
                                        float(probability.item()),
                                        self.config.probability_eps,
                                    )
                                )
                            ),
                            stopped=stopped,
                            stop_reason=stop_reason,
                            context_delta=child_delta,
                        )
                    )

                if trace is not None:
                    trace.record(
                        "beam_distribution_and_expansion",
                        step=step_index + 1,
                        beam_index=beam_index,
                        beam_text=self.decoder.decode_full(beam.token_ids),
                        p_original=probability_diagnostics(
                            step_result.p_original,
                            self.models.gpt_tokenizer,
                            self.config.diagnostic_top_n,
                        ),
                        p_guided_final=probability_diagnostics(
                            step_result.p_guided_final,
                            self.models.gpt_tokenizer,
                            self.config.diagnostic_top_n,
                        ),
                        p_guided_constrained=probability_diagnostics(
                            p_guided_constrained,
                            self.models.gpt_tokenizer,
                            self.config.diagnostic_top_n,
                        ),
                        p_fused=probability_diagnostics(
                            p_final,
                            self.models.gpt_tokenizer,
                            self.config.diagnostic_top_n,
                        ),
                        constraints=constraint_diagnostics,
                        expansions=expansion_rows,
                    )

                non_null_deltas = [
                    delta
                    for delta in sibling_deltas
                    if delta is not None
                ]
                if len(non_null_deltas) > 1:
                    first_ptrs = {
                        delta[0][0].data_ptr()
                        for delta in non_null_deltas
                    }
                    if len(first_ptrs) != len(non_null_deltas):
                        raise AssertionError(
                            "Sibling beams unexpectedly share context delta storage."
                        )

            beams = sorted(
                candidates,
                key=lambda state: state.normalized_score(
                    self.config.beam_length_penalty
                ),
                reverse=True,
            )[: self.config.beam_size]

            if trace is not None:
                trace.record(
                    "beam_step_completed",
                    step=step_index + 1,
                    retained_beams=[
                        {
                            "text": self.decoder.decode_full(beam.token_ids),
                            "generated_token_ids": [
                                int(value) for value in beam.generated_token_ids
                            ],
                            "accumulated_logprob": float(
                                beam.accumulated_logprob
                            ),
                            "normalized_score": float(
                                beam.normalized_score(
                                    self.config.beam_length_penalty
                                )
                            ),
                            "stopped": bool(beam.stopped),
                            "stop_reason": beam.stop_reason,
                        }
                        for beam in beams
                    ],
                )

            if self.config.debug:
                print(
                    f"beam step {step_index + 1}:",
                    [
                        {
                            "text": self.decoder.decode_full(beam.token_ids),
                            "score": beam.accumulated_logprob,
                            "normalized_score": beam.normalized_score(
                                self.config.beam_length_penalty
                            ),
                            "stopped": beam.stopped,
                            "reason": beam.stop_reason,
                        }
                        for beam in beams
                    ],
                )
                print("P_final sum:", float(p_final.sum().item()))

            if beams and all(beam.stopped for beam in beams):
                break

        if loop_iterations > self.config.max_new_tokens:
            raise AssertionError("Generation loop exceeded max_new_tokens.")
        if not beams:
            raise RuntimeError("Beam search produced no beams.")

        for beam in beams:
            if not beam.stopped:
                beam.stopped = True
                beam.stop_reason = "max_tokens"
            if beam.stop_reason not in self.decoder.VALID_STOP_REASONS:
                raise AssertionError(
                    f"Invalid stop reason: {beam.stop_reason!r}"
                )
            if len(beam.generated_token_ids) > self.config.max_new_tokens:
                raise AssertionError("Generated token limit was exceeded.")
            if beam.context_delta is not None:
                for delta_key, delta_value in beam.context_delta:
                    for delta_tensor in (delta_key, delta_value):
                        if (
                            delta_tensor.requires_grad
                            or delta_tensor.grad_fn is not None
                        ):
                            raise AssertionError(
                                "A beam retained a computation graph."
                            )

        full_candidate_texts = [
            self.decoder.decode_full(beam.token_ids)
            for beam in beams
        ]
        candidate_captions = [
            self.decoder.caption_from_full(
                full_text=full_candidate_texts[index],
                prompt_text=prompt_text,
                generated_token_ids=beam.generated_token_ids,
            )
            for index, beam in enumerate(beams)
        ]
        if self.config.final_rerank_text_mode == "caption_only":
            rerank_candidate_texts = candidate_captions
        elif self.config.final_rerank_text_mode == "prompt_plus_caption":
            rerank_candidate_texts = full_candidate_texts
        else:
            raise AssertionError(
                f"Unsupported final rerank text mode: "
                f"{self.config.final_rerank_text_mode!r}."
            )
        normalized_beam_scores = [
            float(
                beam.normalized_score(
                    self.config.beam_length_penalty
                )
            )
            for beam in beams
        ]
        (
            best_index,
            similarities,
            normalized_beam_scores,
            rerank_scores,
            clip_only_index,
            beam_only_index,
        ) = self.guidance.rerank(
            image_feature=image_feature,
            candidate_texts=rerank_candidate_texts,
            normalized_beam_scores=normalized_beam_scores,
        )
        best_beam = beams[best_index]
        caption = candidate_captions[best_index]
        if not caption:
            raise AssertionError("Generated caption is empty.")

        if trace is not None:
            trace.record(
                "final_clip_rerank",
                rerank_mode=self.config.final_rerank_mode,
                rerank_text_mode=self.config.final_rerank_text_mode,
                clip_weight=float(
                    self.config.final_rerank_clip_weight
                ),
                beam_weight=float(
                    self.config.final_rerank_beam_weight
                ),
                clip_only_beam_index=clip_only_index,
                beam_only_beam_index=beam_only_index,
                selected_differs_from_clip_only=(
                    best_index != clip_only_index
                ),
                selected_differs_from_beam_only=(
                    best_index != beam_only_index
                ),
                candidates=[
                    {
                        "beam_index": index,
                        "full_text": full_candidate_texts[index],
                        "caption": candidate_captions[index],
                        "rerank_text": rerank_candidate_texts[index],
                        "clip_similarity": float(similarities[index]),
                        "weighted_clip_component": float(
                            self.config.final_rerank_clip_weight
                            * similarities[index]
                        ),
                        "weighted_beam_component": float(
                            (
                                self.config.final_rerank_beam_weight
                                * normalized_beam_scores[index]
                            )
                            if self.config.final_rerank_mode
                            == "clip_beam"
                            else 0.0
                        ),
                        "rerank_score": float(rerank_scores[index]),
                        "accumulated_logprob": float(
                            beams[index].accumulated_logprob
                        ),
                        "normalized_beam_score": float(
                            normalized_beam_scores[index]
                        ),
                        "stop_reason": beams[index].stop_reason,
                        "selected": index == best_index,
                    }
                    for index in range(len(beams))
                ],
                selected_beam_index=best_index,
                selected_caption=caption,
            )

        if self.config.debug:
            print("Final beam full texts:", full_candidate_texts)
            print("Final candidate captions:", candidate_captions)
            print(
                f"Final rerank texts ({self.config.final_rerank_text_mode}):",
                rerank_candidate_texts,
            )
            print("Final CLIP similarities:", similarities)
            print("Final normalized beam scores:", normalized_beam_scores)
            print(
                f"Final rerank scores ({self.config.final_rerank_mode}):",
                rerank_scores,
            )
            print(
                "CLIP-only / beam-only / selected indices:",
                clip_only_index,
                beam_only_index,
                best_index,
            )
            print("Selected caption:", caption)

        payload = {
            "caption": caption,
            "generated_token_ids": [
                int(value)
                for value in best_beam.generated_token_ids
            ],
            "stop_reason": best_beam.stop_reason,
            "final_clip_similarity": float(similarities[best_index]),
        }
        del beams, candidates
        return payload
