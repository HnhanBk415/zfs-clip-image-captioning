"""CLIP token guidance and final caption reranking."""

import torch
import torch.nn.functional as F


class CLIPGuidance:
    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.device = models.device

    @staticmethod
    def _feature_tensor(value):
        if torch.is_tensor(value):
            return value
        if hasattr(value, "pooler_output"):
            return value.pooler_output
        if isinstance(value, (tuple, list)) and value:
            return value[0]
        raise TypeError(f"Unsupported CLIP text feature output: {type(value)}")

    def _decode_for_clip(self, token_ids):
        visible_ids = [int(value) for value in token_ids]
        bos_token_id = self.models.gpt_tokenizer.bos_token_id
        leading_bos_removed = bool(
            visible_ids
            and bos_token_id is not None
            and visible_ids[0] == int(bos_token_id)
        )
        if self.config.prepend_bos_token and not leading_bos_removed:
            raise AssertionError(
                "A CLIP guidance candidate is missing the required leading BOS."
            )
        if leading_bos_removed:
            visible_ids = visible_ids[1:]
        text = self.models.gpt_tokenizer.decode(
            visible_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text, leading_bos_removed

    def encode_texts(self, texts):
        all_features = []
        chunk_size = self.config.clip_candidate_chunk_size
        for start in range(0, len(texts), chunk_size):
            chunk = list(texts[start: start + chunk_size])
            inputs = self.models.clip_processor(
                text=chunk,
                padding=True,
                truncation=True,
                max_length=self.config.clip_max_text_tokens,
                return_tensors="pt",
            )
            text_inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.no_grad():
                output = self.models.clip_model.get_text_features(
                    **text_inputs
                )
                features = self._feature_tensor(output)
                features = F.normalize(features, dim=-1)
            if not torch.isfinite(features).all():
                raise AssertionError("CLIP text features contain NaN/Inf.")
            all_features.append(features.detach())

        encoded = torch.cat(all_features, dim=0)
        if encoded.shape[0] != len(texts):
            raise AssertionError("CLIP text feature count mismatch.")
        return encoded

    def build_target(
        self,
        image_feature,
        current_ids,
        top_indices,
        vocabulary_size,
        target_dtype,
    ):
        tokenizer = self.models.gpt_tokenizer
        current_token_ids = [
            int(value)
            for value in current_ids[0].detach().cpu().tolist()
        ]
        current_text, current_bos_removed = self._decode_for_clip(
            current_token_ids
        )

        candidate_texts = []
        decoded_tokens = []
        candidate_text_prefix_matches = []
        for token_id in top_indices[0].detach().cpu().tolist():
            token_id = int(token_id)
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            candidate_token_ids = [*current_token_ids, token_id]
            candidate, candidate_bos_removed = self._decode_for_clip(
                candidate_token_ids
            )
            if not candidate_bos_removed:
                raise AssertionError("Leading BOS was not removed before CLIP.")
            if candidate_token_ids[:-1] != current_token_ids:
                raise AssertionError(
                    "Candidate token sequence lost the full current prefix."
                )
            candidate_texts.append(candidate)
            decoded_tokens.append(token_text)
            candidate_text_prefix_matches.append(
                candidate.startswith(current_text)
            )

        text_features = self.encode_texts(candidate_texts)
        if text_features.shape[-1] != image_feature.shape[-1]:
            raise AssertionError(
                "CLIP image/text projection dimensions do not match."
            )
        similarities = (
            text_features @ image_feature.detach().transpose(0, 1)
        ).squeeze(1)
        if not torch.isfinite(similarities).all():
            raise AssertionError("CLIP candidate similarities contain NaN/Inf.")

        target_topk = torch.softmax(
            similarities / self.config.clip_temperature,
            dim=0,
        ).detach()
        target_vocab = torch.zeros(
            (1, vocabulary_size),
            device=self.device,
            dtype=target_dtype,
        )
        target_vocab.scatter_(
            dim=1,
            index=top_indices,
            src=target_topk.unsqueeze(0).to(target_dtype),
        )
        target_vocab = target_vocab.detach()

        if not torch.allclose(
            target_vocab.sum(),
            torch.ones((), device=self.device, dtype=target_dtype),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise AssertionError("target_vocab does not sum to one.")

        outside_topk = target_vocab.clone()
        outside_topk.scatter_(
            1,
            top_indices,
            torch.zeros_like(top_indices, dtype=target_dtype),
        )
        if torch.count_nonzero(outside_topk).item() != 0:
            raise AssertionError("CLIP target is non-zero outside Top-K.")

        diagnostics = {
            "candidate_construction": "full_sequence_decode",
            "leading_bos_removed_before_clip": current_bos_removed,
            "current_text": current_text,
            "decoded_top_tokens": decoded_tokens,
            "candidate_texts": candidate_texts,
            "candidate_text_prefix_matches": (
                candidate_text_prefix_matches
            ),
            "similarities": [
                float(value)
                for value in similarities.detach().cpu().tolist()
            ],
            "target_sum": float(target_vocab.sum().item()),
        }
        return target_vocab, diagnostics

    def rerank(
        self,
        image_feature,
        candidate_texts,
        normalized_beam_scores,
    ):
        if len(candidate_texts) != len(normalized_beam_scores):
            raise AssertionError("Final rerank candidate/score count mismatch.")
        if not candidate_texts:
            raise AssertionError("Final rerank requires at least one candidate.")
        if not all(str(text).strip() for text in candidate_texts):
            raise AssertionError("Final rerank received an empty caption.")
        text_features = self.encode_texts(candidate_texts)
        similarities = (
            text_features @ image_feature.detach().transpose(0, 1)
        ).squeeze(1)
        if not torch.isfinite(similarities).all():
            raise AssertionError("Final CLIP reranking produced NaN/Inf.")

        beam_scores = torch.tensor(
            normalized_beam_scores,
            device=similarities.device,
            dtype=similarities.dtype,
        )
        if not torch.isfinite(beam_scores).all():
            raise AssertionError("Final beam scores contain NaN/Inf.")

        clip_component = (
            self.config.final_rerank_clip_weight * similarities
        )
        if self.config.final_rerank_mode == "clip_only":
            rerank_scores = clip_component
        elif self.config.final_rerank_mode == "clip_beam":
            rerank_scores = (
                clip_component
                + self.config.final_rerank_beam_weight * beam_scores
            )
        else:
            raise AssertionError(
                f"Unsupported final rerank mode: "
                f"{self.config.final_rerank_mode!r}."
            )
        if not torch.isfinite(rerank_scores).all():
            raise AssertionError("Final rerank scores contain NaN/Inf.")

        best_index = int(torch.argmax(rerank_scores).item())
        clip_only_index = int(torch.argmax(similarities).item())
        beam_only_index = int(torch.argmax(beam_scores).item())
        return (
            best_index,
            [float(value) for value in similarities.detach().cpu().tolist()],
            [float(value) for value in beam_scores.detach().cpu().tolist()],
            [float(value) for value in rerank_scores.detach().cpu().tolist()],
            clip_only_index,
            beam_only_index,
        )
