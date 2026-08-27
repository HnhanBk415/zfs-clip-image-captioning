from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel

from src.clipcap.models.clipcap_model import ClipCaptionModel

from .features import extract_feature_tensor
from .records import BeamCandidate, GenerationResult


@torch.inference_mode()
def build_visual_prefix(image_features: Tensor, model: ClipCaptionModel) -> Tensor:
    mapper_parameter = next(model.mapper.parameters())
    features = image_features.to(
        device=mapper_parameter.device,
        dtype=mapper_parameter.dtype,
    )
    if features.ndim == 1:
        features = features.unsqueeze(0)
    prefix = model.mapper(features)
    expected_shape = (
        features.size(0),
        model.mapper.prefix_length,
        model.mapper.embedding_dim,
    )
    if tuple(prefix.shape) != expected_shape:
        raise ValueError(
            f"Visual prefix has shape {tuple(prefix.shape)}, expected {expected_shape}"
        )
    return prefix


def _trim_at_eos(token_ids: list[int], eos_token_id: int) -> tuple[int, ...]:
    trimmed: list[int] = []
    for token_id in token_ids:
        trimmed.append(token_id)
        if token_id == eos_token_id:
            break
    return tuple(trimmed)


@torch.inference_mode()
def beam_search_from_prefix(
    gpt2_model: GPT2LMHeadModel,
    text_tokenizer: Any,
    prefix_embeddings: Tensor,
    max_new_tokens: int,
    num_beams: int = 5,
    num_return_sequences: int = 5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> tuple[BeamCandidate, ...]:
    integer_parameters = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
    }
    for name, value in integer_parameters.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_beams < 2:
        raise ValueError("num_beams must be greater than one")
    if num_return_sequences > num_beams:
        raise ValueError("num_return_sequences cannot exceed num_beams")
    if prefix_embeddings.ndim != 3 or prefix_embeddings.size(0) != 1:
        raise ValueError("prefix_embeddings must have shape [1, P, D]")

    eos_token_id = text_tokenizer.eos_token_id
    pad_token_id = text_tokenizer.pad_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id and pad_token_id")
    prefix_length = int(prefix_embeddings.size(1))
    max_positions = getattr(gpt2_model.config, "n_positions", None)
    if max_positions is None:
        max_positions = getattr(gpt2_model.config, "max_position_embeddings", None)
    if max_positions is not None and prefix_length + max_new_tokens > max_positions:
        raise ValueError("Visual prefix and generated tokens exceed GPT-2 position limit")

    generated = gpt2_model.generate(
        inputs_embeds=prefix_embeddings,
        attention_mask=torch.ones(
            (1, prefix_length),
            dtype=torch.long,
            device=prefix_embeddings.device,
        ),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        length_penalty=float(length_penalty),
        early_stopping=early_stopping,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if generated.sequences.size(0) != num_return_sequences:
        raise ValueError("Beam search returned an unexpected number of sequences")
    if generated.sequences_scores is None:
        raise ValueError("Beam search did not return sequence scores")

    token_rows = generated.sequences.detach().cpu().tolist()
    captions = text_tokenizer.batch_decode(
        generated.sequences,
        skip_special_tokens=True,
    )
    return tuple(
        BeamCandidate(
            beam_rank=index,
            caption=caption.strip(),
            token_ids=_trim_at_eos(token_ids, eos_token_id),
            beam_score=float(score.detach().cpu()),
        )
        for index, (token_ids, caption, score) in enumerate(
            zip(token_rows, captions, generated.sequences_scores),
            start=1,
        )
    )


def compute_clip_similarity_scores(
    image_features: Tensor,
    text_features: Tensor,
) -> Tensor:
    if image_features.ndim != 2 or image_features.size(0) != 1:
        raise ValueError("image_features must have shape [1, clip_dim]")
    if text_features.ndim != 2 or text_features.size(0) < 1:
        raise ValueError("text_features must have shape [N, clip_dim]")
    if image_features.size(1) != text_features.size(1):
        raise ValueError("Image and text features must have the same dimension")
    if not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all():
        raise ValueError("CLIP features contain NaN or Inf")
    if image_features.norm(dim=-1).min() <= 0 or text_features.norm(dim=-1).min() <= 0:
        raise ValueError("CLIP features cannot have zero norm")
    normalized_image = F.normalize(image_features.float(), dim=-1)
    normalized_text = F.normalize(text_features.float(), dim=-1)
    return normalized_text @ normalized_image.transpose(0, 1)


@torch.inference_mode()
def clip_rerank_candidates(
    candidates: tuple[BeamCandidate, ...],
    image_features: Tensor,
    processor: CLIPProcessor,
    encoder: CLIPModel,
    device: torch.device,
) -> GenerationResult:
    if not candidates:
        raise ValueError("At least one beam candidate is required")
    valid_indices = [index for index, item in enumerate(candidates) if item.caption]
    if not valid_indices:
        raise ValueError("All beam candidates are empty")
    text_inputs = processor(
        text=[candidates[index].caption for index in valid_indices],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {name: tensor.to(device) for name, tensor in text_inputs.items()}
    text_features = extract_feature_tensor(encoder.get_text_features(**text_inputs))
    valid_scores = compute_clip_similarity_scores(
        image_features.to(device).reshape(1, -1),
        text_features,
    ).squeeze(1)
    scores = torch.full(
        (len(candidates),),
        -torch.inf,
        dtype=valid_scores.dtype,
        device=valid_scores.device,
    )
    scores[torch.tensor(valid_indices, device=device)] = valid_scores
    rescored = tuple(
        replace(candidate, clip_score=float(scores[index].detach().cpu()))
        for index, candidate in enumerate(candidates)
    )
    selected_index = int(scores.argmax().item())
    return GenerationResult(
        caption=rescored[selected_index].caption,
        selected_beam_rank=rescored[selected_index].beam_rank,
        candidates=rescored,
    )


@torch.inference_mode()
def generate_caption_from_feature(
    image_features: Tensor,
    processor: CLIPProcessor,
    encoder: CLIPModel,
    model: ClipCaptionModel,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 15,
    num_beams: int = 5,
    num_return_sequences: int = 5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> GenerationResult:
    visual_prefix = build_visual_prefix(image_features, model)
    candidates = beam_search_from_prefix(
        model.gpt2,
        tokenizer,
        visual_prefix,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
    )
    return clip_rerank_candidates(
        candidates,
        image_features,
        processor,
        encoder,
        device,
    )
