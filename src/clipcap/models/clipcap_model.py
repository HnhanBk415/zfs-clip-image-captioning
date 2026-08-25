from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import GPT2LMHeadModel

from .mapping_network import TransformerMapper


IGNORE_INDEX = -100


def _validate_prefix_length(prefix_length: int) -> None:
    if (
        isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or prefix_length <= 0
    ):
        raise ValueError("prefix_length must be a positive integer")


def extend_attention_mask(
    attention_mask: Tensor,
    prefix_length: int,
) -> Tensor:
    """Prepend visible prefix positions to a caption attention mask."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, L]")

    _validate_prefix_length(prefix_length)
    prefix_mask = torch.ones(
        (attention_mask.size(0), prefix_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([prefix_mask, attention_mask], dim=1)


def build_extended_labels(
    input_ids: Tensor,
    attention_mask: Tensor,
    prefix_length: int,
) -> Tensor:
    """Create loss labels while ignoring visual-prefix and padding positions."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, L]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long dtype")
    if input_ids.device != attention_mask.device:
        raise ValueError("input_ids and attention_mask must be on the same device")

    _validate_prefix_length(prefix_length)

    caption_labels = input_ids.clone()
    caption_labels.masked_fill_(attention_mask == 0, IGNORE_INDEX)
    prefix_labels = torch.full(
        (input_ids.size(0), prefix_length),
        fill_value=IGNORE_INDEX,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([prefix_labels, caption_labels], dim=1)


class ClipCaptionModel(nn.Module):
    """Connect a CLIP-to-prefix mapper to GPT-2 for caption generation."""

    def __init__(
        self,
        mapper: TransformerMapper,
        gpt2: GPT2LMHeadModel,
    ) -> None:
        super().__init__()
        self.mapper = mapper
        self.gpt2 = gpt2

        gpt_embedding_dim = self.gpt2.get_input_embeddings().embedding_dim
        if self.mapper.embedding_dim != gpt_embedding_dim:
            raise ValueError(
                "Mapper embedding dimension must match GPT-2 embedding dimension"
            )

    @staticmethod
    def _validate_batch(
        image_embed: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None,
    ) -> None:
        if image_embed.ndim != 2:
            raise ValueError("image_embed must have shape [B, clip_dim]")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if image_embed.size(0) != input_ids.size(0):
            raise ValueError("All inputs must have the same batch size")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long dtype")
        if not (
            image_embed.device
            == input_ids.device
            == attention_mask.device
        ):
            raise ValueError("All inputs must be on the same device")

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels and input_ids must have the same shape")
            if labels.dtype != torch.long:
                raise TypeError("labels must use torch.long dtype")
            if labels.device != input_ids.device:
                raise ValueError("labels and input_ids must be on the same device")

    def prepare_gpt2_inputs(
        self,
        image_embed: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Build aligned GPT-2 inputs and create labels only when requested."""
        self._validate_batch(
            image_embed,
            input_ids,
            attention_mask,
            labels,
        )

        prefix_embeddings = self.mapper(image_embed)
        prefix_length = prefix_embeddings.size(1)
        text_embeddings = self.gpt2.get_input_embeddings()(input_ids)

        if prefix_embeddings.size(2) != text_embeddings.size(2):
            raise ValueError("Prefix and text embedding dimensions do not match")
        if prefix_embeddings.dtype != text_embeddings.dtype:
            raise TypeError("Prefix and text embeddings must use the same dtype")

        inputs_embeds = torch.cat(
            [prefix_embeddings, text_embeddings],
            dim=1,
        )
        max_positions = getattr(self.gpt2.config, "n_positions", None)
        if max_positions is None:
            max_positions = getattr(
                self.gpt2.config,
                "max_position_embeddings",
                None,
            )
        if max_positions is not None and inputs_embeds.size(1) > max_positions:
            raise ValueError("Combined sequence exceeds GPT-2 position limit")

        extended_attention_mask = extend_attention_mask(
            attention_mask,
            prefix_length,
        )
        extended_labels = None
        if labels is not None:
            extended_labels = build_extended_labels(
                labels,
                attention_mask,
                prefix_length,
            )

        sequence_length = inputs_embeds.size(1)
        if extended_attention_mask.size(1) != sequence_length:
            raise ValueError("Extended attention mask length mismatch")
        if (
            extended_labels is not None
            and extended_labels.size(1) != sequence_length
        ):
            raise ValueError("Extended labels length mismatch")

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": extended_attention_mask,
            "labels": extended_labels,
        }

    def forward(
        self,
        image_embed: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ):
        """Run GPT-2 with an image-conditioned visual prefix."""
        gpt2_inputs = self.prepare_gpt2_inputs(
            image_embed=image_embed,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return self.gpt2(**gpt2_inputs, return_dict=True)
