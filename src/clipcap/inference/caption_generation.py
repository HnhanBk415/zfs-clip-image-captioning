"""Compatibility facade for the modular ClipCap inference package."""

from .checkpoints import (
    build_mapper_from_checkpoint,
    load_fixed_epoch_checkpoint,
    resolve_fixed_epoch_checkpoints,
    validate_artifact_directory,
)
from .cli import main
from .decoding import (
    beam_search_from_prefix,
    build_visual_prefix,
    clip_rerank_candidates,
    compute_clip_similarity_scores,
    generate_caption_from_feature,
)
from .inference_runner import prepare_run_config, run_evaluation
from .features import (
    encode_evaluation_images,
    encode_image_with_clip,
    load_evaluation_manifest,
    load_feature_cache,
    load_rgb_image,
    save_feature_cache,
)
from .records import (
    BeamCandidate,
    EvaluationItem,
    FixedEpochCheckpoint,
    GenerationResult,
)

__all__ = [
    "BeamCandidate",
    "EvaluationItem",
    "FixedEpochCheckpoint",
    "GenerationResult",
    "beam_search_from_prefix",
    "build_mapper_from_checkpoint",
    "build_visual_prefix",
    "clip_rerank_candidates",
    "compute_clip_similarity_scores",
    "encode_evaluation_images",
    "encode_image_with_clip",
    "generate_caption_from_feature",
    "load_evaluation_manifest",
    "load_feature_cache",
    "load_fixed_epoch_checkpoint",
    "load_rgb_image",
    "main",
    "prepare_run_config",
    "resolve_fixed_epoch_checkpoints",
    "run_evaluation",
    "save_feature_cache",
    "validate_artifact_directory",
]


if __name__ == "__main__":
    main()
