from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, GPT2LMHeadModel

from src.clipcap.models.clipcap_model import ClipCaptionModel
from src.config.clipcap_config import (
    CLIPCAP_DEFAULT_INFERENCE_CONFIG,
    CLIPCAP_FIXED_EPOCH_POLICY,
    CLIPCAP_OUTPUT_ROOT,
    CLIPCAP_TRAIN_SEED,
    CLIPCAP_TRAIN_SUBSETS,
    CLIP_MODEL_NAME,
    ClipCapInferenceConfig,
)

from .checkpoints import (
    FINAL_CHECKPOINT_NAME,
    build_mapper_from_checkpoint,
    resolve_fixed_epoch_checkpoints,
)
from .decoding import generate_caption_from_feature
from .inference_runner import (
    generation_result_to_record,
    prepare_run_config,
    run_evaluation,
)
from .features import (
    encode_evaluation_images,
    encode_image_with_clip,
    load_evaluation_manifest,
    load_feature_cache,
    load_rgb_image,
    save_feature_cache,
)
from .records import EvaluationItem


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_models(
    gpt2_model_name: str,
    device: torch.device,
) -> tuple[CLIPProcessor, CLIPModel, Any, GPT2LMHeadModel]:
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    encoder = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name, use_fast=True)
    gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for component in (encoder, gpt2):
        for parameter in component.parameters():
            parameter.requires_grad = False
        component.eval()
    return processor, encoder, tokenizer, gpt2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate no-prompt ClipCap captions with five-beam decoding and "
            "CLIP reranking from fixed-epoch final checkpoints."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Run one image as a smoke test")
    source.add_argument("--manifest", type=Path, help="JSON evaluation manifest")
    parser.add_argument("--image-dir", type=Path, help="Root containing manifest images")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(CLIPCAP_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=CLIPCAP_TRAIN_SUBSETS,
        default=list(CLIPCAP_TRAIN_SUBSETS),
    )
    parser.add_argument("--seed", type=int, default=CLIPCAP_TRAIN_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(CLIPCAP_OUTPUT_ROOT) / "evaluation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.image_batch_size,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.max_new_tokens,
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_beams,
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.num_return_sequences,
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.length_penalty,
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=CLIPCAP_DEFAULT_INFERENCE_CONFIG.early_stopping,
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def _inference_config_from_args(args: argparse.Namespace) -> ClipCapInferenceConfig:
    return ClipCapInferenceConfig(
        image_batch_size=args.image_batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
    )


def _run_single_image(args: argparse.Namespace) -> None:
    inference_config = _inference_config_from_args(args)
    device = _select_device(args.device)
    checkpoints = resolve_fixed_epoch_checkpoints(
        args.checkpoint_root,
        args.subsets,
        args.seed,
    )
    gpt2_model_name = checkpoints[0].checkpoint["gpt2_model_name"]
    processor, encoder, tokenizer, gpt2 = _load_models(gpt2_model_name, device)
    image = load_rgb_image(args.image)
    image_features = encode_image_with_clip(image, processor, encoder, device)
    for artifact in checkpoints:
        mapper = build_mapper_from_checkpoint(artifact, gpt2, encoder, device)
        model = ClipCaptionModel(mapper=mapper, gpt2=gpt2).to(device)
        result = generate_caption_from_feature(
            image_features,
            processor,
            encoder,
            model,
            tokenizer,
            device,
            max_new_tokens=inference_config.max_new_tokens,
            num_beams=inference_config.num_beams,
            num_return_sequences=inference_config.num_return_sequences,
            length_penalty=inference_config.length_penalty,
            early_stopping=inference_config.early_stopping,
        )
        print(
            json.dumps(
                generation_result_to_record(
                    EvaluationItem(args.image.name, ()),
                    result,
                    artifact.subset_name,
                    artifact.seed,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        del model
        del mapper
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_manifest(args: argparse.Namespace) -> None:
    inference_config = _inference_config_from_args(args)
    if args.image_dir is None and args.feature_cache is None:
        raise ValueError("--manifest requires --image-dir or an existing --feature-cache")
    device = _select_device(args.device)
    checkpoints = resolve_fixed_epoch_checkpoints(
        args.checkpoint_root,
        args.subsets,
        args.seed,
    )
    gpt2_model_name = checkpoints[0].checkpoint["gpt2_model_name"]
    processor, encoder, tokenizer, gpt2 = _load_models(gpt2_model_name, device)
    items = load_evaluation_manifest(args.manifest)
    image_ids = [item.image_id for item in items]
    if args.feature_cache is not None and args.feature_cache.is_file():
        features = load_feature_cache(args.feature_cache, image_ids)
    else:
        if args.image_dir is None:
            raise ValueError("--image-dir is required to build a missing feature cache")
        features = encode_evaluation_images(
            items,
            args.image_dir,
            processor,
            encoder,
            device,
            batch_size=inference_config.image_batch_size,
        )
        if args.feature_cache is not None:
            save_feature_cache(features, args.feature_cache)

    run_metadata = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "image_dir": str(args.image_dir.resolve()) if args.image_dir else None,
        "feature_cache": (
            str(args.feature_cache.resolve()) if args.feature_cache else None
        ),
        "subsets": list(args.subsets),
        "seed": args.seed,
        "checkpoint": FINAL_CHECKPOINT_NAME,
        "training_policy": CLIPCAP_FIXED_EPOCH_POLICY,
        "prompt": None,
        **inference_config.to_dict(),
        "num_images": len(items),
        "clip_model": CLIP_MODEL_NAME,
        "gpt2_model": gpt2_model_name,
    }
    prepare_run_config(
        run_metadata,
        args.output_dir / "run_config.json",
        resume=not args.no_resume,
    )
    output_paths = run_evaluation(
        items,
        features,
        checkpoints,
        processor,
        encoder,
        gpt2,
        tokenizer,
        device,
        args.output_dir,
        max_new_tokens=inference_config.max_new_tokens,
        num_beams=inference_config.num_beams,
        num_return_sequences=inference_config.num_return_sequences,
        length_penalty=inference_config.length_penalty,
        early_stopping=inference_config.early_stopping,
        resume=not args.no_resume,
    )
    for subset_name, path in output_paths.items():
        print(f"{subset_name}: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.image is not None:
        _run_single_image(args)
    else:
        _run_manifest(args)
