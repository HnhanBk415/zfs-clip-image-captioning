from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .cli import main as inference_main
from src.config.clipcap_config import (
    CLIPCAP_CHECKPOINT_ROOT,
    CLIPCAP_DEFAULT_INFERENCE_CONFIG,
    CLIPCAP_OUTPUT_ROOT,
    CLIPCAP_TRAIN_SUBSETS,
    ClipCapInferenceConfig,
    FEATURE_DIR,
)
from src.config.common_config import RAW_IMAGES_DIR, SEED, SPLIT_DIR


DATASET_VALIDATION = "val"
DATASET_FINAL_TEST = "fixed_test_round_001"
SUPPORTED_DATASETS = (DATASET_VALIDATION, DATASET_FINAL_TEST)
RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class InferenceSelection:
    dataset: str
    manifest_path: Path
    output_dir: Path
    effective_run_tag: str
    validation_chunk: int | None


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


def _parse_validation_chunk(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"full", "none"}:
        return None
    try:
        chunk = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "validation chunk must be 'full' or an integer from 1 to 9"
        ) from error
    if not 1 <= chunk <= 9:
        raise argparse.ArgumentTypeError(
            "validation chunk must be 'full' or an integer from 1 to 9"
        )
    return chunk


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-prompt ClipCap inference on validation or the fixed final "
            "test round."
        )
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument(
        "--validation-chunk",
        type=_parse_validation_chunk,
        default=None,
        metavar="{full,1..9}",
        help="Only applies to --dataset val; defaults to the full validation split.",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--split-dir", type=Path, default=Path(SPLIT_DIR))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=_environment_path(
            "ZFS_CLIP_CHECKPOINT_ROOT",
            CLIPCAP_CHECKPOINT_ROOT,
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=_environment_path(
            "ZFS_CLIP_FEATURE_CACHE",
            FEATURE_DIR / "clip_features.pt",
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_environment_path("ZFS_CLIP_IMAGE_DIR", RAW_IMAGES_DIR),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=_environment_path(
            "ZFS_CLIP_INFERENCE_OUTPUT_BASE",
            CLIPCAP_OUTPUT_ROOT / "evaluation",
        ),
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=CLIPCAP_TRAIN_SUBSETS,
        default=list(CLIPCAP_TRAIN_SUBSETS),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=os.environ.get("ZFS_CLIP_DEVICE", "auto"))
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


def resolve_inference_selection(args: argparse.Namespace) -> InferenceSelection:
    if not args.run_tag or RUN_TAG_PATTERN.fullmatch(args.run_tag) is None:
        raise ValueError("run tag may only contain letters, numbers, '.', '_', and '-'")
    split_directory = Path(args.split_dir)
    if args.dataset == DATASET_VALIDATION:
        if args.validation_chunk is None:
            manifest_path = split_directory / "val.json"
            effective_run_tag = args.run_tag
        else:
            manifest_path = (
                split_directory
                / "val_chunks"
                / f"val_chunk_{args.validation_chunk:03d}.json"
            )
            effective_run_tag = (
                f"{args.run_tag}_val_chunk_{args.validation_chunk:03d}"
            )
        split_name = "val"
    else:
        if args.validation_chunk is not None:
            raise ValueError("--validation-chunk only applies to --dataset val")
        if not args.allow_test:
            raise RuntimeError(
                "Final-test inference is locked; pass --allow-test after validation."
            )
        manifest_path = split_directory / "fixed_test_round_001.json"
        effective_run_tag = args.run_tag
        split_name = "test"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Inference manifest not found: {manifest_path}")
    return InferenceSelection(
        dataset=args.dataset,
        manifest_path=manifest_path,
        output_dir=Path(args.output_base) / split_name / effective_run_tag,
        effective_run_tag=effective_run_tag,
        validation_chunk=args.validation_chunk,
    )


def build_inference_arguments(
    args: argparse.Namespace,
    selection: InferenceSelection,
) -> list[str]:
    inference_config = ClipCapInferenceConfig(
        image_batch_size=args.image_batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
    )
    feature_cache = Path(args.feature_cache)
    image_dir = Path(args.image_dir)
    if not feature_cache.is_file() and not image_dir.is_dir():
        raise FileNotFoundError(
            "Feature cache is missing and no image directory is available."
        )
    inference_arguments = [
        "--manifest",
        str(selection.manifest_path),
        "--feature-cache",
        str(feature_cache),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--subsets",
        *args.subsets,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(selection.output_dir),
        "--device",
        args.device,
        "--image-batch-size",
        str(inference_config.image_batch_size),
        "--max-new-tokens",
        str(inference_config.max_new_tokens),
        "--num-beams",
        str(inference_config.num_beams),
        "--num-return-sequences",
        str(inference_config.num_return_sequences),
        "--length-penalty",
        str(inference_config.length_penalty),
    ]
    if image_dir.is_dir():
        inference_arguments.extend(["--image-dir", str(image_dir)])
    inference_arguments.append(
        "--early-stopping"
        if inference_config.early_stopping
        else "--no-early-stopping"
    )
    if args.no_resume:
        inference_arguments.append("--no-resume")
    return inference_arguments


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    selection = resolve_inference_selection(args)
    inference_arguments = build_inference_arguments(args, selection)
    print(f"Dataset: {selection.dataset}")
    print(f"Manifest: {selection.manifest_path}")
    print(f"Run tag: {selection.effective_run_tag}")
    print(f"Output: {selection.output_dir}")
    inference_main(inference_arguments)


if __name__ == "__main__":
    main()
