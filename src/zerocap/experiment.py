"""ZeroCap regression, benchmark, fixed-TEST selection, and prediction runs.

Evaluation is intentionally excluded. ZeroCap and ClipCap must feed their saved
predictions to the same project-level evaluator.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers

from src.config.zerocap_config import ZeroCapRunConfig, seed_everything

from .captioner import ZeroCapCaptioner
from .data import Flickr8kData, deterministic_sample
from .storage import PredictionStore


PINNED_TRANSFORMERS_VERSION = "4.56.2"
LIBRARY_VERSION_NAMES = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "kagglehub",
    "numpy",
    "Pillow",
    "pandas",
    "pycocoevalcap",
    "nltk",
)


@dataclass
class ExperimentContext:
    project_root: Path
    output_root: Path
    config: ZeroCapRunConfig
    data: Flickr8kData
    captioner: ZeroCapCaptioner
    store: PredictionStore
    val_warmup_ids: list[str]
    val_benchmark_ids: list[str]
    test_ids: list[str]


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _split_manifest_sha256(project_root: Path) -> str:
    path = (
        project_root
        / "data"
        / "flickr8k"
        / "metadata"
        / "split_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in LIBRARY_VERSION_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _assert_runtime() -> str:
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "ZeroCap modules require transformers=="
            f"{PINNED_TRANSFORMERS_VERSION}, loaded {transformers.__version__}. "
            "Restart the Colab session after installing pinned dependencies."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a Colab T4 GPU and restart the session."
        )
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", gpu_name)
    return gpu_name


def _fixed_val_ids(data: Flickr8kData, config: ZeroCapRunConfig):
    val_ids = data.split_ids("val")
    smoke_id = deterministic_sample(val_ids, 1, config.seed)[0]
    remaining = [
        image_id
        for image_id in sorted(val_ids)
        if image_id != smoke_id
    ]
    other_ids = deterministic_sample(
        remaining,
        config.benchmark_warmup_images - 1 + config.benchmark_val_images,
        config.seed + 1,
    )
    warmup_ids = [
        smoke_id,
        *other_ids[: config.benchmark_warmup_images - 1],
    ]
    benchmark_ids = other_ids[config.benchmark_warmup_images - 1 :]
    return warmup_ids, benchmark_ids


def _baseline_val_tune_config(base_config: ZeroCapRunConfig):
    grid = dict(ZeroCapRunConfig.val_tune_grid())
    return base_config.for_val_tune_candidate("A_baseline", grid["A_baseline"])


def prepare_experiment(
    mode: str,
    project_root: Path,
    output_root: Path,
    time_budget_hours: float = 4.0,
    benchmark_val_images: int = 5,
) -> ExperimentContext:
    allowed_modes = {"regression", "benchmark", "test_smoke", "final_test"}
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}")

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_name = _assert_runtime()
    git_commit = _git_commit(project_root)
    config_mode = "val_tune" if mode == "regression" else mode
    config = ZeroCapRunConfig.for_mode(
        run_mode=config_mode,
        git_commit=git_commit,
        split_manifest_sha256=_split_manifest_sha256(project_root),
        output_root=output_root,
        time_budget_hours=time_budget_hours,
        benchmark_val_images=benchmark_val_images,
        run_heavy_metrics=False,
    )
    if mode == "regression":
        config = _baseline_val_tune_config(config)

    seed_everything(config.seed)
    data = Flickr8kData(project_root)
    test_ids = data.split_ids("test")
    data.download_and_index_images()
    val_warmup_ids, val_benchmark_ids = _fixed_val_ids(data, config)
    if mode in {"regression", "benchmark"}:
        data.assert_ids_exist([*val_warmup_ids, *val_benchmark_ids])
    if data.references_loaded:
        raise AssertionError("References were loaded before generation.")

    captioner = ZeroCapCaptioner(config)
    store = PredictionStore(
        config=config,
        library_versions=_library_versions(),
        gpu_name=gpu_name,
    )
    print("Project root:", project_root)
    print("Git commit:", git_commit)
    print("Mode:", mode)
    print("Config hash:", config.config_hash)
    print("Run directory:", config.run_dir)
    return ExperimentContext(
        project_root=project_root,
        output_root=output_root,
        config=config,
        data=data,
        captioner=captioner,
        store=store,
        val_warmup_ids=val_warmup_ids,
        val_benchmark_ids=val_benchmark_ids,
        test_ids=test_ids,
    )


def _generate_and_store(context: ExperimentContext, image_id: str):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    image = context.data.load_image(image_id)
    result = context.captioner.generate_caption(image=image, image_id=image_id)
    torch.cuda.synchronize()
    result["end_to_end_time_sec"] = float(time.perf_counter() - started)
    result["peak_vram_mb"] = float(
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    context.store.save_prediction(result)
    del image
    gc.collect()
    return result


def _load_prediction_list(path: Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError("Reference predictions must be a JSON list.")
    rows = {row["image_id"]: row for row in payload}
    if len(rows) != len(payload):
        raise RuntimeError("Reference predictions contain duplicate image IDs.")
    return rows


def run_regression(context: ExperimentContext, reference_predictions: Path):
    image_ids = context.val_benchmark_ids
    for index, image_id in enumerate(image_ids, start=1):
        if image_id in context.store.completed_ids():
            print(f"Regression resume {index}/{len(image_ids)}:", image_id)
            continue
        result = _generate_and_store(context, image_id)
        print(
            f"Regression {index}/{len(image_ids)}:",
            image_id,
            "→",
            result["caption"],
        )

    reference = _load_prediction_list(reference_predictions)
    comparison_fields = ("caption", "generated_token_ids", "stop_reason")
    mismatches = []
    for image_id in image_ids:
        if image_id not in reference:
            mismatches.append({"image_id": image_id, "error": "missing reference"})
            continue
        actual = context.store.predictions[image_id]
        expected = reference[image_id]
        differing_fields = [
            field
            for field in comparison_fields
            if actual.get(field) != expected.get(field)
        ]
        if differing_fields:
            mismatches.append(
                {
                    "image_id": image_id,
                    "differing_fields": differing_fields,
                    "expected_caption": expected.get("caption"),
                    "actual_caption": actual.get("caption"),
                }
            )

    summary = {
        "status": "PASS" if not mismatches else "FAIL",
        "image_ids": image_ids,
        "reference_predictions": str(Path(reference_predictions).resolve()),
        "comparison_fields": list(comparison_fields),
        "mismatches": mismatches,
    }
    context.store.save_artifact_json("regression_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if mismatches:
        raise AssertionError("Module regression differs from notebook predictions.")
    print("MODULE REGRESSION: PASS")
    return summary


def _write_locked_json(path: Path, payload) -> bool:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise RuntimeError(
                f"Locked artifact already exists with different content: {path}. "
                "Use a fresh output root; never overwrite a TEST selection after use."
            )
        return False
    PredictionStore._atomic_json(path, payload)
    return True


def _round_manifest_path(output_root: Path, test_round: int) -> Path:
    return Path(output_root) / f"fixed_test_round_{test_round:03d}.json"


def _round_metadata_path(output_root: Path, test_round: int) -> Path:
    return Path(output_root) / f"fixed_test_round_{test_round:03d}_metadata.json"


def _master_order_path(context: ExperimentContext) -> Path:
    return context.output_root / (
        f"fixed_test_master_order_seed_{context.config.seed}.json"
    )


def _validate_id_list(payload, label: str) -> list[str]:
    if (
        not isinstance(payload, list)
        or not payload
        or len(payload) != len(set(payload))
        or not all(isinstance(value, str) for value in payload)
    ):
        raise RuntimeError(f"{label} must be a non-empty unique string list.")
    return list(payload)


def _read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_master_order(context: ExperimentContext) -> list[str]:
    master_order = deterministic_sample(
        context.test_ids,
        len(context.test_ids),
        context.config.seed,
    )
    path = _master_order_path(context)
    _write_locked_json(path, master_order)
    return master_order


def _read_master_order(context: ExperimentContext) -> list[str]:
    path = _master_order_path(context)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run benchmark to allocate TEST rounds."
        )
    master_order = _validate_id_list(_read_json(path), "Master TEST order")
    if set(master_order) != set(context.test_ids):
        raise RuntimeError("Master TEST order does not match the TEST split.")
    return master_order


def _previous_round_ids(
    context: ExperimentContext,
    test_round: int,
    master_order: list[str],
) -> list[str]:
    previous = []
    for round_index in range(1, test_round):
        path = _round_manifest_path(context.output_root, round_index)
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot allocate round {test_round}: previous round is missing: {path}"
            )
        round_ids = _validate_id_list(
            _read_json(path),
            f"TEST round {round_index}",
        )
        overlap = set(previous).intersection(round_ids)
        if overlap:
            raise RuntimeError(
                f"TEST round {round_index} overlaps earlier rounds: "
                f"{sorted(overlap)[:5]}"
            )
        previous.extend(round_ids)
    if previous != master_order[: len(previous)]:
        raise RuntimeError(
            "Existing TEST rounds are not a contiguous prefix of the locked "
            "master order. Refusing to allocate a potentially overlapping round."
        )
    return previous


def run_benchmark(
    context: ExperimentContext,
    test_round: int = 1,
    test_count: Optional[int] = None,
):
    for index, image_id in enumerate(context.val_warmup_ids, start=1):
        image = context.data.load_image(image_id)
        torch.cuda.synchronize()
        result = context.captioner.generate_caption(image=image, image_id=image_id)
        torch.cuda.synchronize()
        print(
            f"Warm-up {index}/{len(context.val_warmup_ids)}:",
            image_id,
            "→",
            result["caption"],
        )
        del image, result
        gc.collect()

    for index, image_id in enumerate(context.val_benchmark_ids, start=1):
        if image_id in context.store.completed_ids():
            print(f"Benchmark resume {index}/{len(context.val_benchmark_ids)}:", image_id)
            continue
        result = _generate_and_store(context, image_id)
        print(
            f"Benchmark {index}/{len(context.val_benchmark_ids)}:",
            image_id,
            f"{result['end_to_end_time_sec']:.2f}s",
            f"{result['peak_vram_mb']:.1f} MiB",
            "→",
            result["caption"],
        )

    rows = [context.store.predictions[image_id] for image_id in context.val_benchmark_ids]
    times = np.asarray([row["end_to_end_time_sec"] for row in rows], dtype=np.float64)
    generation_times = np.asarray(
        [row["generation_time_sec"] for row in rows],
        dtype=np.float64,
    )
    token_counts = np.asarray(
        [row["num_generated_tokens"] for row in rows],
        dtype=np.float64,
    )
    peaks = np.asarray([row["peak_vram_mb"] for row in rows], dtype=np.float64)
    p95_time = float(np.percentile(times, 95))
    average_time = float(times.mean())
    safe_time = max(p95_time, average_time * 1.2)
    estimated_n_test = min(
        len(context.test_ids),
        int(context.config.time_budget_hours * 3600 / safe_time),
    )
    if estimated_n_test <= 0:
        raise RuntimeError("Benchmark produced an empty TEST selection.")
    master_order = _ensure_master_order(context)
    previous_ids = _previous_round_ids(context, test_round, master_order)
    remaining_count = len(master_order) - len(previous_ids)
    selected_count = (
        estimated_n_test if test_count is None else int(test_count)
    )
    if selected_count <= 0:
        raise ValueError("test_count must be positive.")
    if selected_count > remaining_count:
        raise ValueError(
            f"Requested {selected_count} images for TEST round {test_round}, "
            f"but only {remaining_count} unallocated images remain."
        )
    start = len(previous_ids)
    fixed_ids = master_order[start : start + selected_count]
    context.data.assert_ids_exist(fixed_ids)
    fixed_hash = hashlib.sha256(
        json.dumps(fixed_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "seed": context.config.seed,
        "test_round": test_round,
        "n_test": selected_count,
        "previously_allocated_images": len(previous_ids),
        "total_test_images": len(context.test_ids),
        "safe_time_per_image_sec": safe_time,
        "time_budget_hours": context.config.time_budget_hours,
        "fixed_test_ids_sha256": fixed_hash,
        "source_run_dir": context.config.run_dir,
        "git_commit": context.config.git_commit,
        "split_manifest_sha256": context.config.split_manifest_sha256,
    }
    round_path = _round_manifest_path(context.output_root, test_round)
    created = _write_locked_json(round_path, fixed_ids)
    round_metadata_path = _round_metadata_path(context.output_root, test_round)
    if created or not round_metadata_path.exists():
        _write_locked_json(round_metadata_path, metadata)

    if test_round == 1:
        _write_locked_json(
            context.output_root / "fixed_test_image_ids.json",
            fixed_ids,
        )
        _write_locked_json(
            context.output_root / "fixed_test_image_ids_metadata.json",
            metadata,
        )

    cumulative_ids = [*previous_ids, *fixed_ids]
    cumulative_path = context.output_root / (
        f"fixed_test_cumulative_round_{test_round:03d}.json"
    )
    _write_locked_json(cumulative_path, cumulative_ids)
    context.store.save_artifact_json(round_path.name, fixed_ids)
    context.store.save_artifact_json(round_metadata_path.name, metadata)
    context.store.save_artifact_json(cumulative_path.name, cumulative_ids)

    summary = {
        "count": len(times),
        "mean_sec": average_time,
        "median_sec": float(np.median(times)),
        "p95_sec": p95_time,
        "min_sec": float(times.min()),
        "max_sec": float(times.max()),
        "mean_generation_sec": float(generation_times.mean()),
        "mean_time_per_token_sec": float(
            np.mean(generation_times / np.maximum(token_counts, 1))
        ),
        "peak_vram_mb": float(peaks.max()),
        "safe_time_per_image_sec": safe_time,
        "test_round": test_round,
        "estimated_n_test_by_budget": estimated_n_test,
        "selected_n_test": selected_count,
        "previously_allocated_images": len(previous_ids),
        "cumulative_test_images": len(cumulative_ids),
        "unallocated_test_images": len(master_order) - len(cumulative_ids),
        "total_test_images": len(context.test_ids),
        "fixed_test_ids_path": str(round_path),
        "cumulative_test_ids_path": str(cumulative_path),
        "fixed_test_ids_created": created,
        "fixed_test_ids_sha256": fixed_hash,
    }
    context.store.save_artifact_json("benchmark_summary.json", summary)
    context.store.save_artifact_json(
        f"benchmark_summary_round_{test_round:03d}.json",
        summary,
    )
    if context.data.references_loaded:
        raise AssertionError("Benchmark loaded reference captions.")
    print(json.dumps(summary, indent=2))
    print("Benchmark does not run TEST predictions.")
    return summary


def _load_fixed_test_ids(
    context: ExperimentContext,
    test_round: int,
) -> tuple[list[str], Path]:
    path = _round_manifest_path(context.output_root, test_round)
    if test_round == 1 and not path.is_file():
        legacy_path = context.output_root / "fixed_test_image_ids.json"
        if legacy_path.is_file():
            path = legacy_path
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run benchmark for TEST round {test_round}."
        )
    image_ids = _validate_id_list(_read_json(path), f"TEST round {test_round}")
    if not set(image_ids).issubset(set(context.test_ids)):
        raise RuntimeError("Fixed TEST manifest contains IDs outside the TEST split.")
    master_order = _read_master_order(context)
    previous_ids = _previous_round_ids(context, test_round, master_order)
    expected = master_order[len(previous_ids) : len(previous_ids) + len(image_ids)]
    if image_ids != expected:
        raise RuntimeError(
            f"TEST round {test_round} is not the expected non-overlapping "
            "slice of the master order."
        )
    return image_ids, path


def run_test_predictions(
    context: ExperimentContext,
    smoke: bool = False,
    test_round: int = 1,
):
    fixed_ids, fixed_path = _load_fixed_test_ids(context, test_round)
    requested_ids = (
        fixed_ids[: context.config.test_smoke_images]
        if smoke
        else fixed_ids
    )
    if smoke and len(requested_ids) < context.config.test_smoke_images:
        raise RuntimeError("Fixed TEST manifest contains fewer than five IDs.")
    context.data.assert_ids_exist(requested_ids)
    if context.data.references_loaded:
        raise AssertionError("References were loaded before TEST generation.")

    remaining = [
        image_id
        for image_id in requested_ids
        if image_id not in context.store.completed_ids()
    ]
    resumed = len(requested_ids) - len(remaining)
    print(f"Requested={len(requested_ids)}, resumed={resumed}, remaining={len(remaining)}")
    started = time.perf_counter()
    for index, image_id in enumerate(remaining, start=1):
        result = _generate_and_store(context, image_id)
        average = (time.perf_counter() - started) / index
        eta_minutes = average * (len(remaining) - index) / 60
        print(
            f"[{resumed + index}/{len(requested_ids)}] {image_id} "
            f"| {result['end_to_end_time_sec']:.2f}s "
            f"| ETA {eta_minutes:.1f} min "
            f"| {result['caption']}"
        )

    missing = [
        image_id
        for image_id in requested_ids
        if image_id not in context.store.completed_ids()
    ]
    if missing:
        raise RuntimeError(f"Predictions remain incomplete: {missing[:5]}")
    if context.data.references_loaded:
        raise AssertionError("TEST generation loaded reference captions.")
    manifest = {
        "status": "PASS",
        "mode": "test_smoke" if smoke else "final_test",
        "test_round": test_round,
        "num_predictions": len(requested_ids),
        "prediction_path": str(context.store.predictions_path),
        "fixed_test_ids_path": str(fixed_path),
        "evaluation": "intentionally delegated to the shared evaluator",
    }
    context.store.save_artifact_json("prediction_run_summary.json", manifest)
    context.store.save_artifact_json(
        f"prediction_run_summary_round_{test_round:03d}.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def run_mode(
    mode: str,
    project_root: Path,
    output_root: Path,
    time_budget_hours: float = 4.0,
    benchmark_val_images: int = 5,
    reference_predictions: Optional[Path] = None,
    test_round: int = 1,
    test_count: Optional[int] = None,
):
    if test_round <= 0:
        raise ValueError("test_round must be positive")
    if test_count is not None and test_count <= 0:
        raise ValueError("test_count must be positive")
    if test_count is not None and mode != "benchmark":
        raise ValueError("test_count is only valid in benchmark mode")
    if mode == "regression" and reference_predictions is None:
        raise ValueError("regression mode requires reference_predictions")
    context = prepare_experiment(
        mode=mode,
        project_root=project_root,
        output_root=output_root,
        time_budget_hours=time_budget_hours,
        benchmark_val_images=benchmark_val_images,
    )
    if mode == "regression":
        return run_regression(context, Path(reference_predictions))
    if mode == "benchmark":
        return run_benchmark(
            context,
            test_round=test_round,
            test_count=test_count,
        )
    if mode == "test_smoke":
        return run_test_predictions(
            context,
            smoke=True,
            test_round=test_round,
        )
    if mode == "final_test":
        return run_test_predictions(
            context,
            smoke=False,
            test_round=test_round,
        )
    raise AssertionError(f"Unhandled mode: {mode}")
