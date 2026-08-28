"""Command-line entry point for the migrated ZeroCap pipeline."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zerocap.experiment import run_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("regression", "benchmark", "test_smoke", "final_test"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "zerocap",
    )
    parser.add_argument("--time-budget-hours", type=float, default=4.0)
    parser.add_argument(
        "--benchmark-val-images",
        type=int,
        choices=(5, 10),
        default=5,
    )
    parser.add_argument(
        "--test-round",
        type=int,
        default=1,
        help="One-based immutable TEST allocation round (1, 2, ...).",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        help=(
            "Number of new non-overlapping TEST images to allocate. "
            "Only valid in benchmark mode; omit to use the time-budget estimate."
        ),
    )
    parser.add_argument("--reference-predictions", type=Path)
    args = parser.parse_args()
    if args.mode == "regression" and args.reference_predictions is None:
        parser.error("--reference-predictions is required for regression mode")
    if args.test_round <= 0:
        parser.error("--test-round must be a positive integer")
    if args.test_count is not None and args.test_count <= 0:
        parser.error("--test-count must be a positive integer")
    if args.test_count is not None and args.mode != "benchmark":
        parser.error("--test-count is only valid with --mode benchmark")
    return args


def main():
    args = parse_args()
    run_mode(
        mode=args.mode,
        project_root=args.project_root,
        output_root=args.output_root,
        time_budget_hours=args.time_budget_hours,
        benchmark_val_images=args.benchmark_val_images,
        reference_predictions=args.reference_predictions,
        test_round=args.test_round,
        test_count=args.test_count,
    )


if __name__ == "__main__":
    main()
