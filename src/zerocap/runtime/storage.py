"""Atomic ZeroCap prediction persistence and safe resume."""

import csv
import json
import os
from pathlib import Path


class PredictionStore:
    def __init__(self, config, library_versions, gpu_name):
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / "run_metadata.json"
        self.config_path = self.run_dir / "config.json"
        self.predictions_path = self.run_dir / "predictions.json"
        self.predictions_csv_path = self.run_dir / "predictions.csv"
        self.metrics_path = self.run_dir / "metrics.json"

        expected_metadata = {
            "git_commit": config.git_commit,
            "config_hash": config.config_hash,
            "run_mode": config.run_mode,
            "algorithm_revision": config.algorithm_revision,
            "val_tune_candidate": config.val_tune_candidate,
            "seed": config.seed,
            "gpt_model": config.gpt_model,
            "clip_model": config.clip_model,
            "device": config.device,
            "gpu": gpu_name,
            "library_versions": dict(library_versions),
        }

        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            for key in ("git_commit", "config_hash", "run_mode"):
                if existing.get(key) != expected_metadata[key]:
                    raise RuntimeError(
                        f"Refusing to resume: metadata {key} mismatch in "
                        f"{self.run_dir}."
                    )
        elif self.predictions_path.exists():
            raise RuntimeError(
                "Predictions exist without run_metadata.json; refusing unsafe resume."
            )
        else:
            self._atomic_json(self.metadata_path, expected_metadata)
            self._atomic_json(self.config_path, config.public_dict())

        self.predictions = {}
        if self.predictions_path.exists():
            with self.predictions_path.open("r", encoding="utf-8") as handle:
                existing_predictions = json.load(handle)
            if not isinstance(existing_predictions, list):
                raise RuntimeError("predictions.json must contain a JSON list.")
            for item in existing_predictions:
                if (
                    item.get("config_hash") != config.config_hash
                    or item.get("git_commit") != config.git_commit
                ):
                    raise RuntimeError(
                        "Refusing to resume a prediction with stale config/commit."
                    )
                self.predictions[item["image_id"]] = item

    @staticmethod
    def _atomic_json(path, payload):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def _write_csv(self):
        rows = list(self.predictions.values())
        if not rows:
            return
        fieldnames = sorted(
            set().union(*(row.keys() for row in rows))
        )
        temporary = self.predictions_csv_path.with_suffix(".csv.tmp")
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, self.predictions_csv_path)

    def completed_ids(self):
        return set(self.predictions)

    def save_prediction(self, result):
        json.dumps(result)
        if (
            result.get("config_hash") != self.config.config_hash
            or result.get("git_commit") != self.config.git_commit
        ):
            raise AssertionError("Prediction identity does not match this run.")
        self.predictions[result["image_id"]] = dict(result)
        ordered = list(self.predictions.values())
        self._atomic_json(self.predictions_path, ordered)
        self._write_csv()

    def save_metrics(self, metrics):
        self._atomic_json(self.metrics_path, metrics)

    def save_artifact_json(self, filename, payload):
        path = self.run_dir / filename
        self._atomic_json(path, payload)
        return path
