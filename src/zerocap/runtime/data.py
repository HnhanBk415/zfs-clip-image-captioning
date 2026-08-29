"""Flickr8k access with an explicit reference-caption boundary."""

import json
import random
from pathlib import Path

import kagglehub
from PIL import Image


class Flickr8kData:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / "data" / "flickr8k"
        self.manifest_path = self.data_root / "metadata" / "split_manifest.json"
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.references_loaded = False
        self.images_dir = None
        self.image_paths = {}

    def split_ids(self, split):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        values = self.manifest["image_ids"][split]
        if len(values) != len(set(values)):
            raise AssertionError(f"Duplicate IDs in {split} split.")
        return list(values)

    def download_and_index_images(self):
        tracked_raw_dir = self.data_root / "raw" / "Images"
        if tracked_raw_dir.is_dir() and any(tracked_raw_dir.glob("*.jpg")):
            download_root = tracked_raw_dir.parent
            print("Using existing raw image directory:", tracked_raw_dir)
        else:
            try:
                from src.config.common_config import KAGGLE_DATASET_HANDLE
                print(
                    "Kaggle dataset handle imported from common config:",
                    KAGGLE_DATASET_HANDLE,
                )
            except Exception as exc:
                KAGGLE_DATASET_HANDLE = "adityajn105/flickr8k"
                print(
                    "Common config import unavailable; using verified fallback "
                    f"{KAGGLE_DATASET_HANDLE!r}. Cause: {type(exc).__name__}"
                )

            try:
                download_root = Path(
                    kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
                )
            except Exception as exc:
                raise RuntimeError(
                    "KaggleHub could not download Flickr8k. Check Colab network/"
                    "Kaggle access, then restart and Run all."
                ) from exc

        candidates = [
            path
            for path in Path(download_root).rglob("Images")
            if path.is_dir()
        ]
        if tracked_raw_dir.is_dir():
            candidates.append(tracked_raw_dir)

        candidates = list(dict.fromkeys(path.resolve() for path in candidates))
        if not candidates:
            raise FileNotFoundError(
                f"No directory named Images was found under {download_root}."
            )

        ranked = []
        for candidate in candidates:
            image_files = [
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            ranked.append((len(image_files), candidate, image_files))

        count, selected_dir, image_files = max(ranked, key=lambda item: item[0])
        if count == 0:
            raise RuntimeError(f"Images directory is empty: {selected_dir}")

        paths = {}
        for path in image_files:
            if path.name in paths and paths[path.name] != path:
                raise RuntimeError(f"Duplicate image filename: {path.name}")
            paths[path.name] = path

        self.images_dir = selected_dir
        self.image_paths = paths
        print("Flickr8k Images directory:", selected_dir)
        print("Indexed raw images:", len(paths))
        return selected_dir

    def assert_ids_exist(self, image_ids):
        missing = [image_id for image_id in image_ids if image_id not in self.image_paths]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected image IDs are missing; first: {missing[:5]}"
            )

    def load_image(self, image_id):
        if image_id not in self.image_paths:
            raise KeyError(f"Image ID was not indexed: {image_id}")
        with Image.open(self.image_paths[image_id]) as opened:
            return opened.convert("RGB")

    def load_references(self, split, image_ids):
        if split not in {"val", "test"}:
            raise ValueError("References are only supported for val/test evaluation.")
        split_path = self.data_root / "splits" / f"{split}.json"
        with split_path.open("r", encoding="utf-8") as handle:
            references_by_id = json.load(handle)
        self.references_loaded = True

        selected = {}
        for image_id in image_ids:
            references = references_by_id.get(image_id)
            if references is None:
                raise KeyError(f"No references found for {image_id}")
            if len(references) != 5 or not all(
                isinstance(value, str) and value.strip()
                for value in references
            ):
                raise AssertionError(
                    f"{image_id} must have exactly five non-empty references."
                )
            selected[image_id] = list(references)
        return selected


def deterministic_sample(values, count, seed):
    ordered = sorted(values)
    if count < 0 or count > len(ordered):
        raise ValueError(
            f"Cannot sample {count} items from a population of {len(ordered)}."
        )
    return random.Random(seed).sample(ordered, count)
