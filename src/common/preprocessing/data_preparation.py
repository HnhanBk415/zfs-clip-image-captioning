#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3. Normalize and validate captions.
4. Enforce exactly 5 valid captions per image.
5. Split by IMAGE into train/val/test.
6. Freeze split manifest.
7. Generate nested training subsets.
8. Export audit and validation reports.
9. Verify dataset invariants.

EDA/visualization intentionally remains in the notebook:
    notebook/flickr8k_exploration.ipynbFlickr8k Dataset Preparation Pipeline

Responsibilities
----------------
1. Download/load Flickr8k from KaggleHub.
2. Audit raw image/caption integrity.

"""
import json
import random
import re
import string
import unicodedata
from pathlib import Path

import kagglehub
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.config.common_config import (
    KAGGLE_DATASET_HANDLE,
    METADATA_DIR,
    SEED,
    SPLIT_DIR,
    SUBSET_DIR,
    SUBSET_NAMES,
    TEST_RATIO,
    TRAIN_RATIO,
    TRAIN_SUBSET_RATIOS,
    VAL_RATIO,
    setup_common_directories,
)

# Data-policy parameters belong here rather than being hidden in runtime logic.
STRICT_CAPTIONS_PER_IMAGE = 5
MIN_WORD_COUNT = 2

FLAG_TRANSLATIONS = {
    "EMPTY_OR_NULL": "Rỗng hoặc Null",
    "ONLY_PUNCTUATION": "Chỉ chứa dấu câu",
    "ARTICLE_ONLY": "Chỉ chứa mạo từ/từ đơn lẻ",
    "META_TEXT_JUNK": "Văn bản rác / Lỗi nhãn ảnh",
    "TOO_SHORT": "Caption quá ngắn",
}


# ============================================================
# TEXT CLEANING & AUDIT
# ============================================================

def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace around punctuation."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[^\s\d])", r"\1 ", text)
    return " ".join(text.split()).strip()


def create_comparison_key(text: str) -> str:
    """Create a punctuation-insensitive comparison key."""
    if not isinstance(text, str):
        return ""

    text = text.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    return " ".join(text.split())


def audit_caption(caption: str, min_words: int = MIN_WORD_COUNT) -> dict:
    """Audit one caption and return validity, flags, and word count."""
    if not isinstance(caption, str) or not caption.strip():
        return {
            "comp_key": "",
            "word_count": 0,
            "is_valid": False,
            "flags": ["EMPTY_OR_NULL"],
        }

    comp_key = create_comparison_key(caption)
    words = caption.strip().split()
    flags = []

    if not comp_key:
        flags.append("ONLY_PUNCTUATION")

    if comp_key in {"a", "an", "the", "in", "on", "at", "is"}:
        flags.append("ARTICLE_ONLY")

    junk_patterns = [
        r"^(broken|corrupted|missing|no)\s+(image|photo|picture|file)$",
        r"^(n/a|none|null|unknown)$",
    ]

    for pattern in junk_patterns:
        if re.search(pattern, comp_key):
            flags.append("META_TEXT_JUNK")
            break

    if len(words) < min_words:
        flags.append("TOO_SHORT")

    # IMPORTANT: TOO_SHORT is intentionally invalid.
    invalid_flags = {
        "EMPTY_OR_NULL",
        "ONLY_PUNCTUATION",
        "ARTICLE_ONLY",
        "META_TEXT_JUNK",
        "TOO_SHORT",
    }

    return {
        "comp_key": comp_key,
        "word_count": len(words),
        "is_valid": not any(flag in invalid_flags for flag in flags),
        "flags": flags,
    }


# ============================================================
# DATASET LOADING & RAW AUDIT
# ============================================================

def load_raw_dataset():
    """Download Flickr8k and return image directory + caption DataFrame."""
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)

    images_dir = Path(dataset_path) / "Images"
    captions_file = Path(dataset_path) / "captions.txt"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    df = pd.read_csv(captions_file)
    df.columns = df.columns.str.strip()

    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError("captions.txt must contain 'image' and 'caption' columns.")

    return images_dir, captions_file, df


def audit_raw_dataset(images_dir: Path, captions_file: Path, df: pd.DataFrame):
    """Produce raw-dataset statistics and an invalid-caption audit report."""
    all_image_files = sorted(
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_files_set = set(all_image_files)

    raw_pairs_count = len(df)
    duplicate_pairs = int(
        df.duplicated(subset=["image", "caption"]).sum()
    )

    duplicate_mask = df.duplicated(
        subset=["image", "caption"], keep=False
    )
    duplicate_image_ids = sorted(
        df.loc[duplicate_mask, "image"].dropna().unique().tolist()
    )

    caption_counts = df.groupby("image").size()
    caption_images_set = set(df["image"].dropna().unique())

    missing_images = sorted(caption_images_set - image_files_set)
    images_without_captions = sorted(image_files_set - caption_images_set)

    empty_captions = int(
        df["caption"].fillna("").str.strip().eq("").sum()
    )

    # Caption statistics are kept as metadata/reporting, not mixed into the
    # executable validation policy.
    caption_words = df["caption"].fillna("").str.lower().str.split()
    caption_lengths = caption_words.str.len()

    all_words = [
        word
        for tokens in caption_words
        for word in tokens
    ]

    # Audit every raw caption so filtering decisions remain inspectable.
    invalid_caption_logs = []

    for idx, row in df.iterrows():
        raw_caption = row["caption"]
        cleaned_caption = clean_text(raw_caption)
        audit = audit_caption(cleaned_caption)

        if not audit["is_valid"]:
            vi_flags = [
                FLAG_TRANSLATIONS.get(flag, flag)
                for flag in audit["flags"]
            ]
            invalid_caption_logs.append({
                "index": int(idx),
                "image": row["image"],
                "raw_caption": raw_caption,
                "clean_caption": cleaned_caption,
                "flags": audit["flags"],
                "flags_vi": ", ".join(vi_flags),
                "word_count": audit["word_count"],
            })

    audit_df = pd.DataFrame(invalid_caption_logs)
    audit_file = METADATA_DIR / "caption_audit_violations.csv"
    audit_df.to_csv(audit_file, index=False, encoding="utf-8-sig")

    raw_report = {
        "raw_caption_pairs": raw_pairs_count,
        "duplicate_pairs": duplicate_pairs,
        "unique_images_in_captions": int(df["image"].nunique()),
        "image_files": len(all_image_files),
        "empty_captions": empty_captions,
        "caption_references_missing_images": len(missing_images),
        "images_without_captions": len(images_without_captions),
        "images_with_exactly_5_captions": int((caption_counts == 5).sum()),
        "images_not_having_5_captions": int((caption_counts != 5).sum()),
        "caption_length_words": {
            "min": int(caption_lengths.min()) if len(caption_lengths) else 0,
            "max": int(caption_lengths.max()) if len(caption_lengths) else 0,
            "mean": float(caption_lengths.mean()) if len(caption_lengths) else 0.0,
        },
        "total_word_tokens": len(all_words),
        "vocabulary_size": len(set(all_words)),
        "duplicate_images": duplicate_image_ids,
    }

    return raw_report, audit_df


# ============================================================
# IMAGE/CAPTION VALIDATION
# ============================================================

def validate_image_file(image_path: Path) -> bool:
    """Verify image structure and ensure RGB data can be loaded."""
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.convert("RGB").load()

        return True
    except Exception:
        return False


def build_valid_dataset(df: pd.DataFrame, images_dir: Path):
    """
    Validate each image and its captions.

    Policy:
    - image must exist and be readable;
    - caption must pass audit;
    - duplicate cleaned captions within an image are removed;
    - image is accepted only if exactly 5 clean captions remain.
    """
    valid_data = {}
    invalid_caption_images = set()
    missing_image_files = set()
    invalid_image_files = set()

    # Keep raw duplicate removal as an explicit, measurable policy.
    dedup_df = df.drop_duplicates(
        subset=["image", "caption"]
    ).reset_index(drop=True)

    grouped_raw = (
        dedup_df.groupby("image")["caption"]
        .apply(list)
        .to_dict()
    )

    for img_id, raw_captions in tqdm(
        sorted(grouped_raw.items()),
        desc="Validating images/captions",
    ):
        image_path = images_dir / img_id

        if not image_path.is_file():
            missing_image_files.add(img_id)
            continue

        if not validate_image_file(image_path):
            invalid_image_files.add(img_id)
            continue

        cleaned_captions = []

        for raw_caption in raw_captions:
            cleaned = clean_text(raw_caption)
            if not cleaned:
                continue

            audit = audit_caption(cleaned)

            if audit["is_valid"]:
                cleaned_captions.append(cleaned)

        # Remove exact duplicates after normalization while preserving order.
        cleaned_captions = list(dict.fromkeys(cleaned_captions))

        if len(cleaned_captions) != STRICT_CAPTIONS_PER_IMAGE:
            invalid_caption_images.add(img_id)
            continue

        valid_data[img_id] = cleaned_captions

    validation_report = {
        "raw_unique_images": len(grouped_raw),
        "valid_images": len(valid_data),
        "missing_image_files": len(missing_image_files),
        "invalid_image_files": len(invalid_image_files),
        "invalid_caption_count_images": len(invalid_caption_images),
        "final_valid_caption_pairs": sum(
            len(captions) for captions in valid_data.values()
        ),
    }

    return (
        valid_data,
        validation_report,
        missing_image_files,
        invalid_image_files,
        invalid_caption_images,
    )


# ============================================================
# SPLIT & EXPORT
# ============================================================

def split_by_image(valid_data: dict):
    """Create a deterministic, image-disjoint train/val/test split."""
    image_ids = sorted(valid_data.keys())

    rng = random.Random(SEED)
    rng.shuffle(image_ids)

    total = len(image_ids)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size: train_size + val_size]
    test_ids = image_ids[train_size + val_size:]

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert (train_set | val_set | test_set) == set(image_ids)

    return train_ids, val_ids, test_ids


def build_dataset_dict(valid_data: dict, image_ids: list[str]):
    return {
        image_id: valid_data[image_id]
        for image_id in image_ids
    }


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_splits(valid_data: dict, train_ids, val_ids, test_ids):
    split_data = {
        "train": build_dataset_dict(valid_data, train_ids),
        "val": build_dataset_dict(valid_data, val_ids),
        "test": build_dataset_dict(valid_data, test_ids),
    }

    for split_name, split_dict in split_data.items():
        path = SPLIT_DIR / f"{split_name}.json"
        save_json(split_dict, path)

        print(
            f"✓ Saved {split_name}.json | "
            f"{len(split_dict):,} images | "
            f"{sum(len(v) for v in split_dict.values()):,} captions"
        )

    # Flat train pairs for PyTorch Dataset/DataLoader.
    train_pairs = [
        {"image": image_id, "caption": caption}
        for image_id, captions in split_data["train"].items()
        for caption in captions
    ]
    save_json(train_pairs, SPLIT_DIR / "train_pairs.json")

    manifest = {
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "strict_captions_per_image": STRICT_CAPTIONS_PER_IMAGE,
        "min_word_count": MIN_WORD_COUNT,
        "num_images": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "image_ids": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }

    save_json(manifest, METADATA_DIR / "split_manifest.json")

    return split_data


def generate_nested_subsets(valid_data: dict, train_ids):
    """Generate deterministic nested train subsets."""
    results = {}
    previous_ids = set()

    for ratio, name in zip(TRAIN_SUBSET_RATIOS, SUBSET_NAMES):
        num_images = max(1, int(len(train_ids) * ratio))
        subset_ids = train_ids[:num_images]
        subset_dict = build_dataset_dict(valid_data, subset_ids)

        save_json(subset_dict, SUBSET_DIR / f"{name}.json")

        current_ids = set(subset_ids)
        assert previous_ids.issubset(current_ids)
        previous_ids = current_ids

        results[name] = {
            "images": len(subset_dict),
            "captions": sum(len(v) for v in subset_dict.values()),
        }

        print(
            f"✓ {name}: "
            f"{results[name]['images']:,} images | "
            f"{results[name]['captions']:,} captions"
        )

    return results


# ============================================================
# FINAL INVARIANTS
# ============================================================

def validate_final_invariants(valid_data, split_data):
    """Fail fast if the prepared dataset violates core assumptions."""
    assert valid_data, "No valid images remain after preprocessing."

    assert all(
        len(captions) == STRICT_CAPTIONS_PER_IMAGE
        for captions in valid_data.values()
    )

    for split_name, split_dict in split_data.items():
        expected = len(split_dict) * STRICT_CAPTIONS_PER_IMAGE
        actual = sum(len(captions) for captions in split_dict.values())

        assert actual == expected, (
            f"{split_name}: expected {expected} captions, got {actual}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    setup_common_directories

    print("=" * 60)
    print("FLICKR8K DATASET PREPARATION PIPELINE")
    print("=" * 60)

    print("\n[1/5] Loading Flickr8k...")
    images_dir, captions_file, df = load_raw_dataset()

    print("\n[2/5] Raw dataset audit...")
    raw_report, audit_df = audit_raw_dataset(
        images_dir,
        captions_file,
        df,
    )

    print(
        f"Raw caption pairs: {raw_report['raw_caption_pairs']:,}"
    )
    print(
        f"Duplicate pairs: {raw_report['duplicate_pairs']:,}"
    )
    print(
        f"Invalid caption rows: {len(audit_df):,}"
    )

    print("\n[3/5] Image & caption validation...")
    (
        valid_data,
        validation_report,
        missing_images,
        invalid_images,
        invalid_caption_images,
    ) = build_valid_dataset(df, images_dir)

    print(
        f"Valid images: "
        f"{validation_report['valid_images']:,} / "
        f"{validation_report['raw_unique_images']:,}"
    )
    print(
        f"Missing image files: {len(missing_images):,}"
    )
    print(
        f"Corrupted/unreadable images: {len(invalid_images):,}"
    )
    print(
        f"Images rejected by caption policy: "
        f"{len(invalid_caption_images):,}"
    )

    save_json(
        validation_report,
        METADATA_DIR / "validation_report.json",
    )

    print("\n[4/5] Image-level split & export...")
    train_ids, val_ids, test_ids = split_by_image(valid_data)

    print(f"Train: {len(train_ids):,} images")
    print(f"Val:   {len(val_ids):,} images")
    print(f"Test:  {len(test_ids):,} images")

    split_data = export_splits(
        valid_data,
        train_ids,
        val_ids,
        test_ids,
    )

    print("\nGenerating nested training subsets...")
    subset_results = generate_nested_subsets(
        valid_data,
        train_ids,
    )
    save_json(
        subset_results,
        METADATA_DIR / "subset_manifest.json",
    )

    print("\n[5/5] Final invariant checks...")
    validate_final_invariants(valid_data, split_data)

    final_report = {
        "valid_images": len(valid_data),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "train_captions": sum(
            len(c) for c in split_data["train"].values()
        ),
        "val_captions": sum(
            len(c) for c in split_data["val"].values()
        ),
        "test_captions": sum(
            len(c) for c in split_data["test"].values()
        ),
        "strict_captions_per_image": STRICT_CAPTIONS_PER_IMAGE,
        "min_word_count": MIN_WORD_COUNT,
        "seed": SEED,
    }

    save_json(
        final_report,
        METADATA_DIR / "final_dataset_report.json",
    )

    print("\n" + "=" * 60)
    print("✓ ALL DATASET INVARIANTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
