from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "flickr8k"

RAW_DIR = DATA_ROOT / "raw"
SPLIT_DIR = DATA_ROOT / "splits"
SUBSET_DIR = DATA_ROOT / "subsets"
METADATA_DIR = DATA_ROOT / "metadata"

RAW_IMAGES_DIR = RAW_DIR / "Images"
RAW_CAPTIONS_FILE = RAW_DIR / "captions.txt"

KAGGLE_DATASET_HANDLE = "adityajn105/flickr8k"

TRAIN_SUBSET_RATIOS = [0.01, 0.05, 0.10, 0.25, 1.00]

SUBSET_NAMES = [
    "train_1pct",
    "train_5pct",
    "train_10pct",
    "train_25pct",
    "train_100pct",
]

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def setup_common_directories() -> None:
    directories = (
        DATA_ROOT,
        RAW_DIR,
        SPLIT_DIR,
        SUBSET_DIR,
        METADATA_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)