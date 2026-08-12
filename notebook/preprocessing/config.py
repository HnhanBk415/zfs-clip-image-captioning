import os
import sys
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import unicodedata
from PIL import Image
from tqdm import tqdm
from pyprojroot.here import here
import kagglehub
from collections import Counter
# Import thêm thư viện hỗ trợ model
import torch
import transformers
from transformers import AutoTokenizer
import numpy as np

# 1. Thiết lập Root Directory và thêm vào sys.path
PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Cấu hình Thư mục Dự án
DATA_ROOT = PROJECT_ROOT / "notebook" / "data" / "flickr8k"


RAW_DIR = DATA_ROOT / "raw"
SPLIT_DIR = DATA_ROOT / "splits"
SUBSET_DIR = DATA_ROOT / "subsets"
FEATURE_DIR = DATA_ROOT / "features"
METADATA_DIR = DATA_ROOT / "metadata"
TOKENIZED_DIR = DATA_ROOT / "tokenized"


# 3. Tải & Lưu trữ Dataset
KAGGLE_DATASET_HANDLE = "adityajn105/flickr8k"
RAW_IMAGES_DIR = RAW_DIR / "Images"
RAW_CAPTIONS_FILE = RAW_DIR / "captions.txt"

# 4. Cấu hình Subsets & Ratios
TRAIN_SUBSET_RATIOS = [0.01, 0.05, 0.10, 0.25, 1.00]
SUBSET_NAMES = ["train_1pct", "train_5pct", "train_10pct", "train_25pct", "train_100pct"]

# 5. Tham số Split & Seed
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# 6. Hàm khởi tạo thư mục
def setup_directories():
    for folder in [DATA_ROOT, RAW_DIR, SPLIT_DIR, SUBSET_DIR, FEATURE_DIR, METADATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# 7. Model CLIP
CLIP_MODEL_NAME = ("openai/clip-vit-base-patch32")
CLIP_BATCH_SIZE = 32

# GPT-2
GPT2_MODEL_NAME = "openai-community/gpt2"
GPT2_MAX_LENGTH = 48

