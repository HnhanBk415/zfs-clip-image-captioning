from src.config.common_config import DATA_ROOT


# Giữ nguyên đường dẫn cache hiện tại.
FEATURE_DIR = DATA_ROOT / "features"
TOKENIZED_DIR = DATA_ROOT / "tokenized"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 32

GPT2_MODEL_NAME = "openai-community/gpt2"
GPT2_MAX_LENGTH = 48


def setup_clipcap_directories() -> None:
    directories = (
        FEATURE_DIR,
        TOKENIZED_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)