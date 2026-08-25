from src.config.common_config import PROJECT_ROOT


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GPT2_MODEL_NAME = "openai-community/gpt2"

PROMPT = "Image of a"

MAX_NEW_TOKENS = 48
TOP_K = 512
NUM_ITERATIONS = 5

STEP_SIZE = 0.3
CLIP_TEMPERATURE = 0.01
LANGUAGE_KL_WEIGHT = 0.2
FUSION_WEIGHT = 0.9

BEAM_SIZE = 1
STOP_TOKEN = "."

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "zerocap"


def setup_zerocap_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)