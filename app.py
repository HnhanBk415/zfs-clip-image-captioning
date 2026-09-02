import os
import sys
import subprocess
from pathlib import Path
from google.colab import drive
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2TokenizerFast, DynamicCache
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F

from src.config.zerocap_config import ZeroCapRunConfig
from src.zerocap.models.model_loader import ZeroCapModels
from src.zerocap.core.captioner import ZeroCapCaptioner
from src.zerocap.generation.context_optimizer import ContextOptimizer
from src.clipcap.models.mapping_network import TransformerMapper
from src.clipcap.models.clipcap_model import ClipCaptionModel
from src.clipcap.inference.decoding import generate_caption_from_feature

drive.mount('/content/drive')

# !pip install -q gradio transformers pandas matplotlib

DEMO_DIR = Path("/content/drive/MyDrive/Demo_ZFS")
if (DEMO_DIR / "src").is_dir():
    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))
    print("Đã kết nối src từ Google Drive")
else:
    PROJECT_DIR = Path("/content/zfs-clip-image-captioning")
    REPO_URL = "https://github.com/HnhanBk415/zfs-clip-image-captioning.git"
    BRANCH = "main"
    if not PROJECT_DIR.is_dir():
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True)
    else:
        subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=True)
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    print("Đã clone repo từ GitHub")

print("SẴN SÀNG")

def patched_legacy_cache(self, cache):
    # 1. Nếu đã là tuple/list sẵn
    if isinstance(cache, (tuple, list)) and len(cache) > 0 and isinstance(cache[0], (tuple, list)):
        return tuple((k.detach(), v.detach()) for k, v in cache)

    # 2. Thử to_legacy_cache
    if hasattr(cache, "to_legacy_cache"):
        try:
            res = cache.to_legacy_cache()
            if isinstance(res, (tuple, list)) and len(res) > 0:
                return tuple((k.detach(), v.detach()) for k, v in res)
        except Exception:
            pass

    # 3. Thử key_cache & value_cache (public / private)
    for k_name, v_name in [("key_cache", "value_cache"), ("_key_cache", "_value_cache"), ("keys", "values")]:
        k_list = getattr(cache, k_name, None)
        v_list = getattr(cache, v_name, None)
        if k_list is not None and v_list is not None and len(k_list) > 0:
            return tuple((k.detach(), v.detach()) for k, v in zip(k_list, v_list))

    # 4. Trích xuất từ cache.layers (Bản Transformers mới nhất)
    if hasattr(cache, "layers"):
        extracted = []
        for l in getattr(cache, "layers"):
            # Quét tất cả tensor có trong layer object
            tensors = [v for k, v in vars(l).items() if torch.is_tensor(v)] if hasattr(l, "__dict__") else []
            if len(tensors) >= 2:
                extracted.append((tensors[0].detach(), tensors[1].detach()))
            else:
                k = getattr(l, "key", getattr(l, "key_states", getattr(l, "keys", None)))
                v = getattr(l, "value", getattr(l, "value_states", getattr(l, "values", None)))
                if k is not None and v is not None:
                    extracted.append((k.detach(), v.detach()))
        if len(extracted) == int(self.gpt.config.n_layer):
            return tuple(extracted)

    # 5. Quét đệ quy toàn bộ __dict__ của cache
    if hasattr(cache, "__dict__"):
        for val in cache.__dict__.values():
            if isinstance(val, (list, tuple)) and len(val) == int(self.gpt.config.n_layer):
                first = val[0]
                if hasattr(first, "__dict__"):
                    res = []
                    for item in val:
                        item_tensors = [v for k, v in vars(item).items() if torch.is_tensor(v)]
                        if len(item_tensors) >= 2:
                            res.append((item_tensors[0].detach(), item_tensors[1].detach()))
                    if len(res) == int(self.gpt.config.n_layer):
                        return tuple(res)
                elif isinstance(first, (tuple, list)) and len(first) == 2:
                    return tuple((item[0].detach(), item[1].detach()) for item in val)

    raise RuntimeError(f"Không thể trích xuất cache. Thuộc tính của cache: {dir(cache)}")

@staticmethod
def patched_cache_for_forward(legacy_cache):
    from transformers import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(tuple(legacy_cache))
        except Exception:
            pass
    try:
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(legacy_cache):
            try:
                cache.update(k, v, layer_idx)
            except Exception:
                cache.update(key_states=k, value_states=v, layer_idx=layer_idx)
        return cache
    except Exception:
        return tuple(legacy_cache)

ContextOptimizer._legacy_cache = patched_legacy_cache
ContextOptimizer._cache_for_forward = patched_cache_for_forward
print("Đã kích hoạt bộ chuyển đổi Cache toàn diện cho ZeroCap")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

print("\n[1/3] Đang tải CLIP & GPT-2 Backbone...")
zerocap_config = ZeroCapRunConfig(
    run_mode="benchmark",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=15,
    beam_size=5,
    prompt="Image of a",
)
zerocap_models = ZeroCapModels(zerocap_config)
zerocap_captioner = ZeroCapCaptioner(config=zerocap_config, models=zerocap_models)

clip_model = zerocap_models.clip_model
clip_processor = zerocap_models.clip_processor
gpt2_model = zerocap_models.gpt_model
gpt2_tokenizer = zerocap_models.gpt_tokenizer

print("\n[2/3] Đang nạp 5 Mapper Checkpoints từ Google Drive")
CKPT_DIR = Path("/content/drive/MyDrive/Demo_ZFS/checkpoints")

SUBSET_CHECKPOINTS = {
    "1%": CKPT_DIR / "clipcap_1.pt",
    "5%": CKPT_DIR / "clipcap_5.pt",
    "10%": CKPT_DIR / "clipcap_10.pt",
    "25%": CKPT_DIR / "clipcap_25.pt",
    "100%": CKPT_DIR / "clipcap_100.pt",
}

mappers = {}
for subset_pct, ckpt_path in SUBSET_CHECKPOINTS.items():
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {ckpt_path}")

    mapper = TransformerMapper(
        clip_dim=512,
        embedding_dim=int(gpt2_model.get_input_embeddings().embedding_dim),
        clip_length=10,
        prefix_length=10,
        num_layers=4,
        num_heads=8,
        feedforward_dim=None,
        dropout=0.0,
    ).to(DEVICE)

    ckpt_payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt_payload.get("mapper_state_dict", ckpt_payload)
    mapper.load_state_dict(state_dict, strict=True)
    mapper.eval()
    mappers[subset_pct] = mapper
    print(f"Đã nạp Mapper [{subset_pct}]")

print("\n[3/3] Kiểm tra tài nguyên:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"VRAM: {allocated:.2f} GB / {reserved:.2f} GB")
print("SẴN SÀNG")



def plot_latency_comparison(df: pd.DataFrame):
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        return fig

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    models = [f"{row['Model']} ({row['Tỷ lệ dữ liệu']})" for _, row in df.iterrows()]
    times = [float(t) for t in df["Thời gian (s)"].tolist()]

    colors = ["#4A90E2", "#357ABD", "#2C6DA0", "#1F5B85", "#13486E", "#E94E77"]
    bars = ax.barh(models, times, color=colors[:len(models)], edgecolor="black", height=0.55)

    ax.set_xlabel("Thời gian suy luận (giây)", fontsize=11, fontweight="bold")
    ax.set_title("So sánh độ trễ (Latency): ClipCap vs ZeroCap", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}s",
                ha="left", va="center", fontsize=9, fontweight="bold")

    ax.invert_yaxis()  # Đưa ClipCap 1% lên đầu
    plt.tight_layout()
    return fig

def generate_captions(input_img, progress=gr.Progress()):
    if input_img is None:
        raise gr.Error("Vui lòng chọn hoặc tải lên một bức ảnh trước khi bấm chạy!")

    if isinstance(input_img, str):
        input_img = Image.open(input_img)
    elif isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(input_img)

    if input_img.mode != "RGB":
        input_img = input_img.convert("RGB")

    columns = ["Model", "Tỷ lệ dữ liệu", "Caption sinh ra", "Thời gian (s)", "Peak VRAM (MB)"]
    records = []
    df = pd.DataFrame(columns=columns)

    progress(0.05, desc="[1/3] Đang trích xuất Image Embedding qua CLIP")

    inputs = clip_processor(images=input_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        raw_features = clip_model.get_image_features(pixel_values=pixel_values)
        if hasattr(raw_features, "pooler_output"):
            raw_features = raw_features.pooler_output
        image_embed = F.normalize(raw_features, dim=-1)

    total_mappers = len(mappers)
    for idx, (pct, mapper_module) in enumerate(mappers.items()):
        progress(0.1 + (idx / total_mappers) * 0.4, desc=f"[2/3] ClipCap đang sinh từ với Mapper {pct}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_t = time.perf_counter()

        clipcap_wrapper = ClipCaptionModel(mapper=mapper_module, gpt2=gpt2_model)

        with torch.inference_mode():
            gen_res = generate_caption_from_feature(
                image_features=image_embed,
                processor=clip_processor,
                encoder=clip_model,
                model=clipcap_wrapper,
                tokenizer=gpt2_tokenizer,
                device=DEVICE,
                max_new_tokens=15,
                num_beams=5,
            )

        elapsed_t = time.perf_counter() - start_t
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        records.append({
            "Model": "ClipCap",
            "Tỷ lệ dữ liệu": pct,
            "Caption sinh ra": gen_res.caption,
            "Thời gian (s)": round(elapsed_t, 3),
            "Peak VRAM (MB)": round(peak_vram, 1),
        })
        df = pd.DataFrame(records)
        yield df, None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    progress(0.6, desc="[3/3] ZeroCap đang tối ưu hóa context (15 tokens)")
    start_zc = time.perf_counter()

    # Chạy ZeroCap Generator
    zc_result = zerocap_captioner.generator.generate(image_feature=image_embed)
    zc_caption = zc_result["caption"]

    elapsed_zc = time.perf_counter() - start_zc
    peak_vram_zc = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    records.append({
        "Model": "ZeroCap",
        "Tỷ lệ dữ liệu": "0% (Zero-Shot)",
        "Caption sinh ra": zc_caption,
        "Thời gian (s)": round(elapsed_zc, 3),
        "Peak VRAM (MB)": round(peak_vram_zc, 1),
    })
    df = pd.DataFrame(records)

    progress(1.0, desc="Hoàn tất! Đang xuất biểu đồ")
    fig = plot_latency_comparison(df)
    yield df, fig

# Interface
EXAMPLES_DIR = Path("/content/drive/MyDrive/Demo_ZFS/examples")
example_images = []
if EXAMPLES_DIR.is_dir():
    example_images = [
        [str(p)] for p in EXAMPLES_DIR.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

custom_css = """
#title-header { text-align: center; margin-bottom: 20px; }
#run-btn { font-size: 16px; font-weight: bold; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Demo ClipCap vs ZeroCap") as demo:
    gr.Markdown(
        """
        # Image Captioning Live Demo: ClipCap vs ZeroCap
        ### So sánh khả năng sinh mô tả ảnh giữa **ClipCap (Few-shot/Supervised)** và **ZeroCap (Zero-shot Optimization)**
        """,
        elem_id="title-header"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ảnh Đầu Vào")
            input_image = gr.Image(type="pil", label="Upload ảnh hoặc chọn ảnh mẫu")
            run_button = gr.Button("Bắt Đầu Sinh Caption", variant="primary", elem_id="run-btn")

            if example_images:
                gr.Markdown("#### Ảnh Mẫu (Examples):")
                gr.Examples(
                    examples=example_images,
                    inputs=input_image,
                    label="Bấm vào ảnh để thử nghiệm nhanh",
                )

        with gr.Column(scale=2):
            gr.Markdown("### Kết Quả So Sánh Trực Tiếp")
            output_df = gr.Dataframe(
                headers=["Model", "Tỷ lệ dữ liệu", "Caption sinh ra", "Thời gian (s)", "Peak VRAM (MB)"],
                label="Bảng tiến trình sinh từ theo thời gian thực",
                interactive=False,
                wrap=True,
            )
            gr.Markdown("### Biểu Đồ Độ Trễ (Latency)")
            output_plot = gr.Plot(label="Biểu đồ so sánh thời gian suy luận")

    run_button.click(
        fn=generate_captions,
        inputs=[input_image],
        outputs=[output_df, output_plot],
    )

demo.queue(max_size=5).launch(share=True, debug=True)

# To run: !pip install -q gradio transformers pandas matplotlib
# Haven't cloned: !git clone https://github.com/HnhanBk415/zfs-clip-image-captioning.git
# Already cloned:
# %cd /content/zfs-clip-image-captioning
# !git pull origin main