FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/weights/huggingface

WORKDIR /app

# 1. Cài các thư viện hệ thống cần thiết cho Pillow, OpenCV, Matplotlib và Gradio
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Cài PyTorch CUDA 12.1 trước
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy mã nguồn dự án
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]