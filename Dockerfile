FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/weights/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 + CUDA 12.4
RUN pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt

# Application
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]