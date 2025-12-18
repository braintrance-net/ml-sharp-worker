FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libheif-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy application code first (needed for -e . in requirements.txt)
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3 runpod requests

# Create cache directory and download model checkpoint
RUN mkdir -p /app/.cache/torch/hub/checkpoints \
    && curl -L -o /app/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt \
    https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

# RunPod serverless handler
CMD ["python", "-m", "sharp.handler"]
