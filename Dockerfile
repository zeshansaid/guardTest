# Lightweight, CUDA-enabled PyTorch runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "git+https://github.com/haotian-liu/LLaVA.git@v1.2.2"

# App code
COPY serve.py /app/serve.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Default env (override in RunPod if you like)
ENV MODEL_ID="liuhaotian/llava-v1.5-7b" \
    QUANTIZE="4bit" \
    HF_HOME="/root/.cache/huggingface" \
    PORT=8000

EXPOSE 8000

# Start API
CMD ["/app/start.sh"]
