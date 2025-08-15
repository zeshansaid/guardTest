# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files early to install dependencies first
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download LLaVA model into the image
RUN mkdir -p /app/models/llava-v1.5-7b && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='/app/models/llava-v1.5-7b')"

# Copy handler and schema
COPY rp_handler.py .
COPY rp_schema.py .

# RunPod Serverless Entrypoint
CMD ["python3", "-u", "rp_handler.py"]
