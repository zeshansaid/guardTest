
FROM nvcr.io/nvidia/pytorch:24.01-py3


RUN apt-get update && apt-get install -y \
    git ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/models/llava-v1.5-7b && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='/app/models/llava-v1.5-7b')"


COPY rp_handler.py .
COPY rp_schema.py .

CMD ["python3", "-u", "rp_handler.py"]
