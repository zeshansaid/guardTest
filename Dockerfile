FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg curl && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install


RUN pip install --upgrade pip setuptools wheel


RUN pip install "fastchat>=0.2.37"

# LLaVA
WORKDIR /opt
RUN git clone --depth 1 https://github.com/haotian-liu/LLaVA.git
WORKDIR /opt/LLaVA
RUN pip install -e .


RUN pip install "xformers>=0.0.26.post1" --index-url https://download.pytorch.org/whl/cu121 || true


RUN mkdir -p /models
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models

ENV CONTROLLER_PORT=21001 \
    API_PORT=8000 \
    GRADIO_PORT=7860 \
    WORKER_PORT=31000 \
    MODEL_PATH=llava-hf/llava-v1.6-mistral-7b-hf \
    MODEL_NAME=llava-v1.6-mistral-7b \
    LOAD_4BIT=true

COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh


EXPOSE 8000 7860

CMD ["/opt/start.sh"]
