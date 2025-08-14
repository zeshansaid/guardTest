#!/usr/bin/env bash
set -euo pipefail

# 1) FastChat controller
python3 -m fastchat.serve.controller \
  --host 0.0.0.0 --port "${CONTROLLER_PORT}" &

# 2) OpenAI-compatible API server (serves /v1/* on port 8000)
python3 -m fastchat.serve.openai_api_server \
  --host 0.0.0.0 --port "${API_PORT}" \
  --controller "http://127.0.0.1:${CONTROLLER_PORT}" &

# 3) Gradio UI for quick manual tests
python3 -m llava.serve.gradio_web_server \
  --host 0.0.0.0 --port "${GRADIO_PORT}" \
  --controller "http://127.0.0.1:${CONTROLLER_PORT}" &

# 4) LLaVA model worker 
EXTRA_ARGS=""
if [ "${LOAD_4BIT}" = "true" ]; then
  EXTRA_ARGS="--load-4bit"
fi

python3 -m llava.serve.model_worker \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --controller "http://127.0.0.1:${CONTROLLER_PORT}" \
  --host 0.0.0.0 --port "${WORKER_PORT}" \
  ${EXTRA_ARGS}
