#!/usr/bin/env bash
set -e

echo "Prewarming model cache..."
python - <<'PY'
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_id = os.environ.get("MODEL_ID", "liuhaotian/llava-v1.5-7b")
quant = os.environ.get("QUANTIZE", "4bit").lower()
load_4bit = quant == "4bit"
load_8bit = quant == "8bit"

model_name = get_model_name_from_path(model_id)
print(f"[PRELOAD] {model_id} (quant={quant})")

# This downloads & initializes once to warm the HF cache
load_pretrained_model(
    model_path=model_id,
    model_base=None,
    model_name=model_name,
    load_8bit=load_8bit,
    load_4bit=load_4bit,
    device_map="auto",
)
print("[PRELOAD] Done.")
PY

echo "Starting API on :${PORT:-8000}"
exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
