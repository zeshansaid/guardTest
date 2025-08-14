import os, io, json, requests, torch
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# -----------------------
# Config / Model loading
# -----------------------
MODEL_ID = os.environ.get("MODEL_ID", "liuhaotian/llava-v1.5-7b")
QUANTIZE = os.environ.get("QUANTIZE", "4bit").lower()

load_4bit = QUANTIZE == "4bit"
load_8bit = QUANTIZE == "8bit"

model_name = get_model_name_from_path(MODEL_ID)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_ID,
    model_base=None,
    model_name=model_name,
    load_8bit=load_8bit,
    load_4bit=load_4bit,
    device_map="auto",
)

app = FastAPI(title="LLaVA-Guard", version="1.0")

# -------------
# Data models
# -------------
class GuardRequest(BaseModel):
    image_url: str = Field(..., description="Publicly reachable image URL")
    instruction: Optional[str] = Field(
        None,
        description="Optional custom instruction; defaults to built-in LLaVA-Guard policy prompt."
    )
    temperature: float = 0.0
    max_new_tokens: int = 512


# Default LLaVA-Guard style instruction (concise, JSON-only)
DEFAULT_GUARD_INSTRUCTION = """You are LLaVA-Guard. Analyze the image for safety policy violations and return ONLY valid minified JSON with this schema:

{
  "safe": boolean,
  "categories": {
    "sexual_content": {"score": float, "notes": string},
    "nudity": {"score": float, "notes": string},
    "violence_gore": {"score": float, "notes": string},
    "self_harm": {"score": float, "notes": string},
    "hate_symbols_or_extremism": {"score": float, "notes": string},
    "weapons": {"score": float, "notes": string},
    "minors_risk": {"score": float, "notes": string},
    "drugs": {"score": float, "notes": string},
    "privacy_sensitive": {"score": float, "notes": string}
  },
  "overall_rationale": string
}

Rules:
- scores are 0.0 to 1.0 likelihoods.
- "safe" is true only if all scores < 0.5.
- No markdown, no backticks, no extra text, only valid JSON.
"""

def _download_image(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

def llava_guard_infer(image: Image.Image, instruction: str, temperature: float, max_new_tokens: int) -> str:
    # Build prompt with image tokens
    prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{instruction}"

    # Tokenize text with image placeholder token index
    inputs = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).to(model.device)

    # Preprocess image
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return text

def _force_json(text: str) -> Dict[str, Any]:
    """
    Try to coerce model output to JSON.
    If it contains extra text, attempt to extract the first JSON object.
    """
    try:
        return json.loads(text)
    except Exception:
        # Heuristic: find first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise HTTPException(status_code=500, detail="Model did not return valid JSON.")

# ----------------
# API endpoints
# ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_ID, "quantize": QUANTIZE}

@app.post("/guard")
def guard(req: GuardRequest):
    image = _download_image(req.image_url)
    instruction = req.instruction or DEFAULT_GUARD_INSTRUCTION

    text = llava_guard_infer(
        image=image,
        instruction=instruction,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )
    result = _force_json(text)
    return {
        "model": MODEL_ID,
        "quantize": QUANTIZE,
        "result": result,
    }
