# import runpod
# from PIL import Image
# import requests
# import torch
# from llava import LlavaProcessor, LlavaForConditionalGeneration

# # Load model & processor at startup (cold start)
# MODEL_ID = "liuhaotian/llava-v1.5-7b"
# print(f"Loading model {MODEL_ID}...")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = LlavaForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True
# ).to(device)
# processor = LlavaProcessor.from_pretrained(MODEL_ID)
# print("Model loaded.")

# def handler(event):
#     prompt = event["input"].get("prompt", "")
#     image_url = event["input"].get("image_url", None)
    
#     inputs = {}
#     if image_url:
#         image = Image.open(requests.get(image_url, stream=True).raw)
#         inputs = processor(prompt, image, return_tensors="pt").to(device)
#     else:
#         inputs = processor(prompt, return_tensors="pt").to(device)
    
#     output = model.generate(**inputs, max_new_tokens=256)
#     text_output = processor.decode(output[0], skip_special_tokens=True)
    
#     return {"response": text_output}

# # Start RunPod serverless handler
# runpod.serverless.start({"handler": handler})
import runpod
from PIL import Image
import requests
import torch
from llava import LlavaProcessor, LlavaForConditionalGeneration

MODEL_PATH = "/app/models/llava-v1.5-7b"
print(f"Loading model from {MODEL_PATH}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

processor = LlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")

def handler(event):
    prompt = event["input"].get("prompt", "")
    image_url = event["input"].get("image_url", None)
    
    if image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(prompt, image, return_tensors="pt").to(device)
    else:
        inputs = processor(prompt, return_tensors="pt").to(device)
    
    output = model.generate(**inputs, max_new_tokens=256)
    text_output = processor.decode(output[0], skip_special_tokens=True)
    
    return {"response": text_output}

runpod.serverless.start({"handler": handler})
