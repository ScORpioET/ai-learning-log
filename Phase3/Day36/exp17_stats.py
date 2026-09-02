import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
root = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

image_dir = root / "data"
all_images = sorted(image_dir.glob("*.jpg"))

import random
random.seed(1337)
calib_set = set(random.sample(all_images, 500))
holdout_images = [p for p in all_images if p not in calib_set][:200]

fp32_session = ort.InferenceSession("clip_vision.onnx")
fp16_session = ort.InferenceSession("clip_vision.fp16.onnx")

cos_sims = []
for img_path in holdout_images:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].numpy()

    fp32_out = fp32_session.run(None, {"pixel_values": pv})[0]
    fp16_out = fp16_session.run(None, {"pixel_values": pv})[0]

    cos = np.dot(fp32_out.flatten(), fp16_out.flatten()) / (
        np.linalg.norm(fp32_out) * np.linalg.norm(fp16_out)
    )
    cos_sims.append(cos)

cos_sims = np.array(cos_sims)
print(f"=== 精度對照 (200 張 holdout 圖片, clip_vision.fp16.onnx vs FP32) ===")
print(f"cosine sim mean = {cos_sims.mean():.6f}, min = {cos_sims.min():.6f}, max = {cos_sims.max():.6f}, std = {cos_sims.std():.6f}")
