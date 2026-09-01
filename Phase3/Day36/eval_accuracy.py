import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
root = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"

int8_path = sys.argv[1] if len(sys.argv) > 1 else "clip_vision.int8.onnx"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

image_dir = root / "data"
all_images = sorted(image_dir.glob("*.jpg"))

import random
random.seed(1337)
calib_set = set(random.sample(all_images, 500))  # 跟 calibration 用同個 seed,重現同一批
holdout_images = [p for p in all_images if p not in calib_set][:200]  # 剩下的抽 200 張當精度驗證

fp32_session = ort.InferenceSession("clip_vision.onnx")
int8_session = ort.InferenceSession(int8_path)

cos_sims = []
for img_path in holdout_images:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].numpy()

    fp32_out = fp32_session.run(None, {"pixel_values": pv})[0]
    int8_out = int8_session.run(None, {"pixel_values": pv})[0]

    cos = np.dot(fp32_out.flatten(), int8_out.flatten()) / (
        np.linalg.norm(fp32_out) * np.linalg.norm(int8_out)
    )
    cos_sims.append(cos)

cos_sims = np.array(cos_sims)
print(f"\n=== 精度對照 (200 張 holdout 圖片, {int8_path} vs FP32) ===")
print(f"cosine sim mean = {cos_sims.mean():.6f}, min = {cos_sims.min():.6f}, std = {cos_sims.std():.6f}")
