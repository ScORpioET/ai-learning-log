import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
root = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# ---- 1. 延遲 benchmark：FP32 vs INT8 ----
def benchmark_latency(onnx_path, example_input, n_warmup=20, n_runs=200):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # warm-up：前幾次呼叫通常比較慢（記憶體配置、cache 還沒熱),不計入正式測量
    for _ in range(n_warmup):
        session.run(None, {input_name: example_input})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: example_input})
        times.append((time.perf_counter() - t0) * 1000)  # 轉成 ms

    times = np.array(times)
    return {
        "median_ms": np.median(times),
        "mean_ms": np.mean(times),
        "p95_ms": np.percentile(times, 95),
    }

example_input = np.random.randn(1, 3, 224, 224).astype(np.float32)  # shape 對即可,值不影響延遲測量

fp32_stats = benchmark_latency("clip_vision.onnx", example_input)
int8_stats = benchmark_latency("clip_vision.int8.onnx", example_input)

print("=== 延遲 benchmark (median of 200 runs, ms) ===")
print(f"FP32: {fp32_stats}")
print(f"INT8: {int8_stats}")
print(f"加速比: {fp32_stats['median_ms'] / int8_stats['median_ms']:.2f}x")


# ---- 2. 精度對照：用「沒被拿去 calibration」的圖片,避免污染分數 ----
image_dir = root / "data"
all_images = sorted(image_dir.glob("*.jpg"))

import random
random.seed(1337)
calib_set = set(random.sample(all_images, 500))  # 跟之前 calibration 用同個 seed,重現同一批
holdout_images = [p for p in all_images if p not in calib_set][:200]  # 剩下的抽 200 張當精度驗證

fp32_session = ort.InferenceSession("clip_vision.onnx")
int8_session = ort.InferenceSession("clip_vision.int8.onnx")

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
print(f"\n=== 精度對照 (200 張 holdout 圖片, INT8 vs FP32) ===")
print(f"cosine sim mean = {cos_sims.mean():.6f}, min = {cos_sims.min():.6f}, std = {cos_sims.std():.6f}")