"""
clip_vision.onnx 精度比對:CUDA EP FP32(基準)vs TensorRT EP FP16 vs TensorRT EP INT8。

holdout圖片集的抽樣方式完全照抄Day36/eval_accuracy.py(同一個random.seed(1337),
排除同一批500張calibration圖片,剩下取前200張),這樣算出來的數字才能直接跟
之前CUDA EP上量到的baseline(cosine sim mean=0.547456,INT8 QDQ vs FP32)放在
同一張表比較,不是另外發明一套抽樣方法。
"""
import random
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor

from common import (
    ensure_tensorrt_ld_library_path, IMAGE_ROOT, CLIP_FP32_ONNX, CLIP_INT8_QDQ_ONNX,
)

ensure_tensorrt_ld_library_path()

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
N_HOLDOUT = 200


def build_holdout_list():
    image_dir = IMAGE_ROOT / "data"
    all_images = sorted(image_dir.glob("*.jpg"))
    random.seed(1337)
    calib_set = set(random.sample(all_images, 500))
    return [p for p in all_images if p not in calib_set][:N_HOLDOUT]


def cosine(a, b):
    return float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    holdout = build_holdout_list()
    print(f"[info] holdout images: {len(holdout)} (same seed/split as Day36/eval_accuracy.py)")

    print("[info] loading CUDA EP FP32 baseline session")
    cuda_fp32_sess = ort.InferenceSession(str(CLIP_FP32_ONNX), providers=["CUDAExecutionProvider"])

    print("[info] loading TensorRT EP FP16 session (building engine, may take ~20-30s)")
    t0 = time.perf_counter()
    trt_fp16_sess = ort.InferenceSession(str(CLIP_FP32_ONNX), providers=[
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
    ])
    print(f"[info] TRT FP16 session created in {time.perf_counter() - t0:.1f}s, "
          f"providers={trt_fp16_sess.get_providers()}")

    print("[info] loading TensorRT EP INT8 session from existing QDQ model "
          f"({CLIP_INT8_QDQ_ONNX.name})")
    trt_int8_sess = None
    trt_int8_status = None
    t0 = time.perf_counter()
    try:
        trt_int8_sess = ort.InferenceSession(str(CLIP_INT8_QDQ_ONNX), providers=[
            ("TensorrtExecutionProvider", {"trt_int8_enable": True}),
        ])
        trt_int8_status = "ok (trt_int8_enable alone)"
    except Exception as e:
        print(f"[warn] trt_int8_enable alone failed: {str(e)[:200]}")
        print("[info] retrying with trt_int8_enable + trt_fp16_enable together "
              "(this produces a mixed INT8/FP16 engine, not pure INT8 -- see REPORT.md)")
        try:
            trt_int8_sess = ort.InferenceSession(str(CLIP_INT8_QDQ_ONNX), providers=[
                ("TensorrtExecutionProvider", {"trt_int8_enable": True, "trt_fp16_enable": True}),
            ])
            trt_int8_status = "ok (needed trt_int8_enable + trt_fp16_enable together; mixed-precision engine)"
        except Exception as e2:
            trt_int8_status = f"FAILED even with both flags: {str(e2)[:300]}"
    print(f"[info] TRT INT8 build status: {trt_int8_status} (took {time.perf_counter() - t0:.1f}s)")

    fp16_cos, int8_cos = [], []
    for i, img_path in enumerate(holdout):
        img = Image.open(img_path).convert("RGB")
        pv = processor(images=[img], return_tensors="pt")["pixel_values"].numpy()

        fp32_out = cuda_fp32_sess.run(None, {"pixel_values": pv})[0]
        trt_fp16_out = trt_fp16_sess.run(None, {"pixel_values": pv})[0]
        fp16_cos.append(cosine(fp32_out, trt_fp16_out))

        if trt_int8_sess is not None:
            trt_int8_out = trt_int8_sess.run(None, {"pixel_values": pv})[0]
            int8_cos.append(cosine(fp32_out, trt_int8_out))

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(holdout)}")

    fp16_cos = np.array(fp16_cos)
    print(f"\n=== clip_vision.onnx 精度對照 ({len(holdout)}張holdout圖片, CUDA-FP32基準) ===")
    print(f"TensorRT FP16 : cosine sim mean={fp16_cos.mean():.6f}, min={fp16_cos.min():.6f}, "
          f"std={fp16_cos.std():.6f}")
    if int8_cos:
        int8_cos = np.array(int8_cos)
        print(f"TensorRT INT8 : cosine sim mean={int8_cos.mean():.6f}, min={int8_cos.min():.6f}, "
              f"std={int8_cos.std():.6f}  (build status: {trt_int8_status})")
    else:
        print(f"TensorRT INT8 : 無法建立engine,跳過精度量測。build status: {trt_int8_status}")

    print("\n對照:CUDA EP FP32 vs 同一份QDQ INT8模型(Day36 baseline)= "
          "cosine sim mean 0.547456 (見 Phase3/Day36/quant_experiments.md)")


if __name__ == "__main__":
    main()
