"""
效能量測:TensorRT EP FP16 / TensorRT EP INT8 vs 現行部署用的CUDA EP FP16 基準,
分開記錄engine build時間跟穩態推論延遲(不要混在一起報)。

方法論比照Day41跟Phase1-nanoGPT/Day18那批腳本:穩態延遲用warmup 3次(不計時)+
正式跑10次取median。build時間 = session創建到第一次成功run()完成為止的wall time
(clip_vision是固定shape,build其實發生在session創建當下;gpt.onnx是動態shape,
用trt_profile_min/opt/max_shapes明確指定範圍後,build也是在session創建當下
一次做完,不會每個seq_len重build——這點已經在accuracy_gpt_caption.py之前另外
驗證過)。

clip_vision.onnx:固定輸入shape (1,3,224,224),沒有動態shape的複雜度。
gpt.onnx:選一個有代表性的中庸長度T=20做穩態延遲比較(不是每個生成步驟的每個
長度都測——TensorRT對每個模型只要build一次涵蓋整個shape範圍的engine,不需要
重build,但要在這裡測「更長的T穩態延遲有沒有變化」需要另外設計,這次先用
單一代表值,量出結論後如果有需要再擴充,不是忘記多測幾個長度)。
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import (
    ensure_tensorrt_ld_library_path, CLIP_FP32_ONNX, CLIP_FP16_ONNX,
    GPT_FP32_ONNX, GPT_FP16_ONNX,
)

ensure_tensorrt_ld_library_path()

WARMUP_RUNS = 3
TIMED_RUNS = 10
GPT_BENCH_SEQ_LEN = 20
OUT_PATH = Path(__file__).parent / "benchmark_results.json"

TRT_GPT_SHAPE_PROFILE = {
    "trt_profile_min_shapes": "input_ids:1x1,img_feat:1x512",
    "trt_profile_max_shapes": "input_ids:1x64,img_feat:1x512",
    "trt_profile_opt_shapes": f"input_ids:1x{GPT_BENCH_SEQ_LEN},img_feat:1x512",
}


def time_calls(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    return times_ms


def bench_clip_vision():
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    results = {}

    print("[clip_vision] CUDA EP FP16 (existing deployed conversion, keep_io_types)")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(CLIP_FP16_ONNX), providers=["CUDAExecutionProvider"])
    sess.run(None, {"pixel_values": x})
    build_ms = (time.perf_counter() - t0) * 1000
    steady = time_calls(lambda: sess.run(None, {"pixel_values": x}), WARMUP_RUNS, TIMED_RUNS)
    results["cuda_fp16"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady))}

    print("[clip_vision] TensorRT EP FP16 (built from FP32 graph, trt_fp16_enable)")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(CLIP_FP32_ONNX), providers=[
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
    ])
    sess.run(None, {"pixel_values": x})
    build_ms = (time.perf_counter() - t0) * 1000
    steady = time_calls(lambda: sess.run(None, {"pixel_values": x}), WARMUP_RUNS, TIMED_RUNS)
    results["trt_fp16"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady))}

    print("[clip_vision] TensorRT EP INT8 (existing QDQ model, needs +fp16 flag -- see REPORT.md)")
    from common import CLIP_INT8_QDQ_ONNX
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(str(CLIP_INT8_QDQ_ONNX), providers=[
            ("TensorrtExecutionProvider", {"trt_int8_enable": True, "trt_fp16_enable": True}),
        ])
        sess.run(None, {"pixel_values": x})
        build_ms = (time.perf_counter() - t0) * 1000
        steady = time_calls(lambda: sess.run(None, {"pixel_values": x}), WARMUP_RUNS, TIMED_RUNS)
        results["trt_int8"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady))}
    except Exception as e:
        results["trt_int8"] = {"error": str(e)[:300]}

    return results


def bench_gpt():
    img_feat = np.random.randn(1, 512).astype(np.float32)
    idx = np.random.randint(0, 320, (1, GPT_BENCH_SEQ_LEN)).astype(np.int64)
    inputs = {"input_ids": idx, "img_feat": img_feat}
    results = {}

    print(f"[gpt] CUDA EP FP16 at T={GPT_BENCH_SEQ_LEN} (existing deployed conversion)")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(GPT_FP16_ONNX), providers=["CUDAExecutionProvider"])
    sess.run(None, inputs)
    build_ms = (time.perf_counter() - t0) * 1000
    steady = time_calls(lambda: sess.run(None, inputs), WARMUP_RUNS, TIMED_RUNS)
    results["cuda_fp16"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady))}

    print(f"[gpt] TensorRT EP FP16 at T={GPT_BENCH_SEQ_LEN} (dynamic-shape profile 1..64)")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(GPT_FP32_ONNX), providers=[
        ("TensorrtExecutionProvider", {**TRT_GPT_SHAPE_PROFILE, "trt_fp16_enable": True}),
    ])
    sess.run(None, inputs)
    build_ms = (time.perf_counter() - t0) * 1000
    steady = time_calls(lambda: sess.run(None, inputs), WARMUP_RUNS, TIMED_RUNS)
    results["trt_fp16"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady))}

    print(f"[gpt] TensorRT EP INT8 at T={GPT_BENCH_SEQ_LEN} (existing QDQ model)")
    from common import GPT_INT8_QDQ_ONNX
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(str(GPT_INT8_QDQ_ONNX), providers=[
            ("TensorrtExecutionProvider", {**TRT_GPT_SHAPE_PROFILE, "trt_int8_enable": True}),
        ])
        sess.run(None, inputs)
        build_ms = (time.perf_counter() - t0) * 1000
        steady = time_calls(lambda: sess.run(None, inputs), WARMUP_RUNS, TIMED_RUNS)
        results["trt_int8"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady)), "note": "trt_int8_enable alone succeeded"}
    except Exception as e:
        print(f"  [warn] trt_int8_enable alone failed: {str(e)[:200]}")
        try:
            sess = ort.InferenceSession(str(GPT_INT8_QDQ_ONNX), providers=[
                ("TensorrtExecutionProvider", {**TRT_GPT_SHAPE_PROFILE, "trt_int8_enable": True, "trt_fp16_enable": True}),
            ])
            sess.run(None, inputs)
            build_ms = (time.perf_counter() - t0) * 1000
            steady = time_calls(lambda: sess.run(None, inputs), WARMUP_RUNS, TIMED_RUNS)
            results["trt_int8"] = {"build_ms": build_ms, "steady_median_ms": float(np.median(steady)), "note": "needed +fp16 flag too"}
        except Exception as e2:
            results["trt_int8"] = {"error": str(e2)[:300]}

    return results


def main():
    print("=" * 70)
    print("clip_vision.onnx benchmark (fixed shape 1x3x224x224)")
    print("=" * 70)
    clip_results = bench_clip_vision()

    print("\n" + "=" * 70)
    print(f"gpt.onnx benchmark (T={GPT_BENCH_SEQ_LEN})")
    print("=" * 70)
    gpt_results = bench_gpt()

    OUT_PATH.write_text(json.dumps({
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "gpt_seq_len": GPT_BENCH_SEQ_LEN,
        "clip_vision": clip_results,
        "gpt": gpt_results,
    }, indent=2))

    print("\n" + "=" * 90)
    print(f"{'model':>13} | {'variant':>10} | {'build(ms)':>12} | {'steady median(ms)':>18} | {'speedup vs CUDA-FP16':>20}")
    print("-" * 90)
    for model_name, results in [("clip_vision", clip_results), ("gpt", gpt_results)]:
        baseline = results.get("cuda_fp16", {}).get("steady_median_ms")
        for variant, r in results.items():
            if "error" in r:
                print(f"{model_name:>13} | {variant:>10} | {'FAILED':>12} | {r['error'][:40]:>18} |")
                continue
            speedup = f"{baseline / r['steady_median_ms']:.2f}x" if baseline else "n/a"
            print(f"{model_name:>13} | {variant:>10} | {r['build_ms']:>10.1f}ms | "
                  f"{r['steady_median_ms']:>16.3f}ms | {speedup:>20}")

    print(f"\n[done] results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
