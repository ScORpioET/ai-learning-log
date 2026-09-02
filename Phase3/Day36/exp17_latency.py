import time
import numpy as np
import onnxruntime as ort

example_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

def bench(path, providers, n_warmup=20, n_runs=100):
    try:
        session = ort.InferenceSession(path, providers=providers)
    except Exception as e:
        return {"error": str(e)}
    actual_providers = session.get_providers()
    input_name = session.get_inputs()[0].name
    for _ in range(n_warmup):
        session.run(None, {input_name: example_input})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: example_input})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "providers_used": actual_providers,
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }

models = {
    "FP32": "clip_vision.onnx",
    "FP16": "clip_vision.fp16.onnx",
    "INT8 (exp5 disable_mha_qdq)": "clip_vision.int8.exp5_disable_mha_qdq.onnx",
}

print("=== CPU EP ===")
for name, path in models.items():
    r = bench(path, ["CPUExecutionProvider"])
    print(name, r)

print()
print("=== CUDA EP ===")
for name, path in models.items():
    r = bench(path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(name, r)
