# Day 4 (partial) — TensorRT Engine Build

## Goal
Convert YOLO ONNX models into TensorRT engines and benchmark inference speed.
Understand what TensorRT does internally (not just "run the API").

## What was completed today
- FP32 engine build for yolov8n
- FP32 benchmark: **554 FPS, 1.80 ms latency**
- Refactored `build_engine.py` into a CLI tool taking (onnx_path, engine_path, precision)
- Deep-dived: Build vs Runtime mental model, engine portability (sm_XX + CUDA + TRT version)
- Understood: Constant folding, operator fusion, memory planning, kernel auto-tuning

## Benchmark comparison (RTX 4070, batch=1, 640×640, inference-only)

| Runtime            | yolov8n FPS | Latency (ms) | Speedup vs PyTorch |
|--------------------|-------------|--------------|--------------------|
| PyTorch native     | 117.1       | 8.54         | 1.00×              |
| ONNX Runtime (CUDA)| 178.1       | 5.62         | 1.52×              |
| **TensorRT FP32**  | **554.1**   | **1.80**     | **4.73×**          |

## Files
- `build_engine.py` — General ONNX → TRT engine builder (CLI)
- `bench_trt.py` — TensorRT engine benchmark
- `yolov8n_fp32.engine` — 17 MB, sm_89 (Ada Lovelace)

## Key concepts learned
1. **Build phase vs Runtime phase** are separate: build produces a serialized engine, runtime deserializes and executes it. Analogous to `gcc` vs `./program`.
2. **Kernel auto-tuning** is what makes TensorRT fast — it tests multiple CUDA kernel implementations on the target GPU and picks the fastest.
3. **Engine files are not portable** — they are tied to GPU compute capability (sm_XX), CUDA version, and TensorRT version. Deploying to Jetson requires rebuilding on Jetson.
4. **ONNX is the portable middle layer** — it crosses hardware, engines don't.

## TODO (Day 5)
- Deep dive Runtime phase (bench_trt.py line-by-line)
- Understand FP16 / INT8 precision math (why it works)
- Build FP16 and INT8 engines for yolov8n/s/m
- Complete comparison table with all precisions