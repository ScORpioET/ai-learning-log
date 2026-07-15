# Day 2: YOLOv8 TensorRT Benchmark & Pipeline

Phase 0 Week 1 — 從 PyTorch → ONNX → TensorRT 完整 pipeline，含 FP16 加速對比與即時影片推論。

## What's Inside

| File | Purpose |
|---|---|
| `export_onnx.py` | Export yolov8n/s/m to both FP32 & FP16 ONNX |
| `build_engine.py` | Build TensorRT engine from ONNX (TRT 10+ strongly typed) |
| `bench_trt.py` | Pure inference benchmark (measure raw TRT speed) |
| `webcam_trt.py` | End-to-end video pipeline with NMS & bounding boxes |
| `benchmark_results.txt` | Latest benchmark output |

## Setup

```bash
# 1. Export ONNX (both fp32 & fp16)
python export_onnx.py

# 2. Build all engines
for model in yolov8n yolov8s yolov8m; do
  for prec in fp32 fp16; do
    python build_engine.py "${model}_${prec}.onnx" "${model}_${prec}.engine"
  done
done

# 3. Pure benchmark
python bench_trt.py yolov8n_fp16.engine

# 4. Full pipeline (needs traffic.mp4)
python webcam_trt.py yolov8n_fp16.engine
```

## Benchmark (RTX 4070, pure inference)

| Model | FP32 FPS | FP16 FPS | Speedup |
|---|---|---|---|
| yolov8n | 504.6 | 828.5 | 1.64× |
| yolov8s | 356.8 | 625.2 | 1.75× |
| yolov8m | 184.5 | 368.0 | 1.99× |

## 2K Pipeline Profiling (yolov8n_fp16)

| Stage | Time | % |
|---|---|---|
| cap.read (H.264) | 3.76 ms | 15.6% |
| preprocess | 2.68 ms | 11.2% |
| H→D copy | 1.77 ms | 7.3% |
| **TRT execute** | **4.58 ms** | **19.1%** |
| imshow (WSLg) | 11.24 ms | 46.8% |
| **Total** | **24.03 ms → 41.6 FPS** | |

**Key insight**: Pure inference hits 828 FPS but real pipeline drops to 41 FPS — bottleneck is CPU-bound (74%), dominated by display overhead. TRT itself is only 19%.

## Requirements

- CUDA 12.x
- TensorRT 11.1
- PyTorch with CUDA
- Ultralytics YOLO
- OpenCV
- torchvision (for NMS)

## Notes

- TensorRT 10+ removed `BuilderFlag.FP16/INT8` — precision now determined by ONNX dtype via `STRONGLY_TYPED` network
- Preprocess must include `/255` normalization — otherwise model outputs garbage (silent bug, no crash)
- Engine files are hardware-specific (bound to GPU compute capability + CUDA + TRT version)