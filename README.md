# AI Learning Log — Jack (ScORpioET)

Daily notes and experiments as I transition from CV engineering to multimodal AI (LLM/VLM).

## Background
5+ years CV engineering experience with three real-world deployments:
- Airport thermal fever screening (public health)
- Factory AOI defect detection (manufacturing)
- Landfill fire early warning (environmental)

Familiar with FLIR thermal cameras, YOLO on thermal data, Jetson embedded deployment, and self-compiled OpenCV with GStreamer + CUDA.

Now expanding into LLM/VLM territory to combine language reasoning with visual monitoring pipelines.

## Learning Roadmap
- **Phase 0 (Week 1–2)** — Deployment fundamentals: TensorRT, GStreamer, MLOps
- **Phase 1 (Week 3–6)** — Transformer from scratch (following Karpathy)
- **Phase 2 (Week 7–8)** — Engineering practices (Docker, MLOps, CI/CD)
- **Phase 3 (Week 9–12)** — Multimodal main project: thermal + visible + VLM event reasoning
- **Phase 4 (Week 13–15)** — Portfolio finalization + job hunt

## Environment
- Host: Windows 11
- Dev: WSL2 Ubuntu 22.04 + WSLg
- GPU: RTX 4070 12GB, CUDA 12.4
- PyTorch 2.5 / TensorRT 11 / ONNX Runtime GPU

## Daily Log

### Day 1 — 2026-07-14
Massive setup + first three "days" collapsed into one:
- WSL2 environment from scratch (Ubuntu 22.04, usbipd webcam passthrough, VSCode Remote-WSL)
- YOLO webcam baseline + diagnosed 15 FPS I/O bottleneck (usbipd USB/IP vs isochronous transfer)
- ONNX export + Runtime CUDA library resolution + PyTorch vs ONNX comparison
- TensorRT FP32 engine build + 554 FPS benchmark
- Refactored build_engine.py into a general CLI tool

See `day2-yolo-webcam/`, `day3-onnx-benchmark/`, `day4-tensorrt/` for details.