Broken INT8 pays **quantization overhead** (4 kernel launches + intermediate
memory reads/writes) *without* getting the INT8 speedup (Conv still executes
in FP32 because tensors were dequantized before Conv) — losing on both ends,
making it slower than a clean FP32 pipeline.

**Not verified with:** TRT VERBOSE logger or NVIDIA Nsight Systems profiler
(both are 1–4 hour projects; I judged the ROI wasn't worth it given the
solution path was already clear).

**Lesson:** ORT's QDQ output isn't guaranteed to match TRT's fusion pattern
expectations. NVIDIA maintains `pytorch-quantization` specifically to
sidestep this class of problem — it produces QDQ patterns designed to match
TRT's fusion matcher by construction.

## What's Next (Day 4)

- Rebuild INT8 pipeline with NVIDIA's **`pytorch-quantization`** toolkit
- Expected outcome: INT8 pure inference should drop to ~1.5 ms
  (faster than FP16), completing the three-precision speedup chain
- Compare per-layer precision assignment (VERBOSE logger) between
  ORT-produced ONNX and pytorch-quantization-produced ONNX

## File Manifest

- `webcam_trt.py` — main pipeline: multi-engine hot-swap, CUDA event timing,
  recording, dynamic overlay sizing
- `quantize_int8.py` — ORT INT8 PTQ with `quant_pre_process`, symmetric
  quantization, `TrafficCalibrationReader` (domain-specific calibration
  using 100 evenly-sampled frames from `traffic.mp4`)
- `strip_bias_qdq.py` — ONNX post-processor for TRT compatibility (Gotcha 2)
- `extract_calib_frames.py` — extracts 100 evenly-spaced frames as
  calibration dataset
- `demo_3precision_readme.mp4` — live demo of 3-precision switching
- `outputs/int8_workflow.md` — full concept + workflow reference (self-notes)

## Setup & Reproduction

```bash
# WSL2 Ubuntu 22.04, CUDA 12.x, TensorRT 10.x installed

# 1. Export FP32 + FP16 ONNX from Ultralytics
python ../Day2/export_onnx.py

# 2. Build FP32 + FP16 engines
python ../Day2/build_engine.py yolov8n_fp32.onnx yolov8n_fp32.engine
python ../Day2/build_engine.py yolov8n_fp16.onnx yolov8n_fp16.engine

# 3. Pre-resize source video (isolate GPU cost from CPU-side resize)
ffmpeg -i traffic.mp4 -vf "scale=1280:-2" -c:v libx264 -crf 18 -an traffic_720p.mp4

# 4. Extract calibration frames from same-domain video
python extract_calib_frames.py    # → calibration_data/frame_000.jpg ...

# 5. INT8 quantize (produces yolov8n_int8.onnx with bias Int32 QDQs)
python quantize_int8.py

# 6. Post-process to strip bias Int32 QDQs (TRT compatibility)
python strip_bias_qdq.py          # → yolov8n_int8_clean.onnx

# 7. Build INT8 engine from cleaned ONNX
python ../Day2/build_engine.py yolov8n_int8_clean.onnx yolov8n_int8.engine

# 8. Run three-precision demo
python webcam_trt.py traffic_720p.mp4 \
       yolov8n_fp32.engine yolov8n_fp16.engine yolov8n_int8.engine
```

## Key Takeaways (Résumé-Grade)

1. **INT8 speedup requires kernel fusion, not just quantization.** Broken
   fusion + quantization overhead is worse than no quantization at all.

2. **Every tool's defaults are optimized for some target hardware.**
   ONNX Runtime defaults for CPU int8 inference; TensorRT expects a
   different bias layout. Match defaults to target.

3. **Domain-specific calibration data matters.** ImageNet-calibrated INT8
   deployed on thermal imagery would crash accuracy — activation
   distributions differ per domain.

4. **Diagnosis before repair.** "INT8 slower than FP32" wasn't a bug in my
   code — it was a symptom of ORT/TRT pattern mismatch. Reading symptoms
   before touching code saves debug hours.

5. **Choose the right tool, don't hack around the wrong one.** After three
   workarounds, the correct answer was "use pytorch-quantization" — the tool
   NVIDIA maintains for exactly this use case.

---

Part of my [AI transition learning log](https://github.com/ScORpioET/ai-learning-log).
Building toward multi-modal / VLM-integrated industrial vision systems.