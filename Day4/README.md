# Day 4 — INT8 Quantization via NVIDIA ModelOpt

**Status: environment ready, script validated, benchmark pending.**

Follow-up to Day 3, which used ONNX Runtime PTQ but hit a fusion mismatch —
TensorRT couldn't recognize ORT's QDQ pattern, so INT8 ran slower than FP32.

This day switches to NVIDIA's official toolkit (`nvidia-modelopt`) whose QDQ
output is designed to match TRT's fusion matcher. Environment setup and script
logic are complete; end-to-end benchmark is deferred to next session.

## What worked

- Installed `nvidia-modelopt[torch,onnx]` + hidden dependencies (`huggingface_hub`,
  `transformers`, `datasets`, `accelerate`).
- Upgraded torch / torchvision to `2.13.0+cu130` / `0.28.0+cu130` (modelopt
  pulled newer torch; had to match torchvision or `nms` breaks).
- Wrote `quantize_modelopt.py` — loads Ultralytics YOLOv8n, wraps into
  quantized model via `mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop)`.
- Confirmed **308 quantizers inserted** into the DetectionModel (every Conv
  wrapped with QDQ).
- Exported to ONNX using legacy TorchScript exporter (`dynamo=False`).

## What broke — 4 interview-grade debugging stories

### 1. `pytorch-quantization` is dead
Original plan (from Day 3) was to use `pytorch-quantization` (NVIDIA legacy).
`pip install` fails with `RuntimeError: Bad params` in setup.py. The package
was superseded by `nvidia-modelopt` in 2024. **Decision: switch toolkit.**

### 2. Torch 2.13 dynamo exporter doesn't know custom ops