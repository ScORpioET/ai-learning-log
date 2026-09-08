# CPU FP32 thermal caption demo。輕量base image(不需要CUDA,這版只吃CPU)。
FROM python:3.11-slim

# ultralytics(YOLO疊框,v1.1階段三)拉進 opencv-python,
# 需要這幾個系統動態庫才能import cv2(slim base預設沒有)。
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# v1.1階段三加了ultralytics(YOLO疊框)->帶進torch依賴。plain pip會抓CUDA版torch
# (幾百MB~上GB),CPU image用官方CPU-only wheel index才不會白白裝一份用不到的CUDA。
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app/ ./app/
COPY clip_vision.onnx clip_vision.onnx.data ./
COPY gpt.onnx gpt.onnx.data ./
COPY tokenizer.pkl .
COPY yolov8m.pt .
# v1.1階段四:RGB分支。clip_vision.onnx是domain-agnostic的通用CLIP,thermal/rgb共用,
# 只有gpt(caption decoder)+tokenizer各自一份。
COPY gpt_rgb.onnx gpt_rgb.onnx.data ./
COPY tokenizer_rgb.pkl .

WORKDIR /app/app
EXPOSE 7860

CMD ["python", "app.py"]
