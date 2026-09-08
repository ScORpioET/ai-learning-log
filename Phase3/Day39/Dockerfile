# v1 最小可跑通版本:CPU FP32 thermal caption demo。
# 輕量base image(不需要CUDA,這版只吃CPU)。
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY clip_vision.onnx clip_vision.onnx.data ./
COPY gpt.onnx gpt.onnx.data ./
COPY tokenizer.pkl .

WORKDIR /app/app
EXPOSE 7860

CMD ["python", "app.py"]
