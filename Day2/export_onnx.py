from ultralytics import YOLO
import shutil
import os

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
PRECISIONS = [
    ("fp32", False),   # (suffix, half flag)
    ("fp16", True),
]

for name in MODELS:
    for suffix, half_flag in PRECISIONS:
        model = YOLO(name)
        model.export(
            format="onnx",
            opset=12,
            simplify=True,
            dynamic=False,
            imgsz=640,
            half=half_flag,    # ← 唯一新增的參數
        )
        # ultralytics 固定輸出 <basename>.onnx，我們重新命名
        base = name.replace(".pt", "")           # yolov8m
        src = f"{base}.onnx"
        dst = f"{base}_{suffix}.onnx"            # yolov8m_fp16.onnx
        if os.path.exists(src):
            shutil.move(src, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f">>> {dst}  ({size_mb:.1f} MB)")
        else:
            print(f">>> ERROR: {src} not found")