from ultralytics import YOLO

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

for name in MODELS:
    model = YOLO(name)
    exported = model.export(
        format="onnx",
        opset=12,       # ONNX opset 版本
        simplify=True,  # 用 onnx-simplifier 化簡 graph
        dynamic=False,  # 固定 input shape，讓 ORT 能做更多最佳化
        imgsz=640,      # 輸入解析度
    )
    print(f"Exported: {exported}")