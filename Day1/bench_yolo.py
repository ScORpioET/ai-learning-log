import time
import torch
import numpy as np
from ultralytics import YOLO

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

# 假的 frame（跟 webcam 一樣大小）
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

for name in MODELS:
    model = YOLO(name)
    
    # 暖機 20 張（GPU 第一次跑會建 kernel）
    for _ in range(20):
        model(dummy, verbose=False)
    
    # 量測 100 張
    torch.cuda.synchronize()
    t0 = time.time()
    n = 30 * 30
    for _ in range(n):
        model(dummy, verbose=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    print(f"{name}: {n/elapsed:.1f} FPS (inference-only, no I/O)")