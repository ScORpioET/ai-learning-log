import time
import numpy as np
import onnxruntime as ort

MODELS = ["yolov8n.onnx", "yolov8s.onnx", "yolov8m.onnx"]

# 假的 frame：注意 ONNX 吃 NCHW float32 [0,1]，不是 HWC uint8
# 這跟 PyTorch YOLO 內部前處理不同——這裡我們自己做前處理
dummy_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def preprocess(img_bgr):
    # 1. resize 到 640x640
    import cv2
    img = cv2.resize(img_bgr, (640, 640))
    # 2. BGR → RGB
    img = img[:, :, ::-1]
    # 3. HWC → CHW
    img = img.transpose(2, 0, 1)
    # 4. uint8 → float32 [0,1]
    img = img.astype(np.float32) / 255.0
    # 5. 加 batch 維度：CHW → NCHW
    img = img[np.newaxis, :]
    return img

for name in MODELS:
    session = ort.InferenceSession(
        name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    providers = session.get_providers()

    x = preprocess(dummy_bgr)

    # 暖機
    for _ in range(20):
        session.run(None, {input_name: x})

    # 量測
    t0 = time.time()

    n = 30 * 30

    for _ in range(n):
        session.run(None, {input_name: x})
    elapsed = time.time() - t0

    print(f"{name}: {n/elapsed:.1f} FPS  (providers={providers})")