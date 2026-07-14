import cv2
import time
from ultralytics import YOLO

# 想比較的三個 model 大小
MODEL_NAME = "yolov8m.pt"  # 待會可以改成 yolov8s.pt / yolov8m.pt

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(0)

# 強制設定 MJPG 格式（壓縮，過 usbipd 更順）+ 30 FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 印出實際設到什麼
print(f"Actual FPS setting: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Actual format: {int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode()}")


if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check /dev/video0 exists.")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {w}x{h}")
print(f"Model: {MODEL_NAME}")

# FPS 量測用
prev_time = time.time()
fps_history = []

cv2.namedWindow("YOLO Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Webcam", 960, 720)

timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    # YOLO 推論
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # 算 FPS
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now
    fps_history.append(fps)

    # 疊 FPS 文字到畫面
    cv2.putText(
        annotated, f"FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        annotated, f"Model: {MODEL_NAME}",
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    cv2.imshow("YOLO Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q') or now - timer >= 30:
        break


cap.release()
cv2.destroyAllWindows()

# 收工印平均 FPS（跳過前 10 張暖機）
if len(fps_history) > 10:
    avg_fps = sum(fps_history[10:]) / len(fps_history[10:])
    print(f"\nAverage FPS (after warmup): {avg_fps:.1f}")