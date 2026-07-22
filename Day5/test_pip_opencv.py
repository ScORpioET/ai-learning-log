import cv2
print("OpenCV version:", cv2.__version__)

# 測 GStreamer pipeline
pipeline = "videotestsrc num-buffers=1 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("GStreamer support:", "YES" if cap.isOpened() else "NO")
cap.release()

# 測 CUDA
try:
    n = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA devices: {n}")
except Exception as e:
    print("CUDA support: NO -", type(e).__name__)
