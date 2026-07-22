# ~/ai-transition-2026/Day5/test_gst_video.py
import cv2

pipeline = (
    "filesrc location=/home/jack/ai-transition-2026/Day5/sintel_trailer-720p.mp4 ! "
    "decodebin ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("opened:", cap.isOpened())

ret, frame = cap.read()
print("frame:", frame.shape if ret else "None")

cap.release()