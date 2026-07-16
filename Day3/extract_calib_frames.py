import cv2
import os
from pathlib import Path

VIDEO_PATH = 'traffic_640.mp4'
OUTPUT_DIR = 'calibration_data'
NUM_FRAMES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")
print(f"Extracting {NUM_FRAMES} evenly-spaced frames...")

# 均勻抽：算出要抽的 frame index
frame_indices = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]

for i, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"  Warning: failed to read frame {frame_idx}")
        continue
    out_path = os.path.join(OUTPUT_DIR, f'frame_{i:03d}.jpg')
    cv2.imwrite(out_path, frame)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{NUM_FRAMES} extracted")

cap.release()
print(f"\nDone. {NUM_FRAMES} frames saved to {OUTPUT_DIR}/")