import os
import cv2
import sys
import time
import torch
import numpy as np
import tensorrt as trt
from torchvision.ops import nms

# 1. Load TRT engine（deserialize + create_execution_context）
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.int32: torch.int32,
}

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


class TRTRunner():

    def __init__(self, engine_path):

        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()

        self.model_name = os.path.basename(engine_path)
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        # 2. 讀 engine 的 tensor 資訊，分配 GPU buffer，set_tensor_address
        print('\nEngine tensors:')
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            kind = "INPUT " if is_input else "output"
            print(f"  {kind} {name}: shape={tuple(shape)} dtype={dtype.__name__}")
            (self.input_names if is_input else self.output_names).append((name, tuple(shape), dtype))

        self.buffers = {}
        for name, shape, np_dtype in self.input_names + self.output_names:
            torch_dtype = NP_TO_TORCH[np_dtype]
            self.buffers[name] = torch.zeros(shape, dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())


if len(sys.argv) < 3:
    print("Usage: python webcam_trt.py <video_path> <engine_path> [<engine_path> ...]")
    sys.exit(1)

video_path = sys.argv[1]

runners = []
for i in range(2, len(sys.argv)):
    ENGINE_PATH = sys.argv[i]
    runners.append(TRTRunner(ENGINE_PATH))


# 3. 定義 preprocess(frame_bgr) function
def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (640, 640))
    transpose_frame = resized_frame.transpose((2, 0, 1))
    normalized = transpose_frame.astype(np.float32) / 255.0

    return normalized[np.newaxis, :]

# 4. VideoCapture 開影片，加循環播放
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video.")

# 印實際格式
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(f"FourCC: {fourcc.to_bytes(4, 'little').decode()}")
print(f"Size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

ret, frame = cap.read()
print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

# ---- Dynamic overlay sizing based on frame resolution ----
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FS = W / 1280.0                   # font scale：原本 2560→2，現在自動縮
TH = max(1, int(FS * 2))          # 線粗


stream = torch.cuda.Stream()

first_printed_flag = True

runners_index = 0
count = 0
t0 = time.time()
fps = '0'
trt_ms_list = []          # 累積這 1 秒內所有 trt_ms
trt_ms_display = '0.00'   # 顯示用（1 秒更新一次）

# ---- Recording state ----
writer = None
recording = False
rec_size = (W, H)
rec_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc_out = cv2.VideoWriter_fourcc(*'mp4v')

# while loop
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    pre_frame = preprocess(frame)

    r = runners[runners_index]
    input_name = r.input_names[0][0]
    output_name = r.output_names[0][0]

    r.buffers[input_name].copy_(torch.from_numpy(pre_frame).cuda())

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record(stream)
    r.context.execute_async_v3(stream.cuda_stream)
    end_ev.record(stream)
    stream.synchronize()

    trt_ms = start_ev.elapsed_time(end_ev)   # milliseconds
    trt_ms_list.append(trt_ms)               # 累積

    boxes = r.buffers[output_name][0, :4, :]
    scores = r.buffers[output_name][0, 4:, :]
    max_scores, class_ids = scores.max(dim=0)

    mask = max_scores > 0.5

    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = boxes[:, mask]

    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    boxes_xyxy = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=0)
    boxes_xyxy = boxes_xyxy.T
    keep_indices = nms(boxes_xyxy, max_scores, iou_threshold=0.5)

    final_boxes  = boxes_xyxy[keep_indices].cpu().numpy()
    final_scores = max_scores[keep_indices].cpu().numpy()
    final_class  = class_ids[keep_indices].cpu().numpy()

    for box, cls, conf in zip(final_boxes, final_class, final_scores):
        x1, y1, x2, y2 = box
        x1 = int(x1 * W / 640)
        y1 = int(y1 * H / 640)
        x2 = int(x2 * W / 640)
        y2 = int(y2 * H / 640)
        label = f"{COCO_NAMES[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * FS, (0, 255, 0), max(1, int(FS)))

    # ---- HUD overlay ----
    y_top = int(H * 0.10)
    cv2.putText(frame, r.model_name, (int(W*0.04), y_top),
                cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 255), TH)
    cv2.putText(frame, f'frame rate:{fps}', (int(W*0.35), y_top),
                cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 255), TH)
    cv2.putText(frame, f'TRT: {trt_ms_display} ms', (int(W*0.60), y_top),
                cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 255), TH)

    now = time.time()
    elapsed = now - t0
    if elapsed >= 1:
        fps = f'{(count/elapsed):.1f}'
        trt_ms_display = f'{sum(trt_ms_list)/len(trt_ms_list):.2f}'   # 平均
        t0 = now
        count = 0
        trt_ms_list = []                                              # 清空

    cv2.imshow('', frame)
    key = cv2.waitKey(1) & 0xFF  # 等待 1ms，& 0xFF 確保跨平台相容

    if key == ord('q'):
        break
    elif key == ord('p'):
        runners_index = (runners_index+1) % len(runners)
    elif key == ord('r'):
        if not recording:
            fname = f'demo_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
            writer = cv2.VideoWriter(fname, fourcc_out, rec_fps, rec_size)
            recording = True
            print(f'[REC] start → {fname}')
        else:
            recording = False
            if writer is not None:
                writer.release()
                writer = None
            print('[REC] stop')

    # 錄影中提示 + 寫檔
    if recording:
        cv2.circle(frame, (int(W*0.96), y_top), int(FS*12), (0, 0, 255), -1)
        cv2.putText(frame, 'REC', (int(W*0.87), y_top + int(FS*7)),
                    cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 255), TH + 1)
        if writer is not None:
            writer.write(frame)


elapsed = time.time() - t0

print(f"\nTensorRT pipeline FPS: {(count-30)/elapsed:.1f}")

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()