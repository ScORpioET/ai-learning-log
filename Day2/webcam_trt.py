import cv2
import sys
import time
import torch
import numpy as np
import tensorrt as trt
from torchvision.ops import nms

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

if len(sys.argv) < 2:
    print("Usage: python bench_trt.py <engine_path>")
    sys.exit(1)

ENGINE_PATH = sys.argv[1]

with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

# 1. Load TRT engine（deserialize + create_execution_context）
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()

# 2. 讀 engine 的 tensor 資訊，分配 GPU buffer，set_tensor_address
print('\nEngine tensors:')
input_names, output_names = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    kind = "INPUT " if is_input else "output"
    print(f"  {kind} {name}: shape={tuple(shape)} dtype={dtype.__name__}")
    (input_names if is_input else output_names).append((name, tuple(shape), dtype))

buffers = {}
for name, shape, np_dtype in input_names + output_names:
    torch_dtype = NP_TO_TORCH[np_dtype]
    buffers[name] = torch.zeros(shape, dtype=torch_dtype, device="cuda")
    context.set_tensor_address(name, buffers[name].data_ptr())

# 3. 定義 preprocess(frame_bgr) function
def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (640, 640))
    transpose_frame = resized_frame.transpose((2, 0, 1))
    normalized = transpose_frame.astype(np.float32) / 255.0

    return normalized[np.newaxis, :]

# 4. VideoCapture 開影片，加循環播放
cap = cv2.VideoCapture('traffic.mp4')
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

# 印實際格式
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(f"FourCC: {fourcc.to_bytes(4, 'little').decode()}")
print(f"Size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

ret, frame = cap.read()
print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

pre_frame = preprocess(frame)

stream = torch.cuda.Stream()

first_printed_flag = True

count = 0
input_name = input_names[0][0]
output_name = output_names[0][0]

# while loop
while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    pre_frame = preprocess(frame)

    buffers[input_name].copy_(torch.from_numpy(pre_frame).cuda())
    
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    boxes = buffers[output_name][0, :4, :]
    scores = buffers[output_name][0, 4:, :]
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
        x1 = int(x1 * 2560 / 640)
        y1 = int(y1 * 1440 / 640)
        x2 = int(x2 * 2560 / 640)
        y2 = int(y2 * 1440 / 640)
        label = f"{COCO_NAMES[int(cls)]} {conf:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    if first_printed_flag:
        print(f'output shape : {buffers[output_name].shape}')
        first_printed_flag = False
        


    count += 1

    if count == 30:
        t0 = time.time()

    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # break
    
elapsed = time.time() - t0

print(f"\nTensorRT FP16 FPS: {(count-30)/elapsed:.1f}")

cap.release()
cv2.destroyAllWindows()


