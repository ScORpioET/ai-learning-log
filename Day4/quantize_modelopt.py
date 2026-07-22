import cv2
import glob
import numpy as np
import torch
from ultralytics import YOLO
import modelopt.torch.quantization as mtq


# ---- 這個 preprocess 必須跟 webcam_trt.py 完全一致 ----
def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (640, 640))
    transposed = resized.transpose((2, 0, 1))              # HWC → CHW
    normalized = transposed.astype(np.float32) / 255.0     # [0, 255] → [0, 1]
    return normalized[np.newaxis, :]                       # add batch dim → (1, 3, 640, 640)

def forward_loop(model):
    for img_path in glob.glob("./calib_data/*.jpg"):
        img = cv2.imread(img_path)
        processed_img = preprocess(img)
        t_img = torch.tensor(processed_img, device='cuda')
        model(t_img)



yolo = YOLO("../model/yolov8m.pt")
model = yolo.model    # ← 這個才是純 nn.Module，可以餵給 mtq.quantize
model.cuda().eval()

# Select quantization config
config = mtq.INT8_DEFAULT_CFG 


q_model = mtq.quantize(model, config, forward_loop)

# ★ 新增：切 Detect head 到 export 模式
from ultralytics.nn.modules import Detect
for m in q_model.modules():
    if isinstance(m, Detect):
        m.export = True
        m.format = 'onnx'

dummy = torch.randn(1, 3, 640, 640, device='cuda')

# mtq.print_quant_summary(q_model)
torch.onnx.export(
    q_model, dummy,
    '../model/yolov8m_int8_modelopt.onnx',
    opset_version=17,
    input_names=["images"],
    output_names=["output0"],
    dynamo=False,     # ★ 關鍵，強制走舊 exporter
)