import os
import cv2
import glob
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (640, 640))
    transposed = resized.transpose((2, 0, 1))              # HWC → CHW
    normalized = transposed.astype(np.float32) / 255.0     # [0, 255] → [0, 1]
    return normalized[np.newaxis, :]                       # add batch dim → (1, 3, 640, 640)


class TrafficCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, input_name):
        self.image_paths = sorted(glob.glob(os.path.join(calibration_dir, "*.jpg")))
        self.input_name = input_name
        self.iterator = None
        print(f"[Calibration] Loaded {len(self.image_paths)} images")
        print(f"[Calibration] ONNX input name: '{input_name}'")

    def _generator(self):
        for path in self.image_paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"  Warning: could not read {path}")
                continue
            yield {self.input_name: preprocess(frame)}

    def get_next(self):
        if self.iterator is None:
            self.iterator = self._generator()
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = None


MODEL_DIR = os.path.expanduser("~/ai-transition-2026/model")
FP32_ONNX = os.path.join(MODEL_DIR, "yolov8n_fp32.onnx")
PRE_PROCESSED = FP32_ONNX.replace(".onnx", "_preprocessed.onnx")
INT8_ONNX = os.path.join(MODEL_DIR, "yolov8n_int8.onnx")
CALIB_DIR = "calibration_data"
# ---- Step A: Pre-process（fuse BN、fold constants、shape inference） ----
from onnxruntime.quantization.shape_inference import quant_pre_process

print(f"\nPre-processing: {FP32_ONNX} → {PRE_PROCESSED}")
quant_pre_process(
    input_model_path=FP32_ONNX,
    output_model_path=PRE_PROCESSED,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=False,
)
print("Pre-processing done.")


onnx_model = onnx.load(PRE_PROCESSED)
input_name = onnx_model.graph.input[0].name
print(f"Detected input name: {input_name}")

reader = TrafficCalibrationReader(CALIB_DIR, input_name)

# ---- Step B: Static quantization ----
print("\nRunning static quantization...")
quantize_static(
    model_input=PRE_PROCESSED,
    model_output=INT8_ONNX,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=False,
    op_types_to_quantize=['Conv', 'MatMul'],    
    extra_options={
        'ActivationSymmetric': True,
        'WeightSymmetric': True,
        'DedicatedQDQPair': True,             
    },
)
print(f"\n✅ Done. Wrote {INT8_ONNX}")