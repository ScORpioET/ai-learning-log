import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

model = onnx.load("clip_vision.onnx")
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "clip_vision.fp16.onnx", save_as_external_data=True,
          location="clip_vision.fp16.onnx.data")
print("saved clip_vision.fp16.onnx")
