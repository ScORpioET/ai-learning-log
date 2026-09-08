"""
v1.1 階段一:gpt.onnx 也做一次FP16轉換(沿用 exp17_fp16_convert.py 對
clip_vision.onnx 的同一套方法),避免pipeline一個FP16、一個FP32混用卻沒驗證過。

轉換後用cosine similarity跟原本的gpt.onnx(FP32)比對,confirm沒有跑掉
(結果:cosine sim ~1.0000001, max abs diff ~0.00277,詳見commit message)。
"""
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

model = onnx.load("gpt.onnx")
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "gpt.fp16.onnx", save_as_external_data=True,
          location="gpt.fp16.onnx.data")
print("saved gpt.fp16.onnx")
