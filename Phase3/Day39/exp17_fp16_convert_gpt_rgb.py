"""
RGB分支的gpt也做FP16轉換,跟thermal的exp17_fp16_convert_gpt.py同一套方法,
規格對齊thermal分支(不再是RGB唯獨少一個GPU路徑)。

轉換後用cosine similarity跟原本的gpt_rgb.onnx(FP32)比對,確認沒有跑掉。
"""
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

model = onnx.load("gpt_rgb.onnx")
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "gpt_rgb.fp16.onnx", save_as_external_data=True,
          location="gpt_rgb.fp16.onnx.data")
print("saved gpt_rgb.fp16.onnx")
