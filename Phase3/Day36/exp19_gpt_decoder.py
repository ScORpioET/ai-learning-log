import json
import random
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization import quantize
from minbpe import minbpe

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319

root = Path.home() / "ai-transition-2026" / "thermal_dataset"
image_dir = root / "images_thermal_val" / "data"
captions_path = root / "captions_val.jsonl"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
tokenizer = minbpe.load("tokenizer.pkl")
clip_session = ort.InferenceSession("clip_vision.onnx")

with open(captions_path) as f:
    records = [json.loads(line) for line in f]
print(f"captions_val.jsonl 總共有 {len(records)} 筆")

random.seed(1337)  # 跟 clip vision 實驗用同個 seed,方法論一致
calib_records = random.sample(records, min(500, len(records)))
calib_names = {r["file_name"] for r in calib_records}
holdout_records = [r for r in records if r["file_name"] not in calib_names][:200]
print(f"calibration: {len(calib_records)} 筆, holdout: {len(holdout_records)} 筆")


def build_sample(record):
    img_path = root / "images_thermal_val" / record["file_name"]
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].numpy()
    img_feat = clip_session.run(None, {"pixel_values": pv})[0]  # (1, 512)

    token_groups = tokenizer.encode(record["caption"])
    flat_ids = [tid for group in token_groups for tid in group]
    seq = [IMAGE_TOKEN_ID] + flat_ids + [EOS_TOKEN_ID]
    idx = np.array(seq[:-1], dtype=np.int64)[None, :]     # (1, T)
    targets = np.array(seq[1:], dtype=np.int64)[None, :]  # (1, T)
    return idx, img_feat.astype(np.float32), targets


print("建立 calibration 資料(逐筆算 clip img_feat + tokenize caption)...")
calib_samples = [build_sample(r) for r in calib_records]

print("建立 holdout 資料...")
holdout_samples = [build_sample(r) for r in holdout_records]


class GPTCalibrationReader(CalibrationDataReader):
    """decoder 的 input_ids 長度是變動的(每筆 caption 長度不同),不能像
    clip vision 那樣疊成單一固定 shape 的 array,所以改用
    CalibrationDataReader 逐筆餵資料。

    第一次跑的時候噴 AttributeError: 'GPTCalibrationReader' object has no
    attribute 'get_first' —— modelopt 的 graph_utils.py 內部
    (find_nodes_from_mha_to_exclude -> get_extended_model_outputs)會呼叫
    `calibration_data_reader.get_first()`,這不是標準
    onnxruntime.quantization.calibrate.CalibrationDataReader 介面的一部分
    (那個介面只要求 get_next()),是 modelopt 自己在
    modelopt/onnx/quantization/calib_utils.py 的
    CalibrationDataProvider/RandomDataProvider 裡額外加的方法。照那份原始
    碼補上 get_first() 跟 rewind()(entropy calibration 通常需要不只一輪
    掃描資料,所以 rewind 也一併補上,不是憑空猜的)。"""

    def __init__(self, samples):
        self.samples = samples
        self.idx = 0

    def _make_dict(self, i):
        idx_arr, img_feat, _ = self.samples[i]
        return {"input_ids": idx_arr, "img_feat": img_feat}

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        result = self._make_dict(self.idx)
        self.idx += 1
        return result

    def get_first(self):
        return self._make_dict(0)

    def rewind(self):
        self.idx = 0


reader = GPTCalibrationReader(calib_samples)

# 跟 exp1 一樣的 calibration 設定:calibration_method="entropy",沒有額外
# 排除任何 op/node,是對 gpt.onnx 做的第一次 baseline INT8 量化。
quantize(
    onnx_path="gpt.onnx",
    quantize_mode="int8",
    calibration_data_reader=reader,
    calibration_method="entropy",
    output_path="gpt.int8.exp19_baseline.onnx",
)
print("gpt.int8.exp19_baseline.onnx 量化完成")
