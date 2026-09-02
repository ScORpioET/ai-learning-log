import json
import random
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor

from minbpe import minbpe

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319

root = Path.home() / "ai-transition-2026" / "thermal_dataset"
captions_path = root / "captions_val.jsonl"

int8_path = sys.argv[1] if len(sys.argv) > 1 else "gpt.int8.exp19_baseline.onnx"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
tokenizer = minbpe.load("tokenizer.pkl")
clip_session = ort.InferenceSession("clip_vision.onnx")

with open(captions_path) as f:
    records = [json.loads(line) for line in f]

random.seed(1337)  # 跟量化時用的同一個 seed,重現同一批 calib/holdout 切分
calib_records = random.sample(records, min(500, len(records)))
calib_names = {r["file_name"] for r in calib_records}
holdout_records = [r for r in records if r["file_name"] not in calib_names][:200]
print(f"holdout: {len(holdout_records)} 筆")


def build_sample(record):
    img_path = root / "images_thermal_val" / record["file_name"]
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].numpy()
    img_feat = clip_session.run(None, {"pixel_values": pv})[0]

    token_groups = tokenizer.encode(record["caption"])
    flat_ids = [tid for group in token_groups for tid in group]
    seq = [IMAGE_TOKEN_ID] + flat_ids + [EOS_TOKEN_ID]
    idx = np.array(seq[:-1], dtype=np.int64)[None, :]
    targets = np.array(seq[1:], dtype=np.int64)[None, :]
    return idx, img_feat.astype(np.float32), targets


def cross_entropy(logits, targets):
    # logits: (1, T, V), targets: (1, T) -- 跟 GPT.forward 裡
    # F.cross_entropy(logits.view(-1, V), targets.view(-1)) 算法一樣,手動用
    # numpy 重算(onnx 模型只輸出 logits,沒有算 loss)
    logits = logits[0]
    targets = targets[0]
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(len(targets)), targets]
    return nll.mean()


fp32_session = ort.InferenceSession("gpt.onnx")
int8_session = ort.InferenceSession(int8_path)

fp32_losses, int8_losses = [], []
top1_agree_total, top1_agree_count = 0, 0

for record in holdout_records:
    idx, img_feat, targets = build_sample(record)
    fp32_logits = fp32_session.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
    int8_logits = int8_session.run(None, {"input_ids": idx, "img_feat": img_feat})[0]

    fp32_losses.append(cross_entropy(fp32_logits, targets))
    int8_losses.append(cross_entropy(int8_logits, targets))

    fp32_argmax = fp32_logits[0].argmax(axis=-1)
    int8_argmax = int8_logits[0].argmax(axis=-1)
    top1_agree_total += (fp32_argmax == int8_argmax).sum()
    top1_agree_count += len(fp32_argmax)

fp32_losses = np.array(fp32_losses)
int8_losses = np.array(int8_losses)

print(f"\n=== GPT decoder 精度對照 (200 張 holdout, {int8_path} vs FP32) ===")
print(f"FP32 cross-entropy loss: mean = {fp32_losses.mean():.6f}, std = {fp32_losses.std():.6f}")
print(f"INT8 cross-entropy loss: mean = {int8_losses.mean():.6f}, std = {int8_losses.std():.6f}")
print(f"loss 差值 (INT8 - FP32): mean = {(int8_losses - fp32_losses).mean():.6f}")
print(f"top-1 token 一致率 (INT8 argmax == FP32 argmax): {top1_agree_total / top1_agree_count:.6f} ({top1_agree_total}/{top1_agree_count})")
