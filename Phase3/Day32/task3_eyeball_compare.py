"""
Day36 Task 3:10 張圖,新舊 decoder 生成結果並排對照。
沿用 evaluate_val.py 的 GPT / minbpe / generate_batch / decode_generated,
不重寫生成邏輯。
"""
import json

import torch

from train_vlm import GPT, GPTConfig, minbpe, device  # noqa: F401
from evaluate_val import (
    generate_batch, decode_generated, BASE_VOCAB_SIZE,
)

IMAGE_IDS_DAY35 = [
    "data/video-JhYLiFCieHQHaY8o7-frame-002195-t9p9hLssbg77DLqxH.jpg",
    "data/video-YQpCvGJxowy9uhkCw-frame-006300-RmduwqDEZGaQwjxxM.jpg",
    "data/video-zp8ed5vPKfAJ2fKWh-frame-002073-tQpYPw4iRtcEqAvve.jpg",
    "data/video-zp8ed5vPKfAJ2fKWh-frame-003267-PF4EPNZbETHRX45HG.jpg",
    "data/video-AP7PvpujjZZGLnsJt-frame-003765-HbkL4AHTbGRPyTnGx.jpg",
]
IMAGE_IDS_EXTRA = [
    "data/video-k5bTJAiyEgHismN7Y-frame-004624-WQMJgCPSogQbt8qcq.jpg",
    "data/video-57kWWRyeqqHs3Byei-frame-004451-id9CTPqA5GQNQtDQb.jpg",
    "data/video-57kWWRyeqqHs3Byei-frame-002218-GvFqwFzdrm8aPduB5.jpg",
    "data/video-Qk8msXvMopoYNDdco-frame-005315-oJuSMH6XCuESTWAki.jpg",
    "data/video-57kWWRyeqqHs3Byei-frame-006681-o5NXT3jrAkJr7wmfq.jpg",
]
TARGET_FILES = IMAGE_IDS_DAY35 + IMAGE_IDS_EXTRA


def load_captions(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["file_name"]] = r["caption"]
    return d


def load_model_and_tokenizer(ckpt_path, train_captions_path):
    train_captions = []
    with open(train_captions_path, encoding="utf-8") as f:
        for line in f:
            train_captions.append(json.loads(line))
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model, tokenizer


def generate_for_files(model, tokenizer, feature_by_name, files):
    image_features = torch.stack([feature_by_name[fn] for fn in files]).to(device)
    idx, eos_step = generate_batch(model, image_features)
    out = {}
    for i, fn in enumerate(files):
        cap, _, _ = decode_generated(tokenizer, idx[i], int(eos_step[i]))
        out[fn] = cap
    return out


def main():
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    gt_full_caps = load_captions("captions_val.jsonl")
    gt_filtered_caps = load_captions("captions_val_filtered.jsonl")

    cache = torch.load("clip_features_val.pt", map_location="cpu")
    feature_by_name = {name: cache["features"][i] for i, name in enumerate(cache["file_name"])}

    model_full, tok_full = load_model_and_tokenizer("checkpoints/best_model.pt", "captions_train.jsonl")
    model_filtered, tok_filtered = load_model_and_tokenizer("checkpoints/best_model_filtered.pt", "captions_train_filtered.jsonl")

    with torch.no_grad():
        gen_full = generate_for_files(model_full, tok_full, feature_by_name, TARGET_FILES)
        gen_filtered = generate_for_files(model_filtered, tok_filtered, feature_by_name, TARGET_FILES)

    print("\n" + "=" * 90)
    for fn in TARGET_FILES:
        print(f"\nFILE: {fn}")
        print(f"  GT (full)      : {gt_full_caps.get(fn, '(not in full GT set)')}")
        print(f"  GT (filtered)  : {gt_filtered_caps.get(fn, '(not in filtered GT set -- dropped)')}")
        print(f"  GEN best_model.pt          : {gen_full[fn]}")
        print(f"  GEN best_model_filtered.pt : {gen_filtered[fn]}")


if __name__ == "__main__":
    main()
