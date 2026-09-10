"""
控制實驗:generate_batch() 用 torch.multinomial 抽樣(不是 greedy),代表
「同一張圖片、只是換一次隨機種子」本身就可能生成不同 caption,這個雜訊跟
「flare removal 真的改變了圖片內容」是兩件事,必須先分開估計,不然
before/after 的差異沒辦法歸因。

做法:對 4 張 before 原圖(2 張 high_diff + 2 張 low_diff 各自的 before 版本,
不用 after,避免混入真正的內容差異),同一張圖各生成 5 次 caption(每次
換一個 manual_seed),看同一張圖片自己內部的 caption 有多不穩定。
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
sys.path.insert(0, str(DAY32))

from train_vlm import GPT, GPTConfig, device, minbpe  # noqa: E402,F401
from evaluate_val import generate_batch, decode_generated, BASE_VOCAB_SIZE  # noqa: E402

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CKPT_PATH = DAY32 / "checkpoints" / "best_model_rgb_full_reweight2x.pt"
TRAIN_CAPTIONS_PATH = DAY32 / "captions_rgb_train_full.jsonl"
HERE = Path(__file__).parent
N_REPEATS = 5
SEEDS = [42, 43, 44, 45, 46]


def extract_clip_features(image_paths):
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(clip_device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(clip_device)
        outputs = model.get_image_features(**inputs)
        pooled = outputs.pooler_output.cpu()
    del model
    return pooled


def main():
    samples = json.load(open(HERE / "selected_samples.json"))
    picks = [s for s in samples if s["group"] == "high_diff"][:2] + \
            [s for s in samples if s["group"] == "low_diff"][:2]
    paths = [s["before_path"] for s in picks]

    print(f"[info] 抽 {len(paths)} 張圖的 CLIP 特徵(只算一次,重複用於 {N_REPEATS} 次生成)...")
    feats = extract_clip_features(paths)

    train_captions = [json.loads(l) for l in open(TRAIN_CAPTIONS_PATH, encoding="utf-8")]
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_features = feats.to(device)

    results = {s["file_name"]: [] for s in picks}
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.no_grad():
            idx, eos_step = generate_batch(model, image_features)
        for i, s in enumerate(picks):
            caption, _, _ = decode_generated(tokenizer, idx[i], int(eos_step[i]))
            results[s["file_name"]].append(caption)

    out = {"seeds": SEEDS, "results": results}
    with open(HERE / "sampling_noise_control.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[done] sampling_noise_control.json written\n")
    for fn, caps in results.items():
        n_unique = len(set(caps))
        print(f"- {fn}  ({n_unique}/{len(caps)} unique captions across seeds)")
        for seed, c in zip(SEEDS, caps):
            print(f"    seed={seed}: {c}")


if __name__ == "__main__":
    main()
