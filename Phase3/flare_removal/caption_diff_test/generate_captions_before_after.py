"""
Step 2: 對 selected_samples.json 的 20 張 frame,各自用 before(原圖)、
after(Flare7K++ 去光暈後的 blend 圖)跑同一套 RGB caption pipeline
(checkpoint: best_model_rgb_full_reweight2x.pt),各生成一次 caption。

邏輯逐行沿用 Phase3/caption_fusion/run_model_inference.py 的 run_domain()
(CLIP 特徵 + minbpe tokenizer + train_vlm.GPT + evaluate_val.generate_batch/
decode_generated),不重寫生成邏輯。差異只在於:
  - checkpoint 換成 best_model_rgb_full_reweight2x.pt(非 capfix 版,對齊
    train captions 用 captions_rgb_train_full.jsonl,而不是 capfix 版)
  - 圖片來源是任意路徑(前景不是同一個資料夾的 file_name,而是 before/after
    兩批各自獨立路徑),所以用 PIL 直接開檔,不透過 image_root + file_name
    的組合。
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

    before_paths = [s["before_path"] for s in samples]
    after_paths = [s["after_path"] for s in samples]
    all_paths = before_paths + after_paths
    tags = ["before"] * len(before_paths) + ["after"] * len(after_paths)

    print(f"[info] 抽 {len(all_paths)} 張圖(前 {len(before_paths)} before / 後 {len(after_paths)} after)的 CLIP 特徵...")
    feats = extract_clip_features(all_paths)

    train_captions = [json.loads(l) for l in open(TRAIN_CAPTIONS_PATH, encoding="utf-8")]
    print(f"[info] 用 {len(train_captions)} 筆 {TRAIN_CAPTIONS_PATH.name} 重訓 tokenizer...")
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[info] 載入 {CKPT_PATH.name}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    image_features = feats.to(device)
    with torch.no_grad():
        idx, eos_step = generate_batch(model, image_features)

    gen_captions = []
    for i in range(len(all_paths)):
        caption, gen_len, eos_hit = decode_generated(tokenizer, idx[i], int(eos_step[i]))
        gen_captions.append(caption)

    n = len(samples)
    for i, s in enumerate(samples):
        s["before_caption"] = gen_captions[i]
        s["after_caption"] = gen_captions[n + i]

    out_path = HERE / "captions_before_after.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n[done] {out_path} written")
    for s in samples:
        print(f"- [{s['group']}] {s['file_name']} (diff_mean={s['diff_mean']:.2f})")
        print(f"    before: {s['before_caption']}")
        print(f"    after : {s['after_caption']}")


if __name__ == "__main__":
    main()
