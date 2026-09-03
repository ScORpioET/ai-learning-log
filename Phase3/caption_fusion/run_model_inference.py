"""
Day39:對 build_test_pairs.py 選出的 10 組 test frame pair,各自跑上一輪
(Phase 4 capfix)最新穩定版 checkpoint 的 model inference,拿到 thermal/RGB
各自的「model 生成」caption(不是 GT)。

流程(逐一沿用既有腳本的邏輯,沒有重寫任何生成/tokenizer 邏輯):
  1. CLIP 特徵:跟 precompute_clip_features.py / precompute_clip_features_rgb.py
     同一套(openai/clip-vit-base-patch32,pooled 512 維),只是這次只對
     20 張新圖現算,不存整個 test set 的特徵檔。
  2. tokenizer:跟 evaluate_val.py 一樣,用該 domain 訓練當時的 train
     captions 語料重跑 minbpe(必須跟 checkpoint 訓練時同一份語料+
     vocab_size,不然 token id 對不上 embedding)。
  3. 生成:直接 import evaluate_val.py 的 generate_batch()/decode_generated(),
     邏輯不改。

checkpoint:
  thermal = best_model_full_capfix_reweight2x.pt(Phase 4 capfix,最新穩定版)
  rgb     = best_model_rgb_full_capfix_reweight2x.pt(同上)
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
sys.path.insert(0, str(DAY32))

from train_vlm import GPT, GPTConfig, minbpe, device  # noqa: E402
from evaluate_val import generate_batch, decode_generated, BASE_VOCAB_SIZE  # noqa: E402

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def extract_clip_features(image_root, file_names):
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(clip_device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    images = [Image.open(Path(image_root) / fn).convert("RGB") for fn in file_names]
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(clip_device)
        outputs = model.get_image_features(**inputs)
        pooled = outputs.pooler_output.cpu()
    del model
    return pooled


def run_domain(label, image_root, file_names, train_captions_path, ckpt_path):
    print(f"[{label}] 抽 {len(file_names)} 張圖的 CLIP 特徵...")
    feats = extract_clip_features(image_root, file_names)

    train_captions = [json.loads(l) for l in open(train_captions_path, encoding="utf-8")]
    print(f"[{label}] 用 {len(train_captions)} 筆 train captions 重訓 tokenizer...")
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[{label}] 載入 {ckpt_path.name}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    image_features = feats.to(device)
    with torch.no_grad():
        idx, eos_step = generate_batch(model, image_features)

    results = {}
    for i, fn in enumerate(file_names):
        caption, gen_len, eos_hit = decode_generated(tokenizer, idx[i], int(eos_step[i]))
        results[fn] = {"gen_caption": caption, "gen_len": gen_len, "eos_hit": eos_hit}
    del model
    return results


def main():
    pairs = json.load(open(Path(__file__).parent / "test_pairs.json"))["sampled"]
    thermal_files = [p["thermal_file"] for p in pairs]
    rgb_files = [p["rgb_file"] for p in pairs]

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    thermal_gen = run_domain(
        "thermal", TD / "video_thermal_test", thermal_files,
        DAY32 / "captions_train_full_capfix.jsonl",
        DAY32 / "checkpoints" / "best_model_full_capfix_reweight2x.pt",
    )
    rgb_gen = run_domain(
        "rgb", TD / "video_rgb_test", rgb_files,
        DAY32 / "captions_rgb_train_full_capfix.jsonl",
        DAY32 / "checkpoints" / "best_model_rgb_full_capfix_reweight2x.pt",
    )

    thermal_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
                  for l in open(TD / "captions_test.jsonl", encoding="utf-8")}
    rgb_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
              for l in open(Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation" / "captions_rgb_test.jsonl", encoding="utf-8")}

    out = []
    for p in pairs:
        tf, rf = p["thermal_file"], p["rgb_file"]
        out.append({
            "thermal_file": tf, "rgb_file": rf,
            "thermal_video_id": p["thermal_video_id"], "rgb_video_id": p["rgb_video_id"],
            "frame_index": p["frame_index"],
            "thermal_gt": thermal_gt.get(tf, ""), "rgb_gt": rgb_gt.get(rf, ""),
            "thermal_gen": thermal_gen[tf]["gen_caption"], "rgb_gen": rgb_gen[rf]["gen_caption"],
        })

    with open(Path(__file__).parent / "model_inference_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[done] model_inference_results.json written")
    for r in out:
        print(f"- {r['thermal_file']}")
        print(f"    thermal GT : {r['thermal_gt']}")
        print(f"    thermal GEN: {r['thermal_gen']}")
        print(f"    rgb GT     : {r['rgb_gt']}")
        print(f"    rgb GEN    : {r['rgb_gen']}")


if __name__ == "__main__":
    main()
