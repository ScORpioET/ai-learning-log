"""
對 val_pairs.json 的 872 組 RGB/thermal frame pair,各自跑最新穩定 checkpoint
生成 caption。

checkpoint(比對檔案時間戳確認,capfix 版本比 Jack 原本點名的
best_model_rgb_full_reweight2x.pt / best_model_exp2_reweight2x.pt 更新
——Day38 caption-completeness bug 修好後重訓的版本,是目前實際最新穩定版,
跟 Day39 caption_fusion 用的同一組):
    thermal = best_model_full_capfix_reweight2x.pt
    rgb     = best_model_rgb_full_capfix_reweight2x.pt
GT 直接讀現成的 captions_val_full_capfix.jsonl / captions_rgb_val_full_capfix.jsonl
(跟上面 checkpoint 訓練時用的同一份 GT 語料,不重新呼叫 generate_captions.py
——generate_captions.py 的 build_caption() 在這次對話更早的步驟被改成拿掉
top-2 cap,如果現在重新生成 GT 會跟訓練時的版本對不上,所以直接用既有檔案)。

第一版直接 import caption_fusion/run_model_inference.py 的 run_domain()
(872 張圖一次全部塞進 CLIP 特徵擷取 + generate_batch),在這台機器(15GB
RAM)上把系統記憶體榨乾被 OOM killer 殺掉(dmesg 確認 anon-rss 15.3GB)。
run_domain() 本身沒有分批機制,872 張一次做不下去,所以改成這支腳本自己
拆成小批次(CHUNK_SIZE=100)迴圈呼叫,但底層的 extract_clip_features()/
generate_batch()/decode_generated() 全部照原樣重用,不改生成邏輯本身,
只是加了「模型只載入一次、每批次跑完釋放中間 tensor」這層批次控制。
"""
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent
CHUNK_SIZE = 100

sys.path.insert(0, str(DAY32))
from train_vlm import GPT, GPTConfig, minbpe, device  # noqa: E402
from evaluate_val import generate_batch, decode_generated, BASE_VOCAB_SIZE  # noqa: E402

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

THERMAL_CKPT = DAY32 / "checkpoints" / "best_model_full_capfix_reweight2x.pt"
RGB_CKPT = DAY32 / "checkpoints" / "best_model_rgb_full_capfix_reweight2x.pt"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_domain_chunked(label, image_root, file_names, train_captions_path, ckpt_path):
    print(f"[{label}] 用 {train_captions_path.name} 重訓 tokenizer...")
    train_captions = [json.loads(l) for l in open(train_captions_path, encoding="utf-8")]
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[{label}] 載入 {ckpt_path.name}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(clip_device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    results = {}
    n_chunks = (len(file_names) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci, chunk in enumerate(chunked(file_names, CHUNK_SIZE)):
        images = [Image.open(Path(image_root) / fn).convert("RGB") for fn in chunk]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(clip_device)
            feats = clip_model.get_image_features(**inputs).pooler_output.to(device)
            idx, eos_step = generate_batch(model, feats)
        for i, fn in enumerate(chunk):
            caption, gen_len, eos_hit = decode_generated(tokenizer, idx[i], int(eos_step[i]))
            results[fn] = {"gen_caption": caption, "gen_len": gen_len, "eos_hit": eos_hit}
        del images, inputs, feats, idx, eos_step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ...[{label}] chunk {ci+1}/{n_chunks}", end="\r")
    print()

    del clip_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    pairs = json.load(open(HERE / "val_pairs.json"))
    thermal_files = [p["thermal_file"] for p in pairs]
    rgb_files = [p["rgb_file"] for p in pairs]
    print(f"[info] {len(pairs)} 組樣本, thermal ckpt={THERMAL_CKPT.name}, rgb ckpt={RGB_CKPT.name}, "
          f"chunk_size={CHUNK_SIZE}")

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    thermal_gen = run_domain_chunked(
        "thermal", TD / "images_thermal_val", thermal_files,
        DAY32 / "captions_train_full_capfix.jsonl", THERMAL_CKPT,
    )
    rgb_gen = run_domain_chunked(
        "rgb", TD / "images_rgb_val", rgb_files,
        DAY32 / "captions_rgb_train_full_capfix.jsonl", RGB_CKPT,
    )

    thermal_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
                  for l in open(TD / "captions_val_full_capfix.jsonl", encoding="utf-8")}
    rgb_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
              for l in open(Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation" / "captions_rgb_val_full_capfix.jsonl", encoding="utf-8")}

    out = []
    for p in pairs:
        tf, rf = p["thermal_file"], p["rgb_file"]
        out.append({
            **p,
            "thermal_gt": thermal_gt.get(tf, ""), "rgb_gt": rgb_gt.get(rf, ""),
            "thermal_gen": thermal_gen[tf]["gen_caption"], "rgb_gen": rgb_gen[rf]["gen_caption"],
        })

    with open(HERE / "val_inference_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] val_inference_results.json written, {len(out)} records")


if __name__ == "__main__":
    main()
