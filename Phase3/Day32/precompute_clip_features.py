"""
把 captions_{split}.jsonl 裡列出的每張熱像圖,跑過 frozen CLIP ViT
(openai/clip-vit-base-patch32),把 pooled image embedding(512 維)
預先算好存檔——decoder fine-tune 階段直接讀存好的向量,不用每個 epoch
重跑一次 CLIP forward,省下大量重複運算。

用法:
    python precompute_clip_features.py --split train
    python precompute_clip_features.py --split val

輸出:
    clip_features_{split}.pt
    內容是一個 dict: {"file_name": [...], "features": Tensor[N, 512]}
    "file_name" 跟 "features" 的順序一一對應,file_name 字串跟
    captions_{split}.jsonl 裡的 "file_name" 欄位完全一致,拿來對照用。

【圖片檔案格式的假設,尚未實測驗證,第一次跑務必留意印出來的 log】
coco.json 裡的 file_name 是 .jpg,大機率指向「已經可視化過的」熱像圖
(不是保留原始溫度值的 16-bit raw),所以理論上直接讀成一般 RGB 圖片
就可以餵給 CLIP。這裡用 PIL 的 .convert("RGB") 保險——不管原始檔案是
單通道灰階還是已經是 RGB,.convert("RGB") 都會轉成 CLIP 預期的三通道
格式,所以不管前面那個假設對不對,這支 script 都不會因為通道數量壞掉。
第一次執行時第一張圖的 log 會印出原始 mode(例如 "L" 是單通道灰階,
"RGB" 是三通道),你可以順手確認一下跟你認知的是否一致。
"""
SCRIPT_VERSION = "v0.1 (2026-08-25)"

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def main(split, captions_path, out_path):
    print(f"[script version] {SCRIPT_VERSION}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    root = Path.home() / "ai-transition-2026" / "thermal_dataset" / f"images_thermal_{split}"

    print(f"[info] loading {MODEL_NAME} ...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    file_names = []
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            file_names.append(row["file_name"])
    print(f"[info] {len(file_names)} images to process (from {captions_path})")

    all_features = []
    logged_first_mode = False

    with torch.no_grad():
        for i in range(0, len(file_names), BATCH_SIZE):
            batch_names = file_names[i:i + BATCH_SIZE]
            images = []
            for name in batch_names:
                img_path = root / name
                img = Image.open(img_path)
                if not logged_first_mode:
                    print(f"[info] first image raw mode = {img.mode} ({img_path})")
                    logged_first_mode = True
                images.append(img.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            pooled = outputs.pooler_output  # (batch, 512)，已經過 visual_projection
            all_features.append(pooled.cpu())

            done = min(i + BATCH_SIZE, len(file_names))
            print(f"[progress] {done}/{len(file_names)}", end="\r")

    print()
    features = torch.cat(all_features, dim=0)
    assert features.shape[0] == len(file_names), \
        f"特徵數量 {features.shape[0]} 跟圖片數量 {len(file_names)} 對不上,不要往下走"

    torch.save({"file_name": file_names, "features": features}, out_path)
    print(f"[done] saved {features.shape[0]} features, shape={tuple(features.shape)}, to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--captions", default=None,
                         help="預設是 captions_{split}.jsonl")
    parser.add_argument("--out", default=None,
                         help="預設是 clip_features_{split}.pt")
    args = parser.parse_args()

    data_path = os.path.expanduser('~/ai-transition-2026/thermal_dataset')

    captions_path = args.captions or os.path.join(data_path, f"captions_{args.split}.jsonl")
    out_path = args.out or f"clip_features_{args.split}.pt"
    main(args.split, captions_path, out_path)