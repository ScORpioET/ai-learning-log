"""
跟 Phase3/Day32/precompute_clip_features.py 同一套流程(frozen CLIP ViT
openai/clip-vit-base-patch32,pooled 512 維 embedding),只換 root 路徑成
images_rgb_{split},其他邏輯逐行照抄不改。
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def main(split, captions_path, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    root = Path.home() / "ai-transition-2026" / "thermal_dataset" / f"images_rgb_{split}"

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    file_names = []
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            file_names.append(row["file_name"])
    print(f"[info] {len(file_names)} images to process (from {captions_path})")

    all_features = []
    with torch.no_grad():
        for i in range(0, len(file_names), BATCH_SIZE):
            batch_names = file_names[i:i + BATCH_SIZE]
            images = [Image.open(root / name).convert("RGB") for name in batch_names]
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            pooled = outputs.pooler_output
            all_features.append(pooled.cpu())
            done = min(i + BATCH_SIZE, len(file_names))
            print(f"[progress] {done}/{len(file_names)}", end="\r")
    print()

    features = torch.cat(all_features, dim=0)
    assert features.shape[0] == len(file_names)
    torch.save({"file_name": file_names, "features": features}, out_path)
    print(f"[done] saved {features.shape[0]} features, shape={tuple(features.shape)}, to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.split, args.captions, args.out)
