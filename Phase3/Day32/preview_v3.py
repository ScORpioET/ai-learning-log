"""
preview_v3.py — v0.7 caption + best_model_v3.pt standalone inference 抽樣預覽

對齊 make_inspection_sets.py 的視覺化慣例(圖片下方疊字幕 banner)。
用 checkpoints_v3/best_model.pt + tokenizer_v3.pkl(v0.7 corpus 重訓過的 tokenizer,
跟訓練這顆 checkpoint 用的是同一份 merges,token id 才對得上 embedding)
對 captions_val_v3.jsonl 隨機抽 10 張生成 caption,GT/生成並排寫進同一張圖,
存到 preview_v3/,肉眼檢查有沒有出現 several/many 這些 v0.7 新詞。

直接吃 clip_features_val.pt 裡的 precomputed CLIP feature,不重跑 CLIP
(v0.7 只換了 caption 文字,image 沒變,feature 快取本來就不用重算)。
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from train_vlm import GPT, GPTConfig, device  # noqa: F401
from train_minbpe import minbpe  # train_vlm.py 自己的 minbpe 沒有 save/load,這裡要用 train_minbpe.py 那份

THERMAL_VAL_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"
CAPTIONS_VAL_PATH = "captions_val_v3.jsonl"
FEATURES_VAL_PATH = "clip_features_val.pt"
CKPT_PATH = "checkpoints/best_model_v3.pt"
TOKENIZER_PATH = "tokenizer_v3.pkl"
OUT_DIR = Path("preview_v3")
SEED = 42
N_SAMPLES = 10
MAX_NEW_TOKENS = 40
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def wrap_text(text, max_chars=70):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def draw_caption_banner(img, text, font_size=15):
    img = img.convert("RGB")
    font = load_font(font_size)
    lines = text.split("\n")
    line_h = font_size + 6
    banner_h = line_h * len(lines) + 10

    canvas = Image.new("RGB", (img.width, img.height + banner_h), "black")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, img.height, img.width, img.height + banner_h], fill=(0, 0, 0))
    y = img.height + 5
    for line in lines:
        draw.text((6, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return canvas


@torch.no_grad()
def generate(model, tokenizer, img_feat, max_new_tokens=MAX_NEW_TOKENS):
    image_token_id = 318
    eos_token_id = 319
    idx = torch.tensor([[image_token_id]], dtype=torch.long, device=device)
    feat = img_feat.unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(idx, targets=None, image_feature=feat)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_token_id:
            break

    generated_ids = idx[0].tolist()
    clean_ids = [i for i in generated_ids if i not in (image_token_id, eos_token_id)]
    return tokenizer.decode([clean_ids]) if clean_ids else ""


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = minbpe.load(TOKENIZER_PATH)
    print(f"[1/4] 載入 tokenizer: {TOKENIZER_PATH} ({len(tokenizer.merges)} merges)")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[2/4] 載入 {CKPT_PATH}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    val_captions = []
    with open(CAPTIONS_VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            val_captions.append(json.loads(line))

    cache = torch.load(FEATURES_VAL_PATH, map_location="cpu")
    feature_by_name = {name: cache["features"][i] for i, name in enumerate(cache["file_name"])}
    val_captions = [c for c in val_captions if c["file_name"] in feature_by_name]
    print(f"[3/4] val set: {len(val_captions)} 筆(有對應 CLIP feature)")

    OUT_DIR.mkdir(exist_ok=True)
    sample = random.sample(val_captions, min(N_SAMPLES, len(val_captions)))

    summary_lines = []
    for i, row in enumerate(sample, start=1):
        img_feat = feature_by_name[row["file_name"]]
        gen_caption = generate(model, tokenizer, img_feat)

        img = Image.open(THERMAL_VAL_ROOT / row["file_name"])
        text = wrap_text(f"GT:  {row['caption']}\nGEN: {gen_caption}", max_chars=70)
        out = draw_caption_banner(img, text, font_size=15)
        fname = f"preview_{i:02d}.png"
        out.save(OUT_DIR / fname)

        summary_lines.append(f"{fname} | {row['file_name']}\n  GT : {row['caption']}\n  GEN: {gen_caption}")
        print(f"[{i}/{len(sample)}] {row['file_name']}")
        print(f"  GT : {row['caption']}")
        print(f"  GEN: {gen_caption}")

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines) + "\n")

    print(f"[4/4] 寫出 {len(sample)} 張圖 + summary.txt 到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
