"""
make_inspection_sets.py — Task 4:兩份給人眼看的視覺化抽樣資料集

(a) night_verify/  : val set 隨機抽 20 張 GT=Night + 20 張 GT=非Night,單張熱像圖
                      (RGB 配對資料經查證實際不存在 —— thermal/RGB 是不重疊的獨立
                      影片集合,video_id 完全對不上,沒有可靠的逐幀配對方式,
                      所以只用熱像,不做並排)。疊字印 GT caption 前 60 字元。
(b) gt_inspection/  : val set 隨機抽 30 張熱像圖,疊字印完整 GT caption,
                      另外印 summary.txt 附 annotation_count / classes_in_frame
                      (直接讀 coco.json 原始標註,不經過 caption 模板的 long-tail
                      改名,給人工核對用的是「最原始」的類別清單)。
"""
import json
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

THERMAL_VAL_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"
CAPTIONS_VAL_PATH = "captions_val_v2.jsonl"  # long-tail bug 修好後的版本
OUT_NIGHT_DIR = Path("night_verify")
OUT_GT_DIR = Path("gt_inspection")
SEED = 42
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_caption_banner(img, text, font_size=16):
    """在圖片下方疊一條半透明黑底白字的字幕,text 可能多行(用 \n 分好)。"""
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


def has_night_prefix(caption):
    return caption.startswith("Night:") or caption.startswith("Night,")


def main():
    random.seed(SEED)

    val_captions = []
    with open(CAPTIONS_VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            val_captions.append(json.loads(line))
    print(f"讀到 {len(val_captions)} 筆 val captions(來源: {CAPTIONS_VAL_PATH})")

    coco = json.load(open(THERMAL_VAL_ROOT / "coco.json", "r", encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_file = defaultdict(list)
    img_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    for ann in coco["annotations"]:
        fname = img_id_to_file.get(ann["image_id"])
        if fname:
            anns_by_file[fname].append(id2name[ann["category_id"]])

    # --- (a) night_verify/ ---
    OUT_NIGHT_DIR.mkdir(exist_ok=True)
    night_pool = [c for c in val_captions if has_night_prefix(c["caption"])]
    non_night_pool = [c for c in val_captions if not has_night_prefix(c["caption"])]
    print(f"night_pool={len(night_pool)}  non_night_pool={len(non_night_pool)}")

    night_sample = random.sample(night_pool, min(20, len(night_pool)))
    non_night_sample = random.sample(non_night_pool, min(20, len(non_night_pool)))

    for i, row in enumerate(night_sample, start=1):
        img = Image.open(THERMAL_VAL_ROOT / row["file_name"])
        snippet = row["caption"][:60]
        out = draw_caption_banner(img, f"GT: {snippet}", font_size=16)
        out.save(OUT_NIGHT_DIR / f"night_{i:02d}_YES.png")

    for i, row in enumerate(non_night_sample, start=1):
        img = Image.open(THERMAL_VAL_ROOT / row["file_name"])
        snippet = row["caption"][:60]
        out = draw_caption_banner(img, f"GT: {snippet}", font_size=16)
        out.save(OUT_NIGHT_DIR / f"night_{i:02d}_NO.png")

    print(f"[4a] 寫出 {len(night_sample)} 張 YES + {len(non_night_sample)} 張 NO 到 {OUT_NIGHT_DIR}/")

    # --- (b) gt_inspection/ ---
    OUT_GT_DIR.mkdir(exist_ok=True)
    gt_sample = random.sample(val_captions, min(30, len(val_captions)))

    summary_lines = []
    for i, row in enumerate(gt_sample, start=1):
        img = Image.open(THERMAL_VAL_ROOT / row["file_name"])
        wrapped = wrap_text(f"GT: {row['caption']}", max_chars=70)
        out = draw_caption_banner(img, wrapped, font_size=15)
        fname = f"gt_{i:02d}.png"
        out.save(OUT_GT_DIR / fname)

        classes = anns_by_file.get(row["file_name"], [])
        class_counts = sorted(set(classes))
        summary_lines.append(
            f"{fname} | GT: {row['caption']} | annotation_count: {len(classes)} | classes_in_frame: {class_counts}"
        )

    with open(OUT_GT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[4b] 寫出 {len(gt_sample)} 張圖 + summary.txt 到 {OUT_GT_DIR}/")


if __name__ == "__main__":
    main()
