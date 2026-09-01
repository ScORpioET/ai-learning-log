"""
exp2_reweight2x 5 張 eyeball 比較圖,GT + 生成結果疊在圖片下方 banner。
文字直接取自 log/exp2_reweight2x_stdout.log 裡跑 best_model_exp2_reweight2x.pt
的原始輸出,不重新 inference(該次生成用的是 torch.manual_seed(42) 的確定結果,
log 裡就是這個 checkpoint 對這 5 張圖的真實生成句)。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THERMAL_VAL_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"
OUT_DIR = Path("exp2_eyeball_overlay")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 取自 log/exp2_reweight2x_stdout.log(best_model_exp2_reweight2x.pt 的生成結果)
SAMPLES = [
    {
        "file_name": "data/video-JhYLiFCieHQHaY8o7-frame-000000-xT7BXRKKyuWEsnywX.jpg",
        "gt": "Night: two cars, one nearby ahead.",
        "gen": "Several cars, the nearest on the left; three pedestrians, one ahead.",
    },
    {
        "file_name": "data/video-JhYLiFCieHQHaY8o7-frame-000600-54cr88GJsdAyksdNj.jpg",
        "gt": "Night: a car ahead.",
        "gen": "Night: a car nearby on the left.",
    },
    {
        "file_name": "data/video-JhYLiFCieHQHaY8o7-frame-000900-pdb96S7B7fgmguxPE.jpg",
        "gt": "Night: several cars, one nearby on the right.",
        "gen": "Three cars, the nearest on the right; a pedestrian in the distance ahead.",
    },
    {
        "file_name": "data/video-JhYLiFCieHQHaY8o7-frame-001200-n7sXgfg2LyFCYRzSc.jpg",
        "gt": "Night: a car nearby on the right; a pedestrian on the left.",
        "gen": "Night: several cars, one ahead; a pedestrian ahead.",
    },
    {
        "file_name": "data/video-JhYLiFCieHQHaY8o7-frame-001500-4dRaerqz2cJhjyCHA.jpg",
        "gt": "Night: many cars, one nearby on the right.",
        "gen": "Many cars, the nearest on the right.",
    },
]


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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    summary_lines = []
    for i, row in enumerate(SAMPLES, start=1):
        img = Image.open(THERMAL_VAL_ROOT / row["file_name"])
        text = wrap_text(f"GT:  {row['gt']}\nGEN: {row['gen']}", max_chars=70)
        out = draw_caption_banner(img, text, font_size=15)
        fname = f"exp2_{i:02d}.png"
        out.save(OUT_DIR / fname)

        summary_lines.append(f"{fname} | {row['file_name']}\n  GT : {row['gt']}\n  GEN: {row['gen']}")
        print(f"[{i}/{len(SAMPLES)}] {row['file_name']} -> {OUT_DIR / fname}")

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines) + "\n")

    print(f"完成:{len(SAMPLES)} 張圖 + summary.txt 存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
