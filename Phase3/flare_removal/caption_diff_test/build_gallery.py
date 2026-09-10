"""
Step 4:把 20 張 frame 的 before/after 圖片存到獨立資料夾,方便肉眼複核。
每張 frame 存成一張左右合併圖(左 before / 右 after,上方標 caption),
檔名帶 group + diff_mean,方便排序瀏覽。
"""
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT_DIR = HERE / "gallery"
OUT_DIR.mkdir(exist_ok=True)

CAPTION_BAND_H = 90
PAD = 8


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    samples = json.load(open(HERE / "captions_before_after.json"))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for s in samples:
        before = Image.open(s["before_path"]).convert("RGB")
        after = Image.open(s["after_path"]).convert("RGB")
        h = max(before.height, after.height)
        w = before.width + after.width + PAD

        canvas = Image.new("RGB", (w, h + CAPTION_BAND_H), "white")
        canvas.paste(before, (0, CAPTION_BAND_H))
        canvas.paste(after, (before.width + PAD, CAPTION_BAND_H))

        draw = ImageDraw.Draw(canvas)
        col_w = before.width - 10
        before_lines = wrap_text(draw, f"BEFORE: {s['before_caption']}", font, col_w)
        after_lines = wrap_text(draw, f"AFTER: {s['after_caption']}", font, after.width - 10)
        y = 4
        for line in before_lines[:3]:
            draw.text((4, y), line, fill="black", font=font)
            y += 20
        y = 4
        for line in after_lines[:3]:
            draw.text((before.width + PAD + 4, y), line, fill="black", font=font)
            y += 20

        fname = f"{s['group']}_diffmean{s['diff_mean']:.1f}_{s['file_name']}"
        canvas.save(OUT_DIR / fname, quality=90)

    print(f"[done] {len(samples)} gallery images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
