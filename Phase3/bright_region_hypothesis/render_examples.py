"""
從 A 組(候選漏偵測)裡各挑幾張「重疊」跟「沒重疊」的實際案例,疊框+疊
亮區遮罩,存成 HTML 方便肉眼複核。用 overlap_center 這個判定當挑選依據
(比 overlap_iou 更貼近「亮區到底有沒有蓋到物件中心」的直覺)。
"""
import base64
import html
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
from PIL import Image, ImageDraw

from bright_region_detect import detect_bright_regions

HERE = Path(__file__).parent
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
SEED = 42
N_PER_GROUP = 4


def draw_example(rec):
    rgb_path = TD / "video_rgb_test" / rec["rgb_file"]
    th_path = TD / "video_thermal_test" / rec["thermal_file"]

    mask, region_boxes = detect_bright_regions(rgb_path)
    rgb_img = Image.open(rgb_path).convert("RGB")
    overlay = Image.new("RGBA", rgb_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    yellow = Image.new("RGBA", rgb_img.size, (255, 220, 0, 90))
    overlay.paste(yellow, (0, 0), mask_img)
    rgb_img = Image.alpha_composite(rgb_img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(rgb_img)
    x, y, w, h = rec["projected_bbox"]
    draw.rectangle([x, y, x + w, y + h], outline=(230, 30, 30), width=4)
    draw.text((x, max(0, y - 20)), f"{rec['en_name']} (thermal proj, conf={rec['thermal_conf']:.2f})",
              fill=(230, 30, 30))

    th_img = Image.open(th_path).convert("RGB")
    tdraw = ImageDraw.Draw(th_img)
    tx, ty, tw, th_ = rec["thermal_bbox"]
    tdraw.rectangle([tx, ty, tx + tw, ty + th_], outline=(230, 30, 30), width=3)

    def to_b64(im, w=480):
        ratio = w / im.width
        im = im.resize((w, int(im.height * ratio)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return to_b64(rgb_img), to_b64(th_img)


def main():
    records = [json.loads(l) for l in open(HERE / "classified_objects.jsonl", encoding="utf-8")]
    a_overlap = [r for r in records if r["group"] == "A" and r["overlap_center"]]
    a_no_overlap = [r for r in records if r["group"] == "A" and not r["overlap_center"]]
    print(f"[info] A overlap candidates: {len(a_overlap)}, A no-overlap candidates: {len(a_no_overlap)}")

    rng = random.Random(SEED)
    sample_overlap = rng.sample(a_overlap, min(N_PER_GROUP, len(a_overlap)))
    sample_no_overlap = rng.sample(a_no_overlap, min(N_PER_GROUP, len(a_no_overlap)))

    cards = []
    for tag, samples in (("重疊(overlap_center=True)", sample_overlap), ("沒重疊(overlap_center=False)", sample_no_overlap)):
        for rec in samples:
            rgb_b64, th_b64 = draw_example(rec)
            cards.append(f"""
            <div class="card">
              <div class="tag">{html.escape(tag)} · {rec['en_name']} · thermal_conf={rec['thermal_conf']:.2f}
                · best_iou={rec['best_iou']}</div>
              <div class="imgs">
                <div><div class="lbl">RGB(黃=亮區,紅框=thermal投影位置)</div><img src="data:image/jpeg;base64,{rgb_b64}"/></div>
                <div><div class="lbl">Thermal(原偵測)</div><img src="data:image/jpeg;base64,{th_b64}"/></div>
              </div>
              <div class="fname">{html.escape(rec['rgb_file'])}</div>
            </div>""")

    html_out = f"""<title>Bright Region Hypothesis Examples</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#b5652f; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.88rem; margin-bottom:24px; max-width:760px; line-height:1.6; }}
.grid {{ display:flex; flex-direction:column; gap:18px; max-width:1000px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:14px; }}
.tag {{ font-weight:700; font-size:.85rem; margin-bottom:8px; color:var(--accent); }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.lbl {{ font-size:.7rem; color:var(--muted); margin-bottom:4px; }}
img {{ width:100%; border-radius:8px; display:block; }}
.fname {{ font-family:monospace; font-size:.65rem; color:var(--muted); margin-top:8px; word-break:break-all; }}
</style>
<h1>A 組(候選漏偵測)—— 亮區重疊 vs 沒重疊 案例</h1>
<div class="subtitle">A 組定義:thermal 偵測到、RGB 在投影位置沒有匹配到(或匹配到但信心值明顯偏低)的物件。
黃色是 RGB 圖上偵測到的亮區(強光/耀光),紅框是 thermal bbox 投影到 RGB 座標系的位置(等比例縮放投影,
非精確校準,見報告方法論說明)。</div>
<div class="grid">
{''.join(cards)}
</div>
"""
    out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/bright_region_examples.html"
    Path(out_path).write_text(html_out, encoding="utf-8")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
