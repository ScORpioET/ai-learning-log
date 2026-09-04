import base64
import html
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
BRIGHT_DIR = Path.home() / "ai-transition-2026" / "Phase3" / "bright_region_hypothesis"

rows = json.load(open(HERE / "test_analysis_rows.json"))
rng = random.Random(42)
sampled = rng.sample(rows, 10)
sampled.sort(key=lambda r: r["thermal_file"])
json.dump(sampled, open(HERE / "seed_sample_10_test.json", "w"), ensure_ascii=False, indent=2)

inference_by_key = {r["thermal_file"]: r for r in json.load(open(HERE / "test_inference_results.json"))}


def load_detections(path):
    """detections_*.jsonl 存的 file_name 沒有 'data/' 前綴,補回去跟
    rgb_file/thermal_file 對齊(這個坑在 classify_and_analyze.py 已經踩過
    一次,這裡沿用同樣的修法)。"""
    by_file = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_file[f"data/{r['file_name']}"] = r["detections"]
    return by_file


rgb_dets_by_file = load_detections(BRIGHT_DIR / "detections_rgb_test.jsonl")
th_dets_by_file = load_detections(BRIGHT_DIR / "detections_thermal_test.jsonl")

BOX_COLOR = (230, 60, 30)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()


def draw_boxes(im, detections):
    draw = ImageDraw.Draw(im)
    for d in detections:
        x, y, w, h = d["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=BOX_COLOR, width=3)
        label = f"{d['class_name']} {d['conf']:.2f}"
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x, y - th - 6, x + tw + 8, y], fill=BOX_COLOR)
        draw.text((x + 4, y - th - 4), label, fill=(255, 255, 255), font=FONT)
    return im

RESULT_LABEL = {
    ("tie", "tie"): ("real 打平 · proxy 也打平", "res-neutral"),
    ("tie", "rgb"): ("real 打平 · proxy 猜 RGB(無對錯可言)", "res-neutral"),
    ("tie", "thermal"): ("real 打平 · proxy 猜 thermal(無對錯可言)", "res-neutral"),
}


def result_tag(real, combined):
    if real == "tie":
        return RESULT_LABEL.get((real, combined), ("real 打平", "res-neutral"))
    if real == combined:
        return (f"猜對 · 真實較準:{real.upper()}", "res-correct")
    return (f"猜錯 · 真實較準:{real.upper()}(proxy 猜 {combined.upper()})", "res-wrong")


def to_b64(path, detections, w=440):
    im = Image.open(path).convert("RGB")
    im = draw_boxes(im, detections)
    ratio = w / im.width
    im = im.resize((w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def esc(s):
    return html.escape(s or "(空)")


cards = []
for i, r in enumerate(sampled, 1):
    rec = inference_by_key[r["thermal_file"]]
    rgb_path = TD / "video_rgb_test" / r["rgb_file"]
    th_path = TD / "video_thermal_test" / r["thermal_file"]
    rgb_b64 = to_b64(rgb_path, rgb_dets_by_file.get(r["rgb_file"], []))
    th_b64 = to_b64(th_path, th_dets_by_file.get(r["thermal_file"], []))
    tag_text, tag_cls = result_tag(r["real_winner"], r["combined_winner"])

    card = f"""
    <div class="card">
      <div class="card-head">
        <span class="idx">#{i}</span>
        <span class="meta">{esc(r['thermal_file'])}</span>
        <span class="badge {tag_cls}">{esc(tag_text)}</span>
      </div>
      <div class="imgs">
        <div class="img-block"><div class="img-label">RGB(紅框=YOLO偵測)</div><img src="data:image/jpeg;base64,{rgb_b64}"/></div>
        <div class="img-block"><div class="img-label">Thermal(紅框=YOLO偵測)</div><img src="data:image/jpeg;base64,{th_b64}"/></div>
      </div>
      <table class="proxy-table">
        <tr><th></th><th>RGB</th><th>Thermal</th></tr>
        <tr><td>YOLO 偵測數量</td><td>{r['n_dets_rgb']}</td><td>{r['n_dets_thermal']}</td></tr>
        <tr><td>平均信心值</td><td>{r['avg_conf_rgb']}</td><td>{r['avg_conf_thermal']}</td></tr>
        <tr><td>真實 frame_score</td><td>{r['rgb_score']}</td><td>{r['thermal_score']}</td></tr>
      </table>
      <table class="cap-table">
        <tr><th>RGB GT</th><td>{esc(rec['rgb_gt'])}</td></tr>
        <tr><th>RGB 生成</th><td>{esc(rec['rgb_gen'])}</td></tr>
        <tr><th>Thermal GT</th><td>{esc(rec['thermal_gt'])}</td></tr>
        <tr><th>Thermal 生成</th><td>{esc(rec['thermal_gen'])}</td></tr>
      </table>
      <div class="winners">count_winner={r['count_winner']} · conf_winner={r['conf_winner']} · combined_winner={r['combined_winner']} · real_winner={r['real_winner']}</div>
    </div>
    """
    cards.append(card)

html_out = f"""<title>RGB vs Thermal Proxy(Test Split)</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#b5652f;
  --res-correct-bg:#e4f3e6; --res-correct-fg:#2f7a3d;
  --res-wrong-bg:#fbe4e0; --res-wrong-fg:#b13f2c;
  --res-neutral-bg:#eeece6; --res-neutral-fg:#6b6558; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a;
    --res-correct-bg:#1e3323; --res-correct-fg:#7fca8f;
    --res-wrong-bg:#3a2420; --res-wrong-fg:#e88f7a;
    --res-neutral-bg:#302c25; --res-neutral-fg:#b5ae9e; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a;
  --res-correct-bg:#1e3323; --res-correct-fg:#7fca8f;
  --res-wrong-bg:#3a2420; --res-wrong-fg:#e88f7a;
  --res-neutral-bg:#302c25; --res-neutral-fg:#b5ae9e; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.86rem; margin-bottom:24px; max-width:780px; line-height:1.6; }}
.grid {{ display:flex; flex-direction:column; gap:18px; max-width:960px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:16px; }}
.card-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
.idx {{ font-weight:700; color:var(--accent); }}
.meta {{ font-family:monospace; font-size:.68rem; color:var(--muted); word-break:break-all; }}
.badge {{ margin-left:auto; padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:600; white-space:nowrap; }}
.res-correct {{ background:var(--res-correct-bg); color:var(--res-correct-fg); }}
.res-wrong {{ background:var(--res-wrong-bg); color:var(--res-wrong-fg); }}
.res-neutral {{ background:var(--res-neutral-bg); color:var(--res-neutral-fg); }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }}
.img-block img {{ width:100%; border-radius:8px; display:block; }}
.img-label {{ font-size:.68rem; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:3px; }}
.proxy-table {{ width:100%; border-collapse:collapse; font-size:.78rem; margin-bottom:10px; }}
.proxy-table th, .proxy-table td {{ padding:4px 6px; text-align:center; border:1px solid var(--border); }}
.proxy-table th:first-child {{ text-align:left; color:var(--muted); }}
.cap-table {{ width:100%; border-collapse:collapse; font-size:.82rem; margin-bottom:8px; }}
.cap-table th {{ text-align:left; color:var(--muted); font-weight:600; padding:4px 8px 4px 0; width:90px; vertical-align:top; white-space:nowrap; }}
.cap-table td {{ padding:4px 0; line-height:1.45; }}
.cap-table tr {{ border-top:1px solid var(--border); }}
.cap-table tr:first-child {{ border-top:none; }}
.winners {{ font-family:monospace; font-size:.68rem; color:var(--muted); }}
</style>
<h1>RGB vs Thermal 偵測代理指標驗證 —— Test Split,seed=42 抽樣</h1>
<div class="subtitle">
改用 test split(RGB/thermal 100% frame 對應,同步雙鏡頭,不是 Method A 部分配對)重跑,3749 組全配對。
seed=42 隨機抽 10 筆,含 real 打平、猜對、猜錯各種情況。整體結果比 val 更明確:combined 規則一致率只有
41.97%,比「無腦一律猜 thermal」這個多數類基準線(56.35%)還低,count 更差(37.45%)——在乾淨配對的
大樣本下,這個代理指標沒有預測力,詳見文字回報。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/proxy_seed_gallery_test.html"
Path(out_path).write_text(html_out, encoding="utf-8")
print(f"[done] {out_path}, {len(html_out)} bytes")
for r in sampled:
    print(r["thermal_file"], r["real_winner"], r["combined_winner"])
