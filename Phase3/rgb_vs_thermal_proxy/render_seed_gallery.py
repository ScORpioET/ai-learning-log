import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"

rows = json.load(open(HERE / "seed_sample_10.json"))
inference_by_key = {r["thermal_file"]: r for r in json.load(open(HERE / "val_inference_results.json"))}

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


def to_b64(path, w=440):
    im = Image.open(path).convert("RGB")
    ratio = w / im.width
    im = im.resize((w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def esc(s):
    return html.escape(s or "(空)")


cards = []
for i, r in enumerate(rows, 1):
    rec = inference_by_key[r["thermal_file"]]
    rgb_path = TD / "images_rgb_val" / r["rgb_file"]
    th_path = TD / "images_thermal_val" / r["thermal_file"]
    rgb_b64 = to_b64(rgb_path)
    th_b64 = to_b64(th_path)
    tag_text, tag_cls = result_tag(r["real_winner"], r["combined_winner"])

    card = f"""
    <div class="card">
      <div class="card-head">
        <span class="idx">#{i}</span>
        <span class="meta">{esc(r['thermal_file'])}</span>
        <span class="badge {tag_cls}">{esc(tag_text)}</span>
      </div>
      <div class="imgs">
        <div class="img-block"><div class="img-label">RGB</div><img src="data:image/jpeg;base64,{rgb_b64}"/></div>
        <div class="img-block"><div class="img-label">Thermal</div><img src="data:image/jpeg;base64,{th_b64}"/></div>
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

html_out = f"""<title>RGB vs Thermal Proxy 驗證樣本</title>
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
<h1>RGB vs Thermal 偵測代理指標驗證 —— seed=42 抽樣</h1>
<div class="subtitle">
val set 872 組對齊樣本裡,seed=42 隨機抽 10 筆(含 real 打平、猜對、猜錯各種情況,不是只挑好看的)。
每筆列出 YOLO 偵測數量/平均信心值(代理指標)、Position-Class Binding Accuracy 算出的真實 frame_score、
兩邊 GT/生成 caption,以及 count/conf/combined 三種 proxy 規則各自的判斷跟真實結果的對照。
整體結論:combined 規則一致率 56.7%(基準線 51.3%),count 單獨看幾乎無訊號(50.1%),詳見前一則回報。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/proxy_seed_gallery.html"
Path(out_path).write_text(html_out, encoding="utf-8")
print(f"[done] {out_path}, {len(html_out)} bytes")
