import base64
import io
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
SPLIT_DIR = {"train": "images_rgb_train", "val": "images_rgb_val", "test": "video_rgb_test"}

top30 = json.load(open(HERE / "top30_median_all_splits.json"))


def to_b64(path, w=380):
    im = Image.open(path).convert("RGB")
    ratio = w / im.width
    im = im.resize((w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


cards = []
for i, r in enumerate(top30, 1):
    path = ROOT / SPLIT_DIR[r["split"]] / r["file_name"]
    b64 = to_b64(path)
    is_frame0 = "-frame-000000-" in r["file_name"]
    flag = ' <span class="blank">blank frame-0</span>' if is_frame0 else ""
    cards.append(f"""
    <div class="card{' flagged' if is_frame0 else ''}">
      <div class="head"><span class="idx">#{i}</span><span class="dd">median={r['median']}</span>{flag}</div>
      <img src="data:image/jpeg;base64,{b64}"/>
      <div class="meta"><span class="tag">{r['split']}</span> dark_diff={r['dark_diff']} · {r['file_name']}</div>
    </div>""")

html_out = f"""<title>Top 30 Median(混合三 Split)</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#a23b2e; --warn-bg:#f8e7e4; --flag:#c1652f; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0776a; --warn-bg:#3a2320; --flag:#e08a52; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0776a; --warn-bg:#3a2320; --flag:#e08a52; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.86rem; margin-bottom:16px; max-width:760px; line-height:1.6; }}
.caveat {{ background:var(--warn-bg); border:1px solid var(--accent); border-radius:10px; padding:14px 18px; margin-bottom:24px; max-width:760px; font-size:.85rem; line-height:1.7; }}
.caveat b {{ color:var(--accent); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; max-width:1200px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:10px; padding:10px; }}
.card.flagged {{ border-color:var(--flag); }}
.head {{ display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:6px; font-size:.8rem; flex-wrap:wrap; }}
.idx {{ font-weight:700; color:var(--accent); }}
.dd {{ font-family:monospace; color:var(--muted); }}
.blank {{ font-size:.62rem; font-weight:700; color:#fff; background:var(--flag); padding:1px 6px; border-radius:999px; }}
img {{ width:100%; border-radius:6px; display:block; background:#eee; }}
.meta {{ font-size:.65rem; color:var(--muted); margin-top:6px; word-break:break-all; }}
.tag {{ font-family:monospace; font-weight:700; color:var(--accent); }}
</style>
<h1>Median 最大值(最亮)Top 30 —— Train + Val + Test 混合</h1>
<div class="subtitle">
三個 split 混合(15,153 張),對整張圖算 median luminance,取混合後最大(最亮)的前 30 名。
</div>
<div class="caveat">
⚠️ <b>誠實揭露</b>:前 19 名(median=254,已經接近 8-bit 動態範圍上限 255)幾乎全部是
<b>各支影片的 frame-000000</b>(19 支不同影片各自的第一幀)。實際打開圖檢查(#3
<code>4gwXPBYQZeiYezBZn</code>)——是<b>完全空白的純白畫面</b>,什麼場景內容都沒有,判斷是
dashcam 錄影開始時的相機初始化/校準空幀,不是真實駕駛場景的過曝案例。真正有實際場景內容、
median 也很高的過曝案例要看<b>#20 之後</b>(median 215–237,dark_diff 74–119,實際打開 #27
<code>6JEtSEw5FSuqcp7iS</code> 確認是大太陽天、路面反光明顯的真實過曝街景)。空白幀已經用橘色
「blank frame-0」標籤跟橘框標出來,不是混在裡面沒講。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/top30_median.html"
Path(out_path).write_text(html_out, encoding="utf-8")
print(f"[done] {out_path}, {len(html_out)} bytes")
