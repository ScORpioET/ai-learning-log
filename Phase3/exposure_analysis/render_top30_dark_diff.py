import base64
import io
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
SPLIT_DIR = {"train": "images_rgb_train", "val": "images_rgb_val", "test": "video_rgb_test"}

top30 = json.load(open(HERE / "top30_dark_diff_all_splits.json"))


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
    cards.append(f"""
    <div class="card">
      <div class="head"><span class="idx">#{i}</span><span class="dd">dark_diff={r['dark_diff']}</span></div>
      <img src="data:image/jpeg;base64,{b64}"/>
      <div class="meta"><span class="tag">{r['split']}</span> {r['file_name']}</div>
    </div>""")

html_out = f"""<title>Top 30 Dark Diff(混合三 Split)</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#a23b2e; --warn-bg:#f8e7e4; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0776a; --warn-bg:#3a2320; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0776a; --warn-bg:#3a2320; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.86rem; margin-bottom:16px; max-width:760px; line-height:1.6; }}
.caveat {{ background:var(--warn-bg); border:1px solid var(--accent); border-radius:10px; padding:14px 18px; margin-bottom:24px; max-width:760px; font-size:.85rem; line-height:1.7; }}
.caveat b {{ color:var(--accent); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; max-width:1200px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:10px; padding:10px; }}
.head {{ display:flex; justify-content:space-between; margin-bottom:6px; font-size:.8rem; }}
.idx {{ font-weight:700; color:var(--accent); }}
.dd {{ font-family:monospace; color:var(--muted); }}
img {{ width:100%; border-radius:6px; display:block; }}
.meta {{ font-size:.65rem; color:var(--muted); margin-top:6px; word-break:break-all; }}
.tag {{ font-family:monospace; font-weight:700; color:var(--accent); }}
</style>
<h1>Dark Diff 最大值 Top 30 —— Train + Val + Test 混合</h1>
<div class="subtitle">
三個 split 的 RGB 圖片全部混在一起(train 10,319 + val 1,085 + test 3,749 = 15,153 張),對整張圖算
dark_diff(= median − p1),取混合後最大的前 30 名。
</div>
<div class="caveat">
⚠️ <b>誠實揭露,不是隨便帶過</b>:這 30 張<b>全部來自 train</b>(val/test 一張都沒有),而且集中在
<b>只有 4 支影片</b>(fK9MqP9T52ArLJAGk 佔 20 張、H4LTfF4eGmFxZL6ZS 5 張、k5wBRi7N5x8NsYrjy 4 張、
5zpwfwcv9hXTFxw8m 1 張)。實際打開圖片檢查發現:這 30 張的 dark_diff 數值<b>剛好都等於 median</b>
(代表 p1=0,也就是至少 1% 的像素是純黑 0),肉眼一看是這幾支影片<b>畫面左右兩側有黑色letterbox
色條</b>(影片原始長寬比跟輸出畫布不同,補了黑邊),不是「場景裡真的有一小塊很暗的東西」——
這個指標在整張圖(不分 bbox)算的時候,會被 letterbox 黑邊主導,不是真正反映曝光問題。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/top30_dark_diff.html"
Path(out_path).write_text(html_out, encoding="utf-8")
print(f"[done] {out_path}, {len(html_out)} bytes")
