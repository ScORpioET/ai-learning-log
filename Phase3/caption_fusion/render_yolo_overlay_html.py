import html
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/yolo_overlay_gallery.html")

data = json.load(open(HERE / "yolo_overlay_data.json"))


def esc(s):
    return html.escape(s or "")


def version_block(label_zh, v):
    return f"""
    <div class="version">
      <div class="version-head">{label_zh}<span class="n-boxes">RGB {v['n_boxes_rgb']} 框 · Thermal {v['n_boxes_thermal']} 框</span></div>
      <div class="version-imgs">
        <div class="img-block">
          <div class="img-label">RGB</div>
          <img src="data:image/jpeg;base64,{v['rgb_img']}" />
        </div>
        <div class="img-block">
          <div class="img-label">Thermal</div>
          <img src="data:image/jpeg;base64,{v['thermal_img']}" />
        </div>
      </div>
      <div class="caption">{esc(v['caption'])}</div>
      <div class="classes">classes_used: {', '.join(v['classes_used'])}</div>
    </div>
    """


cards = []
for i, s in enumerate(data, 1):
    card = f"""
    <div class="card">
      <div class="card-head">
        <span class="idx">#{i}</span>
        <span class="meta">thermal video <code>{s['thermal_video_id']}</code> · rgb video <code>{s['rgb_video_id']}</code> · frame {s['frame_index']}</span>
      </div>
      <div class="versions">
        {version_block('RGB 優先版', s['rgb_priority'])}
        {version_block('Thermal 優先版', s['thermal_priority'])}
      </div>
    </div>
    """
    cards.append(card)

html_out = f"""<title>YOLO 融合 Caption 疊圖</title>
<style>
:root {{
  --bg:#f7f6f3; --card-bg:#ffffff; --text:#2a2620; --muted:#7a7468;
  --border:#e7e2d8; --accent:#b5652f; --v-bg:#faf8f4;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c;
    --border:#3a352c; --accent:#e0955a; --v-bg:#211e1a;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c;
  --border:#3a352c; --accent:#e0955a; --v-bg:#211e1a;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans','Noto Sans TC',sans-serif; margin:0; padding:32px 20px 80px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.9rem; margin-bottom:28px; max-width:760px; line-height:1.6; }}
.grid {{ display:flex; flex-direction:column; gap:22px; max-width:1000px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:14px; padding:20px 22px; }}
.card-head {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; }}
.idx {{ font-weight:700; color:var(--accent); }}
.meta {{ color:var(--muted); font-size:.82rem; }}
.meta code {{ font-family:'IBM Plex Mono',monospace; }}
.versions {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.version {{ background:var(--v-bg); border:1px solid var(--border); border-radius:10px; padding:12px; }}
.version-head {{ font-weight:700; font-size:.9rem; margin-bottom:8px; display:flex; justify-content:space-between; align-items:baseline; }}
.n-boxes {{ font-weight:400; font-size:.72rem; color:var(--muted); }}
.version-imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px; }}
.img-block img {{ width:100%; border-radius:6px; display:block; }}
.img-label {{ font-size:.68rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; }}
.caption {{ font-size:.86rem; line-height:1.5; color:var(--accent); font-weight:600; margin-bottom:6px; }}
.classes {{ font-size:.7rem; color:var(--muted); font-family:'IBM Plex Mono',monospace; }}
</style>
<h1>YOLO 融合 Caption 疊圖</h1>
<div class="subtitle">
對步驟 4 的最終融合 caption(RGB 優先版 / thermal 優先版)套 YOLO(yolov8m,COCO
pretrained,沿用 Day35 KEEP_CLASSES/conf=0.25)偵測結果:只畫出「YOLO 有偵測到、
且 caption 文字有提到」的類別框,caption 提到但這次 YOLO 沒偵測到的物件就靜靜地
少一個框。同一版 caption 的框同時畫在 RGB 圖跟 thermal 圖上,因為融合 caption
本身就是兩個 domain 資訊合併出來的。橘色框旁標的是 caption 用的類別名稱
(en_name,例如 pedestrian/car/truck),不是 COCO 原始類別名。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

OUT.write_text(html_out, encoding="utf-8")
print(f"[done] {OUT} written, {len(html_out)} bytes")
