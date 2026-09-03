import html
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/final_fusion_gallery.html")

data = json.load(open(HERE / "final_gallery_data.json"))

SUG_CLASS = {
    "建議 thermal 優先": "sug-thermal",
    "建議 RGB 優先": "sug-rgb",
}


def sug_class(label):
    return SUG_CLASS.get(label, "sug-neutral")


def esc(s):
    return html.escape(s or "(空——這張圖沒有標註 dynamic object)")


cards = []
for i, s in enumerate(data, 1):
    fused = s["fused"]
    sug = s["suggestion"]
    conflict_note = ""
    if s["has_conflict"]:
        conflict_note = f'<div class="conflict-note">⚠ 這組樣本存在 position 衝突的 class,兩版本內容不同</div>'
    else:
        conflict_note = '<div class="conflict-note ok">兩版本字面相同(這組樣本沒有 position 衝突,系統建議只是附加標籤,兩版仍然都列出)</div>'

    card = f"""
    <div class="card">
      <div class="card-head">
        <span class="idx">#{i}</span>
        <span class="meta">thermal video <code>{s['thermal_video_id']}</code> · rgb video <code>{s['rgb_video_id']}</code> · frame {s['frame_index']}</span>
        <span class="badge {sug_class(sug['label'])}">{esc(sug['label'])}</span>
      </div>
      <div class="reason">{esc(sug['reason'])}</div>

      <div class="imgs">
        <div class="img-block">
          <div class="img-label">Thermal</div>
          <img src="data:image/jpeg;base64,{s['thermal_img']}" />
          <div class="fname">{esc(s['thermal_file'])}</div>
        </div>
        <div class="img-block">
          <div class="img-label">RGB</div>
          <img src="data:image/jpeg;base64,{s['rgb_img']}" />
          <div class="fname">{esc(s['rgb_file'])}</div>
        </div>
      </div>

      <table class="cap-table">
        <tr><th>Thermal GT</th><td>{esc(s['thermal_gt'])}</td></tr>
        <tr><th>RGB GT</th><td>{esc(s['rgb_gt'])}</td></tr>
        <tr><th>Thermal 模型生成</th><td>{esc(s['thermal_gen'])}</td></tr>
        <tr><th>RGB 模型生成</th><td>{esc(s['rgb_gen'])}</td></tr>
        <tr class="fused-row"><th>融合 · RGB 優先版</th><td>{esc(fused['rgb_priority']['caption'])}</td></tr>
        <tr class="fused-row"><th>融合 · Thermal 優先版</th><td>{esc(fused['thermal_priority']['caption'])}</td></tr>
      </table>
      {conflict_note}
    </div>
    """
    cards.append(card)

html_out = f"""<title>RGB+Thermal 融合 Caption Gallery</title>
<style>
:root {{
  --bg:#f7f6f3; --card-bg:#ffffff; --text:#2a2620; --muted:#7a7468;
  --border:#e7e2d8; --accent:#b5652f; --mono:#3a362e;
  --sug-rgb-bg:#fdeee0; --sug-rgb-fg:#a6480f;
  --sug-thermal-bg:#e6eef5; --sug-thermal-fg:#2a5d8a;
  --sug-neutral-bg:#eeece6; --sug-neutral-fg:#6b6558;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c;
    --border:#3a352c; --accent:#e0955a; --mono:#d8d2c4;
    --sug-rgb-bg:#3a2414; --sug-rgb-fg:#e8a06b;
    --sug-thermal-bg:#182631; --sug-thermal-fg:#8fc0e8;
    --sug-neutral-bg:#302c25; --sug-neutral-fg:#b5ae9e;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c;
  --border:#3a352c; --accent:#e0955a; --mono:#d8d2c4;
  --sug-rgb-bg:#3a2414; --sug-rgb-fg:#e8a06b;
  --sug-thermal-bg:#182631; --sug-thermal-fg:#8fc0e8;
  --sug-neutral-bg:#302c25; --sug-neutral-fg:#b5ae9e;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans','Noto Sans TC',sans-serif; margin:0; padding:32px 20px 80px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.9rem; margin-bottom:28px; max-width:760px; line-height:1.6; }}
.grid {{ display:flex; flex-direction:column; gap:22px; max-width:960px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:14px; padding:20px 22px; }}
.card-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px; }}
.idx {{ font-weight:700; color:var(--accent); }}
.meta {{ color:var(--muted); font-size:.82rem; }}
.meta code {{ font-family:'IBM Plex Mono',monospace; }}
.badge {{ margin-left:auto; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }}
.sug-rgb {{ background:var(--sug-rgb-bg); color:var(--sug-rgb-fg); }}
.sug-thermal {{ background:var(--sug-thermal-bg); color:var(--sug-thermal-fg); }}
.sug-neutral {{ background:var(--sug-neutral-bg); color:var(--sug-neutral-fg); }}
.reason {{ font-size:.83rem; color:var(--muted); margin-bottom:14px; line-height:1.6; }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }}
.img-block img {{ width:100%; border-radius:8px; border:1px solid var(--border); display:block; }}
.img-label {{ font-size:.75rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
.fname {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; color:var(--muted); margin-top:4px; word-break:break-all; }}
.cap-table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
.cap-table th {{ text-align:left; color:var(--muted); font-weight:600; padding:6px 10px 6px 0; vertical-align:top; width:150px; white-space:nowrap; }}
.cap-table td {{ padding:6px 0; line-height:1.5; }}
.cap-table tr {{ border-top:1px solid var(--border); }}
.cap-table tr:first-child {{ border-top:none; }}
.fused-row th, .fused-row td {{ color:var(--accent); font-weight:600; }}
.conflict-note {{ margin-top:12px; font-size:.78rem; color:var(--sug-rgb-fg); }}
.conflict-note.ok {{ color:var(--muted); }}
</style>
<h1>RGB + Thermal 融合 Caption Gallery</h1>
<div class="subtitle">
Test split,seed=42 抽樣 10 組 RGB/thermal frame_id 對應樣本。每組都列出兩邊 GT(對 GT 標註直接跑
generate_captions.py,不是模型生成)、兩邊模型生成 caption、融合後的 RGB 優先版與 thermal 優先版
(兩版永遠都輸出,不由系統自動二選一)。徽章是根據 Day40 在 train set 上訂出的曝光/背景品質門檻
(RGB bright_diff / median_luminance,thermal object-surround diff)算出的「系統建議」,只是附加提示,
不代表唯一正確答案。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""

OUT.write_text(html_out, encoding="utf-8")
print(f"[done] {OUT} written, {len(html_out)} bytes")
