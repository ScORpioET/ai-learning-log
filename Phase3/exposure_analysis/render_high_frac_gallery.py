import html
import json

data = json.load(open("/tmp/high_frac_gallery_data.json"))
metrics = {m["file_name"]: m for m in json.load(open("high_frac_samples_full_metrics.json"))}

cards = []
for i, e in enumerate(data, 1):
    m = metrics[e["file_name"]]
    cards.append(f"""
    <div class="card">
      <div class="head"><span class="idx">#{i}</span><span class="hf">high_frac={e['high_frac']:.4f}</span></div>
      <img src="data:image/jpeg;base64,{e['img']}" />
      <div class="metrics">
        <span>median <b>{m['median']}</b></span>
        <span>dark_diff <b>{m['dark_diff']}</b></span>
        <span>high_frac <b>{m['high_frac']}</b></span>
        <span>low_frac <b>{m['low_frac']}</b></span>
      </div>
      <div class="fname">{html.escape(e['file_name'])}</div>
    </div>""")

html_out = f"""<title>High Frac Train Samples</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#b5652f; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.88rem; margin-bottom:24px; max-width:700px; line-height:1.6; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; max-width:1200px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:12px; }}
.head {{ display:flex; justify-content:space-between; margin-bottom:8px; font-size:.85rem; }}
.idx {{ font-weight:700; color:var(--accent); }}
.hf {{ font-family:monospace; color:var(--muted); }}
img {{ width:100%; border-radius:8px; display:block; }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; font-size:.78rem; color:var(--muted); margin-top:8px; }}
.metrics b {{ color:var(--accent); font-family:monospace; }}
.fname {{ font-family:monospace; font-size:.65rem; color:var(--muted); margin-top:6px; word-break:break-all; }}
</style>
<h1>RGB Train — high_frac &gt; 0.08 樣本</h1>
<div class="subtitle">全量 10,319 張 RGB train 圖片(整張圖,非 bbox)算 high_frac(px&gt;=240 佔比),
共 3643 張(35.30%)超過 0.08。這是 seed=42 從候選裡抽出的 10 張,依 high_frac 由高到低排列,
供門檻校準用。</div>
<div class="grid">
{''.join(cards)}
</div>
"""

out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/high_frac_gallery.html"
open(out_path, "w").write(html_out)
print("written", len(html_out))
