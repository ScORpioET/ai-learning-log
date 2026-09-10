import json
from pathlib import Path

DATA_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/gallery_data.json")
OUT_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/saturation_inspector.html")

OVERLAP_LO, OVERLAP_HI = 0.1881, 0.2826

rows = json.load(open(DATA_PATH))
rows.sort(key=lambda r: r["saturation_ratio"])

GROUP_LABEL = {"not_severe": "not_severe / mis-flagged", "severe": "genuinely severe"}

def fmt_pct(v):
    return f"{v*100:.2f}%"

def card_html(i, r):
    in_zone = OVERLAP_LO <= r["saturation_ratio"] <= OVERLAP_HI
    group = r["human_severity"]
    zone_badge = '<span class="badge badge-zone">in overlap zone</span>' if in_zone else ""
    return f"""
    <article class="card" id="card-{i}" data-group="{group}">
      <div class="card-head">
        <span class="rank">#{i:02d}</span>
        <span class="pill pill-{group}">{GROUP_LABEL[group]}</span>
        {zone_badge}
      </div>
      <div class="card-body">
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['full_thumb_b64']}" alt="full frame with core bbox" loading="lazy">
          <figcaption>full frame &middot; green = core bbox &middot; amber = measured region (+0.5&times; pad)</figcaption>
        </figure>
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['crop_zoom_b64']}" alt="measured region crop" loading="lazy">
          <figcaption>measured region (actual pixels)</figcaption>
        </figure>
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['blended_zoom_b64']}" alt="saturation mask overlay" loading="lazy">
          <figcaption>red = pixels &ge; 250 (saturated)</figcaption>
        </figure>
      </div>
      <dl class="metrics">
        <div><dt>saturation_ratio</dt><dd>{fmt_pct(r['saturation_ratio'])}</dd></div>
        <div><dt>core_mean_brightness</dt><dd>{r['core_mean_brightness']:.1f}</dd></div>
        <div><dt>laplacian_bbox</dt><dd>{r['laplacian_bbox']:.1f}</dd></div>
      </dl>
      <p class="fname">{r['split']} / {r['file_name']}</p>
    </article>
    """

# ---- number-line strip (SVG) ----
W, H = 860, 120
margin_l, margin_r = 40, 40
plot_w = W - margin_l - margin_r
xmax = 0.45

def xpos(v):
    return margin_l + (v / xmax) * plot_w

zone_x0, zone_x1 = xpos(OVERLAP_LO), xpos(OVERLAP_HI)

dots = []
for i, r in enumerate(rows):
    cx = xpos(r["saturation_ratio"])
    cy = 55 if r["human_severity"] == "severe" else 80
    cls = "dot-severe" if r["human_severity"] == "severe" else "dot-not-severe"
    dots.append(f'<a href="#card-{i}"><circle class="{cls}" cx="{cx:.1f}" cy="{cy}" r="7"/></a>')

ticks = []
for t in [0, 0.1, 0.2, 0.3, 0.4]:
    tx = xpos(t)
    ticks.append(f'<line x1="{tx:.1f}" y1="98" x2="{tx:.1f}" y2="104" class="tick-line"/>')
    ticks.append(f'<text x="{tx:.1f}" y="116" class="tick-label" text-anchor="middle">{t:.1f}</text>')

svg = f"""
<svg viewBox="0 0 {W} {H}" class="strip-svg" role="img" aria-label="saturation ratio number line">
  <rect x="{zone_x0:.1f}" y="20" width="{zone_x1-zone_x0:.1f}" height="86" class="zone-band"/>
  <line x1="{margin_l}" y1="98" x2="{W-margin_r}" y2="98" class="axis-line"/>
  {''.join(ticks)}
  <text x="{(zone_x0+zone_x1)/2:.1f}" y="14" text-anchor="middle" class="zone-label">overlap zone</text>
  {''.join(dots)}
  <text x="{W-margin_r}" y="55" text-anchor="end" class="row-label" dy="4">severe</text>
  <text x="{W-margin_r}" y="80" text-anchor="end" class="row-label" dy="4">not_severe</text>
</svg>
"""

cards = "\n".join(card_html(i, r) for i, r in enumerate(rows))

html = f"""<title>Saturation Ratio Inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg: #f5f3ee;
  --surface: #ffffff;
  --ink: #201d19;
  --ink-dim: #6f6a62;
  --line: #e3ded3;
  --not-severe: #2f5fa8;
  --not-severe-soft: #e8eef7;
  --severe: #b23a2e;
  --severe-soft: #f7e9e6;
  --overlap: #b8791a;
  --overlap-soft: #f6ecd8;
  --mono-num: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17140f;
    --surface: #221e18;
    --ink: #efe9df;
    --ink-dim: #a89e8f;
    --line: #3a352c;
    --not-severe: #7fa6de;
    --not-severe-soft: #22283a;
    --severe: #e08579;
    --severe-soft: #3a2420;
    --overlap: #dba64b;
    --overlap-soft: #3a2f18;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17140f;
  --surface: #221e18;
  --ink: #efe9df;
  --ink-dim: #a89e8f;
  --line: #3a352c;
  --not-severe: #7fa6de;
  --not-severe-soft: #22283a;
  --severe: #e08579;
  --severe-soft: #3a2420;
  --overlap: #dba64b;
  --overlap-soft: #3a2f18;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  padding: 28px 20px 60px;
}}
.wrap {{ max-width: 980px; margin: 0 auto; }}

header h1 {{
  font-size: 1.65rem;
  font-weight: 600;
  margin: 0 0 6px;
  text-wrap: balance;
}}
header p.lede {{
  color: var(--ink-dim);
  max-width: 68ch;
  line-height: 1.55;
  margin: 0 0 20px;
}}
header p.lede code {{
  font-family: var(--mono-num);
  background: var(--surface);
  border: 1px solid var(--line);
  padding: 0 4px;
  border-radius: 4px;
  font-size: 0.9em;
}}

.strip-panel {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 14px 4px;
  margin-bottom: 28px;
}}
.strip-svg {{ width: 100%; height: auto; display: block; }}
.zone-band {{ fill: var(--overlap-soft); }}
.zone-label {{ fill: var(--overlap); font-size: 11px; font-family: var(--mono-num); letter-spacing: 0.02em; }}
.axis-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-label {{ fill: var(--ink-dim); font-size: 11px; font-family: var(--mono-num); }}
.row-label {{ fill: var(--ink-dim); font-size: 11px; font-family: var(--mono-num); }}
.dot-severe {{ fill: var(--severe); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}
.dot-not-severe {{ fill: var(--not-severe); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}

.legend {{
  display: flex; gap: 18px; flex-wrap: wrap;
  font-size: 0.85rem; color: var(--ink-dim);
  margin-bottom: 24px;
}}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.swatch.not-severe {{ background: var(--not-severe); }}
.swatch.severe {{ background: var(--severe); }}
.swatch.zone {{ background: var(--overlap-soft); border: 1px solid var(--overlap); }}

.grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}}

.card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px 16px;
}}
.card-head {{
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 10px;
}}
.rank {{
  font-family: var(--mono-num);
  color: var(--ink-dim);
  font-size: 0.85rem;
  min-width: 2.4em;
}}
.pill {{
  font-size: 0.78rem;
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 600;
}}
.pill-not_severe {{ background: var(--not-severe-soft); color: var(--not-severe); }}
.pill-severe {{ background: var(--severe-soft); color: var(--severe); }}
.badge-zone {{
  font-size: 0.75rem;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--overlap-soft);
  color: var(--overlap);
  font-weight: 600;
}}

.card-body {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 12px;
}}
.shot {{ margin: 0; }}
.shot img {{
  width: 100%;
  display: block;
  border-radius: 8px;
  border: 1px solid var(--line);
  aspect-ratio: 4 / 3;
  object-fit: cover;
  background: var(--bg);
}}
.shot figcaption {{
  font-size: 0.72rem;
  color: var(--ink-dim);
  margin-top: 5px;
  line-height: 1.3;
}}

.metrics {{
  display: flex; flex-wrap: wrap; gap: 20px;
  margin: 0 0 8px;
  padding-top: 10px;
  border-top: 1px dashed var(--line);
}}
.metrics div {{ display: flex; flex-direction: column; gap: 2px; }}
.metrics dt {{
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ink-dim); margin: 0;
}}
.metrics dd {{
  font-family: var(--mono-num); font-variant-numeric: tabular-nums;
  font-size: 1.05rem; margin: 0; font-weight: 600;
}}
.fname {{
  font-family: var(--mono-num);
  font-size: 0.72rem;
  color: var(--ink-dim);
  word-break: break-all;
  margin: 0;
}}

@media (max-width: 640px) {{
  .card-body {{ grid-template-columns: 1fr 1fr; }}
  .card-body .shot:nth-child(3) {{ grid-column: span 2; }}
}}
</style>

<div class="wrap">
  <header>
    <h1>Saturation Ratio Inspector</h1>
    <p class="lede">
      12 張線A人工標記樣本,依 <code>saturation_ratio</code>(核心量測區域內灰階值 &ge;250 的像素比例)由低到高排序。
      每張卡片顯示:全圖(綠框=偵測到的核心黑核bbox,琥珀框=實際量測區域,即bbox外擴0.5倍)、量測區域裁切放大圖、
      以及紅色標出飽和像素的疊圖,方便肉眼核對數字跟畫面是否吻合。琥珀色底的數線區間 <code>[{OVERLAP_LO}, {OVERLAP_HI}]</code>
      是兩組數值重疊的範圍——落在裡面的樣本用這個指標分不出「不嚴重」還是「嚴重」。
    </p>
  </header>

  <div class="strip-panel">
    {svg}
  </div>

  <div class="legend">
    <span><i class="swatch not-severe"></i> not_severe / mis-flagged (n=6)</span>
    <span><i class="swatch severe"></i> genuinely severe (n=6)</span>
    <span><i class="swatch zone"></i> overlap zone (5/12 張落在裡面)</span>
  </div>

  <div class="grid">
    {cards}
  </div>
</div>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"[done] {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.2f} MB)")
