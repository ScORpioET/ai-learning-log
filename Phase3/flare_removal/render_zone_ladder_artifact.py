import json
from pathlib import Path

DATA_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/zone_ladder_data.json")
OUT_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/zone_ladder_inspector.html")

LO, HI = 0.1881, 0.2826
rows = json.load(open(DATA_PATH))
rows.sort(key=lambda r: r["actual_pct"])


def color_for_pct(pct):
    # LO(0%) 藍 -> 中點 灰 -> HI(100%) 紅,線性內插,呼應前兩輪 not_severe/severe 配色
    t = pct / 100
    lo_rgb = (0x2f, 0x5f, 0xa8)
    hi_rgb = (0xb2, 0x3a, 0x2e)
    r = round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * t)
    g = round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * t)
    b = round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * t)
    return f"rgb({r},{g},{b})"


def card_html(i, r):
    c = color_for_pct(r["actual_pct"])
    return f"""
    <article class="card" id="rung-{i}" style="--accent: {c};">
      <div class="card-head">
        <span class="rung-num">{r['target_pct']:.0f}%</span>
        <span class="actual">實際位置 {r['actual_pct']:.1f}%</span>
      </div>
      <div class="card-body">
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['full_thumb_b64']}" alt="full frame with core bbox" loading="lazy">
          <figcaption>full frame</figcaption>
        </figure>
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['crop_zoom_b64']}" alt="measured region crop" loading="lazy">
          <figcaption>measured region</figcaption>
        </figure>
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['blended_zoom_b64']}" alt="saturation mask overlay" loading="lazy">
          <figcaption>saturated pixels (red)</figcaption>
        </figure>
      </div>
      <p class="sat-val">saturation_ratio = {r['saturation_ratio']*100:.2f}%</p>
      <p class="fname">{r['split']} / {r['file_name']}</p>
    </article>
    """


# ---- vertical ladder axis (SVG), LO at top(0%) to HI at bottom(100%) ----
W, H = 120, 44 * 20 + 40
top_y, bottom_y = 20, H - 20
track_h = bottom_y - top_y

ticks = []
for i, r in enumerate(rows):
    y = top_y + (r["actual_pct"] / 100) * track_h
    c = color_for_pct(r["actual_pct"])
    ticks.append(f'<a href="#rung-{i}"><circle cx="60" cy="{y:.1f}" r="6" fill="{c}" stroke="var(--surface)" stroke-width="1.5"/></a>')

svg = f"""
<svg viewBox="0 0 {W} {H}" class="ladder-svg" role="img" aria-label="0 to 100 percent ladder">
  <line x1="60" y1="{top_y}" x2="60" y2="{bottom_y}" class="ladder-line"/>
  <text x="60" y="12" text-anchor="middle" class="ladder-label">LO 0%</text>
  <text x="60" y="{H-6}" text-anchor="middle" class="ladder-label">HI 100%</text>
  {''.join(ticks)}
</svg>
"""

cards = "\n".join(card_html(i, r) for i, r in enumerate(rows))

html = f"""<title>Ambiguous Zone Ladder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg: #f5f3ee; --surface: #ffffff; --ink: #201d19; --ink-dim: #6f6a62; --line: #e3ded3;
  --mono-num: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c; }}
}}
:root[data-theme="dark"] {{ --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; padding: 28px 20px 60px; }}
.wrap {{ max-width: 1040px; margin: 0 auto; }}
header h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0 0 6px; text-wrap: balance; }}
header p.lede {{ color: var(--ink-dim); max-width: 72ch; line-height: 1.55; margin: 0 0 24px; }}
header p.lede code {{ font-family: var(--mono-num); background: var(--surface); border: 1px solid var(--line); padding: 0 4px; border-radius: 4px; font-size: 0.9em; }}

.layout {{ display: grid; grid-template-columns: 70px 1fr; gap: 8px; align-items: start; }}
.ladder-col {{ position: sticky; top: 20px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 6px 0; }}
.ladder-svg {{ width: 100%; height: auto; display: block; }}
.ladder-line {{ stroke: var(--line); stroke-width: 3; }}
.ladder-label {{ fill: var(--ink-dim); font-size: 8px; font-family: var(--mono-num); }}

.grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.card {{
  background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--accent);
  border-radius: 10px; padding: 12px 16px 14px;
}}
.card-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }}
.rung-num {{
  font-family: var(--mono-num); font-weight: 700; font-size: 1.1rem; color: var(--accent);
  min-width: 3.2em;
}}
.actual {{ font-family: var(--mono-num); font-size: 0.78rem; color: var(--ink-dim); }}
.card-body {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }}
.shot {{ margin: 0; }}
.shot img {{ width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); aspect-ratio: 4/3; object-fit: cover; background: var(--bg); }}
.shot figcaption {{ font-size: 0.68rem; color: var(--ink-dim); margin-top: 3px; }}
.sat-val {{ font-family: var(--mono-num); font-variant-numeric: tabular-nums; font-weight: 600; margin: 0 0 2px; }}
.fname {{ font-family: var(--mono-num); font-size: 0.7rem; color: var(--ink-dim); word-break: break-all; margin: 0; }}

@media (max-width: 680px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .ladder-col {{ display: none; }}
  .card-body {{ grid-template-columns: 1fr 1fr; }}
  .card-body .shot:nth-child(3) {{ grid-column: span 2; }}
}}
</style>

<div class="wrap">
  <header>
    <h1>Ambiguous Zone Ladder</h1>
    <p class="lede">
      把模糊帶 <code>[{LO}, {HI}]</code> 用 <b>LO=0% &rarr; HI=100%</b> 的相對量尺切成20個5%寬的區間,
      每個區間中點在141張模糊帶樣本裡找最接近的一張(20張皆不重複)。由上到下就是從「貼近not_severe側」
      漸變到「貼近severe側」的連續畫面,拿來肉眼感受這條邊界帶上的漸變過程是否合理、有沒有明顯斷點。
    </p>
  </header>

  <div class="layout">
    <div class="ladder-col">{svg}</div>
    <div class="grid">{cards}</div>
  </div>
</div>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"[done] {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.2f} MB)")
