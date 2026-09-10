import json
from pathlib import Path

DATA_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/boundary_data.json")
OUT_PATH = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/boundary_inspector.html")

LO, HI = 0.1881, 0.2826
KNOWN_LINE_A = "video-GiDQGbWeWwtNTQEnG-frame-000618-f2ESyX6KKhcvwJkJa.jpg"

rows = json.load(open(DATA_PATH))
above = sorted([r for r in rows if r["side"] == "above_HI"], key=lambda r: r["saturation_ratio"])
below = sorted([r for r in rows if r["side"] == "below_LO"], key=lambda r: -r["saturation_ratio"])

def card_html(i, r, side):
    dist = abs(r["saturation_ratio"] - (HI if side == "above_HI" else LO))
    note = ""
    if r["file_name"] == KNOWN_LINE_A:
        note = '<span class="badge badge-known">同一張是12張線A樣本裡的「確實嚴重」案例,剛好卡在LO邊界上</span>'
    return f"""
    <article class="card" data-side="{side}">
      <div class="card-head">
        <span class="rank">#{i:02d}</span>
        <span class="pill pill-{side}">{'高於門檻 (severe側)' if side=='above_HI' else '低於門檻 (not_severe側)'}</span>
        <span class="dist">距門檻 {dist:.4f}</span>
      </div>
      <div class="card-body">
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['full_thumb_b64']}" alt="full frame with core bbox" loading="lazy">
          <figcaption>full frame &middot; green = core bbox &middot; amber = measured region</figcaption>
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
        <div><dt>saturation_ratio</dt><dd>{r['saturation_ratio']*100:.2f}%</dd></div>
      </dl>
      {note}
      <p class="fname">{r['split']} / {r['file_name']}</p>
    </article>
    """

# ---- number line ----
W, H = 860, 110
margin_l, margin_r = 40, 40
plot_w = W - margin_l - margin_r
xmin, xmax = 0.16, 0.31

def xpos(v):
    return margin_l + ((v - xmin) / (xmax - xmin)) * plot_w

lo_x, hi_x = xpos(LO), xpos(HI)
dots = []
for i, r in enumerate(below):
    dots.append(f'<a href="#below-{i}"><circle class="dot-below" cx="{xpos(r["saturation_ratio"]):.1f}" cy="70" r="7"/></a>')
for i, r in enumerate(above):
    dots.append(f'<a href="#above-{i}"><circle class="dot-above" cx="{xpos(r["saturation_ratio"]):.1f}" cy="70" r="7"/></a>')

ticks = []
for t in [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    tx = xpos(t)
    ticks.append(f'<line x1="{tx:.1f}" y1="88" x2="{tx:.1f}" y2="94" class="tick-line"/>')
    ticks.append(f'<text x="{tx:.1f}" y="106" class="tick-label" text-anchor="middle">{t:.2f}</text>')

svg = f"""
<svg viewBox="0 0 {W} {H}" class="strip-svg" role="img" aria-label="boundary number line">
  <rect x="{lo_x:.1f}" y="20" width="{hi_x-lo_x:.1f}" height="68" class="zone-band"/>
  <line x1="{lo_x:.1f}" y1="20" x2="{lo_x:.1f}" y2="88" class="thresh-line"/>
  <line x1="{hi_x:.1f}" y1="20" x2="{hi_x:.1f}" y2="88" class="thresh-line"/>
  <text x="{lo_x:.1f}" y="14" text-anchor="middle" class="thresh-label">LO {LO}</text>
  <text x="{hi_x:.1f}" y="14" text-anchor="middle" class="thresh-label">HI {HI}</text>
  <line x1="{margin_l}" y1="88" x2="{W-margin_r}" y2="88" class="axis-line"/>
  {''.join(ticks)}
  {''.join(dots)}
</svg>
"""

above_cards = "\n".join(card_html(i, r, "above_HI") for i, r in enumerate(above))
below_cards = "\n".join(card_html(i, r, "below_LO") for i, r in enumerate(below))
# fix ids to match anchors used in svg (below-i / above-i)
above_cards_ids = above_cards
below_cards_ids = below_cards

html = f"""<title>Saturation Threshold Boundary Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg: #f5f3ee;
  --surface: #ffffff;
  --ink: #201d19;
  --ink-dim: #6f6a62;
  --line: #e3ded3;
  --above: #b23a2e;
  --above-soft: #f7e9e6;
  --below: #2f5fa8;
  --below-soft: #e8eef7;
  --zone: #b8791a;
  --zone-soft: #f6ecd8;
  --mono-num: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c;
    --above: #e08579; --above-soft: #3a2420; --below: #7fa6de; --below-soft: #22283a;
    --zone: #dba64b; --zone-soft: #3a2f18;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c;
  --above: #e08579; --above-soft: #3a2420; --below: #7fa6de; --below-soft: #22283a;
  --zone: #dba64b; --zone-soft: #3a2f18;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; padding: 28px 20px 60px; }}
.wrap {{ max-width: 980px; margin: 0 auto; }}
header h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0 0 6px; text-wrap: balance; }}
header p.lede {{ color: var(--ink-dim); max-width: 70ch; line-height: 1.55; margin: 0 0 20px; }}
header p.lede code {{ font-family: var(--mono-num); background: var(--surface); border: 1px solid var(--line); padding: 0 4px; border-radius: 4px; font-size: 0.9em; }}
.strip-panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px 4px; margin-bottom: 24px; }}
.strip-svg {{ width: 100%; height: auto; display: block; }}
.zone-band {{ fill: var(--zone-soft); }}
.thresh-line {{ stroke: var(--zone); stroke-width: 1.5; stroke-dasharray: 4 3; }}
.thresh-label {{ fill: var(--zone); font-size: 11px; font-family: var(--mono-num); }}
.axis-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-label {{ fill: var(--ink-dim); font-size: 11px; font-family: var(--mono-num); }}
.dot-above {{ fill: var(--above); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}
.dot-below {{ fill: var(--below); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.85rem; color: var(--ink-dim); margin-bottom: 24px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.swatch.above {{ background: var(--above); }}
.swatch.below {{ background: var(--below); }}
.swatch.zone {{ background: var(--zone-soft); border: 1px solid var(--zone); }}
section.group {{ margin-bottom: 32px; }}
section.group h2 {{ font-size: 1.05rem; margin: 0 0 12px; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 16px; }}
.card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }}
.rank {{ font-family: var(--mono-num); color: var(--ink-dim); font-size: 0.85rem; min-width: 2.4em; }}
.dist {{ font-family: var(--mono-num); color: var(--ink-dim); font-size: 0.8rem; margin-left: auto; }}
.pill {{ font-size: 0.78rem; padding: 3px 10px; border-radius: 999px; font-weight: 600; }}
.pill-above_HI {{ background: var(--above-soft); color: var(--above); }}
.pill-below_LO {{ background: var(--below-soft); color: var(--below); }}
.badge-known {{ display: block; font-size: 0.78rem; padding: 6px 10px; border-radius: 8px; background: var(--zone-soft); color: var(--zone); margin-bottom: 8px; font-weight: 600; }}
.card-body {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }}
.shot {{ margin: 0; }}
.shot img {{ width: 100%; display: block; border-radius: 8px; border: 1px solid var(--line); aspect-ratio: 4/3; object-fit: cover; background: var(--bg); }}
.shot figcaption {{ font-size: 0.72rem; color: var(--ink-dim); margin-top: 5px; line-height: 1.3; }}
.metrics {{ display: flex; gap: 20px; margin: 0 0 8px; padding-top: 10px; border-top: 1px dashed var(--line); }}
.metrics div {{ display: flex; flex-direction: column; gap: 2px; }}
.metrics dt {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); margin: 0; }}
.metrics dd {{ font-family: var(--mono-num); font-variant-numeric: tabular-nums; font-size: 1.05rem; margin: 0; font-weight: 600; }}
.fname {{ font-family: var(--mono-num); font-size: 0.72rem; color: var(--ink-dim); word-break: break-all; margin: 0; }}
@media (max-width: 640px) {{ .card-body {{ grid-template-columns: 1fr 1fr; }} .card-body .shot:nth-child(3) {{ grid-column: span 2; }} }}
</style>

<div class="wrap">
  <header>
    <h1>Saturation Threshold Boundary Check</h1>
    <p class="lede">
      從934張候選的分類結果裡,挑出最貼近門檻的10張:<b>高於 HI 門檻(<code>{HI}</code>)最接近的5張</b>(剛好被判成severe)、
      <b>低於 LO 門檻(<code>{LO}</code>)最接近的5張</b>(剛好被判成not_severe)。這批是分類結果裡最「一線之隔」的邊界案例,
      拿來肉眼確認門檻切在這個位置合不合理。
    </p>
  </header>

  <div class="strip-panel">{svg}</div>

  <div class="legend">
    <span><i class="swatch below"></i> 低於LO,最接近門檻的5張</span>
    <span><i class="swatch above"></i> 高於HI,最接近門檻的5張</span>
    <span><i class="swatch zone"></i> 模糊帶區間 [{LO}, {HI}]</span>
  </div>

  <section class="group">
    <h2>低於 LO 門檻,最接近的5張(判定 not_severe)</h2>
    <div class="grid" id="below-group">{below_cards_ids}</div>
  </section>

  <section class="group">
    <h2>高於 HI 門檻,最接近的5張(判定 severe)</h2>
    <div class="grid" id="above-group">{above_cards_ids}</div>
  </section>
</div>
"""

# patch card ids so svg anchors resolve (#below-i / #above-i)
html = html.replace('<article class="card" data-side="below_LO">', '')
parts = html.split('<article class="card" data-side="below_LO">')
# simpler: re-render with explicit ids
def card_html2(i, r, side, anchor):
    dist = abs(r["saturation_ratio"] - (HI if side == "above_HI" else LO))
    note = ""
    if r["file_name"] == KNOWN_LINE_A:
        note = '<span class="badge badge-known">同一張是12張線A樣本裡的「確實嚴重」案例,剛好卡在LO邊界上</span>'
    return f"""
    <article class="card" id="{anchor}">
      <div class="card-head">
        <span class="rank">#{i:02d}</span>
        <span class="pill pill-{side}">{'高於門檻 (severe側)' if side=='above_HI' else '低於門檻 (not_severe側)'}</span>
        <span class="dist">距門檻 {dist:.4f}</span>
      </div>
      <div class="card-body">
        <figure class="shot">
          <img src="data:image/jpeg;base64,{r['full_thumb_b64']}" alt="full frame with core bbox" loading="lazy">
          <figcaption>full frame &middot; green = core bbox &middot; amber = measured region</figcaption>
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
        <div><dt>saturation_ratio</dt><dd>{r['saturation_ratio']*100:.2f}%</dd></div>
      </dl>
      {note}
      <p class="fname">{r['split']} / {r['file_name']}</p>
    </article>
    """

below_cards_final = "\n".join(card_html2(i, r, "below_LO", f"below-{i}") for i, r in enumerate(below))
above_cards_final = "\n".join(card_html2(i, r, "above_HI", f"above-{i}") for i, r in enumerate(above))

html = f"""<title>Saturation Threshold Boundary Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg: #f5f3ee;
  --surface: #ffffff;
  --ink: #201d19;
  --ink-dim: #6f6a62;
  --line: #e3ded3;
  --above: #b23a2e;
  --above-soft: #f7e9e6;
  --below: #2f5fa8;
  --below-soft: #e8eef7;
  --zone: #b8791a;
  --zone-soft: #f6ecd8;
  --mono-num: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c;
    --above: #e08579; --above-soft: #3a2420; --below: #7fa6de; --below-soft: #22283a;
    --zone: #dba64b; --zone-soft: #3a2f18;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17140f; --surface: #221e18; --ink: #efe9df; --ink-dim: #a89e8f; --line: #3a352c;
  --above: #e08579; --above-soft: #3a2420; --below: #7fa6de; --below-soft: #22283a;
  --zone: #dba64b; --zone-soft: #3a2f18;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; padding: 28px 20px 60px; }}
.wrap {{ max-width: 980px; margin: 0 auto; }}
header h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0 0 6px; text-wrap: balance; }}
header p.lede {{ color: var(--ink-dim); max-width: 70ch; line-height: 1.55; margin: 0 0 20px; }}
header p.lede code {{ font-family: var(--mono-num); background: var(--surface); border: 1px solid var(--line); padding: 0 4px; border-radius: 4px; font-size: 0.9em; }}
.strip-panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px 4px; margin-bottom: 24px; }}
.strip-svg {{ width: 100%; height: auto; display: block; }}
.zone-band {{ fill: var(--zone-soft); }}
.thresh-line {{ stroke: var(--zone); stroke-width: 1.5; stroke-dasharray: 4 3; }}
.thresh-label {{ fill: var(--zone); font-size: 11px; font-family: var(--mono-num); }}
.axis-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-line {{ stroke: var(--ink-dim); stroke-width: 1; }}
.tick-label {{ fill: var(--ink-dim); font-size: 11px; font-family: var(--mono-num); }}
.dot-above {{ fill: var(--above); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}
.dot-below {{ fill: var(--below); stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.85rem; color: var(--ink-dim); margin-bottom: 24px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.swatch.above {{ background: var(--above); }}
.swatch.below {{ background: var(--below); }}
.swatch.zone {{ background: var(--zone-soft); border: 1px solid var(--zone); }}
section.group {{ margin-bottom: 32px; }}
section.group h2 {{ font-size: 1.05rem; margin: 0 0 12px; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 16px; }}
.card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }}
.rank {{ font-family: var(--mono-num); color: var(--ink-dim); font-size: 0.85rem; min-width: 2.4em; }}
.dist {{ font-family: var(--mono-num); color: var(--ink-dim); font-size: 0.8rem; margin-left: auto; }}
.pill {{ font-size: 0.78rem; padding: 3px 10px; border-radius: 999px; font-weight: 600; }}
.pill-above_HI {{ background: var(--above-soft); color: var(--above); }}
.pill-below_LO {{ background: var(--below-soft); color: var(--below); }}
.badge-known {{ display: block; font-size: 0.78rem; padding: 6px 10px; border-radius: 8px; background: var(--zone-soft); color: var(--zone); margin-bottom: 8px; font-weight: 600; }}
.card-body {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }}
.shot {{ margin: 0; }}
.shot img {{ width: 100%; display: block; border-radius: 8px; border: 1px solid var(--line); aspect-ratio: 4/3; object-fit: cover; background: var(--bg); }}
.shot figcaption {{ font-size: 0.72rem; color: var(--ink-dim); margin-top: 5px; line-height: 1.3; }}
.metrics {{ display: flex; gap: 20px; margin: 0 0 8px; padding-top: 10px; border-top: 1px dashed var(--line); }}
.metrics div {{ display: flex; flex-direction: column; gap: 2px; }}
.metrics dt {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); margin: 0; }}
.metrics dd {{ font-family: var(--mono-num); font-variant-numeric: tabular-nums; font-size: 1.05rem; margin: 0; font-weight: 600; }}
.fname {{ font-family: var(--mono-num); font-size: 0.72rem; color: var(--ink-dim); word-break: break-all; margin: 0; }}
@media (max-width: 640px) {{ .card-body {{ grid-template-columns: 1fr 1fr; }} .card-body .shot:nth-child(3) {{ grid-column: span 2; }} }}
</style>

<div class="wrap">
  <header>
    <h1>Saturation Threshold Boundary Check</h1>
    <p class="lede">
      從934張候選的分類結果裡,挑出最貼近門檻的10張:<b>高於 HI 門檻(<code>{HI}</code>)最接近的5張</b>(剛好被判成severe)、
      <b>低於 LO 門檻(<code>{LO}</code>)最接近的5張</b>(剛好被判成not_severe)。這批是分類結果裡最「一線之隔」的邊界案例,
      拿來肉眼確認門檻切在這個位置合不合理。
    </p>
  </header>

  <div class="strip-panel">{svg}</div>

  <div class="legend">
    <span><i class="swatch below"></i> 低於LO,最接近門檻的5張</span>
    <span><i class="swatch above"></i> 高於HI,最接近門檻的5張</span>
    <span><i class="swatch zone"></i> 模糊帶區間 [{LO}, {HI}]</span>
  </div>

  <section class="group">
    <h2>低於 LO 門檻,最接近的5張(判定 not_severe)</h2>
    <div class="grid" id="below-group">{below_cards_final}</div>
  </section>

  <section class="group">
    <h2>高於 HI 門檻,最接近的5張(判定 severe)</h2>
    <div class="grid" id="above-group">{above_cards_final}</div>
  </section>
</div>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"[done] {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.2f} MB)")
