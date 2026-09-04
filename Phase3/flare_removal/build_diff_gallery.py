"""
對 Flare7K++ 跑完的 test_output/blend/ 逐張圖,跟原圖算 diff(逐 pixel 絕對差值
加總、轉灰階熱圖),產生「原圖 / 模型輸出 / diff 熱圖」三欄並排的 gallery。
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path.home() / "ai-transition-2026" / "Phase3" / "flare_removal" / "Flare7K"
ORIG_DIR = Path.home() / "ai-transition-2026" / "thermal_dataset" / "video_rgb_test" / "data"
BLEND_DIR = REPO / "test_output" / "blend"


def to_b64(im, w=380):
    im = im.convert("RGB")
    ratio = w / im.width
    im = im.resize((w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def diff_heatmap(orig, blend):
    """逐 pixel |orig - blend| 絕對差值,三通道加總後正規化成灰階熱圖(用簡單的
    jet-like 上色:差值大 -> 紅/黃,差值小 -> 深藍/黑)。"""
    o = np.asarray(orig.convert("RGB"), dtype=np.float32)
    b = np.asarray(blend.resize(orig.size).convert("RGB"), dtype=np.float32)
    diff = np.abs(o - b).sum(axis=2)  # HxW,範圍 0~765
    diff_norm = np.clip(diff / diff.max() if diff.max() > 0 else diff, 0, 1)

    # 簡單熱圖上色:0->黑, 0.5->紅, 1.0->黃白
    heat = np.zeros((*diff_norm.shape, 3), dtype=np.uint8)
    r = np.clip(diff_norm * 3, 0, 1)
    g = np.clip(diff_norm * 3 - 1, 0, 1)
    bch = np.clip(diff_norm * 3 - 2, 0, 1)
    heat[..., 0] = (r * 255).astype(np.uint8)
    heat[..., 1] = (g * 255).astype(np.uint8)
    heat[..., 2] = (bch * 255).astype(np.uint8)
    return Image.fromarray(heat), float(diff.mean()), float(diff.max())


def main():
    blend_files = sorted(BLEND_DIR.glob("*.jpg"))
    print(f"[info] {len(blend_files)} 張模型輸出")

    cards = []
    stats = []
    for i, bpath in enumerate(blend_files, 1):
        orig_path = ORIG_DIR / bpath.name
        orig = Image.open(orig_path)
        blend = Image.open(bpath)
        heat, diff_mean, diff_max = diff_heatmap(orig, blend)

        import re
        m = re.search(r"-frame-(\d+)-", bpath.name)
        frame_num = int(m.group(1)) if m else i

        stats.append({"file_name": bpath.name, "frame": frame_num, "diff_mean": round(diff_mean, 2), "diff_max": round(diff_max, 2)})

        card = f"""
        <div class="card">
          <div class="head"><span class="idx">#{i}</span><span class="fn">frame {frame_num}</span>
            <span class="stat">diff_mean={diff_mean:.1f} · diff_max={diff_max:.0f}</span></div>
          <div class="triple">
            <div><div class="lbl">原圖</div><img src="data:image/jpeg;base64,{to_b64(orig)}"/></div>
            <div><div class="lbl">模型輸出(deflare)</div><img src="data:image/jpeg;base64,{to_b64(blend)}"/></div>
            <div><div class="lbl">Diff 熱圖</div><img src="data:image/jpeg;base64,{to_b64(heat)}"/></div>
          </div>
        </div>"""
        cards.append(card)
        print(f"  {i}/{len(blend_files)} frame {frame_num}  diff_mean={diff_mean:.1f}", end="\r")

    print()
    with open(Path(__file__).parent / "diff_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    html_out = f"""<title>Flare7K++ Diff Gallery</title>
<style>
:root {{ --bg:#f7f6f3; --card-bg:#fff; --text:#2a2620; --muted:#7a7468; --border:#e7e2d8; --accent:#b5652f; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
}}
:root[data-theme="dark"] {{ --bg:#1c1a17; --card-bg:#262320; --text:#ece7dd; --muted:#a39c8c; --border:#3a352c; --accent:#e0955a; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; margin:0; padding:28px 20px 60px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:.86rem; margin-bottom:24px; max-width:800px; line-height:1.6; }}
.grid {{ display:flex; flex-direction:column; gap:16px; max-width:1180px; margin:0 auto; }}
.card {{ background:var(--card-bg); border:1px solid var(--border); border-radius:12px; padding:14px; }}
.head {{ display:flex; align-items:baseline; gap:12px; margin-bottom:8px; font-size:.82rem; }}
.idx {{ font-weight:700; color:var(--accent); }}
.fn {{ color:var(--muted); font-family:monospace; }}
.stat {{ margin-left:auto; font-family:monospace; color:var(--muted); font-size:.75rem; }}
.triple {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
.lbl {{ font-size:.66rem; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:3px; }}
img {{ width:100%; border-radius:6px; display:block; }}
</style>
<h1>Flare7K++ 去光斑驗證 —— dvZBYnphN2BwdMKBc</h1>
<div class="subtitle">
test split 影片 dvZBYnphN2BwdMKBc(RGB,共 565 幀)等距抽樣 31 張(含 frame 80,太陽正中央的
sanity-check 幀),用 Flare7K++ 預訓練模型(flare7kpp checkpoint,Uformer,512px tiling)推論,
拿模型輸出跟原圖算 diff 熱圖(逐 pixel 絕對差值加總,越紅/黃代表模型改動越大)。
</div>
<div class="grid">
{''.join(cards)}
</div>
"""
    out_path = "/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/flare_diff_gallery.html"
    Path(out_path).write_text(html_out, encoding="utf-8")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
