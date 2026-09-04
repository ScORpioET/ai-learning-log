"""
Day40:彙整 compute_rgb_exposure.py / compute_thermal_background.py 的原始
數字,畫直方圖(標 10/25/50/75/90 百分位數)。

RGB 優先權判斷門檻(2026-09-04 Jack 明確指定,寫死的固定數字,不是從這批
資料的百分位數算出來的——這裡只是照給定規則套用,不重新推導理由):
    dark_diff > 200 OR median_luminance < 50  ->  判定 RGB 被燈光影響,
    這個 bbox 算「有問題」,否則預設沒問題。
不使用 bright_diff 當判斷依據(bright_diff 直方圖/百分位數還是照算照畫,
留著當診斷資訊,但不進入 flagged 的判斷式)。thermal 側不另外設門檻做
獨立的優先權判斷(見 compute_sample_quality.py),這裡的 thermal diff
分析純粹是背景相近程度的描述統計,不是優先權規則的一部分。
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
PCTS = [25, 50, 75]
PERCENTILES = [10, 25, 50, 75, 90]


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def percentile_table(values, percentiles=PERCENTILES):
    return {p: round(float(np.percentile(values, p)), 3) for p in percentiles}


def plot_hist(values, percentiles, title, xlabel, out_path, color):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=80, color=color, alpha=0.75, edgecolor="none")
    ymax = ax.get_ylim()[1]
    for p, v in percentiles.items():
        ax.axvline(v, color="#333", linestyle="--", linewidth=0.8)
        ax.text(v, ymax * 0.96, f"p{p}={v:.1f}", rotation=90, fontsize=8,
                 va="top", ha="right", color="#333")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("bbox count", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def analyze_rgb(records):
    print("\n" + "=" * 70)
    print("RGB 曝光分析")
    print("=" * 70)
    DARK_DIFF_THRESH = 200
    MEDIAN_THRESH = 50

    summary = {}
    for pct in PCTS:
        key = f"pct_{pct}"
        bright = [r[key]["bright_diff"] for r in records if r.get(key)]
        dark = [r[key]["dark_diff"] for r in records if r.get(key)]
        median_lum = [r[key]["median_luminance"] for r in records if r.get(key)]

        bright_pctile = percentile_table(bright)
        dark_pctile = percentile_table(dark)
        median_pctile = percentile_table(median_lum)

        flagged = 0
        for r in records:
            rec = r.get(key)
            if not rec:
                continue
            if rec["dark_diff"] > DARK_DIFF_THRESH or rec["median_luminance"] < MEDIAN_THRESH:
                flagged += 1
        n = sum(1 for r in records if r.get(key))
        pct_flagged = flagged / n * 100

        print(f"[外擴 {pct}%] n={n}  bright_diff p10/25/50/75/90={bright_pctile}  "
              f"dark_diff p10/25/50/75/90={dark_pctile}  median_luminance p10/25/50/75/90={median_pctile}")
        print(f"           門檻(固定值): dark_diff>{DARK_DIFF_THRESH} or median_luminance<{MEDIAN_THRESH}"
              f"  -> 有問題比例 = {pct_flagged:.2f}% ({flagged}/{n})")

        plot_hist(bright, bright_pctile, f"RGB bright_diff (expand {pct}%) - diagnostic only, not used for flagging",
                  "p99 - median luminance", HERE / f"hist_rgb_bright_diff_{pct}.png", "#c2703a")
        plot_hist(dark, dark_pctile, f"RGB dark_diff (expand {pct}%)",
                  "median - p1 luminance", HERE / f"hist_rgb_dark_diff_{pct}.png", "#a13d3d")
        plot_hist(median_lum, median_pctile, f"RGB median_luminance (expand {pct}%)",
                  "median luminance of region", HERE / f"hist_rgb_median_lum_{pct}.png", "#0d7d8c")

        summary[pct] = {
            "n": n, "bright_diff_percentiles": bright_pctile, "dark_diff_percentiles": dark_pctile,
            "median_luminance_percentiles": median_pctile,
            "dark_diff_thresh": DARK_DIFF_THRESH, "median_thresh": MEDIAN_THRESH,
            "flagged": flagged, "pct_flagged": round(pct_flagged, 3),
        }
    return summary


def analyze_thermal(records):
    print("\n" + "=" * 70)
    print("Thermal 背景相近分析")
    print("=" * 70)
    summary = {}
    for pct in PCTS:
        key = f"pct_{pct}"
        diffs = [r[key]["diff"] for r in records if r.get(key)]
        pctile = percentile_table(diffs)
        thresh = pctile[10]

        n = len(diffs)
        flagged = sum(1 for d in diffs if d <= thresh)
        pct_flagged = flagged / n * 100

        print(f"[外擴 {pct}%] n={n}  diff p10/25/50/75/90={pctile}")
        print(f"           門檻(p10): diff<={thresh:.1f}  -> 融入背景比例 = {pct_flagged:.2f}% ({flagged}/{n})")

        plot_hist(diffs, pctile, f"Thermal object-surround diff (expand {pct}%)",
                  "|object median - surround median|", HERE / f"hist_thermal_diff_{pct}.png", "#5b4b9c")

        summary[pct] = {
            "n": n, "diff_percentiles": pctile, "thresh": thresh,
            "flagged": flagged, "pct_flagged": round(pct_flagged, 3),
        }
    return summary


def main():
    rgb_records = load_jsonl(HERE / "rgb_exposure_results.jsonl")
    thermal_records = load_jsonl(HERE / "thermal_background_results.jsonl")
    print(f"[info] RGB bboxes: {len(rgb_records)}  thermal bboxes: {len(thermal_records)}")

    rgb_summary = analyze_rgb(rgb_records)
    thermal_summary = analyze_thermal(thermal_records)

    with open(HERE / "summary_results.json", "w", encoding="utf-8") as f:
        json.dump({"rgb": rgb_summary, "thermal": thermal_summary}, f, ensure_ascii=False, indent=2)
    print("\n[done] summary_results.json + hist_*.png written")


if __name__ == "__main__":
    main()
