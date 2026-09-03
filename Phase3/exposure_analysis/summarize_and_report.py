"""
Day40:彙整 compute_rgb_exposure.py / compute_thermal_background.py 的原始
數字,畫直方圖(標 10/25/50/75/90 百分位數),並且用「觀察到的百分位數
本身」當門檻決定「有問題」的比例——不是憑感覺挑一個絕對數字。

門檻定義(方法論選擇,不是查出來的事實,理由寫在這裡):
- RGB 過曝:bright_diff 是「這個 bbox 外擴區域裡,最亮的 1% 像素比
  中位數基準偏離多少」。偏離越大代表局部強光訊號越強。用這批 bbox
  自己的 p90 當門檻——超過 p90(全體最極端的前 10%)才算「有強光問題」。
- RGB 欠曝(2026-09-03 換掉 dark_diff,改用 median_luminance):原本
  用 dark_diff(中位數 - 最暗 1% 像素)判斷欠曝,實測發現排出來的
  top 案例反而大多是「場景本身已經過曝(中位亮度 185-250),裡面剛好
  有一小塊暗色元素」——dark_diff 抓到的其實還是過曝訊號,不是欠曝
  (詳見 bright20/extreme_exposure 兩份 artifact 的人工核對)。改成
  直接看這個 bbox 外擴區域「自己的中位亮度」夠不夠暗:用這批 bbox 自己
  的 p10 當門檻,median_luminance 低於 p10(全體最暗的後 10%)才算
  「欠曝」。這個指標量的是「這個區域整體有多暗」,不是「跟自己比有沒有
  局部反差」,兩者不是同一件事,md 亮度低才是真的欠曝。
  過曝(bright_diff)跟欠曝(median_luminance)分開判斷,任一個超標就
  算這個 bbox 有問題(因為題目說兩者可能同時發生,不互斥)。
- thermal:diff 是「物體中位數 vs 外擴環中位數」的差,diff 越小代表物體
  跟背景越像、越難分辨。用這批 bbox 自己的 p10 當門檻——低於 p10(全體
  最不明顯的後 10%)算「跟背景太像」。
用同一批資料自己的百分位數當門檻,好處是不管兩個 domain 的絕對亮度/
溫度尺度差多少,「前 10% 最極端」這個定義都可以套用,不用另外猜一個
跨 domain 都適用的絕對數字。
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
    summary = {}
    for pct in PCTS:
        key = f"pct_{pct}"
        bright = [r[key]["bright_diff"] for r in records if r.get(key)]
        median_lum = [r[key]["median_luminance"] for r in records if r.get(key)]

        bright_pctile = percentile_table(bright)
        median_pctile = percentile_table(median_lum)
        bright_thresh = bright_pctile[90]
        median_thresh = median_pctile[10]

        flagged = 0
        for r in records:
            rec = r.get(key)
            if not rec:
                continue
            if rec["bright_diff"] >= bright_thresh or rec["median_luminance"] <= median_thresh:
                flagged += 1
        n = sum(1 for r in records if r.get(key))
        pct_flagged = flagged / n * 100

        print(f"[外擴 {pct}%] n={n}  bright_diff p10/25/50/75/90={bright_pctile}  "
              f"median_luminance p10/25/50/75/90={median_pctile}")
        print(f"           門檻: bright_diff>={bright_thresh:.1f}(p90) or median_luminance<={median_thresh:.1f}(p10)"
              f"  -> 有問題比例 = {pct_flagged:.2f}% ({flagged}/{n})")

        plot_hist(bright, bright_pctile, f"RGB bright_diff (expand {pct}%)",
                  "p99 - median luminance", HERE / f"hist_rgb_bright_diff_{pct}.png", "#c2703a")
        plot_hist(median_lum, median_pctile, f"RGB median_luminance (expand {pct}%)",
                  "median luminance of region", HERE / f"hist_rgb_median_lum_{pct}.png", "#0d7d8c")

        summary[pct] = {
            "n": n, "bright_diff_percentiles": bright_pctile, "median_luminance_percentiles": median_pctile,
            "bright_thresh": bright_thresh, "median_thresh": median_thresh,
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
