"""
任務:驗證 Laplacian + YOLO 信心值(含落差)能不能當「曝光是否存在」的
二元判別訊號 —— 只用線A既有的12張人工標記樣本,不擴大範圍。

資料來源(依 Jack 確認,合併兩份):
- human_severity 標籤 <- severity_line_a.csv(權威來源,SELECTED名單本身)
- laplacian_bbox / downstream_confidence_avg / downstream_confidence_drop
  數值 <- severity_scores_v2.json(severity_line_a.csv 沒有落差欄位)
  (laplacian_bbox 兩份數字互相核對過,完全一致)

輸出:
- binary_presence_scatter_v1.png(laplacian vs confidence_drop,兩組上色)
- binary_presence_strip_v1.png(三個指標各自的 strip/box plot,兩組分開)
- 文字表格印到 stdout + 存成 binary_presence_stats_v1.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent

COLOR = {"not_severe": "#1f77b4", "severe": "#d62728"}
LABEL = {"not_severe": "not_severe / mis-flagged (n=6)", "severe": "genuinely severe (n=6)"}


def load_merged():
    line_a = list(csv.DictReader(open(HERE / "severity_line_a.csv", encoding="utf-8")))
    scores_v2 = {r["file_name"]: r for r in json.load(open(HERE / "severity_scores_v2.json"))}

    rows = []
    for r in line_a:
        fn = r["file_name"]
        v2 = scores_v2.get(fn)
        assert v2 is not None, f"{fn} not found in severity_scores_v2.json"
        assert abs(float(r["laplacian_bbox"]) - v2["laplacian_bbox"]) < 1e-6, f"laplacian mismatch for {fn}"
        rows.append({
            "file_name": fn,
            "human_severity": r["human_severity"],
            "laplacian_bbox": v2["laplacian_bbox"],
            "downstream_confidence_avg": v2["downstream_confidence_avg"],
            "downstream_confidence_drop": v2["downstream_confidence_drop"],
        })
    return rows


def group_values(rows, key):
    g = {"not_severe": [], "severe": []}
    for r in rows:
        v = r[key]
        if v is not None:
            g[r["human_severity"]].append(v)
    return g


def overlap_stats(g):
    """回傳兩組數值範圍是否重疊、重疊區間內各自有幾個樣本會分類分錯。"""
    a, b = g["not_severe"], g["severe"]
    if not a or not b:
        return {"overlap": None, "note": "某一組缺值,無法比較"}
    lo = max(min(a), min(b))
    hi = min(max(a), max(b))
    overlap = lo <= hi
    if not overlap:
        return {"overlap": False, "overlap_range": None, "n_misclassified_not_severe": 0,
                "n_misclassified_severe": 0, "n_total_misclassified": 0}
    n_a_in = sum(1 for v in a if lo <= v <= hi)
    n_b_in = sum(1 for v in b if lo <= v <= hi)
    return {
        "overlap": True,
        "overlap_range": [round(lo, 2), round(hi, 2)],
        "n_misclassified_not_severe": n_a_in,
        "n_misclassified_severe": n_b_in,
        "n_total_misclassified": n_a_in + n_b_in,
        "n_total_samples_with_value": len(a) + len(b),
    }


def pctiles(vals):
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "min": round(float(np.min(arr)), 3), "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.median(arr)), 3), "p75": round(float(np.percentile(arr, 75)), 3),
        "max": round(float(np.max(arr)), 3), "n": len(vals),
    }


def main():
    rows = load_merged()
    print(f"[info] 合併後 {len(rows)} 筆(應為12筆)")

    metrics = ["laplacian_bbox", "downstream_confidence_avg", "downstream_confidence_drop"]

    # ---- scatter: laplacian vs confidence_drop ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        if r["downstream_confidence_drop"] is None:
            continue
        ax.scatter(r["laplacian_bbox"], r["downstream_confidence_drop"],
                    c=COLOR[r["human_severity"]], s=90, edgecolors="black", linewidths=0.8, zorder=3)
    # 缺 confidence_drop 的那張(太陽,no_detections)另外標在圖邊緣註記
    missing = [r for r in rows if r["downstream_confidence_drop"] is None]
    for r in missing:
        ax.axvline(r["laplacian_bbox"], color=COLOR[r["human_severity"]], linestyle=":", alpha=0.5)
        ax.annotate(f"{r['file_name'][:24]}...\n(no YOLO detections\n-> confidence_drop=N/A)",
                    xy=(r["laplacian_bbox"], 0), xytext=(r["laplacian_bbox"] + 5, -0.05),
                    fontsize=8, color=COLOR[r["human_severity"]])
    for sev in ("not_severe", "severe"):
        ax.scatter([], [], c=COLOR[sev], label=LABEL[sev], s=90, edgecolors="black")
    ax.set_xlabel("laplacian_bbox (local contrast in exposure core)")
    ax.set_ylabel("downstream_confidence_drop (clean_ref_conf - exposed_conf)")
    ax.set_title("Binary Presence Sanity Check: Laplacian vs Confidence Drop\n(12 Line-A manually labeled samples)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "binary_presence_scatter_v1.png", dpi=150)
    plt.close(fig)

    # ---- strip/box plot per metric ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = {"laplacian_bbox": "laplacian_bbox", "downstream_confidence_avg": "downstream_confidence_avg",
              "downstream_confidence_drop": "downstream_confidence_drop"}
    for ax, metric in zip(axes, metrics):
        g = group_values(rows, metric)
        positions = [1, 2]
        box_data = [g["not_severe"], g["severe"]]
        bp = ax.boxplot(box_data, positions=positions, widths=0.5, showmeans=True,
                          patch_artist=True)
        for patch, sev in zip(bp["boxes"], ("not_severe", "severe")):
            patch.set_facecolor(COLOR[sev])
            patch.set_alpha(0.25)
        for pos, sev in zip(positions, ("not_severe", "severe")):
            vals = g[sev]
            xs = np.random.normal(pos, 0.05, size=len(vals))
            ax.scatter(xs, vals, c=COLOR[sev], edgecolors="black", zorder=3, s=60)
        ax.set_xticks(positions)
        ax.set_xticklabels(["not_severe\n(n=6)", "severe\n(n=6)"])
        ax.set_title(titles[metric])
    fig.suptitle("Binary Presence Sanity Check: per-metric group distribution (12 Line-A samples)")
    fig.tight_layout()
    fig.savefig(HERE / "binary_presence_strip_v1.png", dpi=150)
    plt.close(fig)

    # ---- 統計表 + 重疊判斷 ----
    report = {}
    print("\n=== 各指標兩組 min/25%/median/75%/max ===")
    for metric in metrics:
        g = group_values(rows, metric)
        stat_not_severe = pctiles(g["not_severe"])
        stat_severe = pctiles(g["severe"])
        ov = overlap_stats(g)
        report[metric] = {"not_severe": stat_not_severe, "severe": stat_severe, "overlap": ov}
        print(f"\n-- {metric} --")
        print(f"  not_severe: {stat_not_severe}")
        print(f"  severe:     {stat_severe}")
        if ov["overlap"] is None:
            print(f"  [重疊判斷] {ov['note']}")
        elif ov["overlap"] is False:
            print("  [重疊判斷] 兩組數值範圍完全不重疊 -> 理論上存在一條門檻線可以把12張完全分對")
        else:
            print(f"  [重疊判斷] 兩組數值範圍重疊,重疊區間=[{ov['overlap_range'][0]}, {ov['overlap_range'][1]}]"
                  f",落在重疊區間內的樣本共 {ov['n_total_misclassified']}/{ov['n_total_samples_with_value']} 張"
                  f"(not_severe組{ov['n_misclassified_not_severe']}張、severe組{ov['n_misclassified_severe']}張)"
                  f" —— 意味著不管門檻線設在重疊區間哪個位置,都至少會分類分錯這些樣本裡的一部分")

    with open(HERE / "binary_presence_stats_v1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[done] binary_presence_scatter_v1.png / binary_presence_strip_v1.png / binary_presence_stats_v1.json 已寫入")


if __name__ == "__main__":
    main()
