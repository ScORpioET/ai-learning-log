"""
任務:計算「飽和像素比例」與「核心平均亮度」兩個新指標,驗證能否作為
「曝光是否存在」的二元判別訊號 —— 延續上一輪 binary_presence sanity check,
只用同一批12張線A人工標記樣本,核心區域定義完全沿用 laplacian_bbox 那段
(severity_line_a_manual_compare.py / severity_line_b_full_scores.py 共用邏輯):
    blob = max(row["blobs"], key=area) 的 bbox,換算回原圖解析度,
    再往外擴張 0.5*max(ow,oh) 當作核心裁切區域。

兩個新指標,都在這個核心裁切區域的灰階圖上算:
- saturation_ratio: 灰階值 >= 250 的像素數 / 區域總像素數
- core_mean_brightness: 區域內灰階像素值的平均數(單純 mean)

輸出:
- saturation_brightness_scatter_v1.png
- saturation_brightness_strip_v1.png
- saturation_brightness_stats_v1.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}
FLARE_DIRS = {s: HERE / s / "flare" for s in ["out_train", "out_val", "out_test"]}
SAT_THRESH = 250

COLOR = {"not_severe": "#1f77b4", "severe": "#d62728"}
LABEL = {"not_severe": "not_severe / mis-flagged (n=6)", "severe": "genuinely severe (n=6)"}
SUN_FRAME = "video-dvZBYnphN2BwdMKBc-frame-000021-Fte9QvqiE6kenxHWP.jpg"


def core_crop_gray(split, fn):
    """跟 laplacian_bbox 完全相同的核心區域定義(bbox換算回原圖解析度 + 0.5倍pad)。"""
    blob_data = {(r["split"], r["file_name"]): r for r in
                 json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))}
    row = blob_data[(split, fn)]
    blob = max(row["blobs"], key=lambda b: b["area"])
    bx, by, bw, bh = blob["bbox"]

    orig = cv2.imread(str(ORIG_DIRS[split] / fn))
    flare = cv2.imread(str(FLARE_DIRS[split] / fn))
    scale = orig.shape[0] / flare.shape[0]
    ox, oy, ow, oh = [round(v * scale) for v in (bx, by, bw, bh)]
    pad = round(0.5 * max(ow, oh))
    x0, y0 = max(ox - pad, 0), max(oy - pad, 0)
    x1, y1 = min(ox + ow + pad, orig.shape[1]), min(oy + oh + pad, orig.shape[0])
    crop = cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return crop


def compute_metrics(gray):
    saturation_ratio = float(np.mean(gray >= SAT_THRESH))
    core_mean_brightness = float(np.mean(gray))
    return saturation_ratio, core_mean_brightness


def group_values(rows, key):
    g = {"not_severe": [], "severe": []}
    for r in rows:
        g[r["human_severity"]].append(r[key])
    return g


def overlap_stats(g):
    a, b = g["not_severe"], g["severe"]
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
        "overlap_range": [round(lo, 4), round(hi, 4)],
        "n_misclassified_not_severe": n_a_in,
        "n_misclassified_severe": n_b_in,
        "n_total_misclassified": n_a_in + n_b_in,
        "n_total_samples": len(a) + len(b),
    }


def pctiles(vals):
    arr = np.array(vals)
    return {
        "min": round(float(np.min(arr)), 4), "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4), "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4), "n": len(vals),
    }


def main():
    line_a = list(csv.DictReader(open(HERE / "severity_line_a.csv", encoding="utf-8")))

    rows = []
    for r in line_a:
        split, fn = r["split"], r["file_name"]
        gray = core_crop_gray(split, fn)
        sat_ratio, mean_bright = compute_metrics(gray)
        rows.append({
            "split": split, "file_name": fn, "human_severity": r["human_severity"],
            "saturation_ratio": round(sat_ratio, 4), "core_mean_brightness": round(mean_bright, 2),
        })
        print(f"{r['human_severity']:12s} {fn}  saturation_ratio={sat_ratio:.4f}  core_mean_brightness={mean_bright:.2f}")

    metrics = ["saturation_ratio", "core_mean_brightness"]

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        ax.scatter(r["saturation_ratio"], r["core_mean_brightness"],
                    c=COLOR[r["human_severity"]], s=90, edgecolors="black", linewidths=0.8, zorder=3)
    for sev in ("not_severe", "severe"):
        ax.scatter([], [], c=COLOR[sev], label=LABEL[sev], s=90, edgecolors="black")
    sun_row = next(r for r in rows if r["file_name"] == SUN_FRAME)
    ax.annotate("sun frame\n(dvZBYnphN2BwdMKBc)", xy=(sun_row["saturation_ratio"], sun_row["core_mean_brightness"]),
                xytext=(sun_row["saturation_ratio"] - 0.15, sun_row["core_mean_brightness"] - 25),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("saturation_ratio (fraction of core pixels with gray >= 250)")
    ax.set_ylabel("core_mean_brightness (mean gray value in core region)")
    ax.set_title("Saturation/Brightness Sanity Check\n(12 Line-A manually labeled samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "saturation_brightness_scatter_v1.png", dpi=150)
    plt.close(fig)

    # ---- strip/box ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, metrics):
        g = group_values(rows, metric)
        positions = [1, 2]
        bp = ax.boxplot([g["not_severe"], g["severe"]], positions=positions, widths=0.5,
                          showmeans=True, patch_artist=True)
        for patch, sev in zip(bp["boxes"], ("not_severe", "severe")):
            patch.set_facecolor(COLOR[sev])
            patch.set_alpha(0.25)
        for pos, sev in zip(positions, ("not_severe", "severe")):
            vals = g[sev]
            xs = np.random.normal(pos, 0.05, size=len(vals))
            ax.scatter(xs, vals, c=COLOR[sev], edgecolors="black", zorder=3, s=60)
        ax.set_xticks(positions)
        ax.set_xticklabels(["not_severe\n(n=6)", "severe\n(n=6)"])
        ax.set_title(metric)
    fig.suptitle("Saturation/Brightness Sanity Check: per-metric group distribution (12 Line-A samples)")
    fig.tight_layout()
    fig.savefig(HERE / "saturation_brightness_strip_v1.png", dpi=150)
    plt.close(fig)

    # ---- stats + overlap ----
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
        if ov["overlap"] is False:
            print("  [重疊判斷] 兩組數值範圍完全不重疊 -> 理論上存在一條門檻線可以把12張完全分對")
        else:
            print(f"  [重疊判斷] 重疊區間=[{ov['overlap_range'][0]}, {ov['overlap_range'][1]}]"
                  f",落在重疊區間內共 {ov['n_total_misclassified']}/{ov['n_total_samples']} 張"
                  f"(not_severe組{ov['n_misclassified_not_severe']}張、severe組{ov['n_misclassified_severe']}張)")

    sun_rank_note = {
        "saturation_ratio_percentile_in_severe": None,
        "core_mean_brightness_percentile_in_severe": None,
    }
    severe_vals_sat = sorted(group_values(rows, "saturation_ratio")["severe"])
    severe_vals_bri = sorted(group_values(rows, "core_mean_brightness")["severe"])
    sun_rank_note["saturation_ratio_percentile_in_severe"] = round(
        sum(1 for v in severe_vals_sat if v <= sun_row["saturation_ratio"]) / len(severe_vals_sat), 3)
    sun_rank_note["core_mean_brightness_percentile_in_severe"] = round(
        sum(1 for v in severe_vals_bri if v <= sun_row["core_mean_brightness"]) / len(severe_vals_bri), 3)
    report["sun_frame_check"] = {
        "file_name": SUN_FRAME,
        "saturation_ratio": sun_row["saturation_ratio"],
        "core_mean_brightness": sun_row["core_mean_brightness"],
        **sun_rank_note,
    }
    print(f"\n=== 太陽正中央那張在 severe 組內的位置 ===\n{report['sun_frame_check']}")

    with open(HERE / "saturation_brightness_stats_v1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[done] saturation_brightness_scatter_v1.png / _strip_v1.png / _stats_v1.json 已寫入")


if __name__ == "__main__":
    main()
