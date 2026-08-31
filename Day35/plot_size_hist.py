"""Day35 加碼 Task 4: bbox area 直方圖(log-scale x 軸,GT baseline / matched / missed 三條疊)。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path.home() / "ai-transition-2026" / "Day35" / "outputs"
data = json.load(open(OUT_DIR / "size_hist_data.json"))

all_gt = np.array(data["all_gt"]) * 100  # 換成百分比
matched_gt = np.array(data["matched_gt"]) * 100
missed_gt = np.array(data["missed_gt"]) * 100

# log-scale bins,從最小值到最大值切 40 個 bin
lo = max(min(all_gt.min(), 1e-4), 1e-4)
hi = all_gt.max()
bins = np.logspace(np.log10(lo), np.log10(hi), 40)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(all_gt, bins=bins, color="lightgray", alpha=0.7, label=f"All GT boxes (n={len(all_gt)})")
ax.hist(matched_gt, bins=bins, color="green", alpha=0.55, label=f"Matched by YOLO (n={len(matched_gt)})")
ax.hist(missed_gt, bins=bins, color="red", alpha=0.55, label=f"Missed by YOLO (n={len(missed_gt)})")

ax.set_xscale("log")
ax.set_xlabel("bbox area ratio (% of image area, log scale)")
ax.set_ylabel("count")
ax.set_title("GT bbox size distribution: matched vs missed by YOLO (val split, IoU>0.5)")
ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="tiny/small boundary (0.5%)")
ax.legend()
fig.tight_layout()

out_path = OUT_DIR / "gt_vs_yolo_size_hist.png"
fig.savefig(out_path, dpi=150)
print(f"[done] {out_path}")
