import json
import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
OUT = Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation"
OUT.mkdir(parents=True, exist_ok=True)

TH_TRAIN = ROOT / "images_thermal_train"
RGB_TRAIN = ROOT / "images_rgb_train"


def load_coco(d):
    return json.load(open(d / "coco.json"))


th_coco = load_coco(TH_TRAIN)
rgb_coco = load_coco(RGB_TRAIN)

# ---------------------------------------------------------------------------
# A1: 類別定義 + 標註數量比對
# ---------------------------------------------------------------------------
th_cats = {c["id"]: c for c in th_coco["categories"]}
rgb_cats = {c["id"]: c for c in rgb_coco["categories"]}

same_schema = th_coco["categories"] == rgb_coco["categories"]

th_counts = defaultdict(int)
for a in th_coco["annotations"]:
    th_counts[a["category_id"]] += 1
rgb_counts = defaultdict(int)
for a in rgb_coco["annotations"]:
    rgb_counts[a["category_id"]] += 1

all_cat_ids = sorted(set(th_cats) | set(rgb_cats))
a1_rows = []
for cid in all_cat_ids:
    name = th_cats.get(cid, rgb_cats.get(cid))["name"]
    tc = th_counts.get(cid, 0)
    rc = rgb_counts.get(cid, 0)
    ratio = (rc / tc) if tc > 0 else (float("inf") if rc > 0 else None)
    a1_rows.append({"id": cid, "name": name, "thermal": tc, "rgb": rc, "ratio_rgb_over_th": ratio})

json.dump(
    {"same_category_schema": same_schema, "rows": a1_rows},
    open(OUT / "a1_category_counts.json", "w"),
    indent=2,
)
print("[A1] done, same_schema =", same_schema, " total categories:", len(all_cat_ids))

# ---------------------------------------------------------------------------
# A2: bbox 面積佔畫面比例分布(RGB train + thermal train,兩邊都算,方便對照)
# ---------------------------------------------------------------------------
def bbox_area_ratios(coco):
    img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    ratios = []
    for a in coco["annotations"]:
        img_id = a["image_id"]
        if img_id not in img_wh:
            continue
        w_img, h_img = img_wh[img_id]
        x, y, w, h = a["bbox"]
        ratios.append(100.0 * (w * h) / (w_img * h_img))
    return np.array(ratios)


rgb_ratios = bbox_area_ratios(rgb_coco)
th_ratios = bbox_area_ratios(th_coco)

def summarize(ratios, name):
    percentiles = {p: float(np.percentile(ratios, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
    tiny_0025 = float((ratios < 0.025).mean() * 100)
    tiny_005 = float((ratios < 0.05).mean() * 100)
    tiny_05 = float((ratios < 0.5).mean() * 100)
    return {
        "name": name,
        "n": int(len(ratios)),
        "mean_pct": float(ratios.mean()),
        "median_pct": float(np.median(ratios)),
        "percentiles": percentiles,
        "tiny_below_0.025pct_of_all_anns": tiny_0025,
        "tiny_below_0.05pct_of_all_anns": tiny_005,
        "tiny_below_0.5pct_of_all_anns": tiny_05,
    }


a2_summary = {
    "rgb_train": summarize(rgb_ratios, "rgb_train"),
    "thermal_train": summarize(th_ratios, "thermal_train"),
}
json.dump(a2_summary, open(OUT / "a2_bbox_area_summary.json", "w"), indent=2)

# 直方圖(log-scale x 軸,比照 Day35 plot_size_hist.py 的做法)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.5))
lo = max(min(rgb_ratios.min(), th_ratios.min(), 1e-4), 1e-4)
hi = max(rgb_ratios.max(), th_ratios.max())
bins = np.logspace(np.log10(lo), np.log10(hi), 50)
ax.hist(th_ratios, bins=bins, color="steelblue", alpha=0.55, label=f"thermal train (n={len(th_ratios)})")
ax.hist(rgb_ratios, bins=bins, color="orange", alpha=0.55, label=f"RGB train (n={len(rgb_ratios)})")
ax.set_xscale("log")
ax.set_xlabel("bbox area ratio (% of image area, log scale)")
ax.set_ylabel("count")
ax.set_title("bbox area ratio distribution: RGB train vs thermal train")
ax.axvline(0.025, color="black", linestyle="--", linewidth=0.8, label="current filter threshold (0.025%)")
ax.axvline(0.05, color="gray", linestyle=":", linewidth=0.8, label="0.05% (reference line asked for)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "a2_bbox_area_hist.png", dpi=150)
print("[A2] done, hist saved")

# 長寬比分布(RGB vs thermal 畫面本身的長寬比不同,順便列出 bbox 長寬比分布給參考)
def bbox_aspect_ratios(coco):
    ratios = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if h > 0:
            ratios.append(w / h)
    return np.array(ratios)

rgb_ar = bbox_aspect_ratios(rgb_coco)
th_ar = bbox_aspect_ratios(th_coco)
a2_summary["bbox_aspect_ratio"] = {
    "rgb_train": {"median": float(np.median(rgb_ar)), "p10": float(np.percentile(rgb_ar, 10)), "p90": float(np.percentile(rgb_ar, 90))},
    "thermal_train": {"median": float(np.median(th_ar)), "p10": float(np.percentile(th_ar, 10)), "p90": float(np.percentile(th_ar, 90))},
    "image_aspect_ratio": {
        "rgb": rgb_coco["images"][0]["width"] / rgb_coco["images"][0]["height"],
        "thermal": th_coco["images"][0]["width"] / th_coco["images"][0]["height"],
    },
}
json.dump(a2_summary, open(OUT / "a2_bbox_area_summary.json", "w"), indent=2)
print("[A2] aspect ratio summary added")

print("A1+A2 script done")
