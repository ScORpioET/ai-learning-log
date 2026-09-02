import json
from pathlib import Path
import numpy as np

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
OUT = Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation"

TH_W, TH_H = 640, 512
RGB_W, RGB_H = 1800, 1600
TH_AREA = TH_W * TH_H
RGB_AREA = RGB_W * RGB_H

TH_PCT = 0.025  # thermal 目前的設計基準門檻(%)

th_coco = json.load(open(ROOT / "images_thermal_train" / "coco.json"))
rgb_coco = json.load(open(ROOT / "images_rgb_train" / "coco.json"))

rgb_cat_by_id = {c["id"]: c["name"] for c in rgb_coco["categories"]}
PERSON_ID = next(c["id"] for c in rgb_coco["categories"] if c["name"] == "person")

# ---------------------------------------------------------------------------
# Step 1: 0.025% 換算成 thermal 640x512 畫面下的絕對像素面積
# ---------------------------------------------------------------------------
th_abs_px = TH_AREA * TH_PCT / 100.0
print(f"[Step 1] thermal 畫面總像素 = {TH_AREA}, 0.025% 換算絕對面積 = {th_abs_px:.4f} px^2")
print(f"          (相當於邊長約 {th_abs_px**0.5:.2f} x {th_abs_px**0.5:.2f} px 的正方形)")

# ---------------------------------------------------------------------------
# thermal train 用 0.025% 門檻實際濾掉的標註比例(重用 A2 已經算過的邏輯,
# 這裡重新算一次確保跟這份腳本自己的資料流程一致,不是憑印象抄數字)
# ---------------------------------------------------------------------------
def bbox_area_ratios(coco):
    img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    ratios = []
    cat_ids = []
    for a in coco["annotations"]:
        img_id = a["image_id"]
        if img_id not in img_wh:
            continue
        w_img, h_img = img_wh[img_id]
        x, y, w, h = a["bbox"]
        ratios.append(100.0 * (w * h) / (w_img * h_img))
        cat_ids.append(a["category_id"])
    return np.array(ratios), np.array(cat_ids)

th_ratios, th_cat_ids = bbox_area_ratios(th_coco)
rgb_ratios, rgb_cat_ids = bbox_area_ratios(rgb_coco)

th_filtered_frac = float((th_ratios < TH_PCT).mean())
print(f"[baseline] thermal train 用 {TH_PCT}% 門檻濾掉 {th_filtered_frac*100:.4f}% 的標註"
      f"(n={len(th_ratios)})")

rgb_person_total = int((rgb_cat_ids == PERSON_ID).sum())
rgb_total = len(rgb_ratios)
print(f"[baseline] RGB train 標註總數 n={rgb_total}, person 總數={rgb_person_total}")

# ---------------------------------------------------------------------------
# 三種候選校準方式
# ---------------------------------------------------------------------------
def apply_threshold(pct_threshold):
    keep_mask = rgb_ratios >= pct_threshold
    filtered_frac = 1 - keep_mask.mean()
    person_mask = rgb_cat_ids == PERSON_ID
    person_remaining = int((keep_mask & person_mask).sum())
    return filtered_frac, person_remaining

candidates = {}

# (a) 直接沿用同樣的畫面佔比 0.025%
pct_a = TH_PCT
frac_a, person_a = apply_threshold(pct_a)
candidates["a_same_pct"] = {
    "description": "直接沿用同樣的畫面佔比 0.025%",
    "threshold_pct": pct_a,
    "threshold_abs_px": pct_a / 100.0 * RGB_AREA,
    "filtered_frac_pct": frac_a * 100,
    "person_remaining": person_a,
}

# (b) 換算成跟 thermal 相同的絕對像素面積
pct_b = th_abs_px / RGB_AREA * 100.0
frac_b, person_b = apply_threshold(pct_b)
candidates["b_same_abs_px"] = {
    "description": "換算成跟 thermal 相同的絕對像素面積(81.92 px^2)",
    "threshold_pct": pct_b,
    "threshold_abs_px": th_abs_px,
    "filtered_frac_pct": frac_b * 100,
    "person_remaining": person_b,
}

# (c) 用 RGB 自己的分布,取跟 thermal 0.025% 過濾掉相同「標註佔比」的百分位數
target_percentile = th_filtered_frac * 100  # 例如 19.73
pct_c = float(np.percentile(rgb_ratios, target_percentile))
frac_c, person_c = apply_threshold(pct_c)
candidates["c_matched_percentile"] = {
    "description": f"RGB 分布裡濾掉相同標註比例({th_filtered_frac*100:.4f}%)所需的門檻",
    "threshold_pct": pct_c,
    "threshold_abs_px": pct_c / 100.0 * RGB_AREA,
    "filtered_frac_pct": frac_c * 100,
    "person_remaining": person_c,
}

output = {
    "thermal_baseline": {
        "th_pct_threshold": TH_PCT,
        "th_image_area_px": TH_AREA,
        "th_abs_px_threshold": th_abs_px,
        "th_filtered_frac_pct": th_filtered_frac * 100,
        "th_total_anns": int(len(th_ratios)),
    },
    "rgb_baseline": {
        "rgb_image_area_px": RGB_AREA,
        "rgb_total_anns": rgb_total,
        "rgb_person_total": rgb_person_total,
    },
    "candidates": candidates,
}

json.dump(output, open(OUT / "a7_threshold_candidates.json", "w"), indent=2)
print(json.dumps(output, indent=2))
