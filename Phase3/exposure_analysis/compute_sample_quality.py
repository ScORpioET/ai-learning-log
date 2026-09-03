"""
Day39 收尾:對 caption_fusion/test_pairs.json 的 10 組 test frame pair,套用
Day40 已經在 train set 上訂出的曝光/背景品質門檻,算出「系統建議」——
RGB 優先還是 thermal 優先,附理由。

方法(沿用 Day40 的邏輯與門檻,不重新發明):
- bbox 外擴公式、luminance 公式、bright_diff/median_luminance(RGB)、
  object/surround diff(thermal)全部照抄 compute_rgb_exposure.py /
  compute_thermal_background.py,不改任何算法。
- 門檻直接沿用 summary_results.json 裡 pct_50 這一層算出來的門檻(這 10 筆
  抽樣样本量太小,自己重新算百分位數沒有意義,所以借用 train set 全量算出的
  門檻——這是方法論選擇:假設 train/test 的曝光/背景分布相近):
    RGB:   bright_thresh = 123.2   (bright_diff > 這個值 -> 疑似局部強光/過曝)
           median_thresh = 94.862  (median_luminance <= 這個值 -> 疑似整體太暗)
    thermal: thresh = 3.0          (diff <= 這個值 -> 疑似物體融入背景)
- 一張圖(一個 sample 的 RGB 或 thermal 那一側)裡可能有多個 dynamic-class
  bbox,只要「任一個」bbox 踩到門檻,就判定這一側有問題(取最壞情況,
  不是平均掉)。
- 建議規則(這次任務要求「只加標籤+理由,不強制選版本」,所以下面的規則
  只產生一個「系統建議」字串,兩個 fused 版本仍然都輸出):
    RGB 有問題、thermal 沒有 -> 建議 thermal 優先
    thermal 有問題、RGB 沒有 -> 建議 RGB 優先
    兩邊都沒問題 / 兩邊都有問題 -> 沒有明確訊號,理由裡把兩邊的狀態都列出來,
      不硬選一邊(這是誠實揭露,不是漏規則)

輸出:sample_quality.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
RGB_TEST_DIR = TD / "video_rgb_test"
TH_TEST_DIR = TD / "video_thermal_test"
FUSION_DIR = Path(__file__).parent.parent / "caption_fusion"

sys.path.insert(0, str(Path(__file__).parent))
from compute_rgb_exposure import DYNAMIC_CLASSES, expand_bbox  # noqa: E402

RGB_BRIGHT_THRESH = 123.2
RGB_MEDIAN_THRESH = 94.862
THERMAL_DIFF_THRESH = 3.0
EXPAND_PCT = 0.50  # 沿用 summary_results.json 門檻對應的那一層


def load_coco_anns(coco_path, dynamic_classes):
    coco = json.load(open(coco_path))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    dyn_ids = {c["id"] for c in coco["categories"] if c["name"] in dynamic_classes}
    by_file = {}
    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        if a["category_id"] in dyn_ids:
            anns_by_img.setdefault(a["image_id"], []).append(a)
    for img_id, anns in anns_by_img.items():
        im_info = img_meta[img_id]
        by_file[im_info["file_name"]] = (im_info, anns)
    return by_file, id2name


def rgb_quality(file_name, im_info, anns, id2name):
    img_path = RGB_TEST_DIR / file_name
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    img_h, img_w = lum.shape

    per_bbox = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = expand_bbox(x, y, w, h, EXPAND_PCT, img_w, img_h)
        if x1 <= x0 or y1 <= y0:
            continue
        region = lum[y0:y1, x0:x1]
        med = float(np.median(region))
        p99 = float(np.percentile(region, 99))
        bright_diff = round(p99 - med, 2)
        per_bbox.append({
            "category": id2name[ann["category_id"]],
            "bbox": [x, y, w, h],
            "median_luminance": round(med, 2),
            "bright_diff": bright_diff,
            "overexposed": bright_diff > RGB_BRIGHT_THRESH,
            "underexposed": med <= RGB_MEDIAN_THRESH,
        })
    has_over = any(b["overexposed"] for b in per_bbox)
    has_under = any(b["underexposed"] for b in per_bbox)
    return {
        "per_bbox": per_bbox,
        "has_issue": has_over or has_under,
        "overexposed": has_over,
        "underexposed": has_under,
    }


def thermal_quality(file_name, im_info, anns, id2name):
    img_path = TH_TEST_DIR / file_name
    arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    img_h, img_w = arr.shape

    per_bbox = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        xi0, yi0 = max(0, int(round(x))), max(0, int(round(y)))
        xi1, yi1 = min(img_w, int(round(x + w))), min(img_h, int(round(y + h)))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        object_region = arr[yi0:yi1, xi0:xi1]
        object_median = float(np.median(object_region))

        x0, y0, x1, y1 = expand_bbox(x, y, w, h, EXPAND_PCT, img_w, img_h)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
        ix0, iy0 = max(xi0, x0) - x0, max(yi0, y0) - y0
        ix1, iy1 = min(xi1, x1) - x0, min(yi1, y1) - y0
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = False
        ring = arr[y0:y1, x0:x1][mask]
        if ring.size == 0:
            continue
        surround_median = float(np.median(ring))
        diff = round(abs(object_median - surround_median), 2)
        per_bbox.append({
            "category": id2name[ann["category_id"]],
            "bbox": [x, y, w, h],
            "object_median": round(object_median, 2),
            "surround_median": round(surround_median, 2),
            "diff": diff,
            "blends_in": diff <= THERMAL_DIFF_THRESH,
        })
    has_issue = any(b["blends_in"] for b in per_bbox)
    return {"per_bbox": per_bbox, "has_issue": has_issue}


def suggest(rgb_q, th_q):
    rgb_bad, th_bad = rgb_q["has_issue"], th_q["has_issue"]
    if rgb_bad and not th_bad:
        reasons = []
        if rgb_q["overexposed"]:
            reasons.append("RGB 該區域偵測到局部過曝(bright_diff 超過 train set 門檻 123.2)")
        if rgb_q["underexposed"]:
            reasons.append("RGB 整體亮度過低(median_luminance 低於 train set 門檻 94.862)")
        return {"label": "建議 thermal 優先", "reason": "、".join(reasons) + "，thermal 該側未偵測到融入背景問題。"}
    if th_bad and not rgb_bad:
        return {"label": "建議 RGB 優先", "reason": "thermal 偵測到物體與周圍環境像素值接近(diff 低於 train set 門檻 3.0)，疑似融入背景，RGB 該側未偵測到曝光問題。"}
    if rgb_bad and th_bad:
        reasons = []
        if rgb_q["overexposed"]:
            reasons.append("RGB 局部過曝")
        if rgb_q["underexposed"]:
            reasons.append("RGB 整體過暗")
        reasons.append("thermal 疑似融入背景")
        return {"label": "無明確建議(兩側皆有訊號)", "reason": "、".join(reasons) + "——兩側都有品質疑慮，不強制替使用者選邊，兩個版本並列供參考。"}
    return {"label": "無明確建議(兩側皆正常)", "reason": "RGB 與 thermal 這一側都沒有踩到 train set 訂出的曝光/背景門檻，兩個版本並列供參考。"}


def main():
    pairs = json.load(open(FUSION_DIR / "test_pairs.json"))["sampled"]

    rgb_by_file, rgb_id2name = load_coco_anns(RGB_TEST_DIR / "coco.json", DYNAMIC_CLASSES)
    th_by_file, th_id2name = load_coco_anns(TH_TEST_DIR / "coco.json", DYNAMIC_CLASSES)

    out = []
    for p in pairs:
        rf, tf = p["rgb_file"], p["thermal_file"]
        rgb_im, rgb_anns = rgb_by_file.get(rf, (None, []))
        th_im, th_anns = th_by_file.get(tf, (None, []))

        rgb_q = rgb_quality(rf, rgb_im, rgb_anns, rgb_id2name) if rgb_anns else {"per_bbox": [], "has_issue": False, "overexposed": False, "underexposed": False}
        th_q = thermal_quality(tf, th_im, th_anns, th_id2name) if th_anns else {"per_bbox": [], "has_issue": False}

        sug = suggest(rgb_q, th_q)
        print(f"{tf} <-> {rf}: {sug['label']}  ({sug['reason']})")
        out.append({
            "thermal_file": tf, "rgb_file": rf,
            "rgb_quality": rgb_q, "thermal_quality": th_q,
            "suggestion": sug,
        })

    with open(Path(__file__).parent / "sample_quality.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[done] sample_quality.json written")


if __name__ == "__main__":
    main()
