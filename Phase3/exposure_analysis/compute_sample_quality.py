"""
Day39 收尾(2026-09-04 改版):對 caption_fusion/test_pairs.json 的 10 組
test frame pair,套用 Jack 明確指定的固定門檻,算出 RGB/thermal 優先權
建議。

規則(2026-09-04 定案,寫死的固定數字,不是這批資料的百分位數):
    dark_diff > 200 OR median_luminance < 50  ->  判定 RGB 被燈光影響,
    改用 thermal 優先;否則預設用 RGB 優先。
不使用 bright_diff 當判斷依據。thermal 側不另外設門檻、不做獨立失效
檢查——一律看 RGB 側的判斷結果決定優先權(這點跟舊版的差異最大:舊版
thermal diff<=p10 也會單獨觸發「建議 RGB 優先」,這版拿掉了)。

方法:bbox 外擴公式、luminance 公式全部照抄 compute_rgb_exposure.py,
不改算法,只新增 dark_diff(= median - p1)這個原本只在 compute_rgb_exposure.py
裡算過、但這支腳本先前沒有算的欄位。一張圖裡可能有多個 dynamic-class
bbox,只要「任一個」bbox 踩到門檻,就判定 RGB 這一側有問題(取最壞情況)。

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

DARK_DIFF_THRESH = 200
MEDIAN_THRESH = 50
EXPAND_PCT = 0.50  # 跟 compute_rgb_exposure.py 的外擴定義一致,固定門檻本身跟外擴比例無關


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
        p1 = float(np.percentile(region, 1))
        dark_diff = round(med - p1, 2)
        per_bbox.append({
            "category": id2name[ann["category_id"]],
            "bbox": [x, y, w, h],
            "median_luminance": round(med, 2),
            "dark_diff": dark_diff,
            "lighting_affected": dark_diff > DARK_DIFF_THRESH or med < MEDIAN_THRESH,
        })
    has_issue = any(b["lighting_affected"] for b in per_bbox)
    return {
        "per_bbox": per_bbox,
        "has_issue": has_issue,
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
        })
    # thermal 側不設獨立門檻(2026-09-04 定案),這裡只保留 diff 數字供參考,
    # 不產生 has_issue,優先權判斷完全不看這個結果。
    return {"per_bbox": per_bbox}


def suggest(rgb_q):
    """2026-09-04 定案:dark_diff>200 OR median_luminance<50 -> thermal 優先,
    否則預設 RGB 優先。thermal 側不參與判斷。"""
    if rgb_q["has_issue"]:
        reasons = [f"{b['category']}: dark_diff={b['dark_diff']}, median_luminance={b['median_luminance']}"
                   for b in rgb_q["per_bbox"] if b["lighting_affected"]]
        return {
            "label": "建議 thermal 優先",
            "reason": f"RGB 側偵測到 dark_diff>200 或 median_luminance<50(命中的 bbox: {'; '.join(reasons)})，判定 RGB 被燈光影響。",
        }
    return {
        "label": "建議 RGB 優先",
        "reason": "RGB 側所有 bbox 的 dark_diff 都 <=200 且 median_luminance 都 >=50，未偵測到燈光問題，預設用 RGB。",
    }


def main():
    pairs = json.load(open(FUSION_DIR / "test_pairs.json"))["sampled"]

    rgb_by_file, rgb_id2name = load_coco_anns(RGB_TEST_DIR / "coco.json", DYNAMIC_CLASSES)
    th_by_file, th_id2name = load_coco_anns(TH_TEST_DIR / "coco.json", DYNAMIC_CLASSES)

    out = []
    for p in pairs:
        rf, tf = p["rgb_file"], p["thermal_file"]
        rgb_im, rgb_anns = rgb_by_file.get(rf, (None, []))
        th_im, th_anns = th_by_file.get(tf, (None, []))

        rgb_q = rgb_quality(rf, rgb_im, rgb_anns, rgb_id2name) if rgb_anns else {"per_bbox": [], "has_issue": False}
        th_q = thermal_quality(tf, th_im, th_anns, th_id2name) if th_anns else {"per_bbox": []}

        sug = suggest(rgb_q)
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
