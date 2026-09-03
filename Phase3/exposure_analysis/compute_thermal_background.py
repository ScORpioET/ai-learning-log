"""
Day40:thermal train dataset 逐 bbox「物體是否融入背景」分析。

方法(方法論選擇,理由寫清楚):

- 範圍:跟 RGB 那支一樣,只算 DYNAMIC_CLASSES 標註。
- bbox 外擴定義跟 compute_rgb_exposure.py 完全一致(每邊各外擴 X%,
  new_w=w*(1+2X),裁到圖片邊界內)。
- 「跟周圍環境顏色相近」的量化方式:
    object_median  = 原始 bbox 內 pixel 值的中位數
    surround_median = (外擴後區域 - 原始 bbox 區域)這個「環」的中位數
    diff = abs(object_median - surround_median)
  diff 越小,代表物體跟外擴進來的周圍環境在熱像圖上的像素值越接近,
  視覺上/模型上越難靠亮度分辨物體邊界。三個外擴比例(25/50/75%)分別算
  一次,因為外擴越大,「周圍」涵蓋的範圍越大,diff 本來就會系統性變化,
  三個層級的數字不能直接混在一起看。
- 這支腳本一樣只算原始 diff 數字,不下「diff 多小算有問題」的最終判斷,
  留給 summarize_and_report.py 依直方圖/百分位數決定。

輸出:thermal_background_results.jsonl(gitignore)。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
TH_DIR = ROOT / "images_thermal_train"
OUT_PATH = Path(__file__).parent / "thermal_background_results.jsonl"

DYNAMIC_CLASSES = {"person", "bike", "motor", "car", "bus", "truck",
                    "other vehicle", "train", "skateboard", "stroller", "scooter"}
EXPANSIONS = [0.25, 0.50, 0.75]


def expand_bbox(x, y, w, h, pct, img_w, img_h):
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * (1 + 2 * pct), h * (1 + 2 * pct)
    x0 = max(0, int(round(cx - new_w / 2)))
    y0 = max(0, int(round(cy - new_h / 2)))
    x1 = min(img_w, int(round(cx + new_w / 2)))
    y1 = min(img_h, int(round(cy + new_h / 2)))
    return x0, y0, x1, y1


def main():
    coco = json.load(open(TH_DIR / "coco.json"))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    dyn_ids = {c["id"] for c in coco["categories"] if c["name"] in DYNAMIC_CLASSES}

    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in dyn_ids:
            anns_by_img[a["image_id"]].append(a)

    n_images = len(anns_by_img)
    print(f"[info] {n_images} 張 thermal train 圖片有 dynamic-class 標註")

    out = open(OUT_PATH, "w", encoding="utf-8")
    n_bbox = 0
    for i, (img_id, anns) in enumerate(anns_by_img.items()):
        im_info = img_meta[img_id]
        img_path = TH_DIR / im_info["file_name"]
        try:
            arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        except Exception as e:
            print(f"[warn] 讀圖失敗 {img_path}: {e}")
            continue
        img_h, img_w = arr.shape

        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_name = id2name[ann["category_id"]]
            xi0, yi0 = max(0, int(round(x))), max(0, int(round(y)))
            xi1, yi1 = min(img_w, int(round(x + w))), min(img_h, int(round(y + h)))
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            object_region = arr[yi0:yi1, xi0:xi1]
            object_median = float(np.median(object_region))

            record = {"file_name": im_info["file_name"], "ann_id": ann["id"], "category": cat_name,
                      "object_median": round(object_median, 2)}
            inner_mask_box = (xi0, yi0, xi1, yi1)
            for pct in EXPANSIONS:
                x0, y0, x1, y1 = expand_bbox(x, y, w, h, pct, img_w, img_h)
                if x1 <= x0 or y1 <= y0:
                    record[f"pct_{int(pct*100)}"] = None
                    continue
                mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
                # 挖掉原始 bbox 那一塊,只留「環」
                ix0, iy0 = max(inner_mask_box[0], x0) - x0, max(inner_mask_box[1], y0) - y0
                ix1, iy1 = min(inner_mask_box[2], x1) - x0, min(inner_mask_box[3], y1) - y0
                if ix1 > ix0 and iy1 > iy0:
                    mask[iy0:iy1, ix0:ix1] = False
                ring = arr[y0:y1, x0:x1][mask]
                if ring.size == 0:
                    record[f"pct_{int(pct*100)}"] = None
                    continue
                surround_median = float(np.median(ring))
                diff = abs(object_median - surround_median)
                record[f"pct_{int(pct*100)}"] = {
                    "surround_median": round(surround_median, 2),
                    "diff": round(diff, 2),
                    "n_ring_pixels": int(ring.size),
                }
            out.write(json.dumps(record) + "\n")
            n_bbox += 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{n_images} images, {n_bbox} bboxes so far", end="\r")

    out.close()
    print(f"\n[done] {n_bbox} bboxes written to {OUT_PATH}")


if __name__ == "__main__":
    main()
