"""
Day35 Task 5-C:5 張眼睛驗證。每張圖:
- Line 1: GT 全版 caption
- Line 2: GT filtered caption(per-class threshold)
- 圖上疊「filtered 後還留著的」GT 框(綠色)
"""
import json
import random
from pathlib import Path

import cv2

ROOT = Path.home() / "ai-transition-2026"
IMG_DIR = ROOT / "thermal_dataset" / "images_thermal_val" / "data"
OUT_DIR = ROOT / "Day35" / "outputs" / "day35_filter_check"
COCO_PATH = ROOT / "thermal_dataset" / "images_thermal_val" / "coco.json"

CAP_FULL = ROOT / "Day35" / "outputs" / "captions_val_gt_current.jsonl"
CAP_FILTERED = ROOT / "Day35" / "outputs" / "captions_val_filtered.jsonl"

GLOBAL_MIN_AREA_PCT = 0.05
DYNAMIC_NAMES = {"person", "bike", "motor", "car", "bus", "truck",
                  "other vehicle", "train", "skateboard", "stroller", "scooter"}
GREEN = (0, 255, 0)


def load_jsonl(path, key):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r[key]] = r
    return d


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = json.load(open(COCO_PATH))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    img_meta = {im["id"]: im for im in coco["images"]}
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    cap_full = load_jsonl(CAP_FULL, "file_name")
    cap_filtered = load_jsonl(CAP_FILTERED, "file_name")

    # Task 6:沿用 Task 5 選中的同一批 image_id,才能做「新舊 threshold 差異」對照,
    # 不要重新隨機抽(threshold 換了,filtered 集合的交集也會跟著變,隨機抽會抽到
    # 不同圖)。
    FIXED_IMAGE_IDS = [674, 2077, 5452, 5471, 5720]
    sample = [img_meta[iid]["file_name"] for iid in FIXED_IMAGE_IDS]

    if not (set(cap_full) & set(sample)) == set(sample):
        missing = set(sample) - set(cap_full)
        print(f"[warn] 這些檔名不在 cap_full 裡: {missing}")

    for i, fn in enumerate(sample, 1):
        img_id = next(iid for iid, im in img_meta.items() if im["file_name"] == fn)
        im_meta = img_meta[img_id]
        w_img, h_img = im_meta["width"], im_meta["height"]
        bare_name = Path(fn).name

        im = cv2.imread(str(IMG_DIR / bare_name), cv2.IMREAD_COLOR)
        if im is None:
            print(f"[warn] cannot read {bare_name}, skip")
            continue

        kept_n, dropped_n = 0, 0
        for a in anns_by_img.get(img_id, []):
            cat_name = id2name[a["category_id"]]
            if cat_name not in DYNAMIC_NAMES:
                continue
            x, y, w, h = a["bbox"]
            area_pct = 100 * (w * h) / (w_img * h_img)
            if area_pct < GLOBAL_MIN_AREA_PCT:
                dropped_n += 1
                continue
            kept_n += 1
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(im, (x1, y1), (x2, y2), GREEN, 1)
            cv2.putText(im, cat_name, (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)

        canvas = cv2.copyMakeBorder(im, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"image: {bare_name}  (kept {kept_n} / dropped {dropped_n} dynamic boxes)",
                    (8, h_img + 15), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        filtered_caption = cap_filtered[fn]["caption"] if fn in cap_filtered else "(no caption -- 0 boxes survived filter)"
        cv2.putText(canvas, f"FULL:     {cap_full[fn]['caption']}",
                    (8, h_img + 35), font, 0.42, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FILTERED: {filtered_caption}",
                    (8, h_img + 55), font, 0.42, (100, 200, 255), 1, cv2.LINE_AA)

        out_path = OUT_DIR / f"{i:02d}_{img_id}_{bare_name}"
        cv2.imwrite(str(out_path), canvas)
        print(f"[saved] img_id={img_id} {out_path}")

    print(f"\n[done] 5 comparison images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
