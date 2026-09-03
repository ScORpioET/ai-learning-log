"""
Day40:RGB train dataset 逐 bbox 曝光品質分析。

方法(方法論選擇,不是查出來的事實,以下每個決定都寫清楚理由):

- 範圍:只算 DYNAMIC_CLASSES(person/bike/car/...)的標註,跟整個專案
  目前訓練/生成 caption 用的物件範圍一致,不含 sign/light/hydrant 這些
  static context 類別。
- 亮度:luminance = 0.299R + 0.587G + 0.114B(標準攝影亮度換算公式,
  不是隨手取三通道平均)。
- bbox 外擴:「外擴 X%」定義成每一邊各外擴原始寬/高的 X%,即
  new_w = w*(1+2X), new_h = h*(1+2X),以原始 bbox 中心為準,裁切到圖片
  邊界內。這是 CV 領域「context padding」的常見定義,題目沒有明確給
  公式,這裡採用這個版本。
- 每個 bbox 在外擴後的區域內算:
    median_luminance:中位數(題目指定,排除極端亮暗值的干擾,當這個
      區域的「正常亮度基準」)
    bright_diff = p99(區域) - median(區域):最亮的那 1% 像素比基準亮
      多少——強力燈光/過曝是局部的(一顆燈、一小塊反光),不會把整個
      外擴區域都拉爆,所以看「最亮的極端值」偏離中位數基準多少,而不是
      看整個區域的平均值
    dark_diff = median(區域) - p1(區域):對稱地看最暗的 1% 像素比基準暗
      多少——欠曝/陰影的局部訊號
    saturated_frac:亮度 >=250(8-bit 動態範圍裡標準的「highlight
      clipping」門檻,不是隨手訂的數字)的像素比例——輔助判斷,量化
      「有多少像素」而不只是「差多少」
    dark_frac:亮度 <=5(對稱的「shadow clipping」門檻)的像素比例——
      欠曝的輔助判斷
  過曝跟欠曝分開算,同一個 bbox 兩者都可能同時不為 0(題目說的「同時
  發生」),不互斥。bright_diff/dark_diff 就是題目要求「用中位數排除
  極端亮度」之後拿去畫直方圖、標百分位數的那個「pixel 差值」。
- 這支腳本只負責算原始數字,不下「多少 % 算有問題」的最終判斷,那個
  門檻留到 summarize_and_report.py 依直方圖/百分位數決定,並清楚標注是
  方法論選擇。

輸出:rgb_exposure_results.jsonl(gitignore,一行一個 bbox 的結果)。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
RGB_DIR = ROOT / "images_rgb_train"
OUT_PATH = Path(__file__).parent / "rgb_exposure_results.jsonl"

DYNAMIC_CLASSES = {"person", "bike", "motor", "car", "bus", "truck",
                    "other vehicle", "train", "skateboard", "stroller", "scooter"}
EXPANSIONS = [0.25, 0.50, 0.75]
SAT_THRESH = 250
DARK_THRESH = 5


def expand_bbox(x, y, w, h, pct, img_w, img_h):
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * (1 + 2 * pct), h * (1 + 2 * pct)
    x0 = max(0, int(round(cx - new_w / 2)))
    y0 = max(0, int(round(cy - new_h / 2)))
    x1 = min(img_w, int(round(cx + new_w / 2)))
    y1 = min(img_h, int(round(cy + new_h / 2)))
    return x0, y0, x1, y1


def main():
    coco = json.load(open(RGB_DIR / "coco.json"))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    dyn_ids = {c["id"] for c in coco["categories"] if c["name"] in DYNAMIC_CLASSES}

    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in dyn_ids:
            anns_by_img[a["image_id"]].append(a)

    n_images = len(anns_by_img)
    print(f"[info] {n_images} 張 RGB train 圖片有 dynamic-class 標註")

    out = open(OUT_PATH, "w", encoding="utf-8")
    n_bbox = 0
    for i, (img_id, anns) in enumerate(anns_by_img.items()):
        im_info = img_meta[img_id]
        img_path = RGB_DIR / im_info["file_name"]
        try:
            arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
        except Exception as e:
            print(f"[warn] 讀圖失敗 {img_path}: {e}")
            continue
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        img_h, img_w = lum.shape

        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_name = id2name[ann["category_id"]]
            record = {"file_name": im_info["file_name"], "ann_id": ann["id"], "category": cat_name}
            for pct in EXPANSIONS:
                x0, y0, x1, y1 = expand_bbox(x, y, w, h, pct, img_w, img_h)
                if x1 <= x0 or y1 <= y0:
                    record[f"pct_{int(pct*100)}"] = None
                    continue
                region = lum[y0:y1, x0:x1]
                med = float(np.median(region))
                p99 = float(np.percentile(region, 99))
                p1 = float(np.percentile(region, 1))
                sat_frac = float(np.mean(region >= SAT_THRESH))
                dark_frac = float(np.mean(region <= DARK_THRESH))
                record[f"pct_{int(pct*100)}"] = {
                    "median_luminance": round(med, 2),
                    "bright_diff": round(p99 - med, 2),
                    "dark_diff": round(med - p1, 2),
                    "saturated_frac": round(sat_frac, 5),
                    "dark_frac": round(dark_frac, 5),
                    "n_pixels": int(region.size),
                }
            out.write(json.dumps(record) + "\n")
            n_bbox += 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{n_images} images, {n_bbox} bboxes so far", end="\r")

    out.close()
    print(f"\n[done] {n_bbox} bboxes written to {OUT_PATH}")


if __name__ == "__main__":
    main()
