"""
驗證「saturation_ratio(核心量測區域) >= 20% 就算有影響/嚴重」這個理論,
對934張候選全部分類,並輸出可肉眼核對的融合圖(full_frame + 全圖飽和像素疊圖)。

門檻:core saturation_ratio(核心bbox外擴0.5倍區域,灰階>=250比例) >= 0.20 -> 嚴重
                                                              <  0.20 -> 不嚴重
(核心區域定義沿用整個系列實驗一致的邏輯,不重新定義)

輸出結構(每個split各兩個資料夾):
  saturation_theory_verify_v1/out_train/嚴重/*.jpg
  saturation_theory_verify_v1/out_train/不嚴重/*.jpg
  saturation_theory_verify_v1/out_val/嚴重|不嚴重/*.jpg
  saturation_theory_verify_v1/out_test/嚴重|不嚴重/*.jpg

每張輸出圖 = 左:原始full frame / 右:full frame疊紅色飽和遮罩(整張圖灰階>=250的像素,
不只核心區域),方便肉眼看飽和到底出現在哪裡、範圍多大。
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402
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
NEW_THRESH = 0.20
OUT_ROOT = HERE / "saturation_theory_verify_v1"
MAX_PANEL_W = 700


def core_saturation_ratio(split, fn, blob_data):
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
    crop_gray = cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(np.mean(crop_gray >= SAT_THRESH)), orig


def resize_max_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, round(h * scale)), interpolation=cv2.INTER_AREA)


def build_merged(orig):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    overlay = orig.copy()
    overlay[gray >= SAT_THRESH] = (0, 0, 255)
    blended = cv2.addWeighted(overlay, 0.5, orig, 0.5, 0)

    left = resize_max_width(orig, MAX_PANEL_W)
    right = resize_max_width(blended, MAX_PANEL_W)
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (left.shape[1], h))
        right = cv2.resize(right, (right.shape[1], h))
    gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    merged = cv2.hconcat([left, gap, right])

    label_strip = np.full((30, merged.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(label_strip, "original", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(label_strip, "saturated pixels (gray>=250) in red", (left.shape[1] + 20, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return cv2.vconcat([label_strip, merged])


def main():
    blob_data = {(r["split"], r["file_name"]): r for r in
                 json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))}
    rows = []
    for fn in ["saturation_classify_severe_v1.csv", "saturation_classify_not_severe_v1.csv",
               "saturation_classify_ambiguous_v1.csv"]:
        rows += list(csv.DictReader(open(HERE / fn)))

    for s in ["out_train", "out_val", "out_test"]:
        (OUT_ROOT / s / "嚴重").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / s / "不嚴重").mkdir(parents=True, exist_ok=True)

    n_total = len(rows)
    counts = {"嚴重": 0, "不嚴重": 0}
    results_log = []
    for i, r in enumerate(rows):
        split, fn = r["split"], r["file_name"]
        sat_ratio, orig = core_saturation_ratio(split, fn, blob_data)
        label = "嚴重" if sat_ratio >= NEW_THRESH else "不嚴重"
        counts[label] += 1
        merged = build_merged(orig)
        out_path = OUT_ROOT / split / label / fn
        cv2.imwrite(str(out_path), merged, [cv2.IMWRITE_JPEG_QUALITY, 88])
        results_log.append({"split": split, "file_name": fn, "saturation_ratio": round(sat_ratio, 4), "label": label})
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{n_total}", end="\r")
    print()

    with open(HERE / "saturation_theory_verify_v1_log.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    print(f"\n[done] 總計 {n_total} 張,嚴重={counts['嚴重']}({100*counts['嚴重']/n_total:.1f}%),"
          f"不嚴重={counts['不嚴重']}({100*counts['不嚴重']/n_total:.1f}%)")
    for s in ["out_train", "out_val", "out_test"]:
        n_sev = len(list((OUT_ROOT / s / "嚴重").glob("*.jpg")))
        n_not = len(list((OUT_ROOT / s / "不嚴重").glob("*.jpg")))
        print(f"  {s}: 嚴重={n_sev}  不嚴重={n_not}")


if __name__ == "__main__":
    main()
