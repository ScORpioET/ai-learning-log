"""
任務:對934張候選(black_blob_summary_matched_after_param_v2.json)套用
saturation_ratio 規則分類,只分類、不修改任何既有清單。

核心區域定義沿用 sanity check 那兩輪完全相同的邏輯:
    blob = max(row["blobs"], key=area) 的 bbox,換算回原圖解析度,
    再往外擴張 0.5*max(ow,oh) 當作量測區域,轉灰階後計算
    saturation_ratio = 灰階值>=250的像素比例。

分類規則(沿用12張樣本驗證出的重疊區間 [0.1881, 0.2826]):
    > 0.2826            -> severe_candidate(明顯高)
    < 0.1881            -> not_severe_candidate(明顯低)
    [0.1881, 0.2826]    -> ambiguous(模糊帶,待人工複核)

輸出三份 csv,不動任何既有清單檔案:
    saturation_classify_severe_v1.csv
    saturation_classify_not_severe_v1.csv
    saturation_classify_ambiguous_v1.csv
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
LO, HI = 0.1881, 0.2826


def saturation_ratio_for(split, fn, row):
    blob = max(row["blobs"], key=lambda b: b["area"])
    bx, by, bw, bh = blob["bbox"]
    orig = cv2.imread(str(ORIG_DIRS[split] / fn))
    flare = cv2.imread(str(FLARE_DIRS[split] / fn))
    if orig is None or flare is None:
        return None
    scale = orig.shape[0] / flare.shape[0]
    ox, oy, ow, oh = [round(v * scale) for v in (bx, by, bw, bh)]
    pad = round(0.5 * max(ow, oh))
    x0, y0 = max(ox - pad, 0), max(oy - pad, 0)
    x1, y1 = min(ox + ow + pad, orig.shape[1]), min(oy + oh + pad, orig.shape[0])
    crop_gray = cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(np.mean(crop_gray >= SAT_THRESH))


def main():
    data = json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))
    n_total = len(data)

    severe, not_severe, ambiguous = [], [], []
    n_skipped = 0
    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        sat = saturation_ratio_for(split, fn, row)
        if sat is None:
            n_skipped += 1
            continue
        entry = {"split": split, "file_name": fn, "saturation_ratio": round(sat, 4)}
        if sat > HI:
            severe.append(entry)
        elif sat < LO:
            not_severe.append(entry)
        else:
            ambiguous.append(entry)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{n_total}", end="\r")
    print()

    n_valid = n_total - n_skipped
    print(f"\n[info] 總計 {n_total} 張,成功計算 {n_valid} 張,跳過(讀檔失敗) {n_skipped} 張\n")
    print(f"明顯高(severe_candidate, sat>{HI}):      {len(severe):4d} 張 ({100*len(severe)/n_valid:.1f}%)")
    print(f"明顯低(not_severe_candidate, sat<{LO}):  {len(not_severe):4d} 張 ({100*len(not_severe)/n_valid:.1f}%)")
    print(f"模糊帶(ambiguous, [{LO},{HI}]):            {len(ambiguous):4d} 張 ({100*len(ambiguous)/n_valid:.1f}%)")

    for name, rows in [("severe", severe), ("not_severe", not_severe), ("ambiguous", ambiguous)]:
        rows_sorted = sorted(rows, key=lambda r: r["saturation_ratio"])
        out_path = HERE / f"saturation_classify_{name}_v1.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "file_name", "saturation_ratio"])
            writer.writeheader()
            writer.writerows(rows_sorted)
        print(f"[written] {out_path}  ({len(rows_sorted)} 筆)")


if __name__ == "__main__":
    main()
