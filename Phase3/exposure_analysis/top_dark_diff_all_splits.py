"""
把 train/val/test 三個 split 的 RGB 圖片混在一起,對整張圖(不是逐 bbox)算
dark_diff(= median - p1),挑出全部混合起來最大的前 30 名。

重用 compute_rgb_exposure.py 的 rgb_luminance()/median_dark_diff(),跟
calibrate_exposure_ui.py 校準工具、select_high_frac_train_samples.py 同一套
公式,不重新刻邏輯。

三個 split 的圖片目錄名稱不一致(test 是 video_rgb_test,不是
images_rgb_test),照 Day39 已經查證過的規則各自組路徑。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compute_rgb_exposure import rgb_luminance, median_dark_diff

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
SPLITS = {
    "train": ROOT / "images_rgb_train" / "data",
    "val": ROOT / "images_rgb_val" / "data",
    "test": ROOT / "video_rgb_test" / "data",
}
OUT_PATH = Path(__file__).parent / "dark_diff_all_splits_scan.jsonl"
TOP_N = 30


def main():
    out = open(OUT_PATH, "w", encoding="utf-8")
    n_total = 0
    for split, img_dir in SPLITS.items():
        paths = sorted(img_dir.glob("*.jpg"))
        print(f"[info] {split}: {len(paths)} 張")
        for i, p in enumerate(paths):
            try:
                arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            except Exception as e:
                print(f"[warn] 讀圖失敗 {p}: {e}")
                continue
            lum = rgb_luminance(arr)
            median, dark_diff = median_dark_diff(lum)
            out.write(json.dumps({
                "split": split, "file_name": f"data/{p.name}",
                "median": round(median, 2), "dark_diff": round(dark_diff, 2),
            }) + "\n")
            n_total += 1
            if (i + 1) % 2000 == 0:
                print(f"  ...{split} {i+1}/{len(paths)}", end="\r")
        print()
    out.close()
    print(f"[done] {n_total} 張圖片掃完 -> {OUT_PATH}")

    records = [json.loads(l) for l in open(OUT_PATH, encoding="utf-8")]
    records.sort(key=lambda r: -r["dark_diff"])
    top30 = records[:TOP_N]

    from collections import Counter
    print(f"\n[info] 全部 {len(records)} 張(三個 split 混合)的 split 分布: {Counter(r['split'] for r in records)}")
    print(f"[info] top {TOP_N} 的 split 分布: {Counter(r['split'] for r in top30)}")
    for i, r in enumerate(top30, 1):
        print(f"  {i:2d}. [{r['split']:5s}] dark_diff={r['dark_diff']:7.2f}  median={r['median']:7.2f}  {r['file_name']}")

    with open(Path(__file__).parent / "top30_dark_diff_all_splits.json", "w", encoding="utf-8") as f:
        json.dump(top30, f, ensure_ascii=False, indent=2)
    print("\n[done] top30_dark_diff_all_splits.json written")


if __name__ == "__main__":
    main()
