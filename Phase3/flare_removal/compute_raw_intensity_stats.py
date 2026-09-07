"""
對 840 張最終樣本,重跑 Flare7K++(raw_intensity_run/)拿到的、正規化前的
原始 flare 灰階輸出,計算 raw_max/raw_mean/raw_p95(用跟 recolor_flare_
images.py 完全一樣的 PIL .convert("L") 灰階轉換方式,確保跟正規化分母
raw_max 的定義一致),寫入 flare_intensity_raw_stats.json,並 join 進
black_blob_summary_v3_keep.json。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
RAW_DIRS = {
    "out_train": HERE / "raw_intensity_run" / "out_train" / "flare",
    "out_val": HERE / "raw_intensity_run" / "out_val" / "flare",
    "out_test": HERE / "raw_intensity_run" / "out_test" / "flare",
}


def raw_stats(path):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return {
        "raw_max": float(gray.max()),
        "raw_mean": round(float(gray.mean()), 2),
        "raw_p95": round(float(np.percentile(gray, 95)), 2),
    }


def main():
    keep = json.load(open(HERE / "black_blob_summary_v3_keep.json"))
    stats_records = []
    n_missing = 0
    for row in keep:
        split, fn = row["split"], row["file_name"]
        p = RAW_DIRS[split] / fn
        if not p.exists():
            n_missing += 1
            continue
        s = raw_stats(p)
        stats_records.append({"split": split, "file_name": fn, **s})
        row.update(s)

    print(f"[info] 840 張裡,{len(stats_records)} 張成功算出 raw 強度,{n_missing} 張找不到對應的重跑輸出")

    with open(HERE / "flare_intensity_raw_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_records, f, ensure_ascii=False, indent=2)

    with open(HERE / "black_blob_summary_v3_keep.json", "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)

    maxes = [r["raw_max"] for r in stats_records]
    means = [r["raw_mean"] for r in stats_records]
    print(f"raw_max: min={min(maxes):.1f} p50={np.percentile(maxes,50):.1f} max={max(maxes):.1f}")
    print(f"raw_mean: min={min(means):.1f} p50={np.percentile(means,50):.1f} max={max(means):.1f}")
    print("[done] flare_intensity_raw_stats.json 已寫入,black_blob_summary_v3_keep.json 已補上 raw_max/raw_mean/raw_p95 欄位")


if __name__ == "__main__":
    main()
