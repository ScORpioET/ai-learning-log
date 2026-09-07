"""
掃描 find_black_blobs.py 實際操作的對象——out_{train,val,test}/flare/ 這
15,153 張 Flare7K++ 模型輸出圖(不是原始 RGB 圖)的解析度分布。

模型本身會把圖片短邊縮到 512px,所以高度預期固定是 512,寬度隨原圖長寬比
變化——這才是 MIN_AREA/RING_DILATE 這兩個像素參數實際套用的解析度基準。
"""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
DIRS = {
    "train": HERE / "out_train" / "flare",
    "val": HERE / "out_val" / "flare",
    "test": HERE / "out_test" / "flare",
}


def main():
    counter = Counter()
    per_split = {s: Counter() for s in DIRS}
    total = 0
    for split, d in DIRS.items():
        files = sorted(d.glob("*.jpg"))
        print(f"[{split}] {len(files)} 張")
        for i, fp in enumerate(files):
            with Image.open(fp) as im:
                size = im.size
            counter[size] += 1
            per_split[split][size] += 1
            total += 1
            if (i + 1) % 2000 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()

    print(f"\n[info] 共 {total} 張,unique 解析度數量: {len(counter)}")
    print("\n=== 全體解析度分布(前20,依張數降冪) ===")
    for size, n in counter.most_common(20):
        print(f"  {size[0]}x{size[1]}: {n} 張 ({100*n/total:.2f}%)")

    heights = Counter()
    for (w, h), n in counter.items():
        heights[h] += n
    print("\n=== 高度分布 ===")
    for h, n in heights.most_common():
        print(f"  height={h}: {n} 張 ({100*n/total:.2f}%)")

    widths = [w for (w, h) in counter.elements()] if False else None
    all_w = []
    for (w, h), n in counter.items():
        all_w.extend([w] * n)
    import numpy as np
    print(f"\n寬度分布: min={min(all_w)} p25={np.percentile(all_w,25):.0f} p50={np.percentile(all_w,50):.0f} "
          f"p75={np.percentile(all_w,75):.0f} max={max(all_w)}")

    dominant_size, dominant_n = counter.most_common(1)[0]
    dominant_frac = dominant_n / total
    print(f"\n[結論] 最主要的解析度 {dominant_size} 佔 {100*dominant_frac:.2f}%")

    out = {
        "total": total,
        "n_unique_resolutions": len(counter),
        "distribution": [{"width": s[0], "height": s[1], "count": n} for s, n in counter.most_common()],
        "dominant_resolution": {"width": dominant_size[0], "height": dominant_size[1], "fraction": round(dominant_frac, 4)},
    }
    with open(HERE / "flare_resolution_distribution.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[done] flare_resolution_distribution.json 已寫入")


if __name__ == "__main__":
    main()
