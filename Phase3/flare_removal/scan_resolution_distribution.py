"""
掃描 train+val+test 全部 15,153 張 RGB 原圖的解析度分布,統計 unique
(width, height) 組合各自張數,決定 find_black_blobs.py 的 MIN_AREA/
RING_DILATE 要不要改成解析度相對值。
"""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
DIRS = {
    "train": ROOT / "images_rgb_train" / "data",
    "val": ROOT / "images_rgb_val" / "data",
    "test": ROOT / "video_rgb_test" / "data",
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
                size = im.size  # (w, h)
            counter[size] += 1
            per_split[split][size] += 1
            total += 1
            if (i + 1) % 2000 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()

    print(f"\n[info] 共 {total} 張,unique 解析度數量: {len(counter)}")
    print("\n=== 全體解析度分布(依張數降冪) ===")
    for size, n in counter.most_common():
        print(f"  {size[0]}x{size[1]}: {n} 張 ({100*n/total:.2f}%)")

    print("\n=== 分 split ===")
    for split, c in per_split.items():
        print(f"[{split}]")
        for size, n in c.most_common():
            print(f"  {size[0]}x{size[1]}: {n} 張")

    dominant_size, dominant_n = counter.most_common(1)[0]
    dominant_frac = dominant_n / total
    print(f"\n[結論] 最主要的解析度 {dominant_size} 佔 {100*dominant_frac:.2f}%")

    out = {
        "total": total,
        "n_unique_resolutions": len(counter),
        "distribution": [{"width": s[0], "height": s[1], "count": n} for s, n in counter.most_common()],
        "per_split": {split: [{"width": s[0], "height": s[1], "count": n} for s, n in c.most_common()] for split, c in per_split.items()},
        "dominant_resolution": {"width": dominant_size[0], "height": dominant_size[1], "fraction": round(dominant_frac, 4)},
    }
    with open(Path(__file__).parent / "resolution_distribution.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[done] resolution_distribution.json 已寫入")


if __name__ == "__main__":
    main()
