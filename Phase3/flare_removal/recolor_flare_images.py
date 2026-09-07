"""
把 Flare7K++ 輸出的 flare/ 資料夾(抽出的光斑成分,原本是模型自己的配色)
套上跟 build_diff_gallery.py 的 diff_heatmap() 完全一樣的暖色系配色
(黑->紅->黃白,強度越大越暖),方便肉眼看出光斑最強烈的位置。

作法:每張圖轉灰階當作「強度」,對每張圖自己的最大值正規化(跟
diff_heatmap() 的正規化方式一致),套用同一組 r/g/b 公式上色,
原地覆蓋(同檔名、同路徑),不改變其他任何檔案(blend/ 不動)。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
SPLIT_DIRS = [HERE / "out_train" / "flare", HERE / "out_val" / "flare", HERE / "out_test" / "flare"]


def to_heatmap(im):
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    norm = np.clip(gray / gray.max() if gray.max() > 0 else gray, 0, 1)
    heat = np.zeros((*norm.shape, 3), dtype=np.uint8)
    r = np.clip(norm * 3, 0, 1)
    g = np.clip(norm * 3 - 1, 0, 1)
    b = np.clip(norm * 3 - 2, 0, 1)
    heat[..., 0] = (r * 255).astype(np.uint8)
    heat[..., 1] = (g * 255).astype(np.uint8)
    heat[..., 2] = (b * 255).astype(np.uint8)
    return Image.fromarray(heat)


def main():
    total = 0
    for d in SPLIT_DIRS:
        files = sorted(d.glob("*.jpg"))
        print(f"[{d.parent.name}] {len(files)} 張")
        for i, fp in enumerate(files):
            im = Image.open(fp)
            heat = to_heatmap(im)
            heat.save(fp, format="JPEG", quality=92)
            total += 1
            if (i + 1) % 1000 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()
    print(f"[done] 共 {total} 張 flare 圖片改成暖色熱圖配色")


if __name__ == "__main__":
    main()
