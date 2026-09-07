"""
對 v4 最終 1,994 張樣本,把 raw_intensity_run_v4/ 裡的 flare 輸出重新上色
(暖色熱圖),跟原圖、blend 三張左右合併成一張圖。

注意:這次刻意不覆蓋 raw_intensity_run_v4/flare/ 原始檔案(先前
recolor_flare_images.py 原地覆蓋導致 raw 強度資料遺失的教訓)——
先複製一份到 flare_recolored_v4/ 再上色,raw_intensity_run_v4/flare/
保持原封不動,以後还能重新算 raw 強度。
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}
SPLITS = ["out_train", "out_val", "out_test"]


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
    for split in SPLITS:
        raw_flare_dir = HERE / "raw_intensity_run_v4" / split / "flare"
        blend_dir = HERE / "raw_intensity_run_v4" / split / "blend"
        recolor_dir = HERE / "flare_recolored_v4" / split
        merged_dir = HERE / "merged_v4" / split
        recolor_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(raw_flare_dir.glob("*.jpg"))
        print(f"[{split}] {len(files)} 張")
        for i, fp in enumerate(files):
            heat = to_heatmap(Image.open(fp))
            heat.save(recolor_dir / fp.name, format="JPEG", quality=92)

            blend_path = blend_dir / fp.name
            orig_path = ORIG_DIRS[split] / fp.name
            blend = cv2.imread(str(blend_path))
            orig = cv2.imread(str(orig_path))
            flare_bgr = cv2.cvtColor(np.array(heat), cv2.COLOR_RGB2BGR)

            if blend is not None:
                h, w = blend.shape[:2]
                panels = []
                if orig is not None:
                    panels.append(cv2.resize(orig, (w, h)))
                panels.append(blend)
                panels.append(cv2.resize(flare_bgr, (w, h)))
                merged = cv2.hconcat(panels)
                cv2.imwrite(str(merged_dir / fp.name), merged)
            total += 1
            if (i + 1) % 500 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()
    print(f"[done] 共處理 {total} 張,合併圖存在 merged_v4/{{split}}/")


if __name__ == "__main__":
    main()
