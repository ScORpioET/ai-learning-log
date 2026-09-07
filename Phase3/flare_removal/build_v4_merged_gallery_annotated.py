"""
把 merged_v4/ 的合併圖改成加上黑核標註版:flare 那一欄疊上每個 blob 的
框(黑核位置)跟面積(px)文字,原地覆蓋 merged_v4/。

沿用「先放大到跟 blend 同解析度、再把字畫在放大後畫布上」的做法(參考
radial_halo_profile.py 修過的教訓),避免文字被放大模糊。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

HERE = Path(__file__).parent


def annotate_scaled(flare_bgr_upscaled, blobs, scale):
    out = flare_bgr_upscaled.copy()
    for i, b in enumerate(blobs, 1):
        x, y, bw, bh = [round(v * scale) for v in b["bbox"]]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), max(1, round(2 * scale)))
        label = f"#{i} area={b['area']}px"
        font_scale = 0.55 * scale
        thick_fill = max(1, round(1 * scale))
        thick_outline = max(2, round(3 * scale))
        ty = max(y - round(8 * scale), round(14 * scale))
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick_outline, cv2.LINE_AA)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick_fill, cv2.LINE_AA)
    return out


def main():
    data = json.load(open(HERE / "black_blob_summary_v4_keep.json"))
    total = 0
    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        recolor_path = HERE / "flare_recolored_v4" / split / fn
        blend_path = HERE / "raw_intensity_run_v4" / split / "blend" / fn
        merged_path = HERE / "merged_v4" / split / fn

        flare = cv2.imread(str(recolor_path))
        blend = cv2.imread(str(blend_path))
        if flare is None or blend is None:
            continue

        h, w = blend.shape[:2]
        scale = w / flare.shape[1]
        flare_resized = cv2.resize(flare, (w, h))
        annotated_flare = annotate_scaled(flare_resized, row["blobs"], scale)

        merged = cv2.imread(str(merged_path))
        if merged is None:
            continue
        # merged 目前是 [orig | blend | flare] 三欄,寬度各 w,把最後一欄換成標註版
        left_two = merged[:, :2 * w]
        new_merged = cv2.hconcat([left_two, annotated_flare])
        cv2.imwrite(str(merged_path), new_merged)
        total += 1
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(data)}", end="\r")
    print(f"\n[done] 共 {total} 張加上黑核標註,原地覆蓋 merged_v4/")


if __name__ == "__main__":
    main()
