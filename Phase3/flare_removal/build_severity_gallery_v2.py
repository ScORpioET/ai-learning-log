"""
把 severity_scores_v2.json(934張)畫成可以直接用眼睛看的圖庫:
- 原圖上畫綠框標出曝光核心 bbox
- 下方白色 stat strip 標出 core_area / halo_area / laplacian_bbox /
  downstream_confidence_avg / _min / _drop / new_severity_rank
- 檔名前綴用 new_severity_rank 由高到低排序,方便照嚴重度瀏覽

輸出到 severity_gallery_v2/{split}/rank-XXXX_原檔名
"""
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
OUT_ROOT = HERE / "severity_gallery_v2"


def fmt(v, unit=""):
    return f"{v}{unit}" if v is not None else "N/A"


def build_stat_strip(width, row, line_height=44, font_scale=0.9):
    scale = max(width / 1400, 1.0)
    font_scale = font_scale * scale
    line_height = round(line_height * scale)
    thickness = max(1, round(font_scale * 1.6))
    lines = [
        f"new_severity_rank={row['new_severity_rank']}   halo_area_rank={row['halo_area_rank']}",
        f"core_area={row['core_area_px']}px   halo_area={fmt(row['halo_area_px'], 'px')}   laplacian_bbox={row['laplacian_bbox']}",
        f"downstream_conf_avg={fmt(row['downstream_confidence_avg'])}   "
        f"conf_min={fmt(row['downstream_confidence_min'])}   conf_drop={fmt(row['downstream_confidence_drop'])}",
    ]
    strip = np.full((line_height * len(lines) + round(line_height * 0.4), width, 3), 255, dtype=np.uint8)
    for i, text in enumerate(lines):
        ty = round(line_height * 0.2) + (i + 1) * line_height - round(line_height * 0.3)
        cv2.putText(strip, text, (round(line_height * 0.3), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return strip


def main():
    data = json.load(open(HERE / "severity_scores_v2.json"))
    data_ranked = [r for r in data if r["new_severity_rank"] is not None]
    data_ranked.sort(key=lambda r: -r["new_severity_rank"])

    n_total = len(data_ranked)
    n_written = 0
    for i, row in enumerate(data_ranked, 1):
        split, fn = row["split"], row["file_name"]
        orig_path = ORIG_DIRS[split] / fn
        flare_path = FLARE_DIRS[split] / fn
        orig = cv2.imread(str(orig_path))
        flare = cv2.imread(str(flare_path))
        if orig is None or flare is None:
            continue
        scale = orig.shape[0] / flare.shape[0]
        bx, by_, bw, bh = row["bbox"]
        ox, oy, ow, oh = [round(v * scale) for v in (bx, by_, bw, bh)]

        canvas = orig.copy()
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
        tag = f"rank#{i}/{n_total}"
        cv2.putText(canvas, tag, (ox, max(oy - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, tag, (ox, max(oy - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        strip = build_stat_strip(canvas.shape[1], row)
        merged = cv2.vconcat([canvas, strip])

        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"rank{i:04d}_{fn}"
        cv2.imwrite(str(out_dir / out_name), merged)
        n_written += 1
        if i % 100 == 0:
            print(f"  ...{i}/{n_total}", end="\r")

    print(f"\n[done] {n_written}/{n_total} 張圖片寫入 {OUT_ROOT}")


if __name__ == "__main__":
    main()
