"""
給「肉眼觀測」用的臨時網站產生素材:對12張線A樣本,各輸出
1) 全圖縮圖(標出核心bbox跟往外擴張0.5倍的量測區域)
2) 量測區域本身的放大裁切圖(saturation_ratio/core_mean_brightness實際算的那塊)
存成 base64 json,供 HTML artifact 直接內嵌。
"""
import base64
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
OUT_JSON = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/gallery_data.json")


def to_b64_jpg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def resize_max_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, round(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    blob_data = {(r["split"], r["file_name"]): r for r in
                 json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))}
    line_a = list(csv.DictReader(open(HERE / "severity_line_a.csv", encoding="utf-8")))
    sat_stats = json.load(open(HERE / "saturation_brightness_stats_v1.json"))

    results = []
    for r in line_a:
        split, fn = r["split"], r["file_name"]
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

        # 全圖縮圖 + 標框(綠色=核心blob bbox,黃色虛線示意=量測區域)
        full_annot = orig.copy()
        cv2.rectangle(full_annot, (ox, oy), (ox + ow, oy + oh), (0, 220, 0), 3)
        cv2.rectangle(full_annot, (x0, y0), (x1, y1), (0, 200, 255), 3)
        full_thumb = resize_max_width(full_annot, 480)

        # 量測區域裁切放大圖(彩色版,實際算saturation時是轉灰階,這裡保留彩色方便肉眼看)
        crop = orig[y0:y1, x0:x1]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sat_mask = (crop_gray >= SAT_THRESH).astype(np.uint8) * 255
        crop_zoom = resize_max_width(crop, 420)

        # 飽和遮罩疊圖(紅色標出 >=250 的像素,方便肉眼對照 saturation_ratio 數字)
        overlay = crop.copy()
        overlay[crop_gray >= SAT_THRESH] = (0, 0, 255)
        blended = cv2.addWeighted(overlay, 0.45, crop, 0.55, 0)
        blended_zoom = resize_max_width(blended, 420)

        laplacian_bbox = float(r["laplacian_bbox"])
        # 找對應的 saturation/brightness 數值(重算一次,跟輸出腳本邏輯一致)
        sat_ratio = float(np.mean(crop_gray >= SAT_THRESH))
        mean_bright = float(np.mean(crop_gray))

        results.append({
            "split": split, "file_name": fn, "human_severity": r["human_severity"],
            "laplacian_bbox": laplacian_bbox,
            "saturation_ratio": round(sat_ratio, 4),
            "core_mean_brightness": round(mean_bright, 2),
            "full_thumb_b64": to_b64_jpg(full_thumb),
            "crop_zoom_b64": to_b64_jpg(crop_zoom),
            "blended_zoom_b64": to_b64_jpg(blended_zoom),
        })
        print(f"done: {fn}  sat={sat_ratio:.4f}  bright={mean_bright:.1f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)
    total_size_mb = OUT_JSON.stat().st_size / 1e6
    print(f"\n[done] {OUT_JSON}  ({total_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
