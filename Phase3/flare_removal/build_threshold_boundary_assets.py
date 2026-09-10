"""
挑選最接近門檻的10張(高於HI最接近的5張 + 低於LO最接近的5張),
產生跟 saturation_inspector 同款素材(全圖+bbox、量測區域裁切、飽和遮罩疊圖),
供臨時網站肉眼核對邊界案例。
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
LO, HI = 0.1881, 0.2826
OUT_JSON = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/boundary_data.json")


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
    sev = sorted(csv.DictReader(open(HERE / "saturation_classify_severe_v1.csv")),
                 key=lambda r: float(r["saturation_ratio"]))[:5]
    notsev = sorted(csv.DictReader(open(HERE / "saturation_classify_not_severe_v1.csv")),
                     key=lambda r: float(r["saturation_ratio"]))[-5:]

    selected = [dict(r, side="above_HI") for r in sev] + [dict(r, side="below_LO") for r in notsev]

    results = []
    for r in selected:
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

        full_annot = orig.copy()
        cv2.rectangle(full_annot, (ox, oy), (ox + ow, oy + oh), (0, 220, 0), 3)
        cv2.rectangle(full_annot, (x0, y0), (x1, y1), (0, 200, 255), 3)
        full_thumb = resize_max_width(full_annot, 480)

        crop = orig[y0:y1, x0:x1]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_zoom = resize_max_width(crop, 420)

        overlay = crop.copy()
        overlay[crop_gray >= SAT_THRESH] = (0, 0, 255)
        blended = cv2.addWeighted(overlay, 0.45, crop, 0.55, 0)
        blended_zoom = resize_max_width(blended, 420)

        sat_ratio = float(np.mean(crop_gray >= SAT_THRESH))

        results.append({
            "split": split, "file_name": fn, "side": r["side"],
            "saturation_ratio": round(sat_ratio, 4),
            "full_thumb_b64": to_b64_jpg(full_thumb),
            "crop_zoom_b64": to_b64_jpg(crop_zoom),
            "blended_zoom_b64": to_b64_jpg(blended_zoom),
        })
        print(f"done: [{r['side']}] {fn}  sat={sat_ratio:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)
    print(f"\n[done] {OUT_JSON}  ({OUT_JSON.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
