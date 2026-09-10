"""
把模糊帶 [LO, HI] 用 LO=0%、HI=100% 的相對量尺切成20個5%寬的區間,
每個區間中點找模糊帶141張裡最接近的一張,共20張,產生跟前兩輪
同款素材,供臨時網站逐張肉眼核對整個模糊帶的漸變過程。
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
OUT_JSON = Path("/tmp/claude-1000/-home-jack-ai-transition-2026/4e8dba1e-c515-4acd-b797-524f88c07dc5/scratchpad/sat_gallery/zone_ladder_data.json")


def to_b64_jpg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def resize_max_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, round(h * scale)), interpolation=cv2.INTER_AREA)


def select_ladder():
    pool = list(csv.DictReader(open(HERE / "saturation_classify_ambiguous_v1.csv")))
    pool = [dict(r, saturation_ratio=float(r["saturation_ratio"])) for r in pool]
    remaining = pool.copy()
    picks = []
    for k in range(20):
        center_pct = (k + 0.5) * 5
        target = LO + center_pct / 100 * (HI - LO)
        best = min(remaining, key=lambda r: abs(r["saturation_ratio"] - target))
        remaining.remove(best)
        actual_pct = (best["saturation_ratio"] - LO) / (HI - LO) * 100
        picks.append({**best, "target_pct": round(center_pct, 1), "actual_pct": round(actual_pct, 1)})
    return picks


def main():
    blob_data = {(r["split"], r["file_name"]): r for r in
                 json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))}
    picks = select_ladder()

    results = []
    for r in picks:
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
        full_thumb = resize_max_width(full_annot, 420)

        crop = orig[y0:y1, x0:x1]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_zoom = resize_max_width(crop, 360)

        overlay = crop.copy()
        overlay[crop_gray >= SAT_THRESH] = (0, 0, 255)
        blended = cv2.addWeighted(overlay, 0.45, crop, 0.55, 0)
        blended_zoom = resize_max_width(blended, 360)

        results.append({
            "split": split, "file_name": fn,
            "saturation_ratio": round(r["saturation_ratio"], 4),
            "target_pct": r["target_pct"], "actual_pct": r["actual_pct"],
            "full_thumb_b64": to_b64_jpg(full_thumb),
            "crop_zoom_b64": to_b64_jpg(crop_zoom),
            "blended_zoom_b64": to_b64_jpg(blended_zoom),
        })
        print(f"done: {r['target_pct']:5.1f}% -> {fn}  sat={r['saturation_ratio']:.4f} (actual {r['actual_pct']:.1f}%)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)
    print(f"\n[done] {OUT_JSON}  ({OUT_JSON.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
