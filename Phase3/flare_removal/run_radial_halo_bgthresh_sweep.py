"""
把 run_radial_halo_all.py 的邏輯重跑兩次,分別用 BG_THRESH=80、100
(對照原本的 40),看光暈外緣門檻拉高之後,算出來的面積會不會比較貼近
肉眼判斷。輸出到各自獨立的資料夾/json,不覆蓋原本 BG_THRESH=40 的結果。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import radial_halo_profile as rhp  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}

CONFIGS = [80, 100]


def run_one_config(bg_thresh, data):
    rhp.BG_THRESH = bg_thresh
    tag = f"radial_halo_bg{bg_thresh}"
    results = []
    n_total = len(data)
    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        flare_path = HERE / split / "flare" / fn
        blend_path = HERE / split / "blend" / fn
        img = cv2.imread(str(flare_path))
        if img is None:
            continue

        blobs_with_halo = []
        blob_results = []
        for b in row["blobs"]:
            halo = rhp.radial_profile_halo(img, tuple(b["centroid"]))
            blobs_with_halo.append((b["bbox"], halo))
            blob_results.append({
                "area": b["area"], "bbox": b["bbox"],
                "halo_area_px": halo["halo_area_px"],
                "jump_width_median_px": halo["jump_width_median_px"],
            })

        blend = cv2.imread(str(blend_path))
        orig = cv2.imread(str(ORIG_DIRS[split] / fn))
        out_dir = HERE / split / tag
        out_dir.mkdir(exist_ok=True)
        if blend is not None:
            h, w = blend.shape[:2]
            scale = w / img.shape[1]
            flare_resized = cv2.resize(img, (w, h))
            annotated_resized = rhp.annotate(flare_resized, blobs_with_halo, scale=scale)
            panels = []
            if orig is not None:
                panels.append(cv2.resize(orig, (w, h)))
            panels.append(blend)
            panels.append(annotated_resized)
            cv2.imwrite(str(out_dir / fn), cv2.hconcat(panels))
        else:
            cv2.imwrite(str(out_dir / fn), rhp.annotate(img, blobs_with_halo))

        results.append({"split": split, "file_name": fn, "blobs": blob_results})
        if (i + 1) % 200 == 0:
            print(f"  [BG_THRESH={bg_thresh}] ...{i+1}/{n_total}", end="\r")
    print()

    with open(HERE / f"radial_halo_summary_bg{bg_thresh}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    areas = [b["halo_area_px"] for r in results for b in r["blobs"]]
    print(f"[BG_THRESH={bg_thresh}] halo_area_px: min={min(areas):.1f} p25={np.percentile(areas,25):.1f} "
          f"p50={np.percentile(areas,50):.1f} p75={np.percentile(areas,75):.1f} max={max(areas):.1f}")
    return areas


def main():
    data = json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))
    area_by_config = {}
    for bg_thresh in CONFIGS:
        area_by_config[bg_thresh] = run_one_config(bg_thresh, data)

    baseline = json.load(open(HERE / "radial_halo_summary.json"))
    baseline_areas = [b["halo_area_px"] for r in baseline for b in r["blobs"]]
    print(f"\n[BG_THRESH=40(原本的)] halo_area_px: min={min(baseline_areas):.1f} "
          f"p50={np.percentile(baseline_areas,50):.1f} max={max(baseline_areas):.1f}")
    for bg_thresh, areas in area_by_config.items():
        p50_baseline = np.percentile(baseline_areas, 50)
        p50_new = np.percentile(areas, 50)
        print(f"[BG_THRESH={bg_thresh}] p50 面積比 40 的版本縮小了 {100*(1-p50_new/p50_baseline):.1f}%")

    print("\n[done]")


if __name__ == "__main__":
    main()
