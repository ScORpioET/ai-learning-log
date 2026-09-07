"""
對 black_blob_summary.json 裡符合特徵的 1,251 張圖(每張圖裡的每個黑
blob),套用 radial_halo_profile.py 的 radial profile 分析,算出:
- halo_area_px(光暈影響面積)
- jump_width_median_px(跳變寬度中位數,越小代表從黑到亮變化越快)

標註後(blob 框 + 光暈多邊形 + 兩個數字)再跟「原圖 + blend」三張左右合併,
輸出到 {split}/radial_halo/,並把逐 blob 的結果彙整成 radial_halo_summary.json。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

from radial_halo_profile import radial_profile_halo, annotate  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}


def main():
    data = json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))
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
            halo = radial_profile_halo(img, tuple(b["centroid"]))
            blobs_with_halo.append((b["bbox"], halo))
            blob_results.append({
                "area": b["area"], "bbox": b["bbox"],
                "halo_area_px": halo["halo_area_px"],
                "jump_width_median_px": halo["jump_width_median_px"],
            })

        blend = cv2.imread(str(blend_path))
        orig_path = ORIG_DIRS[split] / fn
        orig = cv2.imread(str(orig_path))
        out_dir = HERE / split / "radial_halo"
        out_dir.mkdir(exist_ok=True)
        if blend is not None:
            h, w = blend.shape[:2]
            # 先把 flare 圖放大到跟 blend 一樣的解析度,再把字畫在放大後的畫布上
            # (而不是先在小圖上畫字、畫完才整張放大)——避免文字被放大模糊、
            # 糊成一團看起來像疊字。
            scale = w / img.shape[1]
            flare_resized = cv2.resize(img, (w, h))
            annotated_resized = annotate(flare_resized, blobs_with_halo, scale=scale)
            panels = []
            if orig is not None:
                panels.append(cv2.resize(orig, (w, h)))
            panels.append(blend)
            panels.append(annotated_resized)
            merged = cv2.hconcat(panels)
            cv2.imwrite(str(out_dir / fn), merged)
        else:
            annotated = annotate(img, blobs_with_halo)
            cv2.imwrite(str(out_dir / fn), annotated)

        results.append({"split": split, "file_name": fn, "blobs": blob_results})
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n_total}", end="\r")

    with open(HERE / "radial_halo_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    all_widths = [b["jump_width_median_px"] for r in results for b in r["blobs"] if b["jump_width_median_px"] is not None]
    all_areas = [b["halo_area_px"] for r in results for b in r["blobs"]]
    import numpy as np
    print(f"\n[done] 共處理 {len(results)} 張圖、{len(all_widths)} 個 blob")
    print(f"jump_width_median_px: min={min(all_widths):.1f} p25={np.percentile(all_widths,25):.1f} "
          f"p50={np.percentile(all_widths,50):.1f} p75={np.percentile(all_widths,75):.1f} max={max(all_widths):.1f}")
    print(f"halo_area_px: min={min(all_areas):.1f} p50={np.percentile(all_areas,50):.1f} max={max(all_areas):.1f}")


if __name__ == "__main__":
    main()
