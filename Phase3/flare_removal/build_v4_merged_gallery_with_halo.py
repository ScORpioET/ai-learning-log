"""
在黑核標註的基礎上,加上光暈(radial profile halo)的標記——之前只加了
黑核框跟面積,沒有加光暈範圍,這次補上。邏輯本身沒有變,沿用
radial_halo_profile.py 現有的 radial_profile_halo()/annotate() (v3 用過、
文字重疊 bug 已修好的版本),只是這是第一次套用到 1,994 張的 v4 最終樣本。

原地覆蓋 merged_v4/(flare 那一欄改成黑核框+光暈多邊形+面積+跳變寬度都有
標註的版本),同時輸出 radial_halo_summary_v4.json 存逐 blob 的 halo 數值。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

from radial_halo_profile import radial_profile_halo, annotate, build_stat_strip  # noqa: E402

HERE = Path(__file__).parent


def main():
    data = json.load(open(HERE / "black_blob_summary_v4_keep.json"))
    halo_results = []
    total = 0
    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        recolor_path = HERE / "flare_recolored_v4" / split / fn
        blend_path = HERE / "raw_intensity_run_v4" / split / "blend" / fn
        merged_path = HERE / "merged_v4" / split / fn

        flare = cv2.imread(str(recolor_path))
        blend = cv2.imread(str(blend_path))
        merged = cv2.imread(str(merged_path))
        if flare is None or blend is None or merged is None:
            continue

        blobs_with_halo = []
        blob_halo_records = []
        for b in row["blobs"]:
            halo = radial_profile_halo(flare, tuple(b["centroid"]))
            halo["core_area_px"] = b["area"]
            blobs_with_halo.append((b["bbox"], halo))
            blob_halo_records.append({
                "area": b["area"], "bbox": b["bbox"],
                "halo_area_px": halo["halo_area_px"],
                "jump_width_median_px": halo["jump_width_median_px"],
            })
        halo_results.append({"split": split, "file_name": fn, "blobs": blob_halo_records})

        h, w = blend.shape[:2]
        scale = w / flare.shape[1]
        flare_resized = cv2.resize(flare, (w, h))
        # 框內只畫小小的 #i 編號(不會超出畫布),完整數值改列在下方白色區域,
        # 避免黑核貼近圖片邊緣時文字被裁掉的問題
        annotated_flare = annotate(flare_resized, blobs_with_halo, scale=scale, draw_labels=False)

        left_two = merged[:h, :2 * w]  # 用 h 限制列數,避免如果 merged 已經帶有下方白色區域時被誤抓進來
        row = cv2.hconcat([left_two, annotated_flare])
        strip = build_stat_strip(row.shape[1], blobs_with_halo)
        new_merged = cv2.vconcat([row, strip])
        cv2.imwrite(str(merged_path), new_merged)
        total += 1
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(data)}", end="\r")
    print()

    with open(HERE / "radial_halo_summary_v4.json", "w", encoding="utf-8") as f:
        json.dump(halo_results, f, ensure_ascii=False, indent=2)

    print(f"[done] 共 {total} 張補上光暈標註,原地覆蓋 merged_v4/,radial_halo_summary_v4.json 已寫入")


if __name__ == "__main__":
    main()
