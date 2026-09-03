"""
Day39 收尾:合併三份既有輸出,產生最終 gallery HTML 需要的單一 JSON:
  - gallery_data.json          (thermal/rgb 原圖 base64、GT/gen caption)
  - fused_results.json         (rgb_priority / thermal_priority 兩版融合 caption,永遠都輸出)
  - exposure_analysis/sample_quality.json (系統建議標籤 + 理由)
用 thermal_file+rgb_file 當 key 對齊三份資料,不假設順序相同。
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
EXPOSURE = HERE.parent / "exposure_analysis"


def main():
    gallery = json.load(open(HERE / "gallery_data.json"))
    fused = json.load(open(HERE / "fused_results.json"))
    quality = json.load(open(EXPOSURE / "sample_quality.json"))

    fused_by_key = {(r["thermal_file"], r["rgb_file"]): r for r in fused}
    quality_by_key = {(r["thermal_file"], r["rgb_file"]): r for r in quality}

    out = []
    for g in gallery:
        key = (g["thermal_file"], g["rgb_file"])
        f = fused_by_key[key]
        q = quality_by_key[key]
        out.append({
            **g,
            "fused": f["fused"],
            "has_conflict": f["has_conflict"],
            "suggestion": q["suggestion"],
        })

    with open(HERE / "final_gallery_data.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[done] final_gallery_data.json written, {len(out)} samples")


if __name__ == "__main__":
    main()
