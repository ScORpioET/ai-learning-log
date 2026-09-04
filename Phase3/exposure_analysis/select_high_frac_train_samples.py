"""
Day41:對整張 RGB train 圖片(不是逐 bbox)算 high_frac(px>=240 佔比),
篩出 high_frac > 0.08 的圖片,用 seed=42 抽 10 張給 Jack 看(校準
high_frac 門檻用)。

重用 compute_rgb_exposure.py 的 rgb_luminance() / high_low_frac(),
不重新刻公式 —— 跟 calibrate_exposure_ui.py 同一套邏輯。

這是新的 whole-image 掃描(不是既有 rgb_exposure_results.jsonl 那份逐
bbox 的資料,那份只有 saturated_frac@250,沒有 high_frac@240,而且是
bbox 尺度,不是整張圖),所以整個 RGB train(10,319 張)重新掃一次。
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from compute_rgb_exposure import rgb_luminance, high_low_frac

RGB_TRAIN_DIR = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_rgb_train"
OUT_PATH = Path(__file__).parent / "high_frac_train_scan.jsonl"
HIGH_FRAC_MIN = 0.08
SEED = 42
N_SAMPLES = 10


def main():
    img_paths = sorted((RGB_TRAIN_DIR / "data").glob("*.jpg"))
    print(f"[info] {len(img_paths)} RGB train images")

    out = open(OUT_PATH, "w", encoding="utf-8")
    n_done = 0
    for p in img_paths:
        try:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        except Exception as e:
            print(f"[warn] 讀圖失敗 {p}: {e}")
            continue
        lum = rgb_luminance(arr)
        high_frac, low_frac = high_low_frac(lum)
        out.write(json.dumps({
            "file_name": f"data/{p.name}", "high_frac": round(high_frac, 5), "low_frac": round(low_frac, 5),
        }) + "\n")
        n_done += 1
        if n_done % 1000 == 0:
            print(f"  ...{n_done}/{len(img_paths)}", end="\r")
    out.close()
    print(f"\n[done] {n_done} images scanned -> {OUT_PATH}")

    records = [json.loads(l) for l in open(OUT_PATH, encoding="utf-8")]
    candidates = [r for r in records if r["high_frac"] > HIGH_FRAC_MIN]
    print(f"[info] high_frac > {HIGH_FRAC_MIN}: {len(candidates)}/{len(records)} "
          f"({100*len(candidates)/len(records):.2f}%)")

    rng = random.Random(SEED)
    sampled = rng.sample(candidates, min(N_SAMPLES, len(candidates)))
    sampled.sort(key=lambda r: -r["high_frac"])

    print(f"\n[info] seed={SEED} 抽出的 {len(sampled)} 張:")
    for r in sampled:
        print(f"  {r['file_name']}  high_frac={r['high_frac']}")

    with open(Path(__file__).parent / "high_frac_train_samples.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "high_frac_min": HIGH_FRAC_MIN, "n_candidates": len(candidates),
                    "n_total": len(records), "sampled": sampled}, f, ensure_ascii=False, indent=2)
    print("\n[done] high_frac_train_samples.json written")


if __name__ == "__main__":
    main()
