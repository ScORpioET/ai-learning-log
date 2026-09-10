"""
Step 1: 從 Day40 的 all_frames_diff.json(15,153 張 frame,含 train/val/test 三個
split)挑樣本:diff_mean 最大 10 張 + 最小 10 張,共 20 張。

before 圖:thermal_dataset/images_rgb_{split}/data/{file_name}(train/val)
           thermal_dataset/video_rgb_test/data/{file_name}(test)
after  圖:Phase3/flare_removal/out_{split}/blend/{file_name}

不做任何篩選/去重(照字面「diff 最大/最小的 10 張 frame」),只在 summary
裡註記如果同一支影片佔了多張的情況。
"""
import json
from pathlib import Path

ROOT = Path.home() / "ai-transition-2026"
FLARE = ROOT / "Phase3" / "flare_removal"
TD = ROOT / "thermal_dataset"

SPLIT_DATA = {
    "train": TD / "images_rgb_train" / "data",
    "val": TD / "images_rgb_val" / "data",
    "test": TD / "video_rgb_test" / "data",
}
SPLIT_BLEND = {
    "train": FLARE / "out_train" / "blend",
    "val": FLARE / "out_val" / "blend",
    "test": FLARE / "out_test" / "blend",
}


def main():
    frames = json.load(open(FLARE / "all_frames_diff.json"))
    frames_sorted = sorted(frames, key=lambda x: -x["diff_mean"])
    top10 = frames_sorted[:10]
    bot10 = frames_sorted[-10:]

    samples = []
    for group, recs in (("high_diff", top10), ("low_diff", bot10)):
        for r in recs:
            before = SPLIT_DATA[r["split"]] / r["file_name"]
            after = SPLIT_BLEND[r["split"]] / r["file_name"]
            assert before.exists(), before
            assert after.exists(), after
            samples.append({
                "group": group,
                "split": r["split"],
                "file_name": r["file_name"],
                "video_id": r["video_id"],
                "diff_mean": r["diff_mean"],
                "diff_max": r["diff_max"],
                "before_path": str(before),
                "after_path": str(after),
            })

    out_path = Path(__file__).parent / "selected_samples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print("[high_diff video_id counts]", Counter(s["video_id"] for s in samples if s["group"] == "high_diff"))
    print("[low_diff  video_id counts]", Counter(s["video_id"] for s in samples if s["group"] == "low_diff"))
    print(f"[done] {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
