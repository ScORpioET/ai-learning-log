"""
對 out_train/out_val/out_test 三個 split 的 Flare7K++ 輸出(blend/),逐張圖跟
原圖算 diff(重用 build_diff_gallery.py 的 diff_heatmap,不重寫),存每張的
diff_mean/diff_max,再依 coco.json 的 video_id 分組,把同一支影片所有 frame
的 diff_mean 取平均、diff_max 取最大值,當作「這支影片」的代表值,最後畫
histogram。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_diff_gallery import diff_heatmap  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent

SPLITS = {
    "train": (ROOT / "images_rgb_train", HERE / "out_train"),
    "val": (ROOT / "images_rgb_val", HERE / "out_val"),
    "test": (ROOT / "video_rgb_test", HERE / "out_test"),
}
FRAME_RE = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)-")


def main():
    frame_records = []
    for split, (data_root, out_root) in SPLITS.items():
        coco = json.load(open(data_root / "coco.json"))
        vid_by_file = {im["file_name"].split("/")[-1]: im["extra_info"]["video_id"] for im in coco["images"]}

        blend_dir = out_root / "blend"
        blend_files = sorted(blend_dir.glob("*.jpg"))
        print(f"[{split}] {len(blend_files)} 張模型輸出")
        for i, bpath in enumerate(blend_files):
            orig_path = data_root / "data" / bpath.name
            if not orig_path.exists():
                continue
            orig = Image.open(orig_path)
            blend = Image.open(bpath)
            _, diff_mean, diff_max = diff_heatmap(orig, blend)
            vid = vid_by_file.get(bpath.name)
            frame_records.append({
                "split": split, "file_name": bpath.name, "video_id": vid,
                "diff_mean": diff_mean, "diff_max": diff_max,
            })
            if (i + 1) % 500 == 0:
                print(f"  ...{split} {i+1}/{len(blend_files)}", end="\r")
        print()

    with open(HERE / "all_frames_diff.json", "w", encoding="utf-8") as f:
        json.dump(frame_records, f, ensure_ascii=False)
    print(f"[info] 共 {len(frame_records)} 張 frame 算完 diff")

    # 依 video 聚合
    by_video = defaultdict(list)
    for r in frame_records:
        by_video[(r["split"], r["video_id"])].append(r)

    video_records = []
    for (split, vid), recs in by_video.items():
        means = [r["diff_mean"] for r in recs]
        maxes = [r["diff_max"] for r in recs]
        video_records.append({
            "split": split, "video_id": vid, "n_frames": len(recs),
            "diff_mean": float(np.mean(means)), "diff_max": float(np.max(maxes)),
        })
    video_records.sort(key=lambda r: -r["diff_mean"])
    with open(HERE / "all_videos_diff.json", "w", encoding="utf-8") as f:
        json.dump(video_records, f, ensure_ascii=False, indent=2)
    print(f"[info] {len(video_records)} 支影片聚合完成")

    # histogram
    means = [r["diff_mean"] for r in video_records]
    maxes = [r["diff_max"] for r in video_records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, vals, title, color in (
        (axes[0], means, "Per-video diff_mean (avg over frames)", "#c2703a"),
        (axes[1], maxes, "Per-video diff_max (max over frames)", "#5b4b9c"),
    ):
        ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="none")
        for p in (10, 25, 50, 75, 90):
            v = np.percentile(vals, p)
            ax.axvline(v, color="#333", linestyle="--", linewidth=0.8)
            ax.text(v, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, f"p{p}={v:.1f}",
                    rotation=90, fontsize=7, va="top", ha="right", color="#333")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("n videos", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "video_diff_histograms.png", dpi=140)
    plt.close(fig)

    print("\n[summary]")
    print(f"n videos = {len(video_records)}")
    print(f"diff_mean: min={min(means):.1f} p50={np.percentile(means,50):.1f} max={max(means):.1f}")
    print(f"diff_max : min={min(maxes):.1f} p50={np.percentile(maxes,50):.1f} max={max(maxes):.1f}")
    print("\n[top 10 by diff_mean]")
    for r in video_records[:10]:
        print(f"  [{r['split']:5s}] {r['video_id']}  diff_mean={r['diff_mean']:.1f}  diff_max={r['diff_max']:.1f}  n_frames={r['n_frames']}")
    print("\n[bottom 10 by diff_mean]")
    for r in video_records[-10:]:
        print(f"  [{r['split']:5s}] {r['video_id']}  diff_mean={r['diff_mean']:.1f}  diff_max={r['diff_max']:.1f}  n_frames={r['n_frames']}")

    print("\n[done] video_diff_histograms.png + all_videos_diff.json written")


if __name__ == "__main__":
    main()
