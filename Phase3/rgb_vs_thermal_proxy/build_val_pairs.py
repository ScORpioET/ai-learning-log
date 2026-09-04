"""
Day42:val set(有 GT)的 RGB<->thermal frame-level 配對。

重用 Day40 select_train_pairs.py 的 Method A 邏輯(index.json description
內嵌 {"RGB": "<id>"} JSON blob 配對影片,frame 檔名數字比對配對 frame)
不重寫,只是把目錄從 images_*_train 換成 images_*_val。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "exposure_analysis"))
from select_train_pairs import frame_num, build_video_frame_index, method_a_pairs  # noqa: E402

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
TH_DIR = ROOT / "images_thermal_val"
RGB_DIR = ROOT / "images_rgb_val"
HERE = Path(__file__).parent


def main():
    th_idx = json.load(open(TH_DIR / "index.json"))
    rgb_idx = json.load(open(RGB_DIR / "index.json"))
    th_coco = json.load(open(TH_DIR / "coco.json"))
    rgb_coco = json.load(open(RGB_DIR / "coco.json"))

    rgb_video_ids = {v["id"] for v in rgb_idx["videos"]}
    video_pairs = method_a_pairs(th_idx, rgb_video_ids)
    print(f"[info] Method A 影片配對:{len(video_pairs)} / {len(th_idx['videos'])} 支 thermal 影片")

    th_frames = build_video_frame_index(th_coco)
    rgb_frames = build_video_frame_index(rgb_coco)

    all_pairs = []
    for tvid, rvid in video_pairs:
        t_by_frame = th_frames.get(tvid, {})
        r_by_frame = rgb_frames.get(rvid, {})
        both = sorted(set(t_by_frame) & set(r_by_frame))
        for fn in both:
            all_pairs.append({
                "thermal_video_id": tvid, "rgb_video_id": rvid, "frame_num": fn,
                "thermal_file": t_by_frame[fn], "rgb_file": r_by_frame[fn],
            })

    print(f"[info] {len(all_pairs)} 個 frame-level 配對")
    with open(HERE / "val_pairs.json", "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print("[done] val_pairs.json written")


if __name__ == "__main__":
    main()
