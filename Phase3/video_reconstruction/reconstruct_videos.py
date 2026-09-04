"""
把 train/val/test、RGB/thermal 各自的 frame 圖片,依照原本所屬的 video_id
還原成一支一支的影片(有幾支 video 就還原成幾支,不合併成單一影片)。

規則:
- 每個 domain+split 各自的 coco.json 的 extra_info.video_id 決定一張圖屬於
  哪支影片(跟這整個 session 前面所有 RGB/thermal 配對邏輯用的同一個欄位)。
- 同一支影片裡的 frame,依照檔名裡的 frame 編號由小到大排序後寫進影片——
  frame 編號本來就是遞增的擷取序號,跳號(不連續)沒關係,只要保證輸出順序
  嚴格遞增,不會發生後面的 frame 編號比前面小的情況(用 sort 保證,不依賴
  檔名在檔案系統裡列出來的順序)。
- 每一幀用 cv2.putText 疊字幕「frame {實際編號}」,不是輸出時的第幾張
  (那樣會失去「這裡跳號了」的資訊)。

輸出:Phase3/video_reconstruction/{domain}_{split}/{video_id}.mp4
fps 用 5(這些是稀疏抽樣的 frame,不是連續影格,5 fps 純粹是給人看的
播放節奏,不是原始拍攝 frame rate)。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
OUT_ROOT = Path(__file__).parent
FPS = 5

DOMAIN_SPLIT_DIRS = {
    ("rgb", "train"): "images_rgb_train",
    ("rgb", "val"): "images_rgb_val",
    ("rgb", "test"): "video_rgb_test",
    ("thermal", "train"): "images_thermal_train",
    ("thermal", "val"): "images_thermal_val",
    ("thermal", "test"): "video_thermal_test",
}

FRAME_RE = re.compile(r"-frame-(\d+)-")


def build_video_groups(coco):
    """video_id -> [(frame_num, file_name), ...],依 frame_num 由小到大排序"""
    groups = defaultdict(list)
    for im in coco["images"]:
        vid = im["extra_info"]["video_id"]
        m = FRAME_RE.search(im["file_name"])
        if not m:
            continue
        groups[vid].append((int(m.group(1)), im["file_name"]))
    for vid in groups:
        groups[vid].sort(key=lambda t: t[0])  # 保證輸出嚴格遞增,不依賴檔案系統列出順序
    return groups


def write_video(img_dir, frames, out_path):
    first = cv2.imread(str(img_dir / frames[0][1]))
    if first is None:
        print(f"[warn] 讀不到第一張圖,跳過 {out_path.name}")
        return False
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

    prev_frame_num = -1
    for frame_num, file_name in frames:
        assert frame_num > prev_frame_num, f"frame 順序沒有嚴格遞增: {prev_frame_num} -> {frame_num}"
        prev_frame_num = frame_num

        img = cv2.imread(str(img_dir / file_name))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        cv2.putText(img, f"frame {frame_num}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"frame {frame_num}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return True


def main():
    total_videos = total_frames = 0
    for (domain, split), dirname in DOMAIN_SPLIT_DIRS.items():
        img_dir = ROOT / dirname  # coco.json 的 file_name 本身已經是 "data/xxx.jpg",這裡不要再疊一層 /data
        coco = json.load(open(ROOT / dirname / "coco.json"))
        groups = build_video_groups(coco)

        out_dir = OUT_ROOT / f"{domain}_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{domain}/{split}] {len(groups)} 支影片, {sum(len(v) for v in groups.values())} 張圖")
        n_ok = 0
        for vid, frames in groups.items():
            out_path = out_dir / f"{vid}.mp4"
            ok = write_video(img_dir, frames, out_path)
            if ok:
                n_ok += 1
                total_frames += len(frames)
        total_videos += n_ok
        print(f"  -> {n_ok}/{len(groups)} 支影片寫入 {out_dir}")

    print(f"\n[done] 總共 {total_videos} 支影片, {total_frames} 張圖片")


if __name__ == "__main__":
    main()
