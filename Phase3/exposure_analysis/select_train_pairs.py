"""
Day40:從 train dataset 抽 10 組 RGB<->thermal 配對樣本(seed=42),供人工
檢視曝光/背景相近問題的實際案例。

train 的 RGB<->thermal 配對不像 test 那麼乾淨(test 是同步雙鏡頭,見
Phase3/caption_fusion/build_test_pairs.py 的查證)。train 沿用 A4
(Phase3/rgb_validation/day38_frame_correspondence_audit.py)已經查證過的
Method A(index.json description 欄位內嵌 {"RGB": "<rgb_video_id>"},
74 支影片一一對應,是 Jack 背景認可的那個數字)——這裡直接重用同一段
regex/邏輯,不重新發明配對方法,也不改用 Method B(119 支)。

配對到影片後,同一對影片裡兩邊都有 frame(用檔名裡的 frame 編號比對,
不是 image id——image id 在 train/val 兩邊也是各自獨立編號,跟 test
驗證過的結論一樣不能直接比)的 frame 才收進候選清單,seed=42 抽 10 組。
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
TH_DIR = ROOT / "images_thermal_train"
RGB_DIR = ROOT / "images_rgb_train"
SEED = 42
N_SAMPLES = 10


def frame_num(fn):
    m = re.search(r"frame-(\d+)", fn)
    return int(m.group(1)) if m else None


def build_video_frame_index(coco):
    out = defaultdict(dict)
    for img in coco["images"]:
        vid = img["extra_info"]["video_id"]
        fn = frame_num(img["file_name"])
        if fn is not None:
            out[vid][fn] = img["file_name"]
    return out


def method_a_pairs(th_idx, rgb_video_ids):
    pattern = re.compile(r'\{"RGB":\s*"([a-zA-Z0-9]+)"\}')
    pairs = []
    for v in th_idx["videos"]:
        m = pattern.search(v.get("description", "") or "")
        if m:
            rgb_id = m.group(1)
            if rgb_id in rgb_video_ids:
                pairs.append((v["id"], rgb_id))
    return pairs


def main():
    th_idx = json.load(open(TH_DIR / "index.json"))
    rgb_idx = json.load(open(RGB_DIR / "index.json"))
    th_coco = json.load(open(TH_DIR / "coco.json"))
    rgb_coco = json.load(open(RGB_DIR / "coco.json"))

    rgb_video_ids = {v["id"] for v in rgb_idx["videos"]}
    video_pairs = method_a_pairs(th_idx, rgb_video_ids)
    print(f"[info] Method A 影片配對:{len(video_pairs)} 對(預期 74)")

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

    print(f"[info] {len(all_pairs)} 個 frame-level 配對(兩邊都有 frame,靠檔名 frame 編號比對,"
          f"不是 image id——train 這邊沒有 test 那種同步雙鏡頭保證,"
          f"純粹是「這個 frame 編號兩邊都有抽出來標註」)")

    rng = random.Random(SEED)
    sampled = rng.sample(all_pairs, N_SAMPLES)
    sampled.sort(key=lambda p: (p["thermal_video_id"], p["frame_num"]))

    with open(Path(__file__).parent / "train_pairs_sample.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_total_pairs": len(all_pairs), "sampled": sampled}, f,
                  ensure_ascii=False, indent=2)

    print(f"[done] 抽出 {len(sampled)} 組(seed={SEED}):")
    for p in sampled:
        print(f"  {p['thermal_file']}  <->  {p['rgb_file']}")


if __name__ == "__main__":
    main()
