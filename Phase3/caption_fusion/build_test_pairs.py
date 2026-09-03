"""
Day39:從 thermal/RGB 的 test split(video_thermal_test / video_rgb_test)
建立「同一個真實世界 frame」的 thermal<->RGB 配對清單。

查程式碼/資料確認的事實:
- test split 兩邊各自 coco.json 的 image `id`(0..3748)只是各自獨立的
  流水號,不是共用的 frame_id——直接用 id 對應是錯的(抽樣驗證過,兩邊
  video_id 常常對不上)。
- 真正可信的配對路徑是兩層:
  1. video 層級:index.json 的 `videos[].filename` 前綴數字兩邊一致
     (例:"1636345164_thermal.zip" <-> "1636345164_visible.zip"),
     description 文字也逐字對得上,8 支影片全部一一對應,沒有例外。
  2. frame 層級:同一對影片裡,`videoMetadata.frameIndex` 兩邊 100% 對得
     上(每一對的 t_frames == r_frames == both_idx_match,見查證輸出),
     這是同步雙鏡頭拍攝(DualCapture)的資料,不是 train/val 那種需要
     容忍度的粗配對。

輸出:test_pairs.json,含全部配對好的 (thermal_file_name, rgb_file_name,
video pair, frameIndex) 清單,以及用 seed=42 抽出的 10 組子集合。
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
SEED = 42
N_SAMPLES = 10


def load_video_pairs():
    it = json.load(open(ROOT / "video_thermal_test" / "index.json"))
    ir = json.load(open(ROOT / "video_rgb_test" / "index.json"))

    def num_prefix(fn):
        m = re.match(r"(\d+)_", fn)
        return m.group(1) if m else None

    t_map = {num_prefix(v["filename"]): v["id"] for v in it["videos"]}
    r_map = {num_prefix(v["filename"]): v["id"] for v in ir["videos"]}
    pairs = {t_map[k]: r_map[k] for k in t_map if k in r_map and t_map[k] and r_map[k]}
    assert len(pairs) == len(it["videos"]) == len(ir["videos"]), \
        "預期 8 支影片兩邊一一對應,數量對不上要停下來查,不能悄悄跳過"
    return pairs


def build_frame_index(coco):
    """(video_id, frameIndex) -> file_name"""
    idx = {}
    for im in coco["images"]:
        vid = im["extra_info"]["video_id"]
        m = re.search(r"-frame-(\d+)-", im["file_name"])
        frame_idx = int(m.group(1))
        idx[(vid, frame_idx)] = im["file_name"]
    return idx


def main():
    video_pairs = load_video_pairs()

    ct = json.load(open(ROOT / "video_thermal_test" / "coco.json"))
    cr = json.load(open(ROOT / "video_rgb_test" / "coco.json"))
    t_idx = build_frame_index(ct)
    r_idx = build_frame_index(cr)

    all_pairs = []
    for tvid, rvid in video_pairs.items():
        t_frame_ids = {fi for (vid, fi) in t_idx if vid == tvid}
        r_frame_ids = {fi for (vid, fi) in r_idx if vid == rvid}
        both = sorted(t_frame_ids & r_frame_ids)
        for fi in both:
            all_pairs.append({
                "thermal_video_id": tvid, "rgb_video_id": rvid, "frame_index": fi,
                "thermal_file": t_idx[(tvid, fi)], "rgb_file": r_idx[(rvid, fi)],
            })

    print(f"[info] {len(video_pairs)} 支影片配對,{len(all_pairs)} 個 frame-level 配對(兩邊都有)")

    rng = random.Random(SEED)
    sampled = rng.sample(all_pairs, N_SAMPLES)
    sampled.sort(key=lambda p: (p["thermal_video_id"], p["frame_index"]))

    out = {"seed": SEED, "n_total_pairs": len(all_pairs), "video_pairs": video_pairs, "sampled": sampled}
    with open(Path(__file__).parent / "test_pairs.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[done] 抽出 {len(sampled)} 組(seed={SEED}):")
    for p in sampled:
        print(f"  {p['thermal_file']}  <->  {p['rgb_file']}")


if __name__ == "__main__":
    main()
