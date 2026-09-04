"""
Day39 caption_fusion/build_test_pairs.py 的 load_video_pairs()/build_frame_index()
拿來重用,建出「全部」3749 組 RGB<->thermal frame 對應(build_test_pairs.py
原本只存 n_total_pairs 這個數字跟抽樣的 10 組,沒有存完整清單)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion"))
from build_test_pairs import load_video_pairs, build_frame_index, ROOT  # noqa: E402

HERE = Path(__file__).parent


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

    print(f"[info] {len(all_pairs)} 組全量 frame-level 配對")
    with open(HERE / "full_test_pairs.json", "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False)
    print(f"[done] full_test_pairs.json written")


if __name__ == "__main__":
    main()
