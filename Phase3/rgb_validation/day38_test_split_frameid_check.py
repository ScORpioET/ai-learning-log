"""
檢查 thermal_dataset/video_rgb_test 跟 video_thermal_test 的 data 資料夾:
1. 圖片數量是否一致
2. 每張圖片是否都對應到 index.json 裡的一筆 frame 記錄
3. 依 (影片正規化檔名, frameIndex) 當作 frame id,比對兩邊是否 100% 對齊
   不對齊的檔案複製到 day38_test_split_frameid_mismatch/ 供人工檢查
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
RGB_DIR = ROOT / "video_rgb_test"
TH_DIR = ROOT / "video_thermal_test"
OUT_DIR = Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation" / "day38_test_split_frameid_mismatch"


def norm_video_name(name):
    base = re.sub(r"\.(avi|zip|mp4)$", "", name, flags=re.I)
    base = re.sub(r"_(visible|thermal|cam1|cam2)$", "", base, flags=re.I)
    return base.strip().lower()


def frame_num(fn):
    m = re.search(r"frame-(\d+)", fn)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 1. 圖片數量
# ---------------------------------------------------------------------------
rgb_files = sorted((RGB_DIR / "data").glob("*.jpg"))
th_files = sorted((TH_DIR / "data").glob("*.jpg"))
print(f"[1] RGB data 圖片數: {len(rgb_files)}")
print(f"[1] Thermal data 圖片數: {len(th_files)}")
print(f"[1] 數量一致: {len(rgb_files) == len(th_files)}")

# ---------------------------------------------------------------------------
# 2. 每張圖片是否對應到 index.json 裡的一筆 frame 記錄(完整性檢查)
# ---------------------------------------------------------------------------
rgb_idx = json.load(open(RGB_DIR / "index.json"))
th_idx = json.load(open(TH_DIR / "index.json"))

rgb_vid2name = {v["id"]: norm_video_name(v["filename"]) for v in rgb_idx["videos"]}
th_vid2name = {v["id"]: norm_video_name(v["filename"]) for v in th_idx["videos"]}

# index.json frames 用 datasetFrameId 當檔名裡最後一段 hash,拿來核對每個檔案
# 是否真的在 index.json 出現過(而不是資料夾裡多出來/少掉的孤兒檔案)
rgb_known_hashes = {f["datasetFrameId"] for f in rgb_idx["frames"]}
th_known_hashes = {f["datasetFrameId"] for f in th_idx["frames"]}


def file_hash(path):
    # video-<videoId>-frame-<idx>-<datasetFrameId>.jpg
    return path.stem.split("-")[-1]


rgb_orphan_files = [p for p in rgb_files if file_hash(p) not in rgb_known_hashes]
th_orphan_files = [p for p in th_files if file_hash(p) not in th_known_hashes]
print(f"[2] RGB 檔案裡找不到對應 index.json frame 記錄的孤兒檔案數: {len(rgb_orphan_files)}")
print(f"[2] Thermal 檔案裡找不到對應 index.json frame 記錄的孤兒檔案數: {len(th_orphan_files)}")

# 反向:index.json 裡的 frame 記錄,是否每筆都能在 data 資料夾找到對應檔案
rgb_file_hashes = {file_hash(p) for p in rgb_files}
th_file_hashes = {file_hash(p) for p in th_files}
rgb_missing_files = [f for f in rgb_idx["frames"] if f["datasetFrameId"] not in rgb_file_hashes]
th_missing_files = [f for f in th_idx["frames"] if f["datasetFrameId"] not in th_file_hashes]
print(f"[2] RGB index.json 裡有記錄、但 data 資料夾找不到檔案的筆數: {len(rgb_missing_files)}")
print(f"[2] Thermal index.json 裡有記錄、但 data 資料夾找不到檔案的筆數: {len(th_missing_files)}")

# ---------------------------------------------------------------------------
# 3. frame id 對齊比對:frame id 定義為 (正規化影片名稱, frameIndex)
# ---------------------------------------------------------------------------
def build_frame_id_map(idx, vid2name, data_dir):
    """(norm_video_name, frameIndex) -> file path"""
    out = {}
    for f in idx["frames"]:
        vid = f["videoMetadata"]["videoId"]
        name = vid2name.get(vid)
        fidx = f["videoMetadata"]["frameIndex"]
        if name is None:
            continue
        fn = f"video-{vid}-frame-{fidx:06d}-{f['datasetFrameId']}.jpg"
        out[(name, fidx)] = data_dir / "data" / fn
    return out


rgb_frame_ids = build_frame_id_map(rgb_idx, rgb_vid2name, RGB_DIR)
th_frame_ids = build_frame_id_map(th_idx, th_vid2name, TH_DIR)

rgb_id_set = set(rgb_frame_ids)
th_id_set = set(th_frame_ids)

both = rgb_id_set & th_id_set
rgb_only = rgb_id_set - th_id_set
th_only = th_id_set - rgb_id_set

print(f"[3] frame id 總數 -- RGB: {len(rgb_id_set)}, Thermal: {len(th_id_set)}")
print(f"[3] 兩邊都有的 frame id: {len(both)}")
print(f"[3] 只有 RGB 有的 frame id: {len(rgb_only)}")
print(f"[3] 只有 Thermal 有的 frame id: {len(th_only)}")
fully_aligned = (len(rgb_only) == 0 and len(th_only) == 0
                 and len(rgb_orphan_files) == 0 and len(th_orphan_files) == 0
                 and len(rgb_missing_files) == 0 and len(th_missing_files) == 0)
print(f"[3] 是否 100% 對齊: {fully_aligned}")

# ---------------------------------------------------------------------------
# 把沒對齊的都複製到同一個資料夾供人工檢查
# ---------------------------------------------------------------------------
if not fully_aligned:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "rgb_only").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "thermal_only").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rgb_orphan_files").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "thermal_orphan_files").mkdir(parents=True, exist_ok=True)

    for name, fidx in sorted(rgb_only):
        p = rgb_frame_ids[(name, fidx)]
        if p.exists():
            shutil.copy(p, OUT_DIR / "rgb_only" / p.name)
    for name, fidx in sorted(th_only):
        p = th_frame_ids[(name, fidx)]
        if p.exists():
            shutil.copy(p, OUT_DIR / "thermal_only" / p.name)
    for p in rgb_orphan_files:
        shutil.copy(p, OUT_DIR / "rgb_orphan_files" / p.name)
    for p in th_orphan_files:
        shutil.copy(p, OUT_DIR / "thermal_orphan_files" / p.name)

    manifest = {
        "rgb_image_count": len(rgb_files),
        "thermal_image_count": len(th_files),
        "counts_equal": len(rgb_files) == len(th_files),
        "rgb_only_frame_ids": [f"{n}#{i}" for n, i in sorted(rgb_only)],
        "thermal_only_frame_ids": [f"{n}#{i}" for n, i in sorted(th_only)],
        "rgb_orphan_files": [p.name for p in rgb_orphan_files],
        "thermal_orphan_files": [p.name for p in th_orphan_files],
        "rgb_missing_files_in_index": [f["datasetFrameId"] for f in rgb_missing_files],
        "thermal_missing_files_in_index": [f["datasetFrameId"] for f in th_missing_files],
    }
    json.dump(manifest, open(OUT_DIR / "manifest.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n未對齊的檔案已複製到: {OUT_DIR}")
else:
    print("\n100% 對齊,沒有產生檢查資料夾。")
