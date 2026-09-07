"""
v4 補強:回頭檢查發現「934 張候選重跑新門檻確認沒被踢出去」這個驗證方式,
測不出「有沒有新的候選應該被納入」——因為新的相對門檻在寬 576px 的圖片上
會降到約 422px(比固定 500px 低 16%),理論上應該有一批原本卡在 422-500px
之間被階段2排除的圖片,現在該被重新納入,但這些圖不在舊的 934 張候選裡,
舊驗證方式根本看不到它們。

這次對全部 15,153 張 flare 圖片,同時算出:
- fixed_pass:用固定 BLACK_THRESH=30/MIN_AREA=500/RING_DILATE=15/
  BRIGHT_RING_THRESH=80(v3 定案但從沒真的對全量重跑過的版本)
- relative_pass:用 v4 現在的相對門檻(find_black_blobs.py 目前的邏輯)

只跑一次黑塊偵測(不套 MIN_AREA 過濾),把每個黑塊的面積、邊界狀態、
兩種 RING_DILATE 半徑各自算出的外環亮度都記錄下來,事後才能同時判斷
fixed_pass/relative_pass,不用跑兩次 connectedComponents。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

HERE = Path(__file__).parent
SPLITS = ["out_train", "out_val", "out_test"]

BLACK_THRESH = 30
BRIGHT_RING_THRESH = 80
FIXED_MIN_AREA = 500
FIXED_RING_DILATE = 15
REF_W, REF_H = 682, 512
MIN_AREA_RATIO = 500 / (REF_W * REF_H)
RING_DILATE_RATIO = 15 / np.hypot(REF_W, REF_H)


def ring_brightness(maxchan, comp_mask, dilate_px, w, h):
    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(comp_mask, kernel)
    ring = (dilated > 0) & (comp_mask == 0)
    if ring.sum() == 0:
        return None
    return float(maxchan[ring].mean())


def all_candidates(img_bgr):
    """回傳這張圖裡所有「不貼邊界」的黑色連通區塊(不套 MIN_AREA 篩選),
    每個都算出面積、bbox、centroid,以及固定 dilate=15 / 相對 dilate 兩種
    外環亮度。"""
    h, w = img_bgr.shape[:2]
    rel_dilate_px = max(1, round(RING_DILATE_RATIO * np.hypot(w, h)))
    rel_min_area = MIN_AREA_RATIO * (w * h)

    black_mask = cv2.inRange(img_bgr, (0, 0, 0), (BLACK_THRESH, BLACK_THRESH, BLACK_THRESH))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    maxchan = img_bgr.astype(np.float32).max(axis=2)

    candidates = []
    for lbl in range(1, n_labels):
        x, y, bw, bh, area = stats[lbl]
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        comp_mask = (labels == lbl).astype(np.uint8)
        rb_fixed = ring_brightness(maxchan, comp_mask, FIXED_RING_DILATE, w, h)
        rb_rel = ring_brightness(maxchan, comp_mask, rel_dilate_px, w, h) if rel_dilate_px != FIXED_RING_DILATE else rb_fixed
        candidates.append({
            "area": int(area), "bbox": [int(x), int(y), int(bw), int(bh)],
            "centroid": [round(float(centroids[lbl][0]), 1), round(float(centroids[lbl][1]), 1)],
            "ring_brightness_fixed": round(rb_fixed, 1) if rb_fixed is not None else None,
            "ring_brightness_rel": round(rb_rel, 1) if rb_rel is not None else None,
            "rel_min_area": round(rel_min_area, 1),
        })
    return candidates


def main():
    fixed_records = []
    relative_records = []
    all_candidates_by_image = {}
    n_total = 0

    for split in SPLITS:
        flare_dir = HERE / split / "flare"
        files = sorted(flare_dir.glob("*.jpg"))
        print(f"[{split}] {len(files)} 張")
        for i, fp in enumerate(files):
            n_total += 1
            img = cv2.imread(str(fp))
            if img is None:
                continue
            cands = all_candidates(img)
            key = (split, fp.name)
            all_candidates_by_image[key] = cands

            fixed_blobs = [c for c in cands if c["area"] >= FIXED_MIN_AREA and c["ring_brightness_fixed"] is not None and c["ring_brightness_fixed"] >= BRIGHT_RING_THRESH]
            rel_blobs = [c for c in cands if c["area"] >= c["rel_min_area"] and c["ring_brightness_rel"] is not None and c["ring_brightness_rel"] >= BRIGHT_RING_THRESH]

            if fixed_blobs:
                fixed_records.append({"split": split, "file_name": fp.name, "n_blobs": len(fixed_blobs), "blobs": fixed_blobs})
            if rel_blobs:
                relative_records.append({"split": split, "file_name": fp.name, "n_blobs": len(rel_blobs),
                                          "blobs": [{"area": c["area"], "bbox": c["bbox"], "centroid": c["centroid"],
                                                     "ring_brightness": c["ring_brightness_rel"]} for c in rel_blobs]})
            if (i + 1) % 1000 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()

    print(f"[info] 共檢查 {n_total} 張")
    print(f"[info] fixed_pass(固定門檻,全量重跑): {len(fixed_records)} 張")
    print(f"[info] relative_pass(v4相對門檻): {len(relative_records)} 張")

    fixed_keys = {(r["split"], r["file_name"]) for r in fixed_records}
    relative_keys = {(r["split"], r["file_name"]) for r in relative_records}

    added = sorted(relative_keys - fixed_keys)
    removed = sorted(fixed_keys - relative_keys)
    unchanged = sorted(fixed_keys & relative_keys)

    print(f"\n[比對結果]")
    print(f"新增(relative有,fixed沒有): {len(added)} 張")
    print(f"消失(fixed有,relative沒有): {len(removed)} 張")
    print(f"維持不變(兩邊都有): {len(unchanged)} 張")

    with open(HERE / "black_blob_summary_v4_raw.json", "w", encoding="utf-8") as f:
        json.dump(relative_records, f, ensure_ascii=False, indent=2)
    print("\n[done] black_blob_summary_v4_raw.json 已寫入(v4 相對門檻對全部 15,153 張的完整階段2結果)")

    diff_report = {
        "n_total": n_total,
        "n_fixed_pass": len(fixed_records),
        "n_relative_pass": len(relative_records),
        "n_added": len(added),
        "n_removed": len(removed),
        "n_unchanged": len(unchanged),
        "added_files": [{"split": s, "file_name": f} for s, f in added],
        "removed_files": [{"split": s, "file_name": f} for s, f in removed],
    }

    if len(added) == 0:
        # 附上「fixed 排除掉的圖裡,面積最接近 500 的前20筆」佐證
        near_boundary = []
        for (split, fn), cands in all_candidates_by_image.items():
            if (split, fn) in fixed_keys:
                continue
            # 這張圖在 fixed 版本下沒通過,找它最大的候選黑塊面積(不論是否過亮度門檻)
            if not cands:
                continue
            best = max(cands, key=lambda c: c["area"])
            near_boundary.append({"split": split, "file_name": fn, "max_black_area": best["area"],
                                   "ring_brightness_fixed": best["ring_brightness_fixed"]})
        near_boundary.sort(key=lambda r: -r["max_black_area"])
        diff_report["near_boundary_excluded_top20"] = near_boundary[:20]
        print("\n=== 新增候選為 0,附上 fixed 排除案例裡面積最接近 500 的前20筆 ===")
        for r in near_boundary[:20]:
            print(f"  {r['split']} {r['file_name']}  max_black_area={r['max_black_area']}px  ring_brightness_fixed={r['ring_brightness_fixed']}")

    with open(HERE / "v4_threshold_diff_report.json", "w", encoding="utf-8") as f:
        json.dump(diff_report, f, ensure_ascii=False, indent=2)
    print("\n[done] v4_threshold_diff_report.json 已寫入")


if __name__ == "__main__":
    main()
