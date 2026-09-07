"""
線B:對 934 張候選,算:
- laplacian_bbox:曝光核心 bbox(換算回原圖解析度)區域的 Laplacian 變異數
- downstream_confidence_avg / _min:YOLO 在原圖上所有偵測的信心值平均/最小值
  (沒有偵測到東西就是 null)
- downstream_confidence_drop:同一支影片裡,frame 編號最接近、且 diff_mean
  (借用 all_frames_diff.json,越小代表越少受 flare 影響)最低的「乾淨」
  參考幀,拿它的 YOLO 平均信心值減掉曝光幀的平均信心值(drop 越大代表曝光
  對下游任務影響越大);找不到乾淨參考幀或兩邊都沒偵測到東西就是 null。

輸出 severity_scores_v2.json,並計算三個指標兩兩之間的 Spearman 相關係數、
兩份排序反轉清單。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}
FLARE_DIRS = {s: HERE / s / "flare" for s in ["out_train", "out_val", "out_test"]}
SPLIT_MAP = {"out_train": "train", "out_val": "val", "out_test": "test"}
MODEL_PATH = str(Path.home() / "ai-transition-2026" / "model" / "yolov8m.pt")
KEEP_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train",
                7: "truck", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 36: "skateboard"}
CONF_THRESH = 0.25
FRAME_RE = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)-")
NEARBY_FRAME_WINDOW = 300


def parse_video_frame(fn):
    m = FRAME_RE.search(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def yolo_confidences(model, keep_ids, img_path, cache):
    key = str(img_path)
    if key in cache:
        return cache[key]
    results = model.predict(source=key, conf=CONF_THRESH, classes=keep_ids, verbose=False)
    confs = [float(b.conf.item()) for b in results[0].boxes]
    cache[key] = confs
    return confs


def main():
    data = json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))
    halo = json.load(open(HERE / "radial_halo_summary.json"))
    halo_by_key = {(r["split"], r["file_name"]): r["blobs"] for r in halo}
    diff_records = json.load(open(HERE / "all_frames_diff.json"))

    # 依 video_id 分組,方便找「同影片、frame編號最近、diff_mean最低」的乾淨參考幀
    by_video = {}
    for r in diff_records:
        by_video.setdefault(r["video_id"], []).append(r)

    model = YOLO(MODEL_PATH)
    keep_ids = list(KEEP_CLASSES.keys())
    conf_cache = {}

    results_out = []
    n_total = len(data)
    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        blob = max(row["blobs"], key=lambda b: b["area"])
        bx, by_, bw, bh = blob["bbox"]
        core_area = blob["area"]
        halo_area = max((b["halo_area_px"] for b in halo_by_key.get((split, fn), [])), default=None)

        orig_path = ORIG_DIRS[split] / fn
        flare_path = FLARE_DIRS[split] / fn
        orig = cv2.imread(str(orig_path))
        flare = cv2.imread(str(flare_path))
        if orig is None or flare is None:
            continue
        scale = orig.shape[0] / flare.shape[0]
        ox, oy, ow, oh = [round(v * scale) for v in (bx, by_, bw, bh)]
        pad = round(0.5 * max(ow, oh))
        x0, y0 = max(ox - pad, 0), max(oy - pad, 0)
        x1, y1 = min(ox + ow + pad, orig.shape[1]), min(oy + oh + pad, orig.shape[0])
        crop = cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())

        confs = yolo_confidences(model, keep_ids, orig_path, conf_cache)
        conf_avg = float(np.mean(confs)) if confs else None
        conf_min = float(np.min(confs)) if confs else None

        # 找乾淨參考幀:同影片、frame 編號距離在 NEARBY_FRAME_WINDOW 內、diff_mean 最低
        vid, frame_num = parse_video_frame(fn)
        conf_drop = None
        if vid is not None and vid in by_video:
            cands = [r for r in by_video[vid] if r["file_name"] != fn and abs(
                int(FRAME_RE.search(r["file_name"]).group(2)) - frame_num) <= NEARBY_FRAME_WINDOW]
            if cands:
                clean = min(cands, key=lambda r: r["diff_mean"])
                clean_split_key = {"train": "out_train", "val": "out_val", "test": "out_test"}[clean["split"]]
                clean_path = ORIG_DIRS[clean_split_key] / clean["file_name"]
                if clean_path.exists():
                    clean_confs = yolo_confidences(model, keep_ids, clean_path, conf_cache)
                    if clean_confs and confs:
                        conf_drop = float(np.mean(clean_confs) - np.mean(confs))

        results_out.append({
            "split": split, "file_name": fn, "bbox": blob["bbox"],
            "halo_area_px": round(halo_area, 1) if halo_area else None,
            "core_area_px": core_area,
            "laplacian_bbox": round(laplacian_var, 1),
            "downstream_confidence_avg": round(conf_avg, 3) if conf_avg is not None else None,
            "downstream_confidence_min": round(conf_min, 3) if conf_min is not None else None,
            "downstream_confidence_drop": round(conf_drop, 3) if conf_drop is not None else None,
        })
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{n_total}", end="\r")
    print()

    with open(HERE / "severity_scores_v2.json", "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print(f"[done] severity_scores_v2.json 已寫入,共 {len(results_out)} 筆")

    # 相關係數(Spearman),只用兩邊都有值的樣本
    def spearman(key_a, key_b):
        pairs = [(r[key_a], r[key_b]) for r in results_out if r[key_a] is not None and r[key_b] is not None]
        if len(pairs) < 5:
            return None, len(pairs)
        a, b = zip(*pairs)
        rho, p = stats.spearmanr(a, b)
        return (round(float(rho), 3), round(float(p), 4)), len(pairs)

    corr_report = {}
    for ka, kb in [("halo_area_px", "laplacian_bbox"), ("halo_area_px", "downstream_confidence_drop"),
                   ("laplacian_bbox", "downstream_confidence_drop"), ("core_area_px", "laplacian_bbox"),
                   ("core_area_px", "downstream_confidence_drop")]:
        (rho_p, n) = spearman(ka, kb)
        corr_report[f"{ka}_vs_{kb}"] = {"spearman_rho_p": rho_p, "n_pairs": n}
        print(f"{ka} vs {kb}: {rho_p}  (n={n})")

    with open(HERE / "severity_correlations_v2.json", "w", encoding="utf-8") as f:
        json.dump(corr_report, f, ensure_ascii=False, indent=2)
    print("[done] severity_correlations_v2.json 已寫入")


if __name__ == "__main__":
    main()
