"""
線A:10-20 張人工比對(挑「不嚴重但被挑出來」vs「數值相近但確實嚴重」),
對每張算:
- Laplacian:曝光核心 bbox(換算回原圖解析度)區域的 cv2.Laplacian(...).var()
- 下游任務(YOLO)信心值:曝光bbox附近的偵測信心值,沒有就用全圖平均

輸出 severity_line_a.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
ORIG_DIRS = {
    "out_train": DATASET_ROOT / "images_rgb_train" / "data",
    "out_val": DATASET_ROOT / "images_rgb_val" / "data",
    "out_test": DATASET_ROOT / "video_rgb_test" / "data",
}
FLARE_DIRS = {s: HERE / s / "flare" for s in ["out_train", "out_val", "out_test"]}
MODEL_PATH = str(Path.home() / "ai-transition-2026" / "model" / "yolov8m.pt")
KEEP_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train",
                7: "truck", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 36: "skateboard"}
CONF_THRESH = 0.25

SELECTED = [
    ("out_train", "video-hXJxXmduG5Cjz9FBg-frame-001535-HfPEjYdx8M6iPhjfB.jpg", "not_severe"),
    ("out_train", "video-WPJrL5MznEAgqaD47-frame-000253-ibNPeLcpWmf3niZtK.jpg", "not_severe"),
    ("out_train", "video-BfJLkH7fCsS5Y9vQc-frame-002567-sEtfCzZswjRkyFqqS.jpg", "not_severe"),
    ("out_train", "video-qdYxM3S6eGGsKTzwn-frame-000205-SPNuibkvDs2kqxojE.jpg", "not_severe"),
    ("out_train", "video-vHLKy4kSYoaPYZoQo-frame-004474-kPX4nzEYpEXMtFodi.jpg", "not_severe"),
    ("out_train", "video-23bsd9bsr962GdFBZ-frame-005935-e7u6popFNneGQhG6W.jpg", "not_severe"),
    ("out_train", "video-GiDQGbWeWwtNTQEnG-frame-000618-f2ESyX6KKhcvwJkJa.jpg", "severe"),
    ("out_val", "video-QQZ8wcAm8Y9EPufST-frame-003780-PW7Qp4mjzTFx3mTSh.jpg", "severe"),
    ("out_train", "video-rXCGRrzyh98JMJk5v-frame-006431-yyyqnFhJiNKhBHZ3R.jpg", "severe"),
    ("out_val", "video-7cJxWPFMvPdSiWASY-frame-003468-pDzAKf8tGhPdhmpb3.jpg", "severe"),
    ("out_train", "video-QTT9nPmNSQceyBezK-frame-006008-cmF9Xmm4AFhdMdZDe.jpg", "severe"),
    ("out_test", "video-dvZBYnphN2BwdMKBc-frame-000021-Fte9QvqiE6kenxHWP.jpg", "severe"),
]


def bbox_iou_or_near(bbox_a, bbox_b, expand=40):
    ax, ay, aw, ah = bbox_a
    ax0, ay0, ax1, ay1 = ax - expand, ay - expand, ax + aw + expand, ay + ah + expand
    bx0, by0, bx1, by1 = bbox_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    return ix1 > ix0 and iy1 > iy0


def main():
    import json
    m = {(r["split"], r["file_name"]): r for r in json.load(open(HERE / "black_blob_summary_matched_after_param_v2.json"))}
    halo = json.load(open(HERE / "radial_halo_summary.json"))
    halo_by_key = {(r["split"], r["file_name"]): r["blobs"] for r in halo}

    model = YOLO(MODEL_PATH)
    keep_ids = list(KEEP_CLASSES.keys())

    rows_out = []
    for split, fn, tag in SELECTED:
        row = m[(split, fn)]
        blob = max(row["blobs"], key=lambda b: b["area"])
        bx, by, bw, bh = blob["bbox"]
        core_area = blob["area"]
        halo_area = max((b["halo_area_px"] for b in halo_by_key.get((split, fn), [])), default=None)

        orig_path = ORIG_DIRS[split] / fn
        flare_path = FLARE_DIRS[split] / fn
        orig = cv2.imread(str(orig_path))
        flare = cv2.imread(str(flare_path))
        scale = orig.shape[0] / flare.shape[0]

        ox, oy, ow, oh = [round(v * scale) for v in (bx, by, bw, bh)]
        # 稍微擴張一點再算 Laplacian,避免 bbox 太小、樣本點太少不穩定
        pad = round(0.5 * max(ow, oh))
        x0, y0 = max(ox - pad, 0), max(oy - pad, 0)
        x1, y1 = min(ox + ow + pad, orig.shape[1]), min(oy + oh + pad, orig.shape[0])
        crop = cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())

        results = model.predict(source=str(orig_path), conf=CONF_THRESH, classes=keep_ids, verbose=False)
        r = results[0]
        confs_all = []
        confs_near = []
        exposure_bbox = (ox, oy, ow, oh)
        for box in r.boxes:
            conf = float(box.conf.item())
            x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
            confs_all.append(conf)
            if bbox_iou_or_near(exposure_bbox, (x1b, y1b, x2b, y2b), expand=60):
                confs_near.append(conf)

        if confs_near:
            downstream_conf = float(np.mean(confs_near))
            conf_source = "nearby_detections"
        elif confs_all:
            downstream_conf = float(np.mean(confs_all))
            conf_source = "image_average(no_nearby)"
        else:
            downstream_conf = None
            conf_source = "no_detections"

        rows_out.append({
            "split": split, "file_name": fn, "human_severity": tag,
            "core_area_px": core_area, "halo_area_px": round(halo_area, 1) if halo_area else None,
            "laplacian_bbox": round(laplacian_var, 1),
            "downstream_confidence": round(downstream_conf, 3) if downstream_conf is not None else None,
            "conf_source": conf_source,
        })
        print(f"{tag:12s} {split} {fn}  core={core_area} halo={halo_area}  laplacian={laplacian_var:.1f}  "
              f"conf={downstream_conf}  ({conf_source})")

    with open(HERE / "severity_line_a.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print("\n[done] severity_line_a.csv 已寫入")


if __name__ == "__main__":
    main()
