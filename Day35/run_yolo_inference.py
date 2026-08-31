"""
Day35 - YOLOv8m (COCO pretrained, no fine-tune) inference over FLIR ADAS v2 thermal images.

KEEP_CLASSES 決定過程見 Day35/outputs/task0_class_mapping.md —
三方對照(Day32 generate_captions.py 用的 class name / FLIR ADAS 原始 taxonomy /
COCO 80 類)之後,只保留「Day32 script 有用 且 COCO 有對應類別」的類別,
避免 GT 沒有、YOLO 卻亂偵測出來的雜訊類別汙染下游 caption。

用法:
    python run_yolo_inference.py --img-dir <thermal jpg 目錄> --out outputs/detections_val.jsonl [--limit 20]
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

# COCO class id -> COCO class name,只留 Task 0-B 三方對照後決定要保留的類別。
# 對照表見 outputs/task0_class_mapping.md,alias 對齊(COCO name -> Day32 en_name)
# 留到 generate_captions.py --source yolo 那一層做,這裡只做偵測 + 過濾,不做改名。
KEEP_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    36: "skateboard",
}

CONF_THRESH = 0.25
MODEL_PATH = str(Path.home() / "ai-transition-2026" / "model" / "yolov8m.pt")


def main(img_dir, out_path, limit=None):
    img_dir = Path(img_dir)
    img_paths = sorted(img_dir.glob("*.jpg"))
    if limit:
        img_paths = img_paths[:limit]
    print(f"[info] model={MODEL_PATH}")
    print(f"[info] {len(img_paths)} images from {img_dir}")
    print(f"[info] conf_thresh={CONF_THRESH}, keep_classes={sorted(KEEP_CLASSES.values())}")

    model = YOLO(MODEL_PATH)
    keep_ids = list(KEEP_CLASSES.keys())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_zero_det = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESH,
                classes=keep_ids,
                verbose=False,
            )
            r = results[0]
            h_img, w_img = r.orig_shape

            detections = []
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": KEEP_CLASSES[cls_id],
                    "conf": round(conf, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                })

            if not detections:
                n_zero_det += 1

            record = {
                "file_name": img_path.name,
                "img_width": w_img,
                "img_height": h_img,
                "detections": detections,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] {n_written} records written to {out_path}")
    print(f"[info] zero-detection images: {n_zero_det} ({100 * n_zero_det / max(n_written, 1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 張,小量驗證用")
    args = parser.parse_args()
    main(args.img_dir, args.out, args.limit)
