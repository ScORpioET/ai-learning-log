"""
Day35 加碼 Task 1: GT vs YOLO 並排對照圖(30 張,A/B/C 各 10)。

左 panel:GT,綠色框 + FLIR class name + 該類累計數(e.g. "car #3")
右 panel:YOLO,紅色框 + COCO class name + conf(e.g. "car 0.72")
底部三行 caption:image_id / GT 框統計 / YOLO 框統計
"""
import json
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path.home() / "ai-transition-2026"
IMG_DIR = ROOT / "thermal_dataset" / "images_thermal_val" / "data"
OUT_DIR = ROOT / "Day35" / "outputs" / "day35_gt_vs_yolo"
DETAIL_PATH = ROOT / "Day35" / "outputs" / "coverage_detail.json"
GROUPS_PATH = ROOT / "Day35" / "outputs" / "abc_groups.json"

GT_COLOR = (0, 255, 0)     # 綠
YOLO_COLOR = (0, 0, 255)   # 紅
TEXT_BG = (0, 0, 0)
CAPTION_H = 60


def draw_gt_panel(im, gt_anns):
    im = im.copy()
    class_running_count = Counter()
    for a in gt_anns:
        class_running_count[a["flir_name"]] += 1
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(im, (x1, y1), (x2, y2), GT_COLOR, 1)
        label = f"{a['flir_name']} #{class_running_count[a['flir_name']]}"
        cv2.putText(im, label, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GT_COLOR, 1, cv2.LINE_AA)
    return im


def draw_yolo_panel(im, yolo_dets):
    im = im.copy()
    for d in yolo_dets:
        x, y, w, h = d["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(im, (x1, y1), (x2, y2), YOLO_COLOR, 1)
        label = f"{d['class_name']} {d['conf']:.2f}"
        cv2.putText(im, label, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, YOLO_COLOR, 1, cv2.LINE_AA)
    return im


def class_count_str(anns_or_dets, key):
    c = Counter(x[key] for x in anns_or_dets)
    return ", ".join(f"{n} {name}" for name, n in c.most_common())


def make_panel_pair(img_id, d):
    img_path = IMG_DIR / d["file_name"]
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return None

    left = draw_gt_panel(im, d["gt_anns"])
    right = draw_yolo_panel(im, d["yolo_dets"])
    combined = cv2.hconcat([left, right])

    h, w = combined.shape[:2]
    canvas = cv2.copyMakeBorder(combined, 0, CAPTION_H, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    gt_total = len(d["gt_anns"])
    yolo_total = len(d["yolo_dets"])
    gt_breakdown = class_count_str(d["gt_anns"], "flir_name")
    yolo_breakdown = class_count_str(d["yolo_dets"], "class_name")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"image_id: {img_id}  ({d['file_name']})", (8, h + 15),
                font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"GT: {gt_total} boxes ({gt_breakdown})", (8, h + 33),
                font, 0.42, GT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"YOLO: {yolo_total} boxes ({yolo_breakdown})", (8, h + 51),
                font, 0.42, YOLO_COLOR, 1, cv2.LINE_AA)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = json.load(open(DETAIL_PATH))
    groups = json.load(open(GROUPS_PATH))

    for group_name in ("A", "B", "C"):
        for i, img_id in enumerate(groups[group_name], 1):
            d = detail[img_id]
            canvas = make_panel_pair(img_id, d)
            if canvas is None:
                print(f"[warn] skip {img_id}, image not found")
                continue
            out_path = OUT_DIR / f"{group_name}_{i:02d}_{img_id}.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"[saved] {out_path}")

    print(f"\n[done] 30 panels written to {OUT_DIR}")


if __name__ == "__main__":
    main()
