"""
Day35 Task 4: 抽 10 張 val 圖,疊 YOLO 偵測框 + GT caption / YOLO caption 對照,
存成 png 給 Jack 明天肉眼看,決定 conf threshold 要不要調。
"""
import json
import random
from pathlib import Path

import cv2

ROOT = Path.home() / "ai-transition-2026"
IMG_DIR = ROOT / "thermal_dataset" / "images_thermal_val" / "data"
OUT_DIR = ROOT / "Day35" / "outputs" / "day35_compare"
DET_PATH = ROOT / "Day35" / "outputs" / "detections_val.jsonl"
GT_CAP_PATH = ROOT / "Day35" / "outputs" / "captions_val_gt_current.jsonl"
YOLO_CAP_PATH = ROOT / "Day35" / "outputs" / "captions_val_yolo.jsonl"

BOX_COLOR = (0, 255, 0)
TEXT_BG = (0, 0, 0)


def load_jsonl(path, key):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r[key]] = r
    return d


def put_wrapped_text(img, text, y0, max_width, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)

    y = y0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.rectangle(img, (5, y - th - 4), (10 + tw, y + 4), TEXT_BG, -1)
        cv2.putText(img, line, (8, y), font, scale, color, thickness, cv2.LINE_AA)
        y += th + 10
    return y


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dets = load_jsonl(DET_PATH, "file_name")
    gt_caps = load_jsonl(GT_CAP_PATH, "file_name")
    yolo_caps = load_jsonl(YOLO_CAP_PATH, "file_name")

    common = sorted(set(gt_caps) & set(yolo_caps))
    random.seed(7)
    sample = random.sample(common, 10)

    for i, fn in enumerate(sample, 1):
        bare_name = Path(fn).name
        img_path = IMG_DIR / bare_name
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            print(f"[warn] cannot read {img_path}, skip")
            continue

        det_record = dets.get(bare_name, {"detections": []})
        for d in det_record["detections"]:
            x, y, w, h = d["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(im, (x1, y1), (x2, y2), BOX_COLOR, 1)
            label = f"{d['class_name']} {d['conf']:.2f}"
            cv2.putText(im, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)

        h_img, w_img = im.shape[:2]
        canvas = cv2.copyMakeBorder(im, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        y = h_img + 20
        y = put_wrapped_text(canvas, f"GT:   {gt_caps[fn]['caption']}", y, w_img - 10, (100, 255, 100))
        put_wrapped_text(canvas, f"YOLO: {yolo_caps[fn]['caption']}", y, w_img - 10, (100, 200, 255))

        out_path = OUT_DIR / f"{i:02d}_{bare_name}"
        cv2.imwrite(str(out_path), canvas)
        print(f"[saved] {out_path}")

    print(f"\n[done] {len(sample)} comparison images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
