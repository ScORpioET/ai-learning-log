"""
Day39 步驟 5:把「最終融合 caption」提到的 class,拿 YOLO 偵測結果去畫框。

流程:
  1. YOLO 偵測:沿用 Day35 run_yolo_inference.py 的 KEEP_CLASSES + CONF_THRESH(0.25)
     + MODEL_PATH(yolov8m.pt,COCO pretrained,不重新訓練),對這 10 筆樣本的
     RGB 圖跟 thermal 圖各自跑一次(YOLO 本來就是不分 domain 直接吃 jpg)。
  2. class 名稱轉換:沿用 generate_captions.py 的 COCO_TO_FLIR_ALIAS(COCO name
     -> FLIR name)+ DYNAMIC_CLASSES(FLIR name -> caption 用的 en_name,例如
     "person"->"pedestrian"),兩個表都不改,串起來就是 YOLO 偵測到的
     COCO class -> caption 裡會出現的字。
  3. 每一版融合 caption(rgb_priority / thermal_priority)在 fuse_captions.py
     產生時就已經記錄 classes_used(en_name 清單)。只保留 YOLO 偵測結果裡
     en_name 落在該版 classes_used 的框;caption 提到但 YOLO 沒偵測到的,
     靜靜地少畫一個框,不特別標記。
  4. 同一版的框,RGB 圖跟 thermal 圖都畫(因為融合 caption 本來就是兩個
     domain 的資訊合併出來的,不是只屬於某一邊)。

輸出:yolo_overlay_data.json,每筆樣本內有
  rgb_priority:  {rgb_img_b64, thermal_img_b64, caption, boxes_rgb, boxes_thermal}
  thermal_priority: 同上
"""
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".pylibs"))

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HERE = Path(__file__).parent
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
RGB_TEST_DIR = TD / "video_rgb_test"
TH_TEST_DIR = TD / "video_thermal_test"

DAY35 = Path.home() / "ai-transition-2026" / "Phase3" / "Day35"
sys.path.insert(0, str(DAY35))
from run_yolo_inference import KEEP_CLASSES, CONF_THRESH, MODEL_PATH  # noqa: E402

sys.path.insert(0, str(TD))
import generate_captions as gc  # noqa: E402

# COCO class name -> FLIR name -> caption en_name,串起兩份既有對照表,不改邏輯
COCO_NAME_TO_EN_NAME = {}
for coco_name, flir_name in gc.COCO_TO_FLIR_ALIAS.items():
    en_name = gc.DYNAMIC_CLASSES.get(flir_name) or gc.STATIC_CONTEXT_CLASSES.get(flir_name)
    if en_name:
        COCO_NAME_TO_EN_NAME[coco_name] = en_name

BOX_COLOR = (230, 90, 30)


def detect(model, img_path):
    results = model.predict(source=str(img_path), conf=CONF_THRESH, classes=list(KEEP_CLASSES.keys()), verbose=False)
    r = results[0]
    dets = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        coco_name = KEEP_CLASSES[cls_id]
        en_name = COCO_NAME_TO_EN_NAME.get(coco_name)
        if en_name is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({"en_name": en_name, "conf": round(float(box.conf.item()), 3),
                     "bbox": [x1, y1, x2, y2]})
    return dets


def draw_filtered(img_path, dets, allowed_classes, resize_w=520):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    kept = [d for d in dets if d["en_name"] in allowed_classes]
    for d in kept:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        label = d["en_name"]
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=BOX_COLOR)
        draw.text((x1 + 4, y1 - th - 4), label, fill=(255, 255, 255), font=font)

    ratio = resize_w / im.width
    im = im.resize((resize_w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), len(kept)


def main():
    fused = json.load(open(HERE / "fused_results.json"))
    print(f"[info] YOLO model={MODEL_PATH}, conf>={CONF_THRESH}")
    print(f"[info] COCO->en_name map: {COCO_NAME_TO_EN_NAME}")
    model = YOLO(MODEL_PATH)

    out = []
    for r in fused:
        rf, tf = r["rgb_file"], r["thermal_file"]
        rgb_path, th_path = RGB_TEST_DIR / rf, TH_TEST_DIR / tf
        rgb_dets = detect(model, rgb_path)
        th_dets = detect(model, th_path)
        print(f"\n=== {tf} <-> {rf} ===")
        print(f"  rgb dets: {[(d['en_name'], d['conf']) for d in rgb_dets]}")
        print(f"  thermal dets: {[(d['en_name'], d['conf']) for d in th_dets]}")

        versions = {}
        for version in ("rgb_priority", "thermal_priority"):
            info = r["fused"][version]
            allowed = set(info["classes_used"])
            rgb_b64, n_rgb = draw_filtered(rgb_path, rgb_dets, allowed)
            th_b64, n_th = draw_filtered(th_path, th_dets, allowed)
            print(f"  [{version}] classes_used={sorted(allowed)} -> boxes drawn: rgb={n_rgb} thermal={n_th}")
            versions[version] = {
                "caption": info["caption"],
                "classes_used": sorted(allowed),
                "rgb_img": rgb_b64,
                "thermal_img": th_b64,
                "n_boxes_rgb": n_rgb,
                "n_boxes_thermal": n_th,
            }

        out.append({
            "thermal_file": tf, "rgb_file": rf,
            "thermal_video_id": r["thermal_video_id"], "rgb_video_id": r["rgb_video_id"],
            "frame_index": r["frame_index"],
            **versions,
        })

    with open(HERE / "yolo_overlay_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[done] yolo_overlay_data.json written")


if __name__ == "__main__":
    main()
