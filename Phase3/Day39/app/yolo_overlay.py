"""
v1.1 階段三:YOLO疊框。沿用既有的 KEEP_CLASSES / CONF_THRESH / MODEL_PATH 設定
(Day35/run_yolo_inference.py 定案的COCO-pretrained、11類白名單,決策過程見
Day35/outputs/task0_class_mapping.md),不重新設計一套新的類別對應。

用 ultralytics 內建的 results[0].plot() 畫框+標籤(跟原本專案裡其他YOLO疊圖腳本
一樣直接用ultralytics自己的視覺化,不用手刻一套PIL畫框邏輯)。
"""
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent

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
MODEL_PATH = str(MODEL_DIR / "yolov8m.pt")

_yolo_model = YOLO(MODEL_PATH)


def detect_and_draw(image: Image.Image) -> Image.Image:
    results = _yolo_model.predict(source=image, conf=CONF_THRESH, classes=list(KEEP_CLASSES.keys()), verbose=False)
    annotated_bgr = results[0].plot()  # numpy array, BGR
    return Image.fromarray(annotated_bgr[:, :, ::-1])  # BGR -> RGB


def detect_classes(image: Image.Image):
    """回傳原始偵測結果 [{"class_name": str, "conf": float, "bbox": [x1,y1,x2,y2]}, ...],
    給批次腳本(不需要畫框,只要類別/信心值)用,跟 detect_and_draw 共用同一個
    _yolo_model + KEEP_CLASSES + CONF_THRESH,不重新設計一套。"""
    results = _yolo_model.predict(source=image, conf=CONF_THRESH, classes=list(KEEP_CLASSES.keys()), verbose=False)
    r = results[0]
    dets = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        dets.append({
            "class_name": KEEP_CLASSES[cls_id],
            "conf": round(float(box.conf.item()), 4),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    return dets
