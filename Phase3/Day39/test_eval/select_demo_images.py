"""
從 records_thermal.json / records_rgb.json(全量caption+YOLO偵測結果)裡,
依「caption描述的類別」跟「YOLO實際偵測到的類別」的一致程度(Jaccard),
每支影片各挑3張最接近的圖片當demo展示用。

caption關鍵字 -> KEEP_CLASSES canonical名稱的對照,直接照抄
thermal_dataset/generate_captions.py 裡 DYNAMIC_CLASSES / STATIC_CONTEXT_CLASSES
的英文詞彙(pedestrian/bicycle/motorcycle/car/bus/truck/train/skateboard/
traffic light/fire hydrant/sign),不是自己亂猜的對照表。
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

# caption裡出現的詞(小寫,已含常見複數) -> KEEP_CLASSES canonical名稱
CAPTION_KEYWORD_TO_CLASS = {
    "pedestrian": "person", "pedestrians": "person",
    "bicycle": "bicycle", "bicycles": "bicycle",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "car": "car", "cars": "car",
    "bus": "bus", "buses": "bus",
    "truck": "truck", "trucks": "truck",
    "train": "train", "trains": "train",
    "skateboard": "skateboard", "skateboards": "skateboard",
    "traffic light": "traffic light", "traffic lights": "traffic light",
    "fire hydrant": "fire hydrant", "fire hydrants": "fire hydrant",
    "sign": "stop sign", "signs": "stop sign",
}
# 依長度由長到短排序,避免"traffic light"被"light"這種子字串搶先匹配掉
# (雖然目前對照表沒有單獨的light,但保留這個排序習慣比較保險)
_KEYWORDS_SORTED = sorted(CAPTION_KEYWORD_TO_CLASS.keys(), key=len, reverse=True)


def caption_classes(caption: str):
    text = caption.lower()
    found = set()
    for kw in _KEYWORDS_SORTED:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            found.add(CAPTION_KEYWORD_TO_CLASS[kw])
    return found


def yolo_classes(dets):
    return {d["class_name"] for d in dets}


def jaccard(a, b):
    if not a and not b:
        return 0.0  # 兩邊都空 -> 沒內容可比,不算"接近",demo也不會想選空圖
    union = a | b
    inter = a & b
    return len(inter) / len(union)


def main():
    for domain in ["thermal", "rgb"]:
        records = json.load(open(HERE / f"records_{domain}.json"))
        print(f"[info] {domain}: {len(records)} 筆")

        scored = []
        for r in records:
            cap_cls = caption_classes(r["caption"])
            yol_cls = yolo_classes(r["yolo"])
            score = jaccard(cap_cls, yol_cls)
            scored.append({
                **r,
                "caption_classes": sorted(cap_cls),
                "yolo_classes": sorted(yol_cls),
                "agreement_score": round(score, 4),
                "n_yolo_dets": len(r["yolo"]),
            })

        by_video = {}
        for r in scored:
            by_video.setdefault(r["video_id"], []).append(r)

        selected = []
        for vid, items in by_video.items():
            # 排序:agreement_score高的優先,同分再用偵測數多的優先(畫面內容豐富一點更適合當demo)
            items.sort(key=lambda r: (-r["agreement_score"], -r["n_yolo_dets"]))
            top3 = items[:3]
            selected.extend(top3)
            print(f"  video={vid}  candidates={len(items)}  picked scores={[t['agreement_score'] for t in top3]}")

        out_path = HERE / f"selected_{domain}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        print(f"[done] {domain}: {len(selected)} 張選出,存到 {out_path}\n")


if __name__ == "__main__":
    main()
