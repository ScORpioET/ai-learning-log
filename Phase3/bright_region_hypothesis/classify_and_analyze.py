"""
驗證假設:「亮區(強光/耀光)覆蓋的位置,thermal 偵測到物件但 RGB 沒偵測到
(或信心值明顯偏低)的機率,是否明顯高於『RGB 也有偵測到』的位置」

流程:
1. 讀 detections_rgb_test.jsonl / detections_thermal_test.jsonl(Day35
   run_yolo_inference.py 產出,COCO class 名稱),用 generate_captions.py
   的 COCO_TO_FLIR_ALIAS + DYNAMIC_CLASSES 兩張既有表轉成 en_name,兩邊
   用同一套詞彙比對(沿用 yolo_caption_overlay.py 已經用過的同一套轉換)。
2. RGB/thermal 沒有相機校準/homography 資料(index.json 只標了兩邊
   spectrum 都是同一個 "45hfov" 標籤,代表視野角大致設計成一致,但沒有
   實際的內外參可以做嚴謹的像素對應)。這裡採用「等比例縮放投影」這個
   近似:thermal bbox 各座標乘上 (RGB_W/TH_W, RGB_H/TH_H) 直接投影到 RGB
   1224x1024 座標系,當作兩顆鏡頭視野大致同軸、只差解析度的近似值。
   這是這次分析最大的不確定性來源,不是精確的像素對應,結論要照這個
   限制的信心層級來看,不能當成逐 pixel 精確。
3. 每個 thermal 偵測,在同一幀 RGB 偵測裡找同 en_name、投影後 IoU 最高的
   配對:
     IOU_MATCH_THRESH = 0.10 —— 因為座標對應只是近似(見上),用嚴格的
       IoU(例如 0.3-0.5)會把「其實有對應到、只是投影誤差」的案例誤判成
       「RGB 沒偵測到」,人為灌高 A 組數量。0.10 是在「留一點容忍度」跟
       「不要寬到隨便兩個不相關的框都算配對」之間的取捨,沒有更精細的
       校準依據。
     CONF_GAP_THRESH = 0.15 —— YOLO 對同一個真實物件在不同 domain(外觀
       差異大)算出來的信心值本來就會自然浮動,0.15 訂在「明顯高於這種
       自然浮動、但不用到判斷成完全不同等級」的中間值,同樣是方法論選擇。
   分組規則:
     沒配對到(best_iou < IOU_MATCH_THRESH) -> A 組(候選漏偵測)
     配對到但 conf_thermal - conf_rgb > CONF_GAP_THRESH -> A 組(信心值明顯偏低)
     配對到且信心值接近 -> B 組(對照組)
4. 對每一幀的 RGB 圖跑 bright_region_detect.detect_bright_regions(),
   對每個 thermal 偵測的「投影後 bbox」檢查是否跟任一亮區重疊:
     overlap_iou:投影 bbox 跟任一亮區外接矩形的 IoU > 0
     overlap_center:投影 bbox 中心點落在亮區 mask 內
   兩種都算,回報時都給。

輸出:
    classified_objects.jsonl —— 每個 thermal 偵測一筆(group、overlap 資訊)
    hypothesis_summary.json —— A/B 組統計摘要
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "thermal_dataset"))
import generate_captions as gc  # noqa: E402

from bright_region_detect import detect_bright_regions  # noqa: E402

HERE = Path(__file__).parent
TD = Path.home() / "ai-transition-2026" / "thermal_dataset"

RGB_W, RGB_H = 1224, 1024
TH_W, TH_H = 640, 512
SX, SY = RGB_W / TH_W, RGB_H / TH_H

IOU_MATCH_THRESH = 0.10
CONF_GAP_THRESH = 0.15

COCO_NAME_TO_EN_NAME = {}
for coco_name, flir_name in gc.COCO_TO_FLIR_ALIAS.items():
    en_name = gc.DYNAMIC_CLASSES.get(flir_name) or gc.STATIC_CONTEXT_CLASSES.get(flir_name)
    if en_name:
        COCO_NAME_TO_EN_NAME[coco_name] = en_name


def load_detections(path):
    """run_yolo_inference.py 存的 file_name 是不含 'data/' 前綴的 basename,
    但 full_test_pairs.json / coco.json 的 file_name 都是 'data/xxx.jpg'——
    這裡統一補回前綴,兩邊 key 才對得上(第一版漏了這步,導致 A/B 組全部
    是 0,已修正)。"""
    by_file = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        dets = []
        for d in r["detections"]:
            en = COCO_NAME_TO_EN_NAME.get(d["class_name"])
            if en is None:
                continue
            dets.append({"en_name": en, "conf": d["conf"], "bbox": d["bbox"]})
        by_file[f"data/{r['file_name']}"] = dets
    return by_file


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def project_thermal_bbox(bbox):
    x, y, w, h = bbox
    return (x * SX, y * SY, w * SX, h * SY)


def box_overlaps_any(box, region_boxes):
    return any(iou(box, rb) > 0 for rb in region_boxes)


def center_in_mask(box, mask):
    x, y, w, h = box
    cx, cy = int(round(x + w / 2)), int(round(y + h / 2))
    cy = min(max(cy, 0), mask.shape[0] - 1)
    cx = min(max(cx, 0), mask.shape[1] - 1)
    return bool(mask[cy, cx])


def main():
    pairs = json.load(open(HERE / "full_test_pairs.json"))
    rgb_dets_by_file = load_detections(HERE / "detections_rgb_test.jsonl")
    th_dets_by_file = load_detections(HERE / "detections_thermal_test.jsonl")

    out = open(HERE / "classified_objects.jsonl", "w", encoding="utf-8")
    n_a = n_b = 0
    n_a_overlap_iou = n_a_overlap_center = 0
    n_b_overlap_iou = n_b_overlap_center = 0

    for i, p in enumerate(pairs):
        rf, tf = p["rgb_file"], p["thermal_file"]
        rgb_dets = rgb_dets_by_file.get(rf, [])
        th_dets = th_dets_by_file.get(tf, [])
        if not th_dets:
            continue

        rgb_img_path = TD / "video_rgb_test" / rf
        mask, region_boxes = detect_bright_regions(rgb_img_path)

        for td in th_dets:
            proj = project_thermal_bbox(td["bbox"])
            same_class_rgb = [d for d in rgb_dets if d["en_name"] == td["en_name"]]
            best_iou, best_match = 0.0, None
            for rd in same_class_rgb:
                v = iou(proj, rd["bbox"])
                if v > best_iou:
                    best_iou, best_match = v, rd

            if best_iou < IOU_MATCH_THRESH:
                group = "A"
            else:
                conf_gap = td["conf"] - best_match["conf"]
                group = "A" if conf_gap > CONF_GAP_THRESH else "B"

            ov_iou = box_overlaps_any(proj, region_boxes)
            ov_center = center_in_mask(proj, mask)

            rec = {
                "rgb_file": rf, "thermal_file": tf, "en_name": td["en_name"],
                "thermal_conf": td["conf"], "thermal_bbox": td["bbox"], "projected_bbox": proj,
                "matched_rgb_conf": best_match["conf"] if best_match else None,
                "best_iou": round(best_iou, 4), "group": group,
                "overlap_iou": ov_iou, "overlap_center": ov_center,
                "n_bright_regions": len(region_boxes),
            }
            out.write(json.dumps(rec) + "\n")

            if group == "A":
                n_a += 1
                n_a_overlap_iou += ov_iou
                n_a_overlap_center += ov_center
            else:
                n_b += 1
                n_b_overlap_iou += ov_iou
                n_b_overlap_center += ov_center

        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(pairs)} frames  A={n_a} B={n_b}", end="\r")

    out.close()

    summary = {
        "iou_match_thresh": IOU_MATCH_THRESH, "conf_gap_thresh": CONF_GAP_THRESH,
        "bright_region_params": {"thresh": 200, "blur_ksize": 9, "min_area_px": 400},
        "n_A": n_a, "n_B": n_b,
        "A_overlap_iou_rate": round(n_a_overlap_iou / n_a, 4) if n_a else None,
        "A_overlap_center_rate": round(n_a_overlap_center / n_a, 4) if n_a else None,
        "B_overlap_iou_rate": round(n_b_overlap_iou / n_b, 4) if n_b else None,
        "B_overlap_center_rate": round(n_b_overlap_center / n_b, 4) if n_b else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    with open(HERE / "hypothesis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[done] classified_objects.jsonl + hypothesis_summary.json written")


if __name__ == "__main__":
    main()
