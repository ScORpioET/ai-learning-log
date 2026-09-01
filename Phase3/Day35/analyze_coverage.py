"""
Day35 加碼 Task 2: GT vs YOLO 覆蓋度分析(全 val split)。

⚠️ bbox 格式更正(給 Jack 看):原 prompt 假設 detections_val.jsonl 存的是
[x1,y1,x2,y2](xyxy),但實際檢查 run_yolo_inference.py 的程式碼跟
detections_val.jsonl 第一筆資料,發現早就轉成 [x,y,w,h] 了(跟 GT 的
coco.json bbox 格式一致)。所以下面 IoU 計算不需要做格式轉換,兩邊都是
[x,y,w,h],只在算 IoU 時內部轉成 x1,y1,x2,y2 方便算交集。

class 對應:沿用 Day35 Task 0-B / generate_captions.py 裡的
COCO_TO_FLIR_ALIAS(YOLO COCO name -> FLIR name),這裡取其反向
(FLIR name -> COCO name)用來把 GT annotation 的 class 轉成跟 YOLO
detection 同一個命名空間,才能做「同 class」比對。只有這 11 類
(KEEP_CLASSES 涵蓋的類別)會被拿來配對,GT 裡的 other vehicle/
stroller/scooter/dog/... 這些沒有 COCO 對應的類別,直接算「GT 有、
YOLO 不可能有」,不進 matching 計算。
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.home() / "ai-transition-2026"
COCO_PATH = ROOT / "thermal_dataset" / "images_thermal_val" / "coco.json"
DET_PATH = ROOT / "Day35" / "outputs" / "detections_val.jsonl"
OUT_DIR = ROOT / "Day35" / "outputs"

IOU_THRESH = 0.5  # 沿用 task 指定值,沒有換過

# FLIR name -> COCO name(COCO_TO_FLIR_ALIAS 的反向,對照表見
# thermal_dataset/generate_captions.py 裡的原始定義 + Day35 task0_class_mapping.md)
FLIR_TO_COCO = {
    "person": "person",
    "bike": "bicycle",
    "car": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "skateboard": "skateboard",
    "light": "traffic light",
    "hydrant": "fire hydrant",
    "sign": "stop sign",
}
COCO_CLASSES = sorted(set(FLIR_TO_COCO.values()))

SIZE_BUCKETS = [("tiny", 0, 0.005), ("small", 0.005, 0.02), ("medium", 0.02, 0.08), ("large", 0.08, float("inf"))]


def size_bucket(area_ratio):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area_ratio < hi:
            return name
    return "large"


def xywh_to_xyxy(b):
    x, y, w, h = b
    return x, y, x + w, y + h


def iou(box_a, box_b):
    """box_a, box_b: [x,y,w,h]。內部轉 x1y1x2y2 算交集/聯集。"""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(gt_boxes, yolo_boxes):
    """gt_boxes / yolo_boxes: [(idx, class_name, bbox), ...],同一張圖、
    已經只含同一個 class 的物件(呼叫端先依 class 分組再叫這個函式)。
    貪婪配對:先算所有 pair 的 IoU,由高到低排序,大於 IOU_THRESH 且
    兩邊都還沒被用掉的就配對成功。回傳 (matched_pairs, gt_unmatched, yolo_unmatched)。
    """
    pairs = []
    for gi, (g_idx, _, g_box) in enumerate(gt_boxes):
        for yi, (y_idx, _, y_box) in enumerate(yolo_boxes):
            v = iou(g_box, y_box)
            if v > IOU_THRESH:
                pairs.append((v, gi, yi))
    pairs.sort(reverse=True)

    matched_gt, matched_yolo = set(), set()
    matches = []
    for v, gi, yi in pairs:
        if gi in matched_gt or yi in matched_yolo:
            continue
        matched_gt.add(gi)
        matched_yolo.add(yi)
        matches.append((gt_boxes[gi][0], yolo_boxes[yi][0], v))

    gt_unmatched = [gt_boxes[i][0] for i in range(len(gt_boxes)) if i not in matched_gt]
    yolo_unmatched = [yolo_boxes[i][0] for i in range(len(yolo_boxes)) if i not in matched_yolo]
    return matches, gt_unmatched, yolo_unmatched


def load_gt(coco_path):
    d = json.load(open(coco_path))
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    img_meta = {im["id"]: im for im in d["images"]}
    anns_by_img = defaultdict(list)
    for ann in d["annotations"]:
        cat_name = id2name[ann["category_id"]]
        if cat_name not in FLIR_TO_COCO:
            continue
        anns_by_img[ann["image_id"]].append({
            "flir_name": cat_name,
            "coco_name": FLIR_TO_COCO[cat_name],
            "bbox": ann["bbox"],
            "occluded": (ann.get("extra_info", {}) or {}).get("occluded", ""),
        })
    return img_meta, anns_by_img


def load_yolo(det_path):
    by_filename = {}
    with open(det_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_filename[r["file_name"]] = r
    return by_filename


def main():
    img_meta, gt_by_img = load_gt(COCO_PATH)
    yolo_by_file = load_yolo(DET_PATH)

    csv_rows = []
    all_detail = {}  # image_id -> per-image detail dict, used by viz + finding scripts

    # 全域 class confusion / precision-recall 統計
    gt_class_total = Counter()
    yolo_class_total = Counter()
    matched_class_total = Counter()

    # size-bucket 統計 (表 A: GT bbox size 分布 by class; 表 B: 全class 彙總 match rate by size)
    gt_size_by_class = defaultdict(Counter)  # class -> {bucket: count}
    gt_size_total = Counter()               # bucket -> gt count (全 class 彙總)
    matched_size_total = Counter()          # bucket -> matched count (全 class 彙總)

    # 給 histogram 用的原始 area_ratio 清單
    area_ratios_all_gt = []
    area_ratios_matched_gt = []
    area_ratios_missed_gt = []

    for img_id, im in img_meta.items():
        w_img, h_img = im["width"], im["height"]
        bare_name = Path(im["file_name"]).name
        gt_anns = gt_by_img.get(img_id, [])
        yolo_record = yolo_by_file.get(bare_name, {"detections": []})
        yolo_dets = yolo_record["detections"]

        # 依 class 分組
        gt_by_class = defaultdict(list)
        for i, a in enumerate(gt_anns):
            gt_by_class[a["coco_name"]].append((i, a["coco_name"], a["bbox"]))
        yolo_by_class = defaultdict(list)
        for i, d in enumerate(yolo_dets):
            yolo_by_class[d["class_name"]].append((i, d["class_name"], d["bbox"]))

        img_matched_gt_idx = set()
        img_matched_yolo_idx = set()
        per_class_counts = {}

        for cname in COCO_CLASSES:
            g_list = gt_by_class.get(cname, [])
            y_list = yolo_by_class.get(cname, [])
            matches, gt_unm, yolo_unm = greedy_match(g_list, y_list)
            for g_idx, y_idx, v in matches:
                img_matched_gt_idx.add(g_idx)
                img_matched_yolo_idx.add(y_idx)
            gt_class_total[cname] += len(g_list)
            yolo_class_total[cname] += len(y_list)
            matched_class_total[cname] += len(matches)
            per_class_counts[cname] = {
                "gt": len(g_list), "yolo": len(y_list), "matched": len(matches),
            }

        # size bucket:只針對「有 COCO 對應」的 GT class 做(跟上面 matching 範圍一致)
        for i, a in enumerate(gt_anns):
            x, y, w, h = a["bbox"]
            area_ratio = (w * h) / (w_img * h_img)
            bucket = size_bucket(area_ratio)
            gt_size_by_class[a["coco_name"]][bucket] += 1
            gt_size_total[bucket] += 1
            area_ratios_all_gt.append(area_ratio)
            if i in img_matched_gt_idx:
                matched_size_total[bucket] += 1
                area_ratios_matched_gt.append(area_ratio)
            else:
                area_ratios_missed_gt.append(area_ratio)

        gt_total = len(gt_anns)
        yolo_total = len(yolo_dets)
        matched_total = len(img_matched_gt_idx)

        row = {
            "image_id": img_id,
            "file_name": bare_name,
            "gt_total": gt_total,
            "yolo_total": yolo_total,
            "matched_iou05": matched_total,
            "gt_only": gt_total - matched_total,
            "yolo_only": yolo_total - len(img_matched_yolo_idx),
        }
        for cname in COCO_CLASSES:
            row[f"gt_{cname.replace(' ', '_')}"] = per_class_counts[cname]["gt"]
            row[f"yolo_{cname.replace(' ', '_')}"] = per_class_counts[cname]["yolo"]

        csv_rows.append(row)
        all_detail[img_id] = {
            "file_name": bare_name, "w": w_img, "h": h_img,
            "gt_anns": gt_anns, "yolo_dets": yolo_dets,
            "matched_gt_idx": sorted(img_matched_gt_idx),
            "matched_yolo_idx": sorted(img_matched_yolo_idx),
            "per_class_counts": per_class_counts,
        }

    # --- 寫 CSV ---
    import csv
    fieldnames = ["image_id", "file_name", "gt_total", "yolo_total", "matched_iou05", "gt_only", "yolo_only"]
    for cname in COCO_CLASSES:
        fieldnames.append(f"gt_{cname.replace(' ', '_')}")
        fieldnames.append(f"yolo_{cname.replace(' ', '_')}")
    with open(OUT_DIR / "gt_vs_yolo_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # --- 存 detail (給 viz / finding script reuse,不用重算一次) ---
    with open(OUT_DIR / "coverage_detail.json", "w", encoding="utf-8") as f:
        # image_id key 轉成 str 給 json
        json.dump({str(k): v for k, v in all_detail.items()}, f, ensure_ascii=False)

    # --- 彙總 markdown ---
    total_gt = sum(gt_class_total.values())
    total_yolo = sum(yolo_class_total.values())
    total_matched = sum(matched_class_total.values())

    lines = []
    lines.append("# GT vs YOLO 覆蓋度彙總(val split,IoU>0.5 + 同 class 才算配對成功)\n")
    lines.append(f"- 全 val {len(img_meta)} 張圖")
    lines.append(f"- 平均每張 GT {total_gt/len(img_meta):.2f} 個框(僅算 KEEP_CLASSES 11 類範圍內)")
    lines.append(f"- 平均每張 YOLO {total_yolo/len(img_meta):.2f} 個框")
    lines.append(f"- 全域配對率(matched / gt_total)= {total_matched}/{total_gt} = {100*total_matched/total_gt:.1f}%\n")

    lines.append("## 依類別 precision / recall / F1\n")
    lines.append("(precision = matched/yolo_total,即 YOLO 框裡有幾成真的對到 GT;")
    lines.append("recall = matched/gt_total,即 GT 框裡有幾成被 YOLO 抓到)\n")
    lines.append("| class | gt_total | yolo_total | matched | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    class_stats = []
    for cname in COCO_CLASSES:
        g, y, m = gt_class_total[cname], yolo_class_total[cname], matched_class_total[cname]
        prec = m / y if y else 0.0
        rec = m / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        class_stats.append((cname, g, y, m, prec, rec, f1))
        lines.append(f"| {cname} | {g} | {y} | {m} | {prec*100:.1f}% | {rec*100:.1f}% | {f1*100:.1f}% |")

    lines.append("\n## GT 有、YOLO 缺最嚴重的 top 3 類(recall 最低,且 gt_total>=20 才列入避免小樣本雜訊)\n")
    low_recall = sorted([c for c in class_stats if c[1] >= 20], key=lambda c: c[5])[:3]
    for cname, g, y, m, prec, rec, f1 in low_recall:
        lines.append(f"- **{cname}**: recall {rec*100:.1f}% (GT {g} 個,只配對到 {m} 個)")

    lines.append("\n## YOLO 有、GT 缺最嚴重的 top 3 類(precision 最低,且 yolo_total>=20 才列入)\n")
    low_prec = sorted([c for c in class_stats if c[2] >= 20], key=lambda c: c[4])[:3]
    for cname, g, y, m, prec, rec, f1 in low_prec:
        lines.append(f"- **{cname}**: precision {prec*100:.1f}% (YOLO 偵測 {y} 個,只有 {m} 個對得到 GT)")

    lines.append("\n## 表 A: GT bbox size 分布(依 class,area_ratio = w*h / (img_w*img_h))\n")
    lines.append("- tiny: <0.5% / small: 0.5-2% / medium: 2-8% / large: >8%\n")
    lines.append("| class | tiny | small | medium | large | total |")
    lines.append("|---|---|---|---|---|---|")
    for cname in COCO_CLASSES:
        b = gt_size_by_class[cname]
        total = sum(b.values())
        lines.append(f"| {cname} | {b['tiny']} | {b['small']} | {b['medium']} | {b['large']} | {total} |")

    lines.append("\n## 表 B: YOLO 匹配率(依 size bucket,全 class 彙總)\n")
    lines.append("| size | GT 總數 | YOLO 匹配到 | 匹配率 |")
    lines.append("|---|---|---|---|")
    for bname, _, _ in SIZE_BUCKETS:
        g = gt_size_total[bname]
        m = matched_size_total[bname]
        rate = 100 * m / g if g else 0.0
        lines.append(f"| {bname} | {g} | {m} | {rate:.1f}% |")

    with open(OUT_DIR / "gt_vs_yolo_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # --- histogram 用資料存成 json,給 plot script 讀 ---
    with open(OUT_DIR / "size_hist_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "all_gt": area_ratios_all_gt,
            "matched_gt": area_ratios_matched_gt,
            "missed_gt": area_ratios_missed_gt,
        }, f)

    print("[done] gt_vs_yolo_coverage.csv, gt_vs_yolo_summary.md, coverage_detail.json, size_hist_data.json written")
    print(f"[info] total_gt={total_gt} total_yolo={total_yolo} total_matched={total_matched} "
          f"global_recall={100*total_matched/total_gt:.1f}%")


if __name__ == "__main__":
    main()
