"""
Day35 Task 5-A: GT bbox size threshold 敏感度分析(val + train,跟 YOLO 完全無關)。

範圍:generate_captions.py 實際會用到的所有類別——
DYNAMIC_CLASSES(11 類,決定 caption 文字本身)+ STATIC_CONTEXT_CLASSES
(3 類:light/hydrant/sign,只影響 has_static_context metadata,不影響 caption
文字)。兩者合計 14 類,跟這份分析的欄位對齊,方便 Jack 一次看到全貌。
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.home() / "ai-transition-2026"
OUT_DIR = ROOT / "Day35" / "outputs"

DYNAMIC_CLASSES = ["person", "bike", "motor", "car", "bus", "truck",
                   "other vehicle", "train", "skateboard", "stroller", "scooter"]
STATIC_CONTEXT_CLASSES = ["light", "hydrant", "sign"]
ALL_CLASSES = DYNAMIC_CLASSES + STATIC_CONTEXT_CLASSES

THRESHOLDS = [0.0, 0.10, 0.25, 0.50, 1.00]  # 百分比(%),0.0 = no filter


def load_split(split):
    coco = json.load(open(ROOT / "thermal_dataset" / f"images_thermal_{split}" / "coco.json"))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        cat_name = id2name[a["category_id"]]
        if cat_name not in ALL_CLASSES:
            continue
        w_img, h_img = img_wh[a["image_id"]]
        x, y, w, h = a["bbox"]
        area_pct = 100 * (w * h) / (w_img * h_img)
        anns_by_img[a["image_id"]].append((cat_name, area_pct))
    return coco, anns_by_img


def compute_table(anns_by_img, n_images):
    rows = []
    total_no_filter = sum(len(v) for v in anns_by_img.values())
    for thr in THRESHOLDS:
        kept_total = 0
        img_with_box = 0
        for img_id, anns in anns_by_img.items():
            kept = [a for a in anns if a[1] >= thr]
            kept_total += len(kept)
            if kept:
                img_with_box += 1
        zero_box_img = n_images - img_with_box
        pct = 100 * kept_total / total_no_filter if total_no_filter else 0
        avg_per_img = kept_total / n_images if n_images else 0
        rows.append({
            "threshold": thr, "kept_total": kept_total, "pct": pct,
            "avg_per_img": avg_per_img, "img_with_box": img_with_box,
            "zero_box_img": zero_box_img,
        })
    return rows, total_no_filter


def compute_class_table(anns_by_img):
    """每個 threshold 下,各 class 剩多少框(+ 保留率)。"""
    class_no_filter = Counter()
    for anns in anns_by_img.values():
        for cat_name, _ in anns:
            class_no_filter[cat_name] += 1

    class_rows = {}  # threshold -> {class: (kept, pct)}
    for thr in THRESHOLDS:
        counts = Counter()
        for anns in anns_by_img.values():
            for cat_name, area_pct in anns:
                if area_pct >= thr:
                    counts[cat_name] += 1
        class_rows[thr] = {
            c: (counts.get(c, 0), 100 * counts.get(c, 0) / class_no_filter[c] if class_no_filter[c] else 0)
            for c in ALL_CLASSES if class_no_filter[c] > 0
        }
    return class_rows, class_no_filter


def format_split_section(split_name, coco, anns_by_img):
    n_images = len(coco["images"])
    rows, total_no_filter = compute_table(anns_by_img, n_images)
    class_rows, class_no_filter = compute_class_table(anns_by_img)

    lines = [f"## {split_name} split(共 {n_images} 張圖,{total_no_filter} 個框,"
             f"範圍:{len(ALL_CLASSES)} 類 = 11 dynamic + 3 static-context)\n"]
    lines.append("| threshold | 剩 GT 框數 | 剩比例 | 平均/圖 | 至少1框的圖數 | 零框圖數 | 零框圖% |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        thr_label = "no filter" if r["threshold"] == 0.0 else f">= {r['threshold']:.2f}%"
        zero_pct = 100 * r["zero_box_img"] / n_images
        lines.append(f"| {thr_label} | {r['kept_total']} | {r['pct']:.1f}% | "
                      f"{r['avg_per_img']:.2f} | {r['img_with_box']} | {r['zero_box_img']} | {zero_pct:.1f}% |")

    lines.append(f"\n### {split_name}:依 class 分,各 threshold 下剩多少框(保留率)\n")
    header = "| class | no filter | " + " | ".join(f">={t:.2f}%" for t in THRESHOLDS[1:]) + " |"
    sep = "|---" * (len(THRESHOLDS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for cname in ALL_CLASSES:
        if class_no_filter[cname] == 0:
            continue
        cells = [str(class_no_filter[cname])]
        for thr in THRESHOLDS[1:]:
            kept, pct = class_rows[thr][cname]
            cells.append(f"{kept} ({pct:.0f}%)")
        lines.append(f"| {cname} | " + " | ".join(cells) + " |")

    return "\n".join(lines), rows, class_rows, class_no_filter


def main():
    all_lines = ["# Task 5-A: GT bbox size threshold 敏感度分析\n",
                 "範圍:person/bike/motor/car/bus/truck/other vehicle/train/skateboard/"
                 "stroller/scooter(11 個 DYNAMIC_CLASSES,決定 caption 文字)+ "
                 "light/hydrant/sign(3 個 STATIC_CONTEXT_CLASSES,只影響 has_static_context "
                 "metadata,不影響 caption 文字本身)。area_pct = bbox 面積 / 圖片面積 * 100。\n"]

    results = {}
    for split in ["val", "train"]:
        coco, anns_by_img = load_split(split)
        section, rows, class_rows, class_no_filter = format_split_section(split, coco, anns_by_img)
        all_lines.append(section)
        all_lines.append("")
        results[split] = (rows, class_rows, class_no_filter)

    with open(OUT_DIR / "threshold_sensitivity.md", "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    print("[done] threshold_sensitivity.md written")

    # --- Task 5-B 建議邏輯:自動掃過 THRESHOLDS,依 Jack 給的四條準則挑 ---
    print("\n=== Task 5-B threshold 建議(依準則自動評估,val split 為準)===")
    val_rows, val_class_rows, val_class_no_filter = results["val"]
    n_val_img = len(json.load(open(ROOT / "thermal_dataset" / "images_thermal_val" / "coco.json"))["images"])
    for r in val_rows:
        thr = r["threshold"]
        if thr == 0.0:
            continue
        zero_pct = 100 * r["zero_box_img"] / n_val_img
        avg_per_img = r["avg_per_img"]
        person_kept, person_pct = val_class_rows[thr].get("person", (0, 0))
        car_kept, car_pct = val_class_rows[thr].get("car", (0, 0))
        ok_zero = zero_pct <= 5
        ok_avg = avg_per_img >= 3
        ok_person = person_pct >= 70
        ok_car = car_pct >= 70
        verdict = "PASS" if (ok_zero and ok_avg and ok_person and ok_car) else "FAIL"
        print(f"thr={thr:.2f}%  zero_img%={zero_pct:.1f}({'ok' if ok_zero else 'X'})  "
              f"avg/img={avg_per_img:.2f}({'ok' if ok_avg else 'X'})  "
              f"person_kept%={person_pct:.1f}({'ok' if ok_person else 'X'})  "
              f"car_kept%={car_pct:.1f}({'ok' if ok_car else 'X'})  => {verdict}")


if __name__ == "__main__":
    main()
