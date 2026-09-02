"""
Day38 Phase 2:掃描 thermal/RGB train/val(全量版,不套用面積過濾,對應目前
定案的訓練資料)的 gt_caption,檢查 bbox 涵蓋 >=2 種動態類別的圖片裡,
caption 文字有沒有提到全部出現的類別。純關鍵字比對,不做語意理解。

沿用 generate_captions.py / generate_captions_rgb_full.py 的原始函式
(occlusion 過濾、long-tail 併入 "object"、near/far 距離門檻),邏輯不改,
只是重跑一次 dyn_objs 的計算來取得「這張圖實際出現哪些類別」的清單,
拿去跟已經生成好的 caption 檔案比對文字覆蓋率。
"""
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / "ai-transition-2026"

spec = importlib.util.spec_from_file_location("gc", ROOT / "thermal_dataset" / "generate_captions.py")
gc = importlib.util.module_from_spec(spec)
sys.modules["gc"] = gc
spec.loader.exec_module(gc)


def word_forms(en_name):
    singular = en_name
    plural = gc.plural_of(en_name)
    return {singular.lower(), plural.lower()}


def caption_mentions(caption_lower, en_name):
    for form in word_forms(en_name):
        if re.search(r"\b" + re.escape(form) + r"\b", caption_lower):
            return True
    return False


def compute_image_classes(coco_path, image_root_name, ref_split_coco_path=None):
    """回傳 {image_id: {"file_name":..., "en_names": set(...), "counts": Counter}}"""
    coco = json.load(open(coco_path))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    dynamic_ids = {c["id"] for c in coco["categories"] if c["name"] in gc.DYNAMIC_CLASSES}

    if ref_split_coco_path:
        ref_coco = json.load(open(ref_split_coco_path))
        ref_id2name = {c["id"]: c["name"] for c in ref_coco["categories"]}
        cat_counts_by_name = Counter(ref_id2name[a["category_id"]] for a in ref_coco["annotations"])
    else:
        cat_counts_by_name = Counter(id2name[a["category_id"]] for a in coco["annotations"])

    long_tail_names = {
        name for name in gc.DYNAMIC_CLASSES
        if cat_counts_by_name.get(name, 0) < gc.LONG_TAIL_THRESHOLD
    }

    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    out = {}
    for img_id, im in img_meta.items():
        en_names = set()
        counts = Counter()
        for ann in anns_by_img.get(img_id, []):
            cat_name = id2name[ann["category_id"]]
            if ann["category_id"] not in dynamic_ids:
                continue
            occluded = ann.get("extra_info", {}).get("occluded", "") or ""
            if occluded == gc.OCCLUDED_DIFFICULT:
                continue
            en_name = gc.LONG_TAIL_LABEL if cat_name in long_tail_names else gc.DYNAMIC_CLASSES[cat_name]
            en_names.add(en_name)
            counts[en_name] += 1
        out[im["file_name"]] = {"en_names": en_names, "counts": counts}
    return out


def scan(domain, split, coco_path, ref_coco_path, captions_path):
    img_classes = compute_image_classes(coco_path, None, ref_coco_path)

    captions = {}
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            captions[row["file_name"]] = row["caption"]

    multi_class_images = 0
    images_missing_ge1 = 0
    missed_class_counter = Counter()
    reason_top2_trunc_only = 0   # >=3 classes present, top class count<5 (pure top-2 truncation)
    reason_count5_collapse = 0   # top class count>=5 AND >=2 classes present (the Phase1 bug pattern)

    for file_name, info in img_classes.items():
        en_names = info["en_names"]
        if len(en_names) < 2:
            continue
        caption = captions.get(file_name)
        if caption is None:
            continue
        multi_class_images += 1
        caption_lower = caption.lower()

        missed = [n for n in en_names if not caption_mentions(caption_lower, n)]
        if missed:
            images_missing_ge1 += 1
            for m in missed:
                missed_class_counter[m] += 1

            counts = info["counts"]
            top_count = max(counts.values())
            if top_count >= 5:
                reason_count5_collapse += 1
            elif len(en_names) >= 3:
                reason_top2_trunc_only += 1

    return {
        "domain": domain, "split": split,
        "multi_class_images": multi_class_images,
        "images_missing_ge1": images_missing_ge1,
        "missing_rate": images_missing_ge1 / multi_class_images if multi_class_images else 0.0,
        "missed_class_breakdown": missed_class_counter.most_common(),
        "reason_count5_collapse": reason_count5_collapse,
        "reason_top2_trunc_only": reason_top2_trunc_only,
    }


def main():
    day32 = ROOT / "Phase3" / "Day32"
    td = ROOT / "thermal_dataset"

    jobs = [
        ("thermal", "train", td / "images_thermal_train" / "coco.json", None,
         day32 / "captions_train_full_v2.jsonl"),
        ("thermal", "val", td / "images_thermal_val" / "coco.json", td / "images_thermal_train" / "coco.json",
         day32 / "captions_val_full_v2.jsonl"),
        ("rgb", "train", td / "images_rgb_train" / "coco.json", None,
         day32 / "captions_rgb_train_full.jsonl"),
        ("rgb", "val", td / "images_rgb_val" / "coco.json", td / "images_rgb_train" / "coco.json",
         day32 / "captions_rgb_val_full.jsonl"),
    ]

    results = []
    for domain, split, coco_path, ref_coco_path, captions_path in jobs:
        r = scan(domain, split, coco_path, ref_coco_path, captions_path)
        results.append(r)
        print(f"[{domain}/{split}] multi-class images={r['multi_class_images']}  "
              f"missing>=1 class={r['images_missing_ge1']}  "
              f"missing_rate={r['missing_rate']*100:.2f}%  "
              f"(count>=5 collapse: {r['reason_count5_collapse']}, "
              f"pure top-2 truncation w/ 3+ classes: {r['reason_top2_trunc_only']})")
        print(f"    most-missed classes: {r['missed_class_breakdown'][:8]}")

    with open(ROOT / "Phase3" / "day38_caption_completeness_scan_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
