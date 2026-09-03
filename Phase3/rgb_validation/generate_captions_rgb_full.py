"""
Day38 RGB 全量版(對齊 thermal full_v2 的生成邏輯:不套用任何 tiny-bbox 面積
門檻,等同於 generate_captions.py 在 --filter-tiny 未開啟時的行為)。

跟 generate_captions_rgb.py 唯一的差異:拿掉 RGB_MIN_AREA_PCT 的
area_pct < threshold 過濾判斷,其餘(long-tail 修復、position/distance
label、build_caption 等)逐行照抄不改。

用法:
    python generate_captions_rgb_full.py --split train --out captions_rgb_train_full.jsonl
    python generate_captions_rgb_full.py --split val --long-tail-ref-split train --out captions_rgb_val_full.jsonl
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gc_orig", Path.home() / "ai-transition-2026" / "thermal_dataset" / "generate_captions.py"
)
gc = importlib.util.module_from_spec(spec)
sys.modules["gc_orig"] = gc
spec.loader.exec_module(gc)


def _split_dir(domain, split):
    # Day39:test split 資料夾命名跟 train/val 不同(video_{domain}_test,
    # 不是 images_{domain}_test),跟 thermal_dataset/generate_captions.py
    # 的同一個特例對齊。
    if split == "test":
        return f"video_{domain}_test"
    return f"images_{domain}_{split}"


def main(split, out_path, long_tail_ref_split=None):
    root = Path.home() / "ai-transition-2026" / "thermal_dataset" / _split_dir("rgb", split)
    coco = json.load(open(root / "coco.json"))

    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    dynamic_ids = {c["id"] for c in coco["categories"] if c["name"] in gc.DYNAMIC_CLASSES}
    static_ids = {c["id"] for c in coco["categories"] if c["name"] in gc.STATIC_CONTEXT_CLASSES}

    ref_split = long_tail_ref_split or split
    if ref_split != split:
        ref_root = Path.home() / "ai-transition-2026" / "thermal_dataset" / _split_dir("rgb", ref_split)
        ref_coco = json.load(open(ref_root / "coco.json"))
        ref_id2name = {c["id"]: c["name"] for c in ref_coco["categories"]}
        cat_counts_by_name = Counter(ref_id2name[a["category_id"]] for a in ref_coco["annotations"])
        print(f"[info] long-tail 門檻改用「{ref_split}」split(RGB)的 counts 當基準")
    else:
        cat_counts_by_name = Counter(id2name[a["category_id"]] for a in coco["annotations"])

    long_tail_names = {
        name for name in gc.DYNAMIC_CLASSES
        if cat_counts_by_name.get(name, 0) < gc.LONG_TAIL_THRESHOLD
    }
    if long_tail_names:
        detail = ", ".join(f"{name}({cat_counts_by_name.get(name, 0)})" for name in sorted(long_tail_names))
        print(f"[info] long-tail 併入「{gc.LONG_TAIL_LABEL}」: {detail}")

    near_thresh, far_thresh = gc.compute_area_thresholds(coco, dynamic_ids)
    print(f"[info] near_thresh={near_thresh:.4f}, far_thresh={far_thresh:.4f}")
    print("[info] 全量版:不套用 tiny-bbox 面積門檻(對齊 thermal full_v2 / --filter-tiny 未開啟)")

    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    results = []
    for img_id, im in img_meta.items():
        w_img, h_img = im["width"], im["height"]
        dyn_objs = []
        static_present = set()

        for ann in anns_by_img.get(img_id, []):
            cat_name = id2name[ann["category_id"]]
            if ann["category_id"] in static_ids:
                static_present.add(cat_name)
                continue
            if ann["category_id"] not in dynamic_ids:
                continue

            occluded = ann.get("extra_info", {}).get("occluded", "") or ""
            if occluded == gc.OCCLUDED_DIFFICULT:
                continue

            x, y, w, h = ann["bbox"]
            area = w * h

            pos = gc.position_label(ann["bbox"], w_img, h_img)
            dist = gc.distance_label(ann["bbox"], w_img, h_img, near_thresh, far_thresh)

            if cat_name in long_tail_names:
                en_name = gc.LONG_TAIL_LABEL
            else:
                en_name = gc.DYNAMIC_CLASSES.get(cat_name, gc.LONG_TAIL_LABEL)
            dyn_objs.append((cat_name, en_name, pos, dist, area))

        dyn_objs.sort(key=lambda o: -o[4])

        caption = gc.build_caption(dyn_objs, im.get("extra_info", {}), img_id)
        if caption is None:
            continue

        results.append({
            "file_name": im["file_name"],
            "caption": caption,
            "num_objects": len(dyn_objs),
            "has_static_context": sorted(static_present),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] {len(results)} captions written to {out_path} "
          f"(out of {len(img_meta)} images, {len(img_meta) - len(results)} skipped: zero objects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--long-tail-ref-split", default=None, choices=["train", "val"])
    args = parser.parse_args()
    main(args.split, args.out, args.long_tail_ref_split)
