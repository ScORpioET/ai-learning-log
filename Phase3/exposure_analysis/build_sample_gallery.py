"""
Day40:把 train_pairs_sample.json 選出的 10 組配對,疊上 bbox(依 pct_50
外擴層級的判斷結果上色:紅=有問題/太像背景,綠=正常),連同每個 bbox 的
量化數字,存成 gallery_data.json 給下一步的 artifact 頁面用。
"""
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent
TH_DIR = ROOT / "images_thermal_train"
RGB_DIR = ROOT / "images_rgb_train"

RED = (230, 60, 50)
GREEN = (60, 190, 90)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    pairs = json.load(open(HERE / "train_pairs_sample.json"))["sampled"]
    summary = json.load(open(HERE / "summary_results.json"))
    rgb_bright_thresh = summary["rgb"]["50"]["bright_thresh"]
    rgb_median_thresh = summary["rgb"]["50"]["median_thresh"]
    th_thresh = summary["thermal"]["50"]["thresh"]

    rgb_by_file = defaultdict(list)
    for r in load_jsonl(HERE / "rgb_exposure_results.jsonl"):
        rgb_by_file[r["file_name"]].append(r)
    th_by_file = defaultdict(list)
    for r in load_jsonl(HERE / "thermal_background_results.jsonl"):
        th_by_file[r["file_name"]].append(r)

    th_coco = json.load(open(TH_DIR / "coco.json"))
    rgb_coco = json.load(open(RGB_DIR / "coco.json"))
    th_ann_by_id = {a["id"]: a for a in th_coco["annotations"]}
    rgb_ann_by_id = {a["id"]: a for a in rgb_coco["annotations"]}

    out = []
    for p in pairs:
        tf, rf = p["thermal_file"], p["rgb_file"]

        th_im = Image.open(TH_DIR / tf).convert("RGB")
        th_draw = ImageDraw.Draw(th_im)
        th_boxes = []
        for rec in th_by_file.get(tf, []):
            ann = th_ann_by_id[rec["ann_id"]]
            x, y, w, h = ann["bbox"]
            flagged = rec["pct_50"] is not None and rec["pct_50"]["diff"] <= th_thresh
            color = RED if flagged else GREEN
            th_draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            th_boxes.append({
                "category": rec["category"], "flagged": flagged,
                "diff_25": rec["pct_25"]["diff"] if rec["pct_25"] else None,
                "diff_50": rec["pct_50"]["diff"] if rec["pct_50"] else None,
                "diff_75": rec["pct_75"]["diff"] if rec["pct_75"] else None,
            })

        rgb_im = Image.open(RGB_DIR / rf).convert("RGB")
        rgb_draw = ImageDraw.Draw(rgb_im)
        rgb_boxes = []
        for rec in rgb_by_file.get(rf, []):
            ann = rgb_ann_by_id[rec["ann_id"]]
            x, y, w, h = ann["bbox"]
            pct50 = rec["pct_50"]
            flagged = pct50 is not None and (pct50["bright_diff"] >= rgb_bright_thresh
                                              or pct50["median_luminance"] <= rgb_median_thresh)
            color = RED if flagged else GREEN
            rgb_draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
            rgb_boxes.append({
                "category": rec["category"], "flagged": flagged,
                "bright_diff_50": pct50["bright_diff"] if pct50 else None,
                "median_luminance_50": pct50["median_luminance"] if pct50 else None,
            })

        # resize for artifact size, then save to temp files for the next step to encode
        th_max_w, rgb_max_w = 480, 480
        if th_im.width > th_max_w:
            r = th_max_w / th_im.width
            th_im = th_im.resize((th_max_w, int(th_im.height * r)))
        if rgb_im.width > rgb_max_w:
            r = rgb_max_w / rgb_im.width
            rgb_im = rgb_im.resize((rgb_max_w, int(rgb_im.height * r)))

        th_out_path = HERE / "gallery_imgs" / f"{Path(tf).stem}_th.jpg"
        rgb_out_path = HERE / "gallery_imgs" / f"{Path(rf).stem}_rgb.jpg"
        th_out_path.parent.mkdir(exist_ok=True)
        th_im.save(th_out_path, quality=82)
        rgb_im.save(rgb_out_path, quality=82)

        out.append({
            "thermal_file": tf, "rgb_file": rf,
            "thermal_video_id": p["thermal_video_id"], "rgb_video_id": p["rgb_video_id"],
            "frame_num": p["frame_num"],
            "thermal_img_path": str(th_out_path), "rgb_img_path": str(rgb_out_path),
            "thermal_boxes": th_boxes, "rgb_boxes": rgb_boxes,
        })

    with open(HERE / "gallery_manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(out)} pairs -> gallery_manifest.json + gallery_imgs/")


if __name__ == "__main__":
    main()
