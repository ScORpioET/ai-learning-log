import json
import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
OUT = Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation"

TH_DIR = ROOT / "images_thermal_train"
RGB_DIR = ROOT / "images_rgb_train"


def norm_name(name):
    base = re.sub(r"\.(avi|zip|mp4)$", "", name, flags=re.I)
    base = re.sub(r"[_\s]+", " ", base).strip().lower()
    base = re.sub(r"\bcam\d\b", "", base).strip()
    return base


def frame_num(fn):
    m = re.search(r"frame-(\d+)", fn)
    return int(m.group(1)) if m else None


th_idx = json.load(open(TH_DIR / "index.json"))
rgb_idx = json.load(open(RGB_DIR / "index.json"))
th_id2name = {v["id"]: norm_name(v["filename"]) for v in th_idx["videos"]}
rgb_id2name = {v["id"]: norm_name(v["filename"]) for v in rgb_idx["videos"]}

th_coco = json.load(open(TH_DIR / "coco.json"))
rgb_coco = json.load(open(RGB_DIR / "coco.json"))

th_cat_by_id = {c["id"]: c["name"] for c in th_coco["categories"]}
rgb_cat_by_id = {c["id"]: c["name"] for c in rgb_coco["categories"]}

# ---------------------------------------------------------------------------
# 影片配對(查程式碼/資料確認的事實,不是背景提到的「74 支」——這裡如實記錄
# 用什麼方法配對、配出幾支,對不上「74」這個數字就直接說對不上,不硬湊)
# ---------------------------------------------------------------------------
th_frames_by_name = defaultdict(dict)  # name -> {frame_num: image_dict}
for img in th_coco["images"]:
    vid = img["extra_info"]["video_id"]
    name = th_id2name.get(vid)
    fn = frame_num(img["file_name"])
    if name and fn is not None:
        th_frames_by_name[name][fn] = img

rgb_frames_by_name = defaultdict(dict)
for img in rgb_coco["images"]:
    vid = img["extra_info"]["video_id"]
    name = rgb_id2name.get(vid)
    fn = frame_num(img["file_name"])
    if name and fn is not None:
        rgb_frames_by_name[name][fn] = img

common_names = sorted(set(th_frames_by_name) & set(rgb_frames_by_name))
pairing_stats = {
    "method": "normalize video filename (strip ext/case/spaces/camN), "
              "match thermal vs rgb 'videos' list in images_*_train/index.json, "
              "then require the same frame-XXXXXX number present in both "
              "images_*_train/coco.json",
    "thermal_train_video_count": len(th_idx["videos"]),
    "rgb_train_video_count": len(rgb_idx["videos"]),
    "matched_video_count": len(common_names),
    "note": "background 提到 74 支,這裡用檔名正規化配對配出的是這個數字,"
            "跟 74 對不上,如實記錄,沒有為了湊數字調整配對方法",
}

# 每個配對影片、frameIndex<=1500 的重疊 frame pair
pairs = []
for name in common_names:
    th_frames = th_frames_by_name[name]
    rgb_frames = rgb_frames_by_name[name]
    common_fn = set(th_frames) & set(rgb_frames)
    for fn in common_fn:
        if fn <= 1500:
            pairs.append((name, fn, th_frames[fn], rgb_frames[fn]))

pairing_stats["early_frame_pairs_le_1500"] = len(pairs)
print("[A3] video pairing:", pairing_stats)

# ---------------------------------------------------------------------------
# 找「thermal 有標註、RGB 同位置沒有」的候選案例
# ---------------------------------------------------------------------------
th_anns_by_image = defaultdict(list)
for a in th_coco["annotations"]:
    th_anns_by_image[a["image_id"]].append(a)
rgb_anns_by_image = defaultdict(list)
for a in rgb_coco["annotations"]:
    rgb_anns_by_image[a["image_id"]].append(a)

CENTER_DIST_THRESH = 0.12  # 正規化座標下的中心距離門檻,判定「RGB 有沒有對應標註」用,
# 不要求同類別/同大小(兩邊解析度、長寬比、視角本來就不同),只看「這個位置附近
# RGB 有沒有標任何東西」——寬鬆判準,方法論選擇,不是查出來的事實。

candidates = []
random.seed(1337)
random.shuffle(pairs)

for name, fn, th_img, rgb_img in pairs:
    th_anns = th_anns_by_image.get(th_img["id"], [])
    if not th_anns:
        continue
    rgb_anns = rgb_anns_by_image.get(rgb_img["id"], [])
    tw, th_h = th_img["width"], th_img["height"]
    rw, rh = rgb_img["width"], rgb_img["height"]

    rgb_centers_norm = []
    for a in rgb_anns:
        x, y, w, h = a["bbox"]
        rgb_centers_norm.append(((x + w / 2) / rw, (y + h / 2) / rh))

    for a in th_anns:
        x, y, w, h = a["bbox"]
        cx, cy = (x + w / 2) / tw, (y + h / 2) / th_h
        matched = False
        for (rcx, rcy) in rgb_centers_norm:
            if abs(rcx - cx) < CENTER_DIST_THRESH and abs(rcy - cy) < CENTER_DIST_THRESH:
                matched = True
                break
        if not matched:
            candidates.append({
                "video_name": name, "frame_num": fn,
                "th_image": th_img, "rgb_image": rgb_img,
                "th_ann": a, "th_cat": th_cat_by_id.get(a["category_id"], "?"),
                "norm_bbox": (cx, cy, w / tw, h / th_h),
            })

print(f"[A3] 找到 {len(candidates)} 個「thermal 有標註、RGB 同位置沒對應標註」候選")

random.shuffle(candidates)
sample = candidates[:200]
print(f"[A3] 抽樣 {len(sample)} 組做曝光診斷")

# ---------------------------------------------------------------------------
# 對每個候選案例:把 thermal bbox 的正規化座標映射到 RGB 圖片上,算該區域的
# 像素亮度分布,分成過曝/過暗/其他三類
# ---------------------------------------------------------------------------
def classify_exposure(rgb_path, norm_bbox, rw, rh):
    cx, cy, nw, nh = norm_bbox
    # 給一點 padding,避免 bbox 太小取不到足夠像素;padding 後 clip 到圖片範圍內
    half_w = max(nw * rw / 2, 8)
    half_h = max(nh * rh / 2, 8)
    cx_px, cy_px = cx * rw, cy * rh
    x0 = int(max(cx_px - half_w, 0))
    x1 = int(min(cx_px + half_w, rw))
    y0 = int(max(cy_px - half_h, 0))
    y1 = int(min(cy_px + half_h, rh))
    if x1 <= x0 or y1 <= y0:
        return None

    img = Image.open(rgb_path).convert("L")
    crop = np.array(img.crop((x0, y0, x1, y1)), dtype=np.float32)
    if crop.size == 0:
        return None

    overexposed_frac = float((crop >= 250).mean())
    underexposed_frac = float((crop <= 10).mean())
    mean_brightness = float(crop.mean())

    # 分類規則(方法論選擇,不是查出來的事實):
    # - 該區域超過 40% 的像素落在 >=250(近乎純白)→ 過曝
    # - 該區域超過 40% 的像素落在 <=10(近乎純黑)→ 過暗
    # - 其他 → 亮度正常但仍缺標(距離/遮擋/標註疏漏等其他原因)
    if overexposed_frac >= 0.4:
        label = "overexposed"
    elif underexposed_frac >= 0.4:
        label = "underexposed"
    else:
        label = "other"
    return {
        "label": label,
        "overexposed_frac": overexposed_frac,
        "underexposed_frac": underexposed_frac,
        "mean_brightness": mean_brightness,
    }


results = []
for c in sample:
    rgb_img = c["rgb_image"]
    rgb_path = RGB_DIR / rgb_img["file_name"]
    if not rgb_path.exists():
        continue
    r = classify_exposure(rgb_path, c["norm_bbox"], rgb_img["width"], rgb_img["height"])
    if r is None:
        continue
    r.update({
        "video_name": c["video_name"], "frame_num": c["frame_num"],
        "th_cat": c["th_cat"], "th_image_file": c["th_image"]["file_name"],
        "rgb_image_file": rgb_img["file_name"],
    })
    results.append(r)

print(f"[A3] 實際成功分析 {len(results)} 組(有些檔案可能讀不到)")

labels = [r["label"] for r in results]
label_counts = {l: labels.count(l) for l in set(labels)}
label_props = {l: c / len(results) for l, c in label_counts.items()} if results else {}

a3_output = {
    "pairing_stats": pairing_stats,
    "n_candidates_total": len(candidates),
    "n_sampled": len(sample),
    "n_analyzed": len(results),
    "center_dist_threshold": CENTER_DIST_THRESH,
    "label_counts": label_counts,
    "label_proportions": label_props,
    "per_category_breakdown": {},
}

# 額外拆解:每個類別各自的曝光問題比例(person 尤其重要,因為背景提到 person 落差)
by_cat = defaultdict(list)
for r in results:
    by_cat[r["th_cat"]].append(r["label"])
for cat, labs in by_cat.items():
    a3_output["per_category_breakdown"][cat] = {
        "n": len(labs),
        "overexposed": labs.count("overexposed") / len(labs),
        "underexposed": labs.count("underexposed") / len(labs),
        "other": labs.count("other") / len(labs),
    }

json.dump(a3_output, open(OUT / "a3_exposure_results.json", "w"), indent=2)
json.dump(results, open(OUT / "a3_exposure_raw_samples.json", "w"), indent=2)
print("[A3] label_proportions:", label_props)
print("[A3] per_category_breakdown:", json.dumps(a3_output["per_category_breakdown"], indent=2))
