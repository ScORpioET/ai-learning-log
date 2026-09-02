"""
Day38: frame-correspondence audit for the two RGB<->thermal video-pairing
methods that have been used so far (74 支 via index.json description field,
119 支 via filename normalization). This script only produces numbers and
side-by-side comparison images -- it does NOT decide whether either pairing
method is trustworthy, and does NOT touch/delete any existing pairing record.

Outputs:
  - Phase3/rgb_validation/day38_frame_correspondence_stats.json
  - Phase3/rgb_validation/day38_frame_correspondence_audit_images/<method>/*.jpg
    (NOT committed to git -- too large, folder is just left on disk)
"""
import json
import re
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw

ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
OUT_DIR = Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation"
IMG_OUT = OUT_DIR / "day38_frame_correspondence_audit_images"

TH_DIR = ROOT / "images_thermal_train"
RGB_DIR = ROOT / "images_rgb_train"

TH_COLOR = (255, 60, 60)     # thermal bbox: red
RGB_COLOR = (60, 220, 60)    # rgb bbox: green

N_SUSPICIOUS = 30
N_CONTROL = 10

# ---------------------------------------------------------------------------
# load index.json / coco.json for both sides
# ---------------------------------------------------------------------------
th_idx = json.load(open(TH_DIR / "index.json"))
rgb_idx = json.load(open(RGB_DIR / "index.json"))
th_coco = json.load(open(TH_DIR / "coco.json"))
rgb_coco = json.load(open(RGB_DIR / "coco.json"))

th_cat_by_id = {c["id"]: c["name"] for c in th_coco["categories"]}
rgb_cat_by_id = {c["id"]: c["name"] for c in rgb_coco["categories"]}

rgb_video_ids = set(v["id"] for v in rgb_idx["videos"])
th_video_ids = set(v["id"] for v in th_idx["videos"])


def frame_num(fn):
    m = re.search(r"frame-(\d+)", fn)
    return int(m.group(1)) if m else None


# images/annotations indexed by video id, keyed by frame number
def build_video_frame_index(coco):
    """video_id -> {frame_num: image_dict}"""
    out = defaultdict(dict)
    for img in coco["images"]:
        vid = img["extra_info"]["video_id"]
        fn = frame_num(img["file_name"])
        if fn is not None:
            out[vid][fn] = img
    return out


th_frames_by_vid = build_video_frame_index(th_coco)
rgb_frames_by_vid = build_video_frame_index(rgb_coco)

th_anns_by_image = defaultdict(list)
for a in th_coco["annotations"]:
    th_anns_by_image[a["image_id"]].append(a)
rgb_anns_by_image = defaultdict(list)
for a in rgb_coco["annotations"]:
    rgb_anns_by_image[a["image_id"]].append(a)


# ---------------------------------------------------------------------------
# Method A: index.json videos[].description embeds {"RGB": "<rgb_video_id>"}
# ---------------------------------------------------------------------------
def method_a_pairs():
    pattern = re.compile(r'\{"RGB":\s*"([a-zA-Z0-9]+)"\}')
    pairs = []
    for v in th_idx["videos"]:
        m = pattern.search(v.get("description", ""))
        if m:
            rgb_id = m.group(1)
            if rgb_id in rgb_video_ids:
                pairs.append((v["id"], rgb_id))
    return pairs


# ---------------------------------------------------------------------------
# Method B: normalize videos[].filename (strip ext/case/spaces/camN), match
# ---------------------------------------------------------------------------
def norm_name(name):
    base = re.sub(r"\.(avi|zip|mp4)$", "", name, flags=re.I)
    base = re.sub(r"[_\s]+", " ", base).strip().lower()
    base = re.sub(r"\bcam\d\b", "", base).strip()
    return base


def method_b_pairs():
    th_id2name = {v["id"]: norm_name(v["filename"]) for v in th_idx["videos"]}
    rgb_id2name = {v["id"]: norm_name(v["filename"]) for v in rgb_idx["videos"]}
    rgb_name2id = {n: i for i, n in rgb_id2name.items()}
    pairs = []
    for th_id, name in th_id2name.items():
        if name in rgb_name2id:
            pairs.append((th_id, rgb_name2id[name]))
    return pairs


METHODS = {
    "method_A_description_74": method_a_pairs(),
    "method_B_filename_norm_119": method_b_pairs(),
}


def diff_score(n_th, n_rgb, cats_th, cats_rgb):
    """
    bbox 數量絕對差 + 5 * 類別集合對稱差集大小。
    理由:bbox 數量差可以到幾十(尺度大),類別集合通常只有 0~15 種,直接相加
    類別差異的訊號會被 bbox 數量差蓋掉;乘 5 讓兩種訊號量級接近,「類別完全不同」
    跟「數量差 5 顆」大致算同一等級的異常,方便排序時兩種問題都排得上前面。
    這是方法論選擇,不是查出來的事實。
    """
    return abs(n_th - n_rgb) + 5 * len(cats_th.symmetric_difference(cats_rgb))


def draw_boxed(img_path, anns, cat_by_id, color):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for a in anns:
        x, y, w, h = a["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = cat_by_id.get(a["category_id"], "?")
        draw.text((x + 2, max(y - 12, 0)), label, fill=color)
    return im


def make_side_by_side(th_img_dict, rgb_img_dict, th_anns, rgb_anns):
    th_path = TH_DIR / th_img_dict["file_name"]
    rgb_path = RGB_DIR / rgb_img_dict["file_name"]
    th_im = draw_boxed(th_path, th_anns, th_cat_by_id, TH_COLOR)
    rgb_im = draw_boxed(rgb_path, rgb_anns, rgb_cat_by_id, RGB_COLOR)

    h = max(th_im.height, rgb_im.height)
    canvas = Image.new("RGB", (th_im.width + rgb_im.width, h), (30, 30, 30))
    canvas.paste(th_im, (0, 0))
    canvas.paste(rgb_im, (th_im.width, 0))
    return canvas


stats = {}
for method_name, video_pairs in METHODS.items():
    method_dir = IMG_OUT / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    both_frame_pairs = []  # (th_vid, rgb_vid, frame_num, th_img, rgb_img)
    total_both = 0
    total_th_only = 0
    total_rgb_only = 0
    total_th_frames = 0
    total_rgb_frames = 0
    frame_count_mismatch_videos = 0

    for th_vid, rgb_vid in video_pairs:
        th_frames = th_frames_by_vid.get(th_vid, {})
        rgb_frames = rgb_frames_by_vid.get(rgb_vid, {})
        th_fn = set(th_frames)
        rgb_fn = set(rgb_frames)
        both = th_fn & rgb_fn
        th_only = th_fn - rgb_fn
        rgb_only = rgb_fn - th_fn

        total_both += len(both)
        total_th_only += len(th_only)
        total_rgb_only += len(rgb_only)
        total_th_frames += len(th_fn)
        total_rgb_frames += len(rgb_fn)
        if len(th_fn) != len(rgb_fn):
            frame_count_mismatch_videos += 1

        for fn in both:
            both_frame_pairs.append((th_vid, rgb_vid, fn, th_frames[fn], rgb_frames[fn]))

    union = total_both + total_th_only + total_rgb_only
    frame_stats = {
        "n_video_pairs": len(video_pairs),
        "n_both_frame_pairs": total_both,
        "n_thermal_only_frames": total_th_only,
        "n_rgb_only_frames": total_rgb_only,
        "both_over_union_pct": round(100 * total_both / union, 2) if union else None,
        "thermal_only_over_union_pct": round(100 * total_th_only / union, 2) if union else None,
        "rgb_only_over_union_pct": round(100 * total_rgb_only / union, 2) if union else None,
        "n_videos_with_frame_count_mismatch": frame_count_mismatch_videos,
        "n_videos_total": len(video_pairs),
        "total_thermal_frames_in_paired_videos": total_th_frames,
        "total_rgb_frames_in_paired_videos": total_rgb_frames,
    }

    # -----------------------------------------------------------------
    # per-both-frame bbox count / category diff score
    # -----------------------------------------------------------------
    scored = []
    for th_vid, rgb_vid, fn, th_img, rgb_img in both_frame_pairs:
        th_anns = th_anns_by_image.get(th_img["id"], [])
        rgb_anns = rgb_anns_by_image.get(rgb_img["id"], [])
        cats_th = set(th_cat_by_id.get(a["category_id"], "?") for a in th_anns)
        cats_rgb = set(rgb_cat_by_id.get(a["category_id"], "?") for a in rgb_anns)
        score = diff_score(len(th_anns), len(rgb_anns), cats_th, cats_rgb)
        scored.append({
            "th_vid": th_vid, "rgb_vid": rgb_vid, "frame_num": fn,
            "th_img": th_img, "rgb_img": rgb_img,
            "th_anns": th_anns, "rgb_anns": rgb_anns,
            "n_th": len(th_anns), "n_rgb": len(rgb_anns),
            "cats_th": cats_th, "cats_rgb": cats_rgb,
            "score": score,
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    suspicious = scored[:N_SUSPICIOUS]
    control = scored[-N_CONTROL:] if len(scored) >= N_CONTROL else []

    frame_stats["n_scored_both_frame_pairs"] = len(scored)
    frame_stats["n_suspicious_selected"] = len(suspicious)
    frame_stats["n_control_selected"] = len(control)
    frame_stats["max_diff_score"] = scored[0]["score"] if scored else None
    frame_stats["min_diff_score"] = scored[-1]["score"] if scored else None

    all_scores = [r["score"] for r in scored]
    n_all = len(all_scores) or 1
    frame_stats["score_eq_0_pct"] = round(100 * sum(1 for s in all_scores if s == 0) / n_all, 2)
    frame_stats["score_1_to_5_pct"] = round(100 * sum(1 for s in all_scores if 1 <= s <= 5) / n_all, 2)
    frame_stats["score_6_to_15_pct"] = round(100 * sum(1 for s in all_scores if 6 <= s <= 15) / n_all, 2)
    frame_stats["score_gt_15_pct"] = round(100 * sum(1 for s in all_scores if s > 15) / n_all, 2)

    # -----------------------------------------------------------------
    # render images
    # -----------------------------------------------------------------
    def render_group(items, tag):
        for rank, r in enumerate(items, start=1):
            canvas = make_side_by_side(r["th_img"], r["rgb_img"], r["th_anns"], r["rgb_anns"])
            fname = (
                f"{rank:03d}_{tag}_diff{r['score']:04d}_frame{r['frame_num']:06d}_"
                f"th{r['n_th']}bb_rgb{r['n_rgb']}bb_"
                f"thvid-{r['th_vid']}_rgbvid-{r['rgb_vid']}.jpg"
            )
            canvas.save(method_dir / fname, quality=85)

    render_group(suspicious, "SUSPECT")
    render_group(control, "CONTROL")

    stats[method_name] = frame_stats
    print(f"[{method_name}]", json.dumps(frame_stats, indent=2, ensure_ascii=False))

json.dump(stats, open(OUT_DIR / "day38_frame_correspondence_stats.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved stats to", OUT_DIR / "day38_frame_correspondence_stats.json")
print("saved images under", IMG_OUT)
