"""
互動校準工具:丟一張圖片進來(可以是原圖、blend、或 flare 圖,只要檔名
跟我們本地已經跑好的 Flare7K++ 結果同名,就會自動配對出「原圖/blend/flare」
三張),用滑桿即時調整黑塊偵測 + radial profile 光暈分析的所有參數,
即時看標註結果跟光暈面積怎麼變。

執行方式:
    cd Phase3/flare_removal
    PYTHONPATH=/home/jack/ai-transition-2026/Phase3/caption_fusion/.pylibs \
        streamlit run calibrate_flare_halo_ui.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

HERE = Path(__file__).parent
DATASET_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"

SPLIT_DIRS = {
    "out_train": {"orig": DATASET_ROOT / "images_rgb_train" / "data", "blend": HERE / "out_train" / "blend", "flare": HERE / "out_train" / "flare"},
    "out_val": {"orig": DATASET_ROOT / "images_rgb_val" / "data", "blend": HERE / "out_val" / "blend", "flare": HERE / "out_val" / "flare"},
    "out_test": {"orig": DATASET_ROOT / "video_rgb_test" / "data", "blend": HERE / "out_test" / "blend", "flare": HERE / "out_test" / "flare"},
}


def find_trio(filename):
    """依檔名在三個 split 裡找對應的原圖/blend/flare,回傳 (split, {orig,blend,flare} 路徑) 或 None"""
    for split, dirs in SPLIT_DIRS.items():
        fp = dirs["flare"] / filename
        if fp.exists():
            return split, {k: v / filename for k, v in dirs.items()}
    return None, None


# flare/ 輸出解析度不統一(高度固定512,寬度576-768,見
# flare_resolution_distribution.json),MIN_AREA/RING_DILATE 改用相對
# REF_W x REF_H 這個參考解析度的比例值,套用時依實際圖片大小換算 px。
REF_W, REF_H = 682, 512


def find_blobs(flare_bgr, black_thresh, min_area_ref_px, ring_dilate_ref_px, bright_ring_thresh):
    h, w = flare_bgr.shape[:2]
    min_area_ratio = min_area_ref_px / (REF_W * REF_H)
    ring_dilate_ratio = ring_dilate_ref_px / np.hypot(REF_W, REF_H)
    min_area = min_area_ratio * (w * h)
    ring_dilate = max(1, round(ring_dilate_ratio * np.hypot(w, h)))

    black_mask = cv2.inRange(flare_bgr, (0, 0, 0), (black_thresh, black_thresh, black_thresh))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    maxchan = flare_bgr.astype(np.float32).max(axis=2)
    kernel = np.ones((ring_dilate * 2 + 1, ring_dilate * 2 + 1), np.uint8)

    blobs = []
    for lbl in range(1, n_labels):
        x, y, bw, bh, area = stats[lbl]
        if area < min_area:
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        comp_mask = (labels == lbl).astype(np.uint8)
        dilated = cv2.dilate(comp_mask, kernel)
        ring = (dilated > 0) & (comp_mask == 0)
        if ring.sum() == 0:
            continue
        ring_mean = float(maxchan[ring].mean())
        if ring_mean < bright_ring_thresh:
            continue
        cx, cy = centroids[lbl]
        blobs.append({
            "area": int(area), "bbox": [int(x), int(y), int(bw), int(bh)],
            "centroid": (float(cx), float(cy)), "ring_brightness": round(ring_mean, 1),
        })
    blobs.sort(key=lambda b: -b["area"])
    return blobs


def radial_profile_halo(flare_bgr, centroid, n_dirs, max_dist, bg_thresh, sustain):
    intensity = flare_bgr.astype(np.float32).max(axis=2)
    h, w = intensity.shape
    cx, cy = centroid

    outer_points = []
    widths = []
    for k in range(n_dirs):
        theta = 2 * np.pi * k / n_dirs
        dx, dy = np.cos(theta), np.sin(theta)
        dists = np.arange(1, max_dist)
        xs = (cx + dists * dx).round().astype(int)
        ys = (cy + dists * dy).round().astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys, dists = xs[valid], ys[valid], dists[valid]
        if len(xs) < 10:
            outer_points.append((cx, cy))
            continue
        vals = intensity[ys, xs]

        diffs = np.diff(vals)
        jump_idx = int(np.argmax(diffs)) if len(diffs) else 0

        local_min = float(vals[:jump_idx + 1].min()) if jump_idx > 0 else float(vals[0])
        local_max = float(vals[jump_idx:jump_idx + 20].max()) if jump_idx + 1 < len(vals) else float(vals[jump_idx])
        span = max(local_max - local_min, 1e-6)
        lo_thresh, hi_thresh = local_min + 0.1 * span, local_min + 0.9 * span

        start_idx = jump_idx
        while start_idx > 0 and vals[start_idx] > lo_thresh:
            start_idx -= 1
        end_idx = jump_idx
        while end_idx < len(vals) - 1 and vals[end_idx] < hi_thresh:
            end_idx += 1
        widths.append(float(dists[end_idx] - dists[start_idx]))

        outer_idx = len(vals) - 1
        for idx in range(jump_idx, max(len(vals) - sustain, jump_idx + 1)):
            if np.all(vals[idx:idx + sustain] < bg_thresh):
                outer_idx = idx
                break
        r = float(dists[outer_idx])
        outer_points.append((cx + r * dx, cy + r * dy))

    poly = np.array(outer_points, dtype=np.float32)
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = float(abs(area) / 2.0)

    return {
        "halo_area_px": round(area, 1),
        "jump_width_median_px": round(float(np.median(widths)), 2) if widths else None,
        "polygon": poly,
    }


def annotate(flare_bgr, blobs_with_halo):
    out = flare_bgr.copy()
    for i, (bbox, halo) in enumerate(blobs_with_halo, 1):
        x, y, bw, bh = bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        poly = halo["polygon"].astype(np.int32)
        cv2.polylines(out, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
        line1 = f"#{i} halo={int(halo['halo_area_px'])}px"
        line2 = f"jump_w={halo['jump_width_median_px']}px"
        y1 = max(y - 28, 14)
        y2 = y1 + 22
        for text, ty in ((line1, y1), (line2, y2)):
            cv2.putText(out, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
    return out


def bgr_to_rgb_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


st.set_page_config(layout="wide", page_title="Flare 光暈參數校準")
st.title("Flare7K++ 光暈偵測參數校準工具")
st.caption("丟一張圖片進來(原圖/blend/flare 任一張都可以,靠檔名自動配對本地已跑好的三張圖),"
           "用左側滑桿調整參數,即時看黑塊偵測跟光暈面積怎麼變。")

uploaded = st.file_uploader("丟圖片進來(檔名要跟本地已處理過的圖片一致)", type=["jpg", "jpeg", "png"])

with st.sidebar:
    st.header("黑色核心偵測")
    black_thresh = st.slider("BLACK_THRESH(黑色門檻)", 0, 100, 30)
    min_area_ref = st.slider(f"MIN_AREA(以 {REF_W}x{REF_H} 為基準的參考 px,套用時依實際圖片解析度換算)", 0, 1000, 500, step=10)
    ring_dilate_ref = st.slider(f"RING_DILATE(以 {REF_W}x{REF_H} 為基準的參考 px,套用時依實際圖片解析度換算)", 1, 40, 15)
    bright_ring_thresh = st.slider("BRIGHT_RING_THRESH(外環亮度門檻)", 0, 255, 80)

    st.header("Radial Profile 光暈分析")
    n_dirs = st.slider("N_DIRECTIONS(取樣方向數)", 8, 72, 36, step=4)
    max_dist = st.slider("MAX_DIST(單方向最大搜尋距離 px)", 50, 400, 250, step=10)
    bg_thresh = st.slider("BG_THRESH(背景亮度門檻)", 0, 255, 40)
    sustain = st.slider("SUSTAIN(連續幾點才算回背景)", 1, 15, 5)

if uploaded is None:
    st.info("請上傳一張圖片開始。")
else:
    filename = uploaded.name
    split, paths = find_trio(filename)
    if split is None:
        st.error(f"在本地的 out_train/out_val/out_test 裡找不到跟「{filename}」同名的已處理結果,"
                 f"這個工具只能校準已經跑過 Flare7K++ 的圖片,沒辦法即時跑新圖片的模型推論。")
    else:
        st.success(f"找到對應資料,split = {split}")
        orig = cv2.imread(str(paths["orig"]))
        blend = cv2.imread(str(paths["blend"]))
        flare = cv2.imread(str(paths["flare"]))

        blobs = find_blobs(flare, black_thresh, min_area_ref, ring_dilate_ref, bright_ring_thresh)
        blobs_with_halo = []
        table_rows = []
        for b in blobs:
            halo = radial_profile_halo(flare, b["centroid"], n_dirs, max_dist, bg_thresh, sustain)
            blobs_with_halo.append((b["bbox"], halo))
            table_rows.append({
                "黑核面積(px)": b["area"], "外環亮度": b["ring_brightness"],
                "光暈面積(px)": halo["halo_area_px"], "跳變寬度(px)": halo["jump_width_median_px"],
            })
        annotated_flare = annotate(flare, blobs_with_halo)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("原圖")
            st.image(bgr_to_rgb_pil(orig), use_container_width=True)
        with col2:
            st.subheader("Blend(去光斑後)")
            st.image(bgr_to_rgb_pil(blend), use_container_width=True)
        with col3:
            st.subheader(f"Flare + 標註(偵測到 {len(blobs)} 個 blob)")
            st.image(bgr_to_rgb_pil(annotated_flare), use_container_width=True)

        if table_rows:
            st.subheader("每個 blob 的數值")
            st.dataframe(table_rows, use_container_width=True)
        else:
            st.warning("目前參數下沒有偵測到任何符合條件的黑塊,調整左側滑桿試試看。")
