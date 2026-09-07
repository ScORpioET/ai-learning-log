"""
沿著 blob centroid 往外用多個方向角(radial profile)取樣亮度,量化兩件事:
1. 陡峭度:亮度從暗到亮的「跳變寬度」(10%->90% 爬升所花的 px 數),越小
   越像瞬間跳變,越大越像漸層。
2. 光暈影響面積:每個方向找到「亮度重新掉回背景值」的距離,當作那個方向
   的光暈外緣半徑,24 個方向的外緣點連成一個多邊形,面積就是光暈影響面積
   (含黑色核心本身)。

只是先做小範圍測試(先驗證方法有沒有用),還沒套用到全部 1,251 張圖片。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

N_DIRECTIONS = 36
MAX_DIST = 250
BG_THRESH = 40          # 亮度掉回這個值以下,算「回到背景」
SUSTAIN = 5             # 要連續這麼多個取樣點都低於 BG_THRESH 才算真的回到背景


def radial_profile_halo(img_bgr, centroid, max_dist=MAX_DIST, n_dirs=N_DIRECTIONS):
    intensity = img_bgr.astype(np.float32).max(axis=2)
    h, w = intensity.shape
    cx, cy = centroid

    outer_points = []
    widths = []
    n_reached_bg = 0
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

        # 跳變位置:單步差分最大的地方
        diffs = np.diff(vals)
        jump_idx = int(np.argmax(diffs)) if len(diffs) else 0

        # 跳變寬度:以 jump_idx 為中心,往前找 10% 爬升點、往後找 90% 爬升點
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

        # 光暈外緣:跳變之後,亮度重新連續掉回背景值的位置
        outer_idx = len(vals) - 1
        reached_bg = False
        for idx in range(jump_idx, max(len(vals) - SUSTAIN, jump_idx + 1)):
            if np.all(vals[idx:idx + SUSTAIN] < BG_THRESH):
                outer_idx = idx
                reached_bg = True
                break
        if reached_bg:
            n_reached_bg += 1
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
        "jump_width_mean_px": round(float(np.mean(widths)), 2) if widths else None,
        "polygon": poly.tolist(),
        "frac_reached_bg": round(n_reached_bg / n_dirs, 3),
    }


def annotate(img_bgr, blobs_with_halo, scale=1.0, draw_labels=True):
    """scale: 如果 img_bgr 是先放大過的畫布(例如跟 blend 對齊到同一解析度),
    bbox/polygon 的座標(來自原始 flare 解析度)要乘上這個倍率才會對齊;
    直接在放大後的畫布上畫字,而不是先在小圖畫字再整張圖放大,才不會因為
    放大模糊讓文字糊成一團、看起來像疊字。

    blobs_with_halo 裡的 halo dict 如果帶有 "core_area_px"(黑核本身的
    面積,不是光暈面積),會多標一行黑核面積,方便對照綠框實際多大;
    沒帶這個欄位的舊呼叫方式也相容,只是不會多這一行。

    draw_labels=False:黑核貼近圖片邊緣時,文字往上疊會超出畫布被裁掉,
    這種情況改成只在框內左上角畫小小的編號「#i」(不會超出畫布),完整的
    數值改由呼叫端用 build_stat_strip() 畫在圖片下方額外的白色區域。"""
    out = img_bgr.copy()
    font_scale = 0.55 * scale
    thickness_fill = max(1, round(1 * scale))
    thickness_outline = max(2, round(3 * scale))
    line_gap = round(22 * scale)

    for i, (bbox, halo) in enumerate(blobs_with_halo, 1):
        x, y, bw, bh = [round(v * scale) for v in bbox]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), max(1, round(2 * scale)))
        poly = (np.array(halo["polygon"], dtype=np.float32) * scale).astype(np.int32)
        cv2.polylines(out, [poly], isClosed=True, color=(255, 200, 0), thickness=max(1, round(2 * scale)))

        if not draw_labels:
            tag = f"#{i}"
            tx, ty = x + round(4 * scale), y + round(18 * scale)
            cv2.putText(out, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
            cv2.putText(out, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness_fill, cv2.LINE_AA)
            continue

        lines = [f"#{i} halo_area={int(halo['halo_area_px'])}px", f"jump_w={halo['jump_width_median_px']}px"]
        if "core_area_px" in halo:
            lines.insert(0, f"core(bbox)={halo['core_area_px']}px")
        y0 = max(y - line_gap * len(lines) - 6, line_gap)
        for j, text in enumerate(lines):
            ty = y0 + j * line_gap
            cv2.putText(out, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
            cv2.putText(out, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 0), thickness_fill, cv2.LINE_AA)

    return out


def build_stat_strip(width, blobs_with_halo, line_height=60, font_scale=1.3):
    """在合併圖下方加一條白色區域,把每個 blob 的完整數值(編號對應框上的
    #i 小標籤)橫向列出來,不會被畫布邊界裁掉。字級跟著合併圖寬度(通常是
    三張原始解析度圖片並排,寬度動輒幾千 px)放大,不然縮圖看起來會太小。"""
    n = max(len(blobs_with_halo), 1)
    font_scale = font_scale * max(width / 1800, 1.0)
    line_height = round(line_height * max(width / 1800, 1.0))
    thickness = max(1, round(font_scale))
    strip = np.full((line_height * n + round(line_height * 0.3), width, 3), 255, dtype=np.uint8)
    for i, (bbox, halo) in enumerate(blobs_with_halo, 1):
        parts = [f"#{i}"]
        if "core_area_px" in halo:
            parts.append(f"core(bbox)={halo['core_area_px']}px")
        parts.append(f"halo_area={int(halo['halo_area_px'])}px")
        parts.append(f"jump_w={halo['jump_width_median_px']}px")
        text = "   ".join(parts)
        ty = round(line_height * 0.15) + i * line_height - round(line_height * 0.25)
        cv2.putText(strip, text, (round(line_height * 0.3), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return strip
    return out
