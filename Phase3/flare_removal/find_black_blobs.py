"""
在 flare/(光斑抽出圖,已經是暖色熱圖:黑->紅->黃白)裡找 Jack 描述的特徵:
光源本身是一塊黑色,但緊貼著它的光暈立刻變成亮色(沒有中間漸層過渡)。

作法:
1. cv2.inRange 抓出接近黑色的像素(三通道都 <= BLACK_THRESH)。
2. connectedComponentsWithStats 找黑色 blob,濾掉:
   - 面積太小的雜訊(< MIN_AREA)
   - 貼到圖片邊界的(這種通常是背景本身是黑的,不是被光暈包圍的洞)
3. 對每個留下來的黑 blob,往外擴張 RING_DILATE px 取一圈「外環」像素,
   算外環的平均灰階亮度——如果外環不夠亮(< BRIGHT_RING_THRESH),代表
   這個黑塊外面沒有立刻接亮色,不符合「直接變亮、沒有漸層」的描述,排除。
4. 剩下的才算符合特徵的 blob,在圖上標框、標編號+面積(px),輸出到
   {split}/flare_blobs/,並把每張圖的 blob 清單彙整成 json。

注意:只掃 out_train/out_val/out_test 三個 split 的 flare/(已經是暖色化
之後的版本),黑色在暖色化前後沒有變(強度 0 兩種配色都對應黑色),
所以直接在現有檔案上偵測沒問題。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

HERE = Path(__file__).parent
SPLITS = ["out_train", "out_val", "out_test"]

BLACK_THRESH = 30
BRIGHT_RING_THRESH = 80

# MIN_AREA/RING_DILATE 原本是寫死的像素值,但 flare_resolution_distribution.json
# 確認過 out_{train,val,test}/flare/ 這 15,153 張圖高度雖然固定 512(Flare7K++
# 內部把短邊縮到 512),寬度卻有 576/612/682/640/768 五種、最大差 1.33 倍
# (最主要那組只佔 38.9%,沒過 95% 門檻),所以改成「相對參考解析度」的比例值,
# 不同寬度的圖片會各自算出對應的實際 px 門檻。
# 參考解析度:682x512,是當初校準 BLACK_THRESH/MIN_AREA/RING_DILATE/
# BRIGHT_RING_THRESH 這組參數時用的太陽正中央測試圖(video-dvZBYnphN2BwdMKBc-
# frame-000080)的 flare 輸出解析度。
REF_W, REF_H = 682, 512
MIN_AREA_REF_PX = 500
RING_DILATE_REF_PX = 15
MIN_AREA_RATIO = MIN_AREA_REF_PX / (REF_W * REF_H)
RING_DILATE_RATIO = RING_DILATE_REF_PX / np.hypot(REF_W, REF_H)


def find_blobs(img_bgr):
    """亮度判斷用 3 通道的 max(不是灰階 luma):這是暖色熱圖自己的配色規則
    (強度越高,先漲 R,再漲 G,最後漲 B),純紅色 (0,0,255) 灰階 luma 只有
    ~76,會被灰階誤判成「不夠亮」,但在這個配色裡純紅已經代表中高強度。
    另外光暈邊緣是漸層(不是真的瞬間跳變),所以外環半徑要抓大一點
    才會落在漸層走完、真正進入亮色高原的位置。

    MIN_AREA/RING_DILATE 這兩個門檻依這張圖實際的寬高,用 MIN_AREA_RATIO/
    RING_DILATE_RATIO 換算成這張圖自己的 px 值,不同解析度的圖片標準一致。"""
    h, w = img_bgr.shape[:2]
    min_area_px = MIN_AREA_RATIO * (w * h)
    ring_dilate_px = max(1, round(RING_DILATE_RATIO * np.hypot(w, h)))

    black_mask = cv2.inRange(img_bgr, (0, 0, 0), (BLACK_THRESH, BLACK_THRESH, BLACK_THRESH))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    maxchan = img_bgr.astype(np.float32).max(axis=2)
    kernel = np.ones((ring_dilate_px * 2 + 1, ring_dilate_px * 2 + 1), np.uint8)

    blobs = []
    for lbl in range(1, n_labels):
        x, y, bw, bh, area = stats[lbl]
        if area < min_area_px:
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        comp_mask = (labels == lbl).astype(np.uint8)
        dilated = cv2.dilate(comp_mask, kernel)
        ring = (dilated > 0) & (comp_mask == 0)
        if ring.sum() == 0:
            continue
        ring_mean = float(maxchan[ring].mean())
        if ring_mean < BRIGHT_RING_THRESH:
            continue
        cx, cy = centroids[lbl]
        blobs.append({
            "area": int(area), "bbox": [int(x), int(y), int(bw), int(bh)],
            "centroid": [round(float(cx), 1), round(float(cy), 1)],
            "ring_brightness": round(ring_mean, 1),
        })
    blobs.sort(key=lambda b: -b["area"])
    return blobs


def annotate(img_bgr, blobs):
    out = img_bgr.copy()
    for i, b in enumerate(blobs, 1):
        x, y, bw, bh = b["bbox"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"#{i} area={b['area']}px"
        cv2.putText(out, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    summary = []
    n_matched = 0
    n_total = 0
    for split in SPLITS:
        flare_dir = HERE / split / "flare"
        out_dir = HERE / split / "flare_blobs"
        out_dir.mkdir(exist_ok=True)

        files = sorted(flare_dir.glob("*.jpg"))
        print(f"[{split}] {len(files)} 張")
        for i, fp in enumerate(files):
            n_total += 1
            img = cv2.imread(str(fp))
            if img is None:
                continue
            blobs = find_blobs(img)
            if blobs:
                n_matched += 1
                annotated = annotate(img, blobs)
                cv2.imwrite(str(out_dir / fp.name), annotated)
                summary.append({
                    "split": split, "file_name": fp.name,
                    "n_blobs": len(blobs), "blobs": blobs,
                })
            if (i + 1) % 1000 == 0:
                print(f"  ...{i+1}/{len(files)}  (目前符合特徵 {n_matched} 張)", end="\r")
        print()

    summary.sort(key=lambda r: -max(b["area"] for b in r["blobs"]))
    with open(HERE / "black_blob_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[done] 共檢查 {n_total} 張,符合特徵(有 >=1 個被亮光暈包圍的黑塊)的有 {n_matched} 張")
    print(f"標註後的圖片存在各 split 底下 flare_blobs/,清單存在 black_blob_summary.json")


if __name__ == "__main__":
    main()
