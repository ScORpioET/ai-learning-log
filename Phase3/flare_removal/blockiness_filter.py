"""
判斷一張 flare 圖是不是「JPEG 量化區塊雜訊」造成的假訊號(而不是真的
光暈),而不是拿去當光源看待。

背景:低訊號影片(整張圖幾乎沒有真的強光)的 flare 輸出本身強度非常
微弱、接近全黑,經過我們自己的「每張圖各自除以自己最大值」正規化上色
後,微弱的 JPEG 8x8 壓縮量化區塊被硬拉伸成大片邊界銳利的純色色塊
(紅/黃/黑),看起來完全沒有漸層、沒有場景輪廓——這種圖不該被當作
「有效的過曝訊號」。

判斷方式:把顏色量化成粗網格(每 8 階一格),找「同一種顏色(排除純黑
背景)佔滿整張圖的最大連續比例」。真正的光暈是連續漸層,同一個精確顏色
不會佔到很大面積;雜訊區塊化的圖,同一個顏色常常會佔到 30-70% 以上。
"""
import numpy as np

BLOCKINESS_THRESH = 0.15


def blockiness_score(img_bgr, quant=8):
    h, w = img_bgr.shape[:2]
    q = (img_bgr // quant * quant)
    key = (q[:, :, 0].astype(np.int32) << 16) | (q[:, :, 1].astype(np.int32) << 8) | q[:, :, 2].astype(np.int32)
    uniq, counts = np.unique(key, return_counts=True)
    mask = uniq != 0  # 排除純黑背景,背景本來就常大片黑,不算異常
    if mask.sum() == 0:
        return 0.0
    return float(counts[mask].max() / (h * w))


def is_blockiness_failure(img_bgr, thresh=BLOCKINESS_THRESH):
    return blockiness_score(img_bgr, quant=8) >= thresh
