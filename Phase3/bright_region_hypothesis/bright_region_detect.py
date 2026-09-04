"""
RGB 圖片亮區(強光/耀光)偵測。步驟照任務指定:灰階化 -> 高斯模糊 ->
固定門檻二值化 -> erode/dilate 去雜訊 -> connected component 找夠大的亮區。

參數選擇(方法論選擇,理由寫清楚,不是查出來的事實):
- BRIGHT_THRESH = 200:題目指定的起始值,留參數之後可調。
- BLUR_KSIZE = 9(奇數,高斯模糊 kernel):目的是把單一雜訊亮點跟真正
  一整片的強光/耀光區分開——雜訊點模糊後亮度會被鄰近暗像素拉低、
  掉到門檻以下,真正大面積的亮區模糊後中心還是維持在門檻以上。9 是
  常見的「中等強度去雜訊」kernel size,沒有更精細的依據,先用這個,
  之後如果亮區偵測跟肉眼判斷差太多再調。
- ERODE/DILATE:先 erode 3x3 一次去掉二值化後殘留的小雜訊塊,再 dilate
  3x3 兩次把亮區邊緣補回來、順便把同一光源被切成好幾小塊的區域接起來
  (強光/耀光邊緣常有漸層,二值化後容易斷裂)。iteration 數量是常見的
  OpenCV「先侵蝕去噪、再膨脹補洞」標準組合,沒有另外調參數搜尋最佳值。
- MIN_AREA_PX = 400(20x20 像素等效面積):RGB 圖是 1224x1024
  (=1,253,376 px),400px 約佔全圖 0.032%——選這個數字是因為題目關心的
  是「一顆燈/一片反光造成的區域級強光」,不是個位數雜訊像素,20x20 這個
  量級大概是一個遠處小物件的大小,小於這個的 connected component 視為
  雜訊或太小不足以造成「物件被蓋住」的實際影響,直接丟棄。這個門檻也是
  方法論選擇,不是從資料分布反推出來的。

輸出:detect_bright_regions() 回傳 (mask, boxes)。
    mask: HxW bool array,亮區為 True
    boxes: [(x, y, w, h), ...] 每個亮區的外接矩形(像素座標)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

BRIGHT_THRESH = 200
BLUR_KSIZE = 9
ERODE_ITER = 1
DILATE_ITER = 2
MIN_AREA_PX = 400


def detect_bright_regions(img_path_or_array, thresh=BRIGHT_THRESH, min_area=MIN_AREA_PX):
    if isinstance(img_path_or_array, np.ndarray):
        bgr = img_path_or_array
    else:
        bgr = cv2.imread(str(img_path_or_array))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=ERODE_ITER)
    cleaned = cv2.dilate(cleaned, kernel, iterations=DILATE_ITER)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    boxes = []
    mask = np.zeros(gray.shape, dtype=bool)
    for i in range(1, n_labels):  # label 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((int(x), int(y), int(w), int(h)))
        mask |= (labels == i)
    return mask, boxes
