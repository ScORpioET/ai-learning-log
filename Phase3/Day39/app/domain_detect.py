"""
自動判斷輸入圖片是thermal還是RGB分支,取代手動選擇分支的下拉選單。

方法/門檻依據 Phase3/domain_autodetect_check/REPORT.md 的驗證結果:這個資料集
裡thermal圖片100%存成PIL mode='L'單通道灰階,RGB圖片100%是mode='RGB'三通道,
強制.convert("RGB")之後算channel_diff,thermal精確等於0.0,RGB(含夜間、含
最暗15%子集)最小值1.65,中間有乾淨間隙,2000張抽樣0誤判。門檻設0.5,落在
驗證過的乾淨間隙(0.0001~1.65)正中間,對兩邊都留安全餘裕。

已知限制(REPORT.md裡寫明的):這驗證的是「這個資料集的檔案儲存慣例」,不是
「畫面內容本質上是不是灰階」這個更廣義的問題。使用者上傳的圖片如果被某個工具
強制轉成3通道存檔(內容還是灰階)、或RGB照片被刻意大幅降低飽和度,自動判斷
可能誤判,而且系統不會自己發現——這是app.py裡保留手動覆蓋選項的原因,不是
自動判斷可以無腦全信。
"""
import numpy as np
from PIL import Image

CHANNEL_DIFF_THRESHOLD = 0.5


def detect_domain(image: Image.Image) -> str:
    """回傳 "thermal" 或 "rgb"。"""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    diff = max(float(np.abs(r - g).mean()), float(np.abs(g - b).mean()), float(np.abs(r - b).mean()))
    return "thermal" if diff < CHANNEL_DIFF_THRESHOLD else "rgb"
