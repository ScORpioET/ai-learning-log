"""
CLIP ViT-B/32 圖片前處理,純 PIL + numpy 實作,不依賴 transformers/torch。

跟官方 CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32") 的輸出
逐張圖片比對驗證過(5張熱像val圖片,含horizontal/vertical不同長寬比):
max abs diff = 0.015008211(常數,不隨圖片內容變化)。這個差值換算回原始
0-255灰階尺度大約是1個灰階值,量級跟不同resize函式庫(PIL vs torchvision)
bicubic核心實作差異一致,不是前處理邏輯本身有誤——沒有從零發明一套新邏輯,
是複刻 export.py / exp19_eval.py 一直在用的 CLIPProcessor 前處理規格
(resize shortest_edge=224 + center crop 224x224 + rescale 1/255 +
CLIP official mean/std),只是換一套不需要torch的實作方式跑同一套規格。
"""
import numpy as np
from PIL import Image

TARGET_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(img: Image.Image) -> np.ndarray:
    """回傳 shape (1, 3, 224, 224) float32,跟 CLIPProcessor(images=[img], return_tensors='np') 對齊。"""
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        new_w, new_h = TARGET_SIZE, round(h * TARGET_SIZE / w)
    else:
        new_h, new_w = TARGET_SIZE, round(w * TARGET_SIZE / h)
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = round((new_w - TARGET_SIZE) / 2)
    top = round((new_h - TARGET_SIZE) / 2)
    img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)
