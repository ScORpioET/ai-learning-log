"""
驗證「用像素通道差異自動判斷thermal vs RGB」構想:抽樣+量測。

只做量測,不碰app.py或任何正式部署程式碼。

重要:抽第一批thermal圖片就發現一個比原本假設更根本的事實(見REPORT.md)——
thermal圖片存檔時本來就是PIL mode='L'的單通道JPEG,不是「三通道但數值相近」,
是檔案格式層級就只有一個通道。這件事在寫這支腳本前用全量掃描
(11886張thermal + 11404張rgb)驗證過,兩邊都是100%,沒有例外。這改變了
分析的重點:thermal這邊的diff指標永遠精確等於0.0(不是"接近0"),真正需要
擔心的是RGB那邊(尤其night子集)有沒有diff異常低、低到跟thermal的0重疊。

夜間RGB子集合的判定用兩種方法交叉驗證:
1. index.json裡video-level的"night" tag(主要依據,people-labeled)
2. 平均亮度(mean of R,G,B across全圖)由暗到亮排序取最暗15%當第二個獨立子集
   (捕捉可能沒被標"night" tag但實際偏暗的影片,例如隧道/黃昏)
"""
import glob
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent
N_PER_SPLIT = 500
SEED = 1337

FILENAME_RE = re.compile(r"^video-([^-]+)-frame-")


def video_id_from_filename(fn: str):
    m = FILENAME_RE.match(fn)
    return m.group(1) if m else None


def load_night_video_ids(split_dir: Path):
    index = json.loads((split_dir / "index.json").read_text())
    return set(v["id"] for v in index["videos"] if "night" in v.get("tags", []))


def channel_diff_metric(arr: np.ndarray) -> float:
    """arr: HxWx3 float32. 跟user提供的範例寫法一致:三個pairwise通道差異的
    絕對值取平均後,取三者最大值。"""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(max(np.abs(r - g).mean(), np.abs(g - b).mean(), np.abs(r - b).mean()))


def mean_brightness(arr: np.ndarray) -> float:
    return float(arr.mean())


def sample_split(split_name: str, is_rgb: bool):
    split_dir = DATA_ROOT / split_name
    files = sorted((split_dir / "data").glob("*.jpg"))
    random.seed(SEED)
    sampled = random.sample(files, min(N_PER_SPLIT, len(files)))

    night_ids = load_night_video_ids(split_dir) if is_rgb else set()

    rows = []
    for f in sampled:
        with Image.open(f) as img:
            pil_mode = img.mode
            arr = np.asarray(img.convert("RGB")).astype(np.float32)
        diff = channel_diff_metric(arr)
        brightness = mean_brightness(arr)
        vid = video_id_from_filename(f.name)
        is_night_tag = vid in night_ids if is_rgb else False
        rows.append({
            "split": split_name,
            "domain": "rgb" if is_rgb else "thermal",
            "file": f.name,
            "pil_mode": pil_mode,
            "video_id": vid,
            "channel_diff": diff,
            "mean_brightness": brightness,
            "is_night_tag": is_night_tag,
        })
    return rows


def main():
    all_rows = []
    for split, is_rgb in [
        ("images_thermal_train", False),
        ("images_thermal_val", False),
        ("images_rgb_train", True),
        ("images_rgb_val", True),
    ]:
        rows = sample_split(split, is_rgb)
        print(f"[info] {split}: sampled {len(rows)} images")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # 第二種night判定法:亮度最暗15%的RGB圖片(獨立於video tag,交叉驗證用)
    rgb_mask = df["domain"] == "rgb"
    dark_threshold = df.loc[rgb_mask, "mean_brightness"].quantile(0.15)
    df["is_dark_brightness"] = False
    df.loc[rgb_mask, "is_dark_brightness"] = df.loc[rgb_mask, "mean_brightness"] <= dark_threshold

    out_path = HERE / "sampled_measurements.csv"
    df.to_csv(out_path, index=False)
    print(f"[done] {len(df)} rows written to {out_path}")
    print(f"[info] dark-brightness threshold (15th percentile of RGB mean_brightness) = {dark_threshold:.2f}")

    # 全量掃描確認thermal是不是100% mode='L'、rgb是不是100% mode='RGB'(這是
    # 這次分析裡最關鍵的背景事實,抽樣本身也順便再核對一次抽到的這批)
    print("\n[check] pil_mode breakdown by domain (抽樣批次內):")
    print(df.groupby(["domain", "pil_mode"]).size())


if __name__ == "__main__":
    main()
