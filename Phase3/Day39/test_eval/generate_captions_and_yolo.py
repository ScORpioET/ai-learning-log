"""
對test集全部圖片(thermal 3749張 + rgb 3749張)各自生成caption + 跑YOLO偵測,
存成json,给下一步「挑caption跟YOLO偵測結果接近的圖片當demo」用。

沿用Day39既有的模型/tokenizer/YOLO設定,不重新設計一套:
- clip_vision.onnx(domain-agnostic,thermal/rgb共用)
- thermal: gpt.onnx + tokenizer.pkl
- rgb: gpt_rgb.onnx + tokenizer_rgb.pkl
- YOLO: app/yolo_overlay.py 的 KEEP_CLASSES/CONF_THRESH/yolov8m.pt

有GPU就用CUDAExecutionProvider(FP32,沒有另外做FP16 -- 這裡是離線批次生成,
不是v1.1那個對延遲敏感的demo,不需要FP16)加速,沒有就退回CPU。

輸出:test_eval/records_thermal.json, test_eval/records_rgb.json,
每筆 {file_name, video_id, caption, yolo: [{class_name, conf, bbox}]}
"""
import json
import os
import re
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from minbpe import minbpe  # noqa: E402
from preprocess import preprocess  # noqa: E402
from yolo_overlay import detect_classes  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent
TD = Path(os.environ.get("DATASET_ROOT", str(Path.home() / "ai-transition-2026" / "thermal_dataset")))
OUT_DIR = Path(__file__).parent

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

DOMAINS = {
    "thermal": {
        "img_dir": TD / "video_thermal_test" / "data",
        "gpt_onnx": MODEL_DIR / "gpt.onnx",
        "tokenizer": MODEL_DIR / "tokenizer.pkl",
    },
    "rgb": {
        "img_dir": TD / "video_rgb_test" / "data",
        "gpt_onnx": MODEL_DIR / "gpt_rgb.onnx",
        "tokenizer": MODEL_DIR / "tokenizer_rgb.pkl",
    },
}

VIDEO_RE = re.compile(r"^video-([A-Za-z0-9]+)-frame-")


def make_sessions():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        clip_sess = ort.InferenceSession(str(MODEL_DIR / "clip_vision.onnx"), providers=providers)
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        clip_sess.run(None, {"pixel_values": dummy})  # 真的跑一次確認可用
        active = clip_sess.get_providers()[0]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] CUDA session failed ({e}), falling back to CPU only")
        providers = ["CPUExecutionProvider"]
        clip_sess = ort.InferenceSession(str(MODEL_DIR / "clip_vision.onnx"), providers=providers)
        active = "CPUExecutionProvider"
    print(f"[info] clip_vision using provider: {active}")
    return clip_sess, providers


def generate_caption(clip_sess, gpt_sess, tokenizer, image):
    pixel_values = preprocess(image)
    img_feat = clip_sess.run(None, {"pixel_values": pixel_values})[0]
    idx = np.array([[IMAGE_TOKEN_ID]], dtype=np.int64)
    for _ in range(MAX_NEW_TOKENS):
        logits = gpt_sess.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
        next_id = int(np.argmax(logits[0, -1]))
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == EOS_TOKEN_ID:
            break
    clean_ids = [i for i in idx[0].tolist() if i not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    return tokenizer.decode([clean_ids])


def main():
    clip_sess, providers = make_sessions()

    for domain, cfg in DOMAINS.items():
        out_path = OUT_DIR / f"records_{domain}.json"
        gpt_sess = ort.InferenceSession(str(cfg["gpt_onnx"]), providers=providers)
        tokenizer = minbpe.load(str(cfg["tokenizer"]))

        img_paths = sorted(cfg["img_dir"].glob("*.jpg"))
        n_total = len(img_paths)
        print(f"\n[info] domain={domain}  n_images={n_total}  gpt_provider={gpt_sess.get_providers()[0]}")

        records = []
        t_start = time.time()
        for i, img_path in enumerate(img_paths):
            fn = img_path.name
            m = VIDEO_RE.match(fn)
            video_id = m.group(1) if m else None

            image = Image.open(img_path).convert("RGB")
            caption = generate_caption(clip_sess, gpt_sess, tokenizer, image)
            yolo_dets = detect_classes(image)

            records.append({
                "file_name": fn,
                "video_id": video_id,
                "caption": caption,
                "yolo": yolo_dets,
            })

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta_min = (n_total - (i + 1)) / rate / 60 if rate > 0 else float("nan")
                print(f"  [{domain}] {i+1}/{n_total}  ({rate:.2f} img/s, ETA {eta_min:.1f} min)", end="\r")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n[done] {domain}: {out_path}  ({len(records)} records)")


if __name__ == "__main__":
    main()
