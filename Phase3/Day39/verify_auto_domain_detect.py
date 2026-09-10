"""
驗證v1.2「自動判斷thermal/RGB」的整合:
1. sample/(test_eval/selected_demo_images/)裡thermal、rgb兩個資料夾的圖片,
   detect_domain()判斷結果是不是都對(thermal資料夾裡的圖被判成thermal,
   rgb資料夾裡的圖被判成rgb,包含rgb裡的夜間影片)。
2. 用app.py同一套run_generation()邏輯生成caption,跟test_eval/selected_thermal.json
   /selected_rgb.json裡之前(手動選分支時代)記錄的caption逐字比對,確認改成
   自動判斷之後,同一張圖生成的文字沒有變。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "app"))

import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from domain_detect import detect_domain  # noqa: E402
from minbpe import minbpe  # noqa: E402
from preprocess import preprocess  # noqa: E402

# 不直接 `from app import ...`:app.py頂層會import yolo_overlay -> ultralytics
# -> cv2,這台機器的本地.venv裡cv2跟numpy 2.x有已知的ABI衝突(之前跑批次
# caption+YOLO時也踩過,當時用Docker內的python環境繞過)。這裡只需要
# run_generation()這段跟YOLO完全無關的邏輯,直接複製過來,不用為了避開
# import鏈就去改app.py本身的import順序。
IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

MODEL_DIR = Path(__file__).parent
DOMAINS = {
    "thermal": {"gpt_cpu": "gpt.onnx", "gpt_gpu_fp16": "gpt.fp16.onnx", "tokenizer": "tokenizer.pkl"},
    "rgb": {"gpt_cpu": "gpt_rgb.onnx", "gpt_gpu_fp16": "gpt_rgb.fp16.onnx", "tokenizer": "tokenizer_rgb.pkl"},
}


def run_generation(image, clip_sess, gpt_sess, tokenizer):
    pixel_values = preprocess(image)
    img_feat = clip_sess.run(None, {"pixel_values": pixel_values})[0]
    idx = np.array([[IMAGE_TOKEN_ID]], dtype=np.int64)
    for _ in range(MAX_NEW_TOKENS):
        logits = gpt_sess.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
        next_id = int(np.argmax(logits[0, -1]))
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == EOS_TOKEN_ID:
            break
    generated_ids = idx[0].tolist()
    clean_ids = [i for i in generated_ids if i not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    return tokenizer.decode([clean_ids]), None


TEST_EVAL = Path(__file__).parent / "test_eval"
SAMPLE_DIR = TEST_EVAL / "selected_demo_images"

tokenizers = {name: minbpe.load(str(MODEL_DIR / cfg["tokenizer"])) for name, cfg in DOMAINS.items()}


def load_recorded_captions(domain):
    path = TEST_EVAL / f"selected_{domain}.json"
    records = json.loads(path.read_text())
    return {r["file_name"]: r["caption"] for r in records}


def main():
    total = 0
    detect_ok = 0
    caption_ok = 0
    mismatches = []

    for domain in ["thermal", "rgb"]:
        recorded = load_recorded_captions(domain)
        clip_sess = ort.InferenceSession(str(MODEL_DIR / "clip_vision.onnx"), providers=["CPUExecutionProvider"])
        gpt_sess = ort.InferenceSession(str(MODEL_DIR / DOMAINS[domain]["gpt_cpu"]), providers=["CPUExecutionProvider"])
        tokenizer = tokenizers[domain]

        for video_dir in sorted((SAMPLE_DIR / domain).iterdir()):
            if not video_dir.is_dir():
                continue
            for img_path in sorted(video_dir.glob("*.jpg")):
                total += 1
                img = Image.open(img_path).convert("RGB")

                detected = detect_domain(img)
                is_detect_ok = detected == domain
                detect_ok += is_detect_ok

                caption, _ = run_generation(img, clip_sess, gpt_sess, tokenizer)
                expected = recorded.get(img_path.name)
                is_caption_ok = expected is not None and caption.strip() == expected.strip()
                caption_ok += is_caption_ok

                if not is_detect_ok or not is_caption_ok:
                    mismatches.append({
                        "domain": domain, "file": img_path.name,
                        "detected": detected, "detect_ok": is_detect_ok,
                        "generated_caption": caption, "expected_caption": expected,
                        "caption_ok": is_caption_ok,
                    })

    print(f"總測試圖片數: {total}")
    print(f"自動判斷正確: {detect_ok}/{total}")
    print(f"caption跟先前記錄一致: {caption_ok}/{total}")

    if mismatches:
        print("\n=== 不一致的案例 ===")
        for m in mismatches:
            print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        print("\n[PASS] 全部圖片自動判斷正確,且生成的caption跟先前(手動選分支時代)記錄的完全一致。")


if __name__ == "__main__":
    main()
