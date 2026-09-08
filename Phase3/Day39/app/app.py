"""
Thermal caption demo。

v1(CPU FP32): onnxruntime(CPUExecutionProvider),不依賴PyTorch。
v1.1 階段一新增:DEVICE_MODE=gpu 時改用 clip_vision.fp16.onnx + gpt.fp16.onnx,
provider 換成 CUDAExecutionProvider(GPU FP16 路徑)。CPU/GPU 兩份模型都是從
best_model_full_capfix_reweight2x.pt 這同一個checkpoint匯出,FP16版本各自都做過
cosine similarity sanity check(clip_vision ~0.9999992, gpt ~1.0000001),不是沒驗證
就混用精度上線。

模型:tokenizer 用同一輪訓練重建出來的 tokenizer.pkl(見
reconstruct_capfix_tokenizer.py 的驗證紀錄,配對正確)。
"""
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from minbpe import minbpe
from preprocess import preprocess

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent  # onnx檔案放在上一層(Day39/)

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

DEVICE_MODE = os.environ.get("DEVICE_MODE", "cpu")  # "cpu" 或 "gpu"


def load_sessions(device_mode: str):
    if device_mode == "gpu":
        clip_path = MODEL_DIR / "clip_vision.fp16.onnx"
        gpt_path = MODEL_DIR / "gpt.fp16.onnx"
        providers = ["CUDAExecutionProvider"]
    else:
        clip_path = MODEL_DIR / "clip_vision.onnx"
        gpt_path = MODEL_DIR / "gpt.onnx"
        providers = ["CPUExecutionProvider"]

    clip_sess = ort.InferenceSession(str(clip_path), providers=providers)
    gpt_sess = ort.InferenceSession(str(gpt_path), providers=providers)
    return clip_sess, gpt_sess


clip_session, gpt_session = load_sessions(DEVICE_MODE)
tokenizer = minbpe.load(str(MODEL_DIR / "tokenizer.pkl"))

print(f"[info] DEVICE_MODE={DEVICE_MODE}")
print(f"[info] clip_session providers = {clip_session.get_providers()}")
print(f"[info] gpt_session providers  = {gpt_session.get_providers()}")


def run_generation(image: Image.Image, clip_sess, gpt_sess):
    """回傳 (caption, elapsed_ms)。"""
    t0 = time.perf_counter()
    pixel_values = preprocess(image)
    # clip_vision.fp16.onnx / gpt.fp16.onnx 用 keep_io_types=True 轉換,
    # 輸入輸出仍是 float32,只有內部計算是fp16,所以兩種DEVICE_MODE都餵float32即可。
    img_feat = clip_sess.run(None, {"pixel_values": pixel_values})[0]

    idx = np.array([[IMAGE_TOKEN_ID]], dtype=np.int64)
    for _ in range(MAX_NEW_TOKENS):
        logits = gpt_sess.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
        next_id = int(np.argmax(logits[0, -1]))  # greedy decoding,demo用途求輸出穩定可重現
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == EOS_TOKEN_ID:
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000

    generated_ids = idx[0].tolist()
    clean_ids = [i for i in generated_ids if i not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    caption = tokenizer.decode([clean_ids])
    return caption, elapsed_ms


def generate_caption(image: Image.Image) -> str:
    if image is None:
        return ""
    caption, elapsed_ms = run_generation(image, clip_session, gpt_session)
    return caption


demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Thermal image"),
    outputs=gr.Textbox(label="Generated caption"),
    title=f"Thermal Caption Demo (v1.1, {'GPU FP16' if DEVICE_MODE == 'gpu' else 'CPU FP32'})",
    description="上傳一張熱像圖片,生成場景描述caption。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
