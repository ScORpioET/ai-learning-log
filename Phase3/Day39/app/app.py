"""
v1 最小可跑通的 thermal caption demo:單張圖片上傳 -> 生成caption文字。
CPU FP32,onnxruntime(CPUExecutionProvider),不依賴PyTorch。

模型:clip_vision.onnx + gpt.onnx(都從 best_model_full_capfix_reweight2x.pt 匯出,
Day39定案版本),tokenizer 用同一輪訓練重建出來的 tokenizer.pkl(見
reconstruct_capfix_tokenizer.py 的驗證紀錄,配對正確)。
"""
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from minbpe import minbpe
from preprocess import preprocess

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent  # clip_vision.onnx / gpt.onnx 放在上一層(Day39/)

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

_sess_opts = ort.SessionOptions()
clip_session = ort.InferenceSession(
    str(MODEL_DIR / "clip_vision.onnx"), sess_options=_sess_opts, providers=["CPUExecutionProvider"])
gpt_session = ort.InferenceSession(
    str(MODEL_DIR / "gpt.onnx"), sess_options=_sess_opts, providers=["CPUExecutionProvider"])
tokenizer = minbpe.load(str(MODEL_DIR / "tokenizer.pkl"))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def generate_caption(image: Image.Image) -> str:
    if image is None:
        return ""

    pixel_values = preprocess(image)
    img_feat = clip_session.run(None, {"pixel_values": pixel_values})[0]

    idx = np.array([[IMAGE_TOKEN_ID]], dtype=np.int64)
    for _ in range(MAX_NEW_TOKENS):
        logits = gpt_session.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
        next_id = int(np.argmax(logits[0, -1]))  # greedy decoding,demo用途求輸出穩定可重現
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == EOS_TOKEN_ID:
            break

    generated_ids = idx[0].tolist()
    clean_ids = [i for i in generated_ids if i not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    return tokenizer.decode([clean_ids])


demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Thermal image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Thermal Caption Demo (v1, CPU FP32)",
    description="上傳一張熱像圖片,生成場景描述caption。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
