"""
Thermal caption demo。

v1(CPU FP32): onnxruntime(CPUExecutionProvider),不依賴PyTorch。
v1.1 階段一:GPU/FP16路徑(DEVICE_MODE=gpu 時預設走 CUDAExecutionProvider+FP16)。
v1.1 階段二:CPU vs GPU 效能比較模式 -- 開機時嘗試載入GPU FP16 sessions(真的跑一次
推論確認GPU可用,不是只看 provider 有沒有列在清單裡),成功的話UI改成「同一張圖
兩條路徑並排」+各自延遲(ms);沒有GPU(或GPU sessions載入/推論失敗)就退回v1的
單一路徑介面,不會讓沒GPU的機器看到壞掉的按鈕。
v1.1 階段三:YOLO疊框(見 yolo_overlay.py,沿用既有KEEP_CLASSES/CONF_THRESH設定)。

CPU 跟 GPU 兩份模型都是從 best_model_full_capfix_reweight2x.pt 這同一個checkpoint
匯出,FP16版本各自都做過cosine similarity sanity check(clip_vision ~0.9999992,
gpt ~1.0000001),不是沒驗證就混用精度上線。

tokenizer 用同一輪訓練重建出來的 tokenizer.pkl(見 reconstruct_capfix_tokenizer.py
的驗證紀錄,配對正確)。
"""
import time
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from minbpe import minbpe
from preprocess import preprocess
from yolo_overlay import detect_and_draw

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent  # onnx檔案放在上一層(Day39/)

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

tokenizer = minbpe.load(str(MODEL_DIR / "tokenizer.pkl"))


def load_cpu_sessions():
    providers = ["CPUExecutionProvider"]
    clip_sess = ort.InferenceSession(str(MODEL_DIR / "clip_vision.onnx"), providers=providers)
    gpt_sess = ort.InferenceSession(str(MODEL_DIR / "gpt.onnx"), providers=providers)
    return clip_sess, gpt_sess


def try_load_gpu_sessions():
    """回傳 (clip_sess, gpt_sess) 或 None。真的跑一次dummy推論確認GPU能動,
    不是只檢查 CUDAExecutionProvider 有沒有出現在 get_available_providers()
    (那個清單只代表onnxruntime有沒有編譯進CUDA支援,不代表這台機器有GPU可用)。"""
    clip_path = MODEL_DIR / "clip_vision.fp16.onnx"
    gpt_path = MODEL_DIR / "gpt.fp16.onnx"
    if not (clip_path.exists() and gpt_path.exists()):
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return None
    try:
        providers = ["CUDAExecutionProvider"]
        clip_sess = ort.InferenceSession(str(clip_path), providers=providers)
        gpt_sess = ort.InferenceSession(str(gpt_path), providers=providers)
        dummy_pixels = np.zeros((1, 3, 224, 224), dtype=np.float32)
        img_feat = clip_sess.run(None, {"pixel_values": dummy_pixels})[0]
        gpt_sess.run(None, {"input_ids": np.array([[IMAGE_TOKEN_ID]], dtype=np.int64), "img_feat": img_feat})
        return clip_sess, gpt_sess
    except Exception as e:  # noqa: BLE001 -- 任何GPU初始化/推論失敗都視為"沒有GPU可用",退回CPU-only介面
        print(f"[info] GPU sessions unavailable, falling back to CPU-only UI: {e}")
        return None


cpu_clip_session, cpu_gpt_session = load_cpu_sessions()
gpu_sessions = try_load_gpu_sessions()
GPU_AVAILABLE = gpu_sessions is not None
if GPU_AVAILABLE:
    gpu_clip_session, gpu_gpt_session = gpu_sessions

print(f"[info] GPU_AVAILABLE = {GPU_AVAILABLE}")


def run_generation(image: Image.Image, clip_sess, gpt_sess):
    """回傳 (caption, elapsed_ms)。"""
    t0 = time.perf_counter()
    pixel_values = preprocess(image)
    # FP16模型用keep_io_types=True轉換,輸入輸出仍是float32,兩條路徑都餵float32即可。
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


def generate_caption_single(image: Image.Image):
    if image is None:
        return "", None
    caption, _ = run_generation(image, cpu_clip_session, cpu_gpt_session)
    annotated = detect_and_draw(image)
    return caption, annotated


def generate_caption_compare(image: Image.Image):
    if image is None:
        return "", "", "", "", None
    cpu_caption, cpu_ms = run_generation(image, cpu_clip_session, cpu_gpt_session)
    gpu_caption, gpu_ms = run_generation(image, gpu_clip_session, gpu_gpt_session)
    annotated = detect_and_draw(image)
    return cpu_caption, f"{cpu_ms:.1f} ms", gpu_caption, f"{gpu_ms:.1f} ms", annotated


if GPU_AVAILABLE:
    with gr.Blocks(title="Thermal Caption Demo (v1.1, CPU vs GPU)") as demo:
        gr.Markdown("# Thermal Caption Demo (v1.1)\n"
                    "偵測到GPU可用,同一張圖會同時跑 **CPU FP32** 跟 **GPU FP16** 兩條路徑,並排比較延遲。")
        with gr.Row():
            image_in = gr.Image(type="pil", label="Thermal image")
        run_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### CPU FP32")
                cpu_caption_out = gr.Textbox(label="Caption")
                cpu_latency_out = gr.Textbox(label="Latency")
            with gr.Column():
                gr.Markdown("### GPU FP16")
                gpu_caption_out = gr.Textbox(label="Caption")
                gpu_latency_out = gr.Textbox(label="Latency")
        yolo_out = gr.Image(type="pil", label="YOLO detections")
        run_btn.click(
            fn=generate_caption_compare,
            inputs=[image_in],
            outputs=[cpu_caption_out, cpu_latency_out, gpu_caption_out, gpu_latency_out, yolo_out],
        )
else:
    demo = gr.Interface(
        fn=generate_caption_single,
        inputs=gr.Image(type="pil", label="Thermal image"),
        outputs=[gr.Textbox(label="Generated caption"), gr.Image(type="pil", label="YOLO detections")],
        title="Thermal Caption Demo (v1.1, CPU FP32)",
        description="上傳一張熱像圖片,生成場景描述caption跟YOLO偵測框。",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
