"""
Thermal caption demo。

v1(CPU FP32): onnxruntime(CPUExecutionProvider),不依賴PyTorch。
v1.1 階段一:GPU/FP16路徑(DEVICE_MODE=gpu 時預設走 CUDAExecutionProvider+FP16)。
v1.1 階段二:CPU vs GPU 效能比較模式 -- 開機時嘗試載入GPU FP16 sessions(真的跑一次
推論確認GPU可用,不是只看 provider 有沒有列在清單裡),成功的話UI改成「同一張圖
兩條路徑並排」+各自延遲(ms);沒有GPU(或GPU sessions載入/推論失敗)就退回v1的
單一路徑介面,不會讓沒GPU的機器看到壞掉的按鈕。
v1.1 階段三:YOLO疊框(見 yolo_overlay.py,沿用既有KEEP_CLASSES/CONF_THRESH設定)。
v1.1 階段四:RGB分支切換。盤點結論(見 export_rgb_gpt.py 開頭註解):CLIP視覺
encoder是domain-agnostic的通用預訓練權重,訓練時從沒被微調過(train_vlm.py只吃
預先算好的CLIP特徵),所以RGB分支不需要另外的clip_vision.onnx,只有gpt_rgb.onnx
(從best_model_rgb_full_reweight2x.pt匯出)+ tokenizer_rgb.pkl(同一套「重建
訓練時tokenizer」方法+實際生成驗證,見reconstruct_rgb_tokenizer.py跟
test_rgb_tokenizer_pairing.py)。RGB目前只有CPU FP32版本,沒有另外做FP16轉換
(沒被要求也沒驗證過,不要沒驗證就上線),所以RGB選項底下GPU比較欄位會顯示
「尚無RGB GPU模型」而不是硬套thermal的GPU session。

CPU 跟 GPU 兩份 thermal 模型都是從 best_model_full_capfix_reweight2x.pt 這同一個
checkpoint匯出,FP16版本各自都做過cosine similarity sanity check(clip_vision
~0.9999992, gpt ~1.0000001),不是沒驗證就混用精度上線。
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

# clip_vision.onnx是domain-agnostic的通用CLIP權重,thermal/rgb共用同一份,
# 只有gpt(caption decoder)+tokenizer各自domain一份。
DOMAINS = {
    "thermal": {"gpt_cpu": "gpt.onnx", "gpt_gpu_fp16": "gpt.fp16.onnx", "tokenizer": "tokenizer.pkl"},
    "rgb": {"gpt_cpu": "gpt_rgb.onnx", "gpt_gpu_fp16": None, "tokenizer": "tokenizer_rgb.pkl"},
}

tokenizers = {name: minbpe.load(str(MODEL_DIR / cfg["tokenizer"])) for name, cfg in DOMAINS.items()}
_clip_cpu_providers = ["CPUExecutionProvider"]
shared_clip_cpu_session = ort.InferenceSession(str(MODEL_DIR / "clip_vision.onnx"), providers=_clip_cpu_providers)


def load_cpu_sessions(domain: str):
    providers = ["CPUExecutionProvider"]
    gpt_sess = ort.InferenceSession(str(MODEL_DIR / DOMAINS[domain]["gpt_cpu"]), providers=providers)
    return shared_clip_cpu_session, gpt_sess


def try_load_gpu_sessions(domain: str):
    """回傳 (clip_sess, gpt_sess) 或 None。真的跑一次dummy推論確認GPU能動,
    不是只檢查 CUDAExecutionProvider 有沒有出現在 get_available_providers()
    (那個清單只代表onnxruntime有沒有編譯進CUDA支援,不代表這台機器有GPU可用)。
    domain沒有對應的FP16 gpt(目前只有thermal有)就直接回傳None。"""
    gpt_fp16_name = DOMAINS[domain]["gpt_gpu_fp16"]
    clip_path = MODEL_DIR / "clip_vision.fp16.onnx"
    if gpt_fp16_name is None or not clip_path.exists():
        return None
    gpt_path = MODEL_DIR / gpt_fp16_name
    if not gpt_path.exists():
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
        print(f"[info] GPU sessions unavailable for domain={domain}, falling back: {e}")
        return None


# 系統層級的「這台機器有沒有GPU可用」旗標,只用thermal(唯一有FP16版本的domain)測一次。
_gpu_probe = try_load_gpu_sessions("thermal")
GPU_AVAILABLE = _gpu_probe is not None
print(f"[info] GPU_AVAILABLE = {GPU_AVAILABLE}")

_gpu_session_cache = {}


def get_gpu_sessions(domain: str):
    if domain not in _gpu_session_cache:
        _gpu_session_cache[domain] = try_load_gpu_sessions(domain)
    return _gpu_session_cache[domain]


def run_generation(image: Image.Image, clip_sess, gpt_sess, tokenizer):
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


def generate_caption_single(image: Image.Image, domain: str):
    if image is None:
        return "", None
    clip_sess, gpt_sess = load_cpu_sessions(domain)
    caption, _ = run_generation(image, clip_sess, gpt_sess, tokenizers[domain])
    annotated = detect_and_draw(image)
    return caption, annotated


def generate_caption_compare(image: Image.Image, domain: str):
    if image is None:
        return "", "", "", "", None
    clip_cpu, gpt_cpu = load_cpu_sessions(domain)
    cpu_caption, cpu_ms = run_generation(image, clip_cpu, gpt_cpu, tokenizers[domain])

    gpu_sess = get_gpu_sessions(domain)
    if gpu_sess is None:
        gpu_caption, gpu_ms_label = "(尚無此分支的GPU FP16模型)", "N/A"
    else:
        gpu_clip, gpu_gpt = gpu_sess
        gpu_caption, gpu_ms = run_generation(image, gpu_clip, gpu_gpt, tokenizers[domain])
        gpu_ms_label = f"{gpu_ms:.1f} ms"

    annotated = detect_and_draw(image)
    return cpu_caption, f"{cpu_ms:.1f} ms", gpu_caption, gpu_ms_label, annotated


if GPU_AVAILABLE:
    with gr.Blocks(title="Thermal Caption Demo (v1.1, CPU vs GPU)") as demo:
        gr.Markdown("# Thermal Caption Demo (v1.1)\n"
                    "偵測到GPU可用,同一張圖會同時跑 **CPU FP32** 跟 **GPU FP16** 兩條路徑,並排比較延遲。"
                    "選RGB時,GPU那欄會顯示尚無GPU模型(目前只有thermal轉過FP16)。")
        domain_in = gr.Radio(list(DOMAINS.keys()), value="thermal", label="Domain")
        with gr.Row():
            image_in = gr.Image(type="pil", label="Input image")
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
            inputs=[image_in, domain_in],
            outputs=[cpu_caption_out, cpu_latency_out, gpu_caption_out, gpu_latency_out, yolo_out],
        )
else:
    demo = gr.Interface(
        fn=generate_caption_single,
        inputs=[gr.Image(type="pil", label="Input image"), gr.Radio(list(DOMAINS.keys()), value="thermal", label="Domain")],
        outputs=[gr.Textbox(label="Generated caption"), gr.Image(type="pil", label="YOLO detections")],
        title="Thermal Caption Demo (v1.1, CPU FP32)",
        description="上傳一張圖片(thermal或RGB),生成場景描述caption跟YOLO偵測框。",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
