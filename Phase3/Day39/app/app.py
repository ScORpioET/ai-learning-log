"""
Thermal caption demo。

v1(CPU FP32): onnxruntime(CPUExecutionProvider),不依賴PyTorch。
v1.1 階段一:GPU/FP16路徑(DEVICE_MODE=gpu 時預設走 CUDAExecutionProvider+FP16)。
v1.1 階段二(後改版):CPU/GPU 用切換式(不是並排比較)。開機時嘗試載入GPU FP16
sessions(真的跑一次推論確認GPU可用,不是只看provider有沒有列在清單裡),有GPU
才會出現Device切換鈕,預設選GPU;沒有GPU(或GPU sessions載入/推論失敗)的機器
直接不出現這個切換鈕,只能用CPU,不會讓使用者切到一個壞掉的GPU選項。
v1.1 階段三:YOLO疊框(見 yolo_overlay.py,沿用既有KEEP_CLASSES/CONF_THRESH設定)。
v1.1 階段四:RGB分支切換。盤點結論(見 export_rgb_gpt.py 開頭註解):CLIP視覺
encoder是domain-agnostic的通用預訓練權重,訓練時從沒被微調過(train_vlm.py只吃
預先算好的CLIP特徵),所以RGB分支不需要另外的clip_vision.onnx,只有gpt_rgb.onnx
+ tokenizer_rgb.pkl 各自一份。
v1.1 階段四(後改版):RGB也補上GPU/FP16路徑,規格對齊thermal。改用
best_model_rgb_full_capfix_reweight2x.pt(跟thermal同一輪修caption-completeness
bug後retrain的capfix版本,best epoch 6, val_loss=0.4142,取代原本用的舊版
best_model_rgb_full_reweight2x.pt),tokenizer配對用同一套「重建訓練時tokenizer +
實際生成雙重驗證」方法確認過(見reconstruct_rgb_capfix_tokenizer.py跟
test_rgb_capfix_tokenizer_pairing.py),gpt_rgb.fp16.onnx也做過cosine similarity
sanity check(~1.0000001)。RGB分支現在跟thermal一樣有Device切換鈕可選GPU。

v1.2:拿掉手動選thermal/RGB分支的下拉選單,改成上傳圖片後自動判斷(見
domain_detect.py,門檻/驗證方法見Phase3/domain_autodetect_check/REPORT.md)。
自動判斷不是100%保證正確(它驗證的是「這個資料集的檔案儲存慣例」,不是任意
輸入圖片畫面內容本質上是不是灰階),所以保留一個不顯眼的手動覆蓋選項
(進階選項的Accordion裡,預設"auto"跟隨自動判斷,誤判時使用者可以自己切換)
當安全網,平常不需要用到。

CPU/GPU的thermal跟rgb模型都各自從對應的capfix checkpoint匯出、各自FP16版本都
做過cosine similarity sanity check,不是沒驗證就混用精度上線。
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
from domain_detect import detect_domain

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent  # onnx檔案放在上一層(Day39/)

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

# clip_vision.onnx是domain-agnostic的通用CLIP權重,thermal/rgb共用同一份,
# 只有gpt(caption decoder)+tokenizer各自domain一份。
DOMAINS = {
    "thermal": {"gpt_cpu": "gpt.onnx", "gpt_gpu_fp16": "gpt.fp16.onnx", "tokenizer": "tokenizer.pkl"},
    "rgb": {"gpt_cpu": "gpt_rgb.onnx", "gpt_gpu_fp16": "gpt_rgb.fp16.onnx", "tokenizer": "tokenizer_rgb.pkl"},
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


def resolve_domain(image: Image.Image, override: str) -> tuple[str, str]:
    """回傳 (detected, resolved)。detected是自動判斷結果(不受override影響,
    純粹給使用者看系統偵測到什麼),resolved是實際要拿去生成用的domain
    (override="auto"時等於detected,否則是使用者手動指定的值)。"""
    detected = detect_domain(image)
    resolved = detected if override == "auto" else override
    return detected, resolved


def format_detected_label(detected: str) -> str:
    label = "Thermal" if detected == "thermal" else "RGB"
    return f"偵測為:{label}"


def preview_detected_domain(image: Image.Image):
    """上傳圖片(或清空)時觸發,只更新偵測結果顯示,不跑生成。"""
    if image is None:
        return ""
    detected, _ = resolve_domain(image, "auto")
    return format_detected_label(detected)


def generate_caption(image: Image.Image, override: str, device: str):
    if image is None:
        return "", "", None, ""

    detected, domain = resolve_domain(image, override)
    detected_label = format_detected_label(detected)
    if override != "auto" and override != detected:
        detected_label += f"(已手動覆蓋為 {('Thermal' if domain == 'thermal' else 'RGB')})"

    used_device = device
    if device == "gpu":
        gpu_sess = get_gpu_sessions(domain)
        if gpu_sess is None:
            # 這個domain沒有GPU模型(目前只有thermal轉過FP16)-> 自動退回CPU,
            # 不要假裝跑了GPU或直接壞掉。
            clip_sess, gpt_sess = load_cpu_sessions(domain)
            used_device = "cpu (此分支尚無GPU模型,已自動改用CPU)"
        else:
            clip_sess, gpt_sess = gpu_sess
    else:
        clip_sess, gpt_sess = load_cpu_sessions(domain)

    caption, elapsed_ms = run_generation(image, clip_sess, gpt_sess, tokenizers[domain])
    annotated = detect_and_draw(image)
    return caption, f"{elapsed_ms:.1f} ms  ({used_device})", annotated, detected_label


with gr.Blocks(title="Thermal Caption Demo (v1.0)") as demo:
    gr.Markdown("# Thermal Caption Demo (v1.0)")
    with gr.Row():
        if GPU_AVAILABLE:
            # 只有偵測到GPU可用才出現這個切換鈕,預設選GPU;沒GPU的機器完全不會看到它,
            # 不可能切到一個壞掉的選項。
            device_in = gr.Radio(["gpu", "cpu"], value="gpu", label="Device")
        else:
            device_in = gr.State("cpu")
    image_in = gr.Image(type="pil", label="Input image", sources=["upload"])
    detected_out = gr.Markdown("")
    with gr.Accordion("進階選項 / Advanced", open=False):
        gr.Markdown(
            "系統會自動判斷上傳的是thermal還是RGB圖片,不需要手動選擇。"
            "如果判斷結果看起來不對,可以在這裡手動覆蓋(平常不需要用到)。"
        )
        override_in = gr.Radio(
            ["auto"] + list(DOMAINS.keys()), value="auto", label="Domain override"
        )
    run_btn = gr.Button("Generate", variant="primary")
    caption_out = gr.Textbox(label="Generated caption")
    latency_out = gr.Textbox(label="Latency")
    yolo_out = gr.Image(type="pil", label="YOLO detections")

    image_in.change(fn=preview_detected_domain, inputs=[image_in], outputs=[detected_out])
    run_btn.click(
        fn=generate_caption,
        inputs=[image_in, override_in, device_in],
        outputs=[caption_out, latency_out, yolo_out, detected_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
