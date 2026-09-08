"""
從HuggingFace Hub下載thermal-caption-demo v1.0所需的模型權重。

權重(.onnx/.onnx.data/.pt,合計約1.6GB)不進git歷史(見.gitignore的
*.pt/*.onnx/*.onnx.data規則),改放在HF Hub的public model repo:
https://huggingface.co/ScORpioET/thermal-caption-demo

用法:
    python download_weights.py

不需要HF token(repo是public的,snapshot_download對public repo匿名下載即可)。
下載完成後檔案會落在跟這支腳本同一層目錄(Phase3/Day39/),跟Dockerfile裡
COPY指令預期的路徑一致。
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ScORpioET/thermal-caption-demo"
HERE = Path(__file__).parent

# 只抓demo實際要用的檔案,避免repo以後加了其他東西也一起抓下來。
ALLOW_PATTERNS = [
    "clip_vision.onnx", "clip_vision.onnx.data",
    "clip_vision.fp16.onnx", "clip_vision.fp16.onnx.data",
    "gpt.onnx", "gpt.onnx.data",
    "gpt.fp16.onnx", "gpt.fp16.onnx.data",
    "gpt_rgb.onnx", "gpt_rgb.onnx.data",
    "gpt_rgb.fp16.onnx", "gpt_rgb.fp16.onnx.data",
    "tokenizer.pkl", "tokenizer_rgb.pkl",
    "yolov8m.pt",
]


def main():
    print(f"[info] downloading weights from https://huggingface.co/{REPO_ID}")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(HERE),
    )
    print(f"[done] weights downloaded into: {local_dir}")

    missing = [f for f in ALLOW_PATTERNS if not (HERE / f).exists()]
    if missing:
        print(f"[warn] 以下檔案在repo裡沒找到,請確認HF repo內容: {missing}")
    else:
        print("[ok] 所有預期檔案都已就緒,可以進行 docker build。")


if __name__ == "__main__":
    main()
