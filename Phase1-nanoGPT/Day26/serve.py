# serve.py
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tiktoken
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# ---- import GPTConfig 讓 pickle 反序列化找得到 ----
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "Day18"))
from train_gpt2 import GPT, GPTConfig
import __main__
__main__.GPTConfig = GPTConfig

# ===================== 路徑設定 =====================
DAY18 = Path(__file__).parent.parent / "Day18"
CHECKPOINT_PATH = DAY18 / "checkpoints" / "model_00400.pt"
ONNX_PATH = DAY18 / "gpt2_step_cache.onnx"

# ===================== 模組載入時只執行一次 =====================
# 【重要】這一整段是「server 啟動時」執行，不是「每個 request」執行
# tokenizer、checkpoint config、ONNX session 都是重物件，載入很慢
# 如果放在 endpoint 函式裡面，每個 request 都要重載一次，效能會崩

print("Loading tokenizer...")
enc = tiktoken.get_encoding('gpt2')

print("Loading checkpoint config...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']
n_layer = config.n_layer
n_head = config.n_head
head_size = config.n_embd // config.n_head
B = 1

print("Loading ONNX session...")
session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
print(f"Providers: {session.get_providers()}")
print("Ready.")


# ===================== 輔助函式（從 onnx_benchmark.py 搬過來的 step()）=====================
def step(token_id: int, past_kv_ort: list, past_length: int):
    """跑 ONNX model 一步，回傳 (next_token, new_past_kv_ort)"""
    io_binding = session.io_binding()

    input_ids_ort = ort.OrtValue.ortvalue_from_numpy(
        np.array([[token_id]], dtype=np.int64), 'cuda', 0
    )
    io_binding.bind_ortvalue_input('input_ids', input_ids_ort)

    past_length_ort = ort.OrtValue.ortvalue_from_numpy(
        np.array(past_length, dtype=np.int64), 'cuda', 0
    )
    io_binding.bind_ortvalue_input('past_length', past_length_ort)

    for i in range(n_layer):
        io_binding.bind_ortvalue_input(f'past_key_{i}', past_kv_ort[2*i])
        io_binding.bind_ortvalue_input(f'past_value_{i}', past_kv_ort[2*i+1])

    io_binding.bind_output('logits', 'cuda', 0)
    for i in range(n_layer):
        io_binding.bind_output(f'present_key_{i}', 'cuda', 0)
        io_binding.bind_output(f'present_value_{i}', 'cuda', 0)

    session.run_with_iobinding(io_binding)

    outputs = io_binding.get_outputs()
    logits_ort = outputs[0]
    new_past_kv_ort = list(outputs[1:])

    logits_np = logits_ort.numpy()
    next_token = int(logits_np[0, -1, :].argmax())
    return next_token, new_past_kv_ort


def init_past_kv():
    """建立空的 past_kv（cache 一開始沒有東西）"""
    past_kv_ort = []
    for _ in range(n_layer):
        for _ in range(2):
            empty = np.zeros((B, n_head, 0, head_size), dtype=np.float32)
            past_kv_ort.append(ort.OrtValue.ortvalue_from_numpy(empty, 'cuda', 0))
    return past_kv_ort


# ===================== Pydantic 模型（宣告 API 介面）=====================
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    elapsed_ms: float


tokens_generated_total = Counter(
    'tokens_generated_total',
    'Total tokens generated across all requests'
)
generation_duration_ms = Histogram(
    'generation_duration_ms',
    'Time to generate a full response, in milliseconds',
    buckets=(100, 250, 500, 1000, 2500, 5000, 10000),
)
generation_tokens_per_second = Gauge(
    'generation_tokens_per_second',
    'Tokens/sec of the most recent generation'
)

# ===================== FastAPI app =====================
app = FastAPI(title="GPT-2 ONNX Inference Server", version="0.1.0")

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    """健康檢查 endpoint，讓外部確認 server 有起來、GPU 有接到"""
    return {
        "status": "ok",
        "providers": session.get_providers(),
        "n_layer": n_layer,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:

    # prompt: str
    # max_new_tokens: int = 100

    # generated_text: str
    # tokens_generated: int
    # elapsed_ms: float

    t0 = time.perf_counter()

    prompt_tokens = enc.encode(req.prompt)
    past_kv_ort = init_past_kv()
    past_length = 0

    generate_tokens = []

    for tok in prompt_tokens:
        next_token, past_kv_ort = step(tok, past_kv_ort, past_length)
        past_length += 1

    generate_tokens.append(next_token)
    for _ in range(req.max_new_tokens - 1):
        next_token, past_kv_ort = step(next_token, past_kv_ort, past_length)
        generate_tokens.append(next_token)
        past_length += 1

    generated_text = enc.decode(generate_tokens)

    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000

    tokens_generated_total.inc(req.max_new_tokens)
    generation_duration_ms.observe(elapsed_ms)
    generation_tokens_per_second.set(req.max_new_tokens / (t1 - t0))

    return {
        "generated_text" : generated_text,
        "tokens_generated": req.max_new_tokens,
        "elapsed_ms": elapsed_ms,
    }
