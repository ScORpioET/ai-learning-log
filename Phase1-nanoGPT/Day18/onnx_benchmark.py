import time
import numpy as np
import onnxruntime as ort
import tiktoken
import torch
from train_gpt2 import GPT, GPTConfig
import __main__
__main__.GPTConfig = GPTConfig 

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

enc = tiktoken.get_encoding('gpt2')


# 只需要 config 資訊，不需要真的載入權重
checkpoint = torch.load("checkpoints/model_00400.pt", map_location='cuda', weights_only=False)
config = checkpoint['config']
n_layer, n_head = config.n_layer, config.n_head
head_size = config.n_embd // config.n_head
B = 1

prompt = "Hello, I am a language model,"
prompt_tokens = enc.encode(prompt)
num_new_tokens = 100
num_runs = 3

# ---------------- 純 PyTorch 版本(不透過 ONNX，直接用原始模型) ----------------
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

def run_pytorch_once():
    seq = torch.tensor([prompt_tokens], dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_new_tokens):
            logits, _ = model(seq)
            next_token = logits[0, -1, :].argmax().item()
            seq = torch.cat([seq, torch.tensor([[next_token]])], dim=1)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

# warm-up
run_pytorch_once()
pytorch_times = [run_pytorch_once() for _ in range(num_runs)]
pytorch_median = float(np.median(pytorch_times))

# ---------------- 無 cache 版本 ----------------
sess_nocache = ort.InferenceSession("gpt2_step_nocache.onnx", providers=providers)

def run_nocache_once():
    input_ids = np.array([prompt_tokens], dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(num_new_tokens):
        outputs = sess_nocache.run(None, {'input_ids': input_ids})
        next_token = int(outputs[0][0, -1, :].argmax())
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms

# warm-up
run_nocache_once()
nocache_times = [run_nocache_once() for _ in range(num_runs)]
nocache_median = float(np.median(nocache_times))

# ---------------- 有 cache 版本 ----------------
sess_cache = ort.InferenceSession("gpt2_step_cache.onnx", providers=providers)

def run_cache_once():
    past_kv = []
    for _ in range(n_layer):
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))

    def step(token_id, past_kv):
        ort_inputs = {'input_ids': np.array([[token_id]], dtype=np.int64)}
        for i in range(n_layer):
            ort_inputs[f'past_key_{i}'] = past_kv[2*i]
            ort_inputs[f'past_value_{i}'] = past_kv[2*i+1]
        outputs = sess_cache.run(None, ort_inputs)
        return outputs[0], list(outputs[1:])

    t0 = time.perf_counter()
    logits = None
    for tok in prompt_tokens:
        logits, past_kv = step(tok, past_kv)
    next_token = int(logits[0, -1, :].argmax())
    for _ in range(num_new_tokens - 1):
        logits, past_kv = step(next_token, past_kv)
        next_token = int(logits[0, -1, :].argmax())
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

# warm-up
run_cache_once()
cache_times = [run_cache_once() for _ in range(num_runs)]
cache_median = float(np.median(cache_times))

print(f"純 PyTorch(無 cache），median 耗時: {pytorch_median:.2f} ms")
print(f"ONNX 無 cache，       median 耗時: {nocache_median:.2f} ms")
print(f"ONNX 有 cache，       median 耗時: {cache_median:.2f} ms")
print()
print(f"ONNX 無 cache vs 純 PyTorch： {pytorch_median / nocache_median:.2f}x")
print(f"ONNX 有 cache vs ONNX 無 cache： {nocache_median / cache_median:.2f}x")
print(f"ONNX 有 cache vs 純 PyTorch： {pytorch_median / cache_median:.2f}x")

print(sess_nocache.get_providers())
print(sess_cache.get_providers())