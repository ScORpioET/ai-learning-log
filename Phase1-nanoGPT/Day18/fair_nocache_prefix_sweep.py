"""
排除 exporter 干擾的補測：per_step_time_sweep_prefix_vs_postfix.py 用的
pre-fix no-cache 模型(gpt2_step_nocache.onnx)是 dynamo exporter,
post-fix no-cache 模型(gpt2_step_nocache_postfix.onnx)是 legacy(dynamo=False),
兩者 exporter 不同,會混入跟 Memcpy 修復無關的變因。

這裡只補測「用跟 post-fix 一樣的 legacy exporter 匯出的 pre-fix no-cache
模型」(fair_cmp_nocache_prefix.onnx)的 per-step 曲線,拿去替換掉原本
confounded 的 no-cache 曲線,重新算一次「乾淨」的 pre-fix 交叉點。
pre-fix cache 曲線不用重測,gpt2_step.onnx 本來就已經是 legacy exporter,
跟 post-fix cache 是公平對照。
"""
import json
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
checkpoint = torch.load("checkpoints/model_00400.pt", map_location='cuda', weights_only=False)
config = checkpoint['config']
prompt_tokens = enc.encode("Hello, I am a language model,")

WARMUP = 10
NUM_RUNS = 20
T = 500

sess = ort.InferenceSession("fair_cmp_nocache_prefix.onnx", providers=providers)


def run_nocache_per_step(sess, num_new_tokens):
    input_ids = np.array([prompt_tokens], dtype=np.int64)
    per_step = np.zeros(num_new_tokens)
    for i in range(num_new_tokens):
        t0 = time.perf_counter()
        outputs = sess.run(None, {'input_ids': input_ids})
        t1 = time.perf_counter()
        per_step[i] = (t1 - t0) * 1000
        next_token = int(outputs[0][0, -1, :].argmax())
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
    return per_step


def moving_avg(arr, window=20):
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


print("warm-up...", flush=True)
for w in range(WARMUP):
    run_nocache_per_step(sess, T)
    print(f"  warm-up {w+1}/{WARMUP} 完成", flush=True)

print("正式測量...", flush=True)
runs = []
for r in range(NUM_RUNS):
    runs.append(run_nocache_per_step(sess, T))
    print(f"  repeat {r+1}/{NUM_RUNS} 完成", flush=True)

runs = np.array(runs)
median = np.median(runs, axis=0)
smooth = moving_avg(median, 20)

with open('fair_nocache_prefix_sweep_raw.json', 'w') as f:
    json.dump({'t_values': list(range(1, T+1)), 'nocache_median_ms': median.tolist(), 'nocache_smooth_ms': smooth.tolist()}, f, indent=2)

# 用已存的 pre-fix cache 曲線重算「乾淨」交叉點
with open('per_step_time_sweep_prefix_vs_postfix_raw.json') as f:
    old = json.load(f)
cache_smooth_prefix = np.array(old['results']['prefix']['cache_smooth_ms'])
postfix_crossover = old['results']['postfix']['crossover_t']
original_confounded_crossover = old['results']['prefix']['crossover_t']

fair_crossover = None
for i in range(T):
    if cache_smooth_prefix[i] <= smooth[i]:
        fair_crossover = i + 1
        break

print("\n" + "=" * 70)
print(f"原本(exporter 不一致,confounded)pre-fix 交叉點: t = {original_confounded_crossover}")
print(f"用同一個 exporter 重測後的「乾淨」pre-fix 交叉點: t = {fair_crossover}")
print(f"post-fix 交叉點(未變): t = {postfix_crossover}")
print("=" * 70)
