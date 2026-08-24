"""
per_step_time_cache_vs_nocache.png 的來源腳本（per_step_time_sweep.py）用的是
gpt2_step_nocache_postfix.onnx / gpt2_step_cache.onnx —— 兩者檔名都明確帶
postfix/沒有 checkout 過任何舊 commit，可以百分之百確認那張圖是 post-fix
（main）的結果，而且是舊標準（warm-up=1、NUM_RUNS=5）。

這支腳本重新用 Day28 定案的新標準（warm-up=10、NUM_RUNS=20）分別對 pre-fix
（gpt2_step_nocache.onnx / gpt2_step.onnx，跟稍早 stable_reconcile_blog_numbers.py
用的是同一組已驗證檔案）跟 post-fix（gpt2_step_nocache_postfix.onnx /
gpt2_step_cache.onnx）各跑一次完整 per-step 掃描到 T=500，輸出格式跟舊圖
一致，並且疊在同一張圖上比較兩者的交叉點位置。
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
n_layer, n_head = config.n_layer, config.n_head
head_size = config.n_embd // config.n_head
B = 1

prompt = "Hello, I am a language model,"
prompt_tokens = enc.encode(prompt)

WARMUP = 10
NUM_RUNS = 20
T = 500

print("載入四個模型...", flush=True)
sess_nocache_pre = ort.InferenceSession("gpt2_step_nocache.onnx", providers=providers)
sess_cache_pre = ort.InferenceSession("gpt2_step.onnx", providers=providers)
sess_nocache_post = ort.InferenceSession("gpt2_step_nocache_postfix.onnx", providers=providers)
sess_cache_post = ort.InferenceSession("gpt2_step_cache.onnx", providers=providers)


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


def run_cache_pre_per_step(sess, num_new_tokens):
    """pre-fix cache：沒有 past_length 這個 input。"""
    past_kv = []
    for _ in range(n_layer):
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))

    def step(tok, past_kv):
        oi = {'input_ids': np.array([[tok]], dtype=np.int64)}
        for i in range(n_layer):
            oi[f'past_key_{i}'] = past_kv[2 * i]
            oi[f'past_value_{i}'] = past_kv[2 * i + 1]
        t0 = time.perf_counter()
        out = sess.run(None, oi)
        t1 = time.perf_counter()
        return out[0], list(out[1:]), (t1 - t0) * 1000

    logits = None
    for tok in prompt_tokens:
        logits, past_kv, _dur = step(tok, past_kv)  # 建 cache，不計時

    per_step = np.zeros(num_new_tokens)
    nt = int(logits[0, -1, :].argmax())
    for i in range(num_new_tokens):
        logits, past_kv, dur = step(nt, past_kv)
        per_step[i] = dur
        nt = int(logits[0, -1, :].argmax())
    return per_step


def run_cache_post_per_step(sess, num_new_tokens):
    """post-fix cache：多一個 past_length input。"""
    past_kv = []
    for _ in range(n_layer):
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))
        past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))

    def step(tok, past_kv, pl):
        oi = {'input_ids': np.array([[tok]], dtype=np.int64), 'past_length': np.array(pl, dtype=np.int64)}
        for i in range(n_layer):
            oi[f'past_key_{i}'] = past_kv[2 * i]
            oi[f'past_value_{i}'] = past_kv[2 * i + 1]
        t0 = time.perf_counter()
        out = sess.run(None, oi)
        t1 = time.perf_counter()
        return out[0], list(out[1:]), (t1 - t0) * 1000

    logits, pl = None, 0
    for tok in prompt_tokens:
        logits, past_kv, _dur = step(tok, past_kv, pl)
        pl += 1

    per_step = np.zeros(num_new_tokens)
    nt = int(logits[0, -1, :].argmax())
    for i in range(num_new_tokens):
        logits, past_kv, dur = step(nt, past_kv, pl)
        per_step[i] = dur
        pl += 1
        nt = int(logits[0, -1, :].argmax())
    return per_step


def moving_avg(arr, window=20):
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


def find_crossover(t_values, nocache_smooth, cache_smooth):
    """回傳第一個 cache_smooth <= nocache_smooth 的 t（None 代表整個範圍內沒交叉）"""
    for i in range(len(t_values)):
        if cache_smooth[i] <= nocache_smooth[i]:
            return t_values[i]
    return None


def sweep(name, nocache_fn, cache_fn):
    print(f"\n===== {name}：warm-up={WARMUP}, NUM_RUNS={NUM_RUNS}, T={T} =====", flush=True)
    print("warm-up...", flush=True)
    for w in range(WARMUP):
        nocache_fn(T)
        cache_fn(T)
        print(f"  warm-up {w+1}/{WARMUP} 完成", flush=True)

    print("正式測量...", flush=True)
    nocache_runs, cache_runs = [], []
    for r in range(NUM_RUNS):
        nocache_runs.append(nocache_fn(T))
        cache_runs.append(cache_fn(T))
        print(f"  repeat {r+1}/{NUM_RUNS} 完成", flush=True)

    nocache_runs = np.array(nocache_runs)
    cache_runs = np.array(cache_runs)
    nocache_median = np.median(nocache_runs, axis=0)
    cache_median = np.median(cache_runs, axis=0)
    nocache_smooth = moving_avg(nocache_median, 20)
    cache_smooth = moving_avg(cache_median, 20)

    t_values = list(range(1, T + 1))
    crossover = find_crossover(t_values, nocache_smooth, cache_smooth)
    print(f"  {name} 交叉點（20-step 平滑後，cache per-step <= no-cache per-step 的第一個 t）: {crossover}")

    return {
        't_values': t_values,
        'nocache_median_ms': nocache_median.tolist(),
        'cache_median_ms': cache_median.tolist(),
        'nocache_smooth_ms': nocache_smooth.tolist(),
        'cache_smooth_ms': cache_smooth.tolist(),
        'crossover_t': crossover,
    }


results = {}
results['prefix'] = sweep(
    "pre-fix (a14ca01)",
    lambda t: run_nocache_per_step(sess_nocache_pre, t),
    lambda t: run_cache_pre_per_step(sess_cache_pre, t),
)
results['postfix'] = sweep(
    "post-fix (main)",
    lambda t: run_nocache_per_step(sess_nocache_post, t),
    lambda t: run_cache_post_per_step(sess_cache_post, t),
)

with open('per_step_time_sweep_prefix_vs_postfix_raw.json', 'w') as f:
    json.dump({'WARMUP': WARMUP, 'NUM_RUNS': NUM_RUNS, 'T': T, 'results': results}, f, indent=2)

with open('per_step_time_sweep_prefix_vs_postfix.csv', 'w') as f:
    f.write("t,nocache_prefix_smooth_ms,cache_prefix_smooth_ms,nocache_postfix_smooth_ms,cache_postfix_smooth_ms\n")
    for i in range(T):
        f.write(f"{i+1},{results['prefix']['nocache_smooth_ms'][i]:.4f},{results['prefix']['cache_smooth_ms'][i]:.4f},"
                f"{results['postfix']['nocache_smooth_ms'][i]:.4f},{results['postfix']['cache_smooth_ms'][i]:.4f}\n")

print("\n" + "=" * 70)
print(f"pre-fix 交叉點: t = {results['prefix']['crossover_t']}")
print(f"post-fix 交叉點: t = {results['postfix']['crossover_t']}")
print("=" * 70)
print("\n原始數字存在 per_step_time_sweep_prefix_vs_postfix_raw.json / .csv")

# ---------------- 畫圖 ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ts = np.arange(1, T + 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(ts, results['prefix']['nocache_smooth_ms'], color='#1f77b4', linewidth=1.5, linestyle='--', label='pre-fix no-cache')
axes[0].plot(ts, results['prefix']['cache_smooth_ms'], color='#d62728', linewidth=1.5, linestyle='--', label='pre-fix cache')
axes[0].plot(ts, results['postfix']['nocache_smooth_ms'], color='#1f77b4', linewidth=2.0, label='post-fix no-cache')
axes[0].plot(ts, results['postfix']['cache_smooth_ms'], color='#d62728', linewidth=2.0, label='post-fix cache')
if results['prefix']['crossover_t']:
    axes[0].axvline(results['prefix']['crossover_t'], color='gray', linestyle=':', alpha=0.7)
    axes[0].annotate(f"pre-fix crossover t={results['prefix']['crossover_t']}",
                      xy=(results['prefix']['crossover_t'], 0), xytext=(5, 5), textcoords='offset points',
                      fontsize=8, color='gray')
if results['postfix']['crossover_t']:
    axes[0].axvline(results['postfix']['crossover_t'], color='black', linestyle=':', alpha=0.7)
    axes[0].annotate(f"post-fix crossover t={results['postfix']['crossover_t']}",
                      xy=(results['postfix']['crossover_t'], 0), xytext=(5, 15), textcoords='offset points',
                      fontsize=8, color='black')
axes[0].set_xlabel('t (step index)')
axes[0].set_ylabel('per-step median time (ms, 20-step moving avg)')
axes[0].set_title('Per-step time: pre-fix vs post-fix (cache vs no-cache)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 右邊：cache only 放大看交叉點附近細節
axes[1].plot(ts, results['prefix']['cache_smooth_ms'], color='#d62728', linewidth=1.5, linestyle='--', label='pre-fix cache')
axes[1].plot(ts, results['postfix']['cache_smooth_ms'], color='#d62728', linewidth=2.0, label='post-fix cache')
axes[1].plot(ts, results['prefix']['nocache_smooth_ms'], color='#1f77b4', linewidth=1.5, linestyle='--', label='pre-fix no-cache')
axes[1].plot(ts, results['postfix']['nocache_smooth_ms'], color='#1f77b4', linewidth=2.0, label='post-fix no-cache')
axes[1].set_xlabel('t (step index)')
axes[1].set_ylabel('per-step median time (ms, 20-step moving avg)')
axes[1].set_title('Same data, no crossover markers (cleaner view)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('per_step_time_prefix_vs_postfix.png', dpi=150)
print("\n圖存到 per_step_time_prefix_vs_postfix.png")
