"""
效能量化比較:cache vs no-cache,跨3種生成長度、跨CPU/GPU分開記錄。

方法論比照Phase1-nanoGPT/Day18那批benchmark腳本(optimize_and_bench.py等):
每個設定先warmup幾次(排除第一次呼叫的cold-start/cudnn autotune開銷),
重複跑N次記錄wall-clock時間,取median(不是單次數字,避免偶發的系統雜訊
被當成真實差異)。GPU計時前後都呼叫torch.cuda.synchronize(),不然只會測到
kernel launch時間,不是真正執行完成的時間。

生成長度用force_full_length=True強制跑滿(不因為遇到EOS提早停),這樣
cache/no-cache兩邊在同一個長度設定下做的計算量才是真的可比的一樣多,
不會因為某一版本剛好提早生成EOS而少算幾步、拉低平均。

CPU/GPU的數字分開記錄成兩個表,不混在一起報一個籠統的數字。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import IMAGE_ROOT, TEST_IMAGES, load_gpt_model, load_clip, get_image_feature, \
    generate_no_cache, generate_with_cache

LENGTHS = [20, 50, 100]
WARMUP_RUNS = 3
TIMED_RUNS = 10
OUT_PATH = Path(__file__).parent / "benchmark_results.json"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_calls(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    sync(device)
    times_ms = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    return times_ms


def measure_peak_mem_mb(fn, device):
    if device != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_device(device):
    print(f"\n{'='*70}\ndevice = {device}\n{'='*70}")
    gpt_model = load_gpt_model(device)
    clip_model, processor = load_clip(device)
    img_feat = get_image_feature(clip_model, processor, IMAGE_ROOT / TEST_IMAGES[0], device)

    rows = []
    for length in LENGTHS:
        fn_no_cache = lambda: generate_no_cache(
            gpt_model, img_feat, max_new_tokens=length, mode="greedy",
            device=device, force_full_length=True)
        fn_cache = lambda: generate_with_cache(
            gpt_model, img_feat, max_new_tokens=length, mode="greedy",
            device=device, force_full_length=True)

        times_no_cache = time_calls(fn_no_cache, WARMUP_RUNS, TIMED_RUNS, device)
        times_cache = time_calls(fn_cache, WARMUP_RUNS, TIMED_RUNS, device)

        mem_no_cache = measure_peak_mem_mb(fn_no_cache, device)
        mem_cache = measure_peak_mem_mb(fn_cache, device)

        med_no_cache = float(np.median(times_no_cache))
        med_cache = float(np.median(times_cache))

        row = {
            "device": device,
            "length": length,
            "no_cache_median_ms": med_no_cache,
            "cache_median_ms": med_cache,
            "no_cache_per_token_ms": med_no_cache / length,
            "cache_per_token_ms": med_cache / length,
            "speedup_x": med_no_cache / med_cache if med_cache > 0 else None,
            "no_cache_all_ms": times_no_cache,
            "cache_all_ms": times_cache,
            "no_cache_peak_mem_mb": mem_no_cache,
            "cache_peak_mem_mb": mem_cache,
        }
        rows.append(row)
        print(f"length={length:>4}  no_cache={med_no_cache:8.2f}ms  cache={med_cache:8.2f}ms  "
              f"speedup={row['speedup_x']:.2f}x" if row['speedup_x'] else
              f"length={length:>4}  no_cache={med_no_cache:8.2f}ms  cache={med_cache:8.2f}ms")
    return rows


def main():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("[info] CUDA not available on this run -- skipping GPU benchmark.")

    all_rows = []
    for device in devices:
        all_rows.extend(run_device(device))

    OUT_PATH.write_text(json.dumps({
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "lengths": LENGTHS,
        "rows": all_rows,
    }, indent=2))
    print(f"\n[done] results written to {OUT_PATH}")

    print("\n" + "=" * 100)
    print(f"{'device':>6} | {'length':>6} | {'no_cache med(ms)':>17} | {'cache med(ms)':>14} | "
          f"{'no_cache/tok(ms)':>17} | {'cache/tok(ms)':>14} | {'speedup':>8}")
    print("-" * 100)
    for r in all_rows:
        speedup_str = f"{r['speedup_x']:.2f}x" if r['speedup_x'] else "n/a"
        print(f"{r['device']:>6} | {r['length']:>6} | {r['no_cache_median_ms']:>17.2f} | "
              f"{r['cache_median_ms']:>14.2f} | {r['no_cache_per_token_ms']:>17.3f} | "
              f"{r['cache_per_token_ms']:>14.3f} | {speedup_str:>8}")


if __name__ == "__main__":
    main()
