"""
Custom TRT engine profiler using IProfiler API.
Replaces trtexec --dumpProfile with full control over output.

Usage:
    python profile_engine.py <engine1> [<engine2> ...]
"""
import os
import sys
import torch
import numpy as np
import tensorrt as trt
from collections import defaultdict

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.int32: torch.int32,
}


class LayerProfiler(trt.IProfiler):
    """收集每層 latency（每次 execute 都會呼叫 report_layer_time）"""
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_times = defaultdict(list)

    def report_layer_time(self, layer_name, ms):
        self.layer_times[layer_name].append(ms)

    def reset(self):
        self.layer_times.clear()


def profile_engine(engine_path, iterations=500, warmup=50, num_runs=3):
    """跑 num_runs 輪，每輪 iterations 次，回傳最後一輪的 layer 細節 + 三輪 total 統計"""
    import statistics

    print(f"\nLoading {engine_path}...")
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        torch_dtype = NP_TO_TORCH[dtype]
        buffers[name] = torch.zeros(shape, dtype=torch_dtype, device='cuda')
        context.set_tensor_address(name, buffers[name].data_ptr())

    stream = torch.cuda.Stream()

    print(f"Warmup {warmup} iterations...")
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    all_totals = []
    last_results = None

    for run in range(num_runs):
        profiler = LayerProfiler()
        context.profiler = profiler
        print(f"Run {run+1}/{num_runs}: profiling {iterations} iterations...")
        for _ in range(iterations):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        results = []
        total_ms = 0.0
        for layer, times in profiler.layer_times.items():
            avg = sum(times) / len(times)
            total_ms += avg
            results.append((layer, avg))
        results.sort(key=lambda x: -x[1])
        all_totals.append(total_ms)
        last_results = results

    stats = {
        'median': statistics.median(all_totals),
        'min': min(all_totals),
        'max': max(all_totals),
        'runs': all_totals,
    }
    return last_results, stats


def classify_layer(name):
    name_lower = name.lower()
    # Reformatting = TRT boundary conversion (INT8 overhead source)
    if 'reformatting' in name_lower:
        return 'Reformatting'
    # Fused Conv：名字含 conv（不管有沒有 weight_quantized 前綴）
    if '/conv' in name_lower or '.conv' in name_lower or 'conv/' in name_lower or 'conv +' in name_lower or 'conv$' in name_lower:
        return 'ConvFused'
    # Standalone Q/DQ（沒跟其他 op fuse 的）
    if 'quantizelinear' in name_lower and 'conv' not in name_lower:
        return 'QDQ_standalone'
    # Softmax / activation
    if 'softmax' in name_lower or 'sigmoid' in name_lower or 'silu' in name_lower or 'relu' in name_lower:
        return 'Activation'
    return 'Other'


def print_summary(engine_name, results, stats):
    counts = defaultdict(int)
    times = defaultdict(float)
    for layer, ms in results:
        cat = classify_layer(layer)
        counts[cat] += 1
        times[cat] += ms

    print(f"\n{'='*80}")
    print(f"Summary: {engine_name}")
    print(f"{'='*80}")
    print(f"Total layer count: {len(results)}")
    print(f"Total time — median: {stats['median']:.3f} ms | min: {stats['min']:.3f} | max: {stats['max']:.3f} | range: {(stats['max']-stats['min']):.3f} ms")
    print(f"All runs: {[f'{t:.3f}' for t in stats['runs']]}")
    print()
    print(f"{'Category':<10} {'Count':<8} {'Time (ms)':<12} {'% of total':<12}")
    print(f"{'-'*10} {'-'*8} {'-'*12} {'-'*12}")
    for cat in ['ConvFused', 'Reformatting', 'QDQ_standalone', 'Activation', 'Other']:
        pct = 100 * times[cat] / stats['median'] if stats['median'] > 0 else 0
        print(f"{cat:<10} {counts[cat]:<8} {times[cat]:<12.4f} {pct:<12.1f}")

    print(f"\nTop 15 layers by time:")
    print(f"{'Layer':<70} {'ms':<10} {'%':<8}")
    print(f"{'-'*70} {'-'*10} {'-'*8}")
    for layer, ms in results[:15]:
        pct = 100 * ms / stats['median'] if stats['median'] > 0 else 0
        print(f"{layer[:68]:<70} {ms:<10.4f} {pct:<8.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_engine.py <engine1> [<engine2> ...]")
        sys.exit(1)

    all_totals = {}
    for engine_path in sys.argv[1:]:
        name = os.path.basename(engine_path)
        results, stats = profile_engine(engine_path)
        print_summary(name, results, stats)
        all_totals[name] = (len(results), stats['median'])

    # Cross-engine 對照
    if len(all_totals) > 1:
        print(f"\n{'='*80}")
        print(f"Cross-engine comparison")
        print(f"{'='*80}")
        print(f"{'Engine':<40} {'Layers':<10} {'Total ms':<12}")
        for name, (n, t) in all_totals.items():
            print(f"{name:<40} {n:<10} {t:<12.3f}")


if __name__ == '__main__':
    main()