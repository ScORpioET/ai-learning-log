import time
import sys
import torch
import numpy as np
import tensorrt as trt

NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.int32: torch.int32,
}

if len(sys.argv) < 2:
    print("Usage: python bench_trt.py <engine_path> [N_iters]")
    sys.exit(1)

ENGINE_PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. Load engine
print(f"Loading {ENGINE_PATH}...")
with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()

# 2. List tensors
print("\nEngine tensors:")
input_names, output_names = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    kind = "INPUT " if is_input else "OUTPUT"
    print(f"  {kind} {name}: shape={tuple(shape)} dtype={dtype.__name__}")
    (input_names if is_input else output_names).append((name, tuple(shape), dtype))

# 3. GPU buffers（dtype 從 engine 讀）
buffers = {}
for name, shape, np_dtype in input_names + output_names:
    torch_dtype = NP_TO_TORCH[np_dtype]
    buffers[name] = torch.zeros(shape, dtype=torch_dtype, device="cuda")
    context.set_tensor_address(name, buffers[name].data_ptr())

# 4. Fake input（用對的 dtype）
input_name = input_names[0][0]
buffers[input_name].copy_(torch.rand_like(buffers[input_name]))

# 5. CUDA stream
stream = torch.cuda.Stream()

# 6. Warmup
print(f"\nWarming up (20 iters)...")
for _ in range(20):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

# 7. Benchmark
print(f"Benchmarking {N} iters...")
torch.cuda.synchronize()
t0 = time.time()
# 7. Benchmark（多次跑取 median）
import statistics
NUM_RUNS = 5

all_fps = []
all_ms = []
for run in range(NUM_RUNS):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    elapsed = time.time() - t0
    fps = N / elapsed
    latency_ms = elapsed * 1000 / N
    all_fps.append(fps)
    all_ms.append(latency_ms)
    print(f"  Run {run+1}/{NUM_RUNS}: {fps:.1f} FPS ({latency_ms:.3f} ms)")

elapsed = time.time() - t0
print(f"\n{ENGINE_PATH}")
print(f"  FPS    — median: {statistics.median(all_fps):.1f} | min: {min(all_fps):.1f} | max: {max(all_fps):.1f}")
print(f"  Latency — median: {statistics.median(all_ms):.3f} ms | min: {min(all_ms):.3f} | max: {max(all_ms):.3f} | range: {max(all_ms)-min(all_ms):.3f} ms")
print(f"  All runs (ms): {[f'{m:.3f}' for m in all_ms]}")
