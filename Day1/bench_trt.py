import time
import torch
import tensorrt as trt

ENGINE_PATH = "yolov8m_fp32.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. Load engine
print(f"Loading {ENGINE_PATH}...")
with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()

# 2. 列出所有 tensor 資訊
print("\nEngine tensors:")
input_names, output_names = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    kind = "INPUT " if is_input else "OUTPUT"
    print(f"  {kind} {name}: shape={tuple(shape)} dtype={dtype.__name__}")
    (input_names if is_input else output_names).append((name, tuple(shape)))

# 3. 用 torch tensor 當 GPU buffer
buffers = {}
for name, shape in input_names + output_names:
    buffers[name] = torch.zeros(shape, dtype=torch.float32, device="cuda")
    context.set_tensor_address(name, buffers[name].data_ptr())

# 4. 塞假輸入
input_name = input_names[0][0]
buffers[input_name].copy_(torch.rand_like(buffers[input_name]))

# 5. 建 CUDA stream
stream = torch.cuda.Stream()

# 6. Warmup
print("\nWarming up (20 iters)...")
for _ in range(20):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

# 7. Benchmark
N = 500
print(f"Benchmarking {N} iters...")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
elapsed = time.time() - t0

print(f"\nTensorRT FP32 FPS: {N/elapsed:.1f}")
print(f"Latency per inference: {elapsed*1000/N:.2f} ms")