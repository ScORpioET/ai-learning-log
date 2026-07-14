import tensorrt as trt
import time
import sys



# TensorRT 建 engine 有三個核心物件
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(ONNX_PATH, ENGINE_PATH, PRECISION):
    # 1. 建 Builder
    builder = trt.Builder(TRT_LOGGER)

    # 2. 建 Network（explicit batch，TRT 10+ 唯一支援模式）
    network = builder.create_network()

    # 3. 建 ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 4. 讀 ONNX，餵給 parser
    print(f"Parsing {ONNX_PATH}...")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR parsing ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)
    print(f"Network layers: {network.num_layers}")

    # 5. 建 BuilderConfig（設定 workspace、精度）
    config = builder.create_builder_config()
    # workspace：build 過程中 TensorRT 可用的暫存 GPU 記憶體上限
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 31)  # 4 GB

    if PRECISION == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building with FP16")
    else:
        print("Building with FP32")

    # 6. 開始 build engine（這是最花時間的一步：kernel auto-tuning）
    print(f"Building engine... (this takes 3–10 minutes)")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized_engine is None:
        print("ERROR: engine build failed")
        sys.exit(1)

    print(f"Build succeeded in {build_time:.1f}s")

    engine_bytes = bytes(serialized_engine)
    print(f"Engine size: {len(engine_bytes) / 1e6:.1f} MB")

    # 7. 存到檔案
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine_bytes)
    print(f"Saved to {ENGINE_PATH}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Please input the input ONNX path, output engine path and percision')
        sys.exit()

    ONNX_PATH = sys.argv[1]
    ENGINE_PATH = sys.argv[2]
    PRECISION = sys.argv[3]  if len(sys.argv) > 3 else 'FP32' # 之後改成 fp16 就好

    build_engine(ONNX_PATH, ENGINE_PATH, PRECISION)