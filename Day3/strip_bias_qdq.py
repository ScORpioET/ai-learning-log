"""
Post-process INT8 ONNX to strip bias Int32 QDQ nodes for TRT compatibility.

TRT GPU 的 int8 conv kernel 要 FP32 bias（int8×int8 → int32 accumulator + FP32 bias）,
但 ONNX Runtime quantize_static 硬把 bias 也量化成 Int32，導致 TRT parse 失敗。
這個 script 把 bias 的 Int32 initializer + DQ node 拆掉，換成 FP32 initializer。
"""
import onnx
import numpy as np
from onnx import numpy_helper
import sys
import os


def strip_bias_qdq(input_path, output_path):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    graph = model.graph

    # 建 initializer 的名字 → 物件 索引，方便查
    init_by_name = {init.name: init for init in graph.initializer}

    # 找所有 bias 相關的 DequantizeLinear 節點
    bias_dq_nodes = [
        n for n in graph.node
        if n.op_type == 'DequantizeLinear' and 'bias' in n.name.lower()
    ]
    print(f"Found {len(bias_dq_nodes)} bias DequantizeLinear nodes to strip")

    if len(bias_dq_nodes) == 0:
        print("Nothing to strip. Exit.")
        return

    inits_to_remove = set()

    for dq_node in bias_dq_nodes:
        # DQ 節點格式：inputs = [int_tensor, scale, zero_point (optional)]
        #              outputs = [fp32_tensor]
        int_name = dq_node.input[0]
        scale_name = dq_node.input[1]
        zp_name = dq_node.input[2] if len(dq_node.input) > 2 else None
        fp32_output_name = dq_node.output[0]

        int_init = init_by_name.get(int_name)
        scale_init = init_by_name.get(scale_name)
        zp_init = init_by_name.get(zp_name) if zp_name else None

        if int_init is None or scale_init is None:
            print(f"  [SKIP] {dq_node.name}: 找不到 initializer")
            continue

        # 反量化：FP32 = (Int - zero_point) × scale
        int_arr = numpy_helper.to_array(int_init).astype(np.float32)
        scale_arr = numpy_helper.to_array(scale_init).astype(np.float32)
        zp_arr = numpy_helper.to_array(zp_init).astype(np.float32) if zp_init is not None else 0.0

        fp32_arr = (int_arr - zp_arr) * scale_arr

        # 建一個新 FP32 initializer，用 DQ 節點的 OUTPUT 名字
        # （這樣下游的 Conv 找 fp32_output_name 時直接拿到 FP32 initializer）
        new_init = numpy_helper.from_array(fp32_arr, name=fp32_output_name)
        graph.initializer.append(new_init)

        # 標記要刪的舊 initializer（bias 的 int32/scale/zp 通常不會 share，可以安全刪）
        inits_to_remove.add(int_name)
        inits_to_remove.add(scale_name)
        if zp_name:
            inits_to_remove.add(zp_name)

        print(f"  [OK]   {dq_node.name}: Int32({int_arr.shape}) → FP32 initializer")

    # 從 graph 拔掉這些 DQ 節點
    for dq_node in bias_dq_nodes:
        graph.node.remove(dq_node)

    # 從 initializer 拔掉舊的 Int32/scale/zp（先重掃一次確認沒被別人 reference）
    all_input_names = set()
    for n in graph.node:
        for i in n.input:
            all_input_names.add(i)

    kept = []
    for init in graph.initializer:
        if init.name in inits_to_remove and init.name not in all_input_names:
            continue
        kept.append(init)
    del graph.initializer[:]
    graph.initializer.extend(kept)

    print(f"\nRemoved {len(bias_dq_nodes)} DQ nodes and {len(inits_to_remove)} obsolete initializers")

    # Validate + Save
    print("Checking model validity...")
    onnx.checker.check_model(model)
    print("Model is valid.")

    onnx.save(model, output_path)
    print(f"\n✅ Saved cleaned model to {output_path}")

    # 對照檔案大小
    in_mb = os.path.getsize(input_path) / 1024**2
    out_mb = os.path.getsize(output_path) / 1024**2
    print(f"\nFile size:")
    print(f"  Before: {in_mb:.2f} MB")
    print(f"  After:  {out_mb:.2f} MB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python strip_bias_qdq.py <model_base>  e.g. yolov8n, yolov8m")
        sys.exit(1)
    MODEL_BASE = sys.argv[1]
    MODEL_DIR = os.path.expanduser("~/ai-transition-2026/model")
    INPUT = os.path.join(MODEL_DIR, f"{MODEL_BASE}_int8.onnx")
    OUTPUT = os.path.join(MODEL_DIR, f"{MODEL_BASE}_int8_clean.onnx")

    strip_bias_qdq(INPUT, OUTPUT)