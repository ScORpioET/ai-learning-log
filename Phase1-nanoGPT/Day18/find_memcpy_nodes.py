"""
排查 ONNX Runtime 計算圖裡 Memcpy node 的完整流程。

用法：
    python3 find_memcpy_nodes.py <model.onnx> [--seq-len 8] [--steps 10] [--cpu]

流程：
    1. 用 enable_profiling + optimized_model_filepath 建立 session、跑幾步 forward，
       存出優化後的計算圖（.onnx）跟逐 node 的 timing（.json）。
    2. 打開優化後的圖，找出所有 Memcpy node（MemcpyFromHost / MemcpyToHost）。
    3. 對每個 Memcpy node，往上游追 producer（誰算出這個要被搬運的值）、
       往下游追 consumer（這個值搬過去之後給誰用），印出完整因果鏈。
    4. 額外印出 profiling JSON 裡 Memcpy 相關事件的耗時統計，跟總耗時的佔比。

這支腳本假設 model 的 input 是 input_ids（[batch, seq_len]的 int64），如果你的模型
input 名稱或格式不一樣，改 `build_dummy_inputs()` 就好，其他部分不用動。
"""

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def build_dummy_inputs(sess, seq_len):
    """根據 session 宣告的 input，組出一組隨機的 dummy input。"""
    inputs = {}
    for inp in sess.get_inputs():
        shape = [
            (seq_len if isinstance(d, str) or d is None else d)
            for d in inp.shape
        ]
        # 常見狀況：batch 維度也是動態的，固定成 1
        if len(shape) >= 1 and (
            isinstance(inp.shape[0], str) or inp.shape[0] is None
        ):
            shape[0] = 1
        if "int64" in inp.type:
            inputs[inp.name] = np.random.randint(0, 50257, shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
    return inputs


def run_and_profile(model_path, seq_len, steps, providers):
    """跑 session，順便存出 optimized graph + profiling json，回傳兩個檔案路徑。"""
    so = ort.SessionOptions()
    so.enable_profiling = True
    optimized_path = model_path.replace(".onnx", "_optimized_graph.onnx")
    so.optimized_model_filepath = optimized_path

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    inputs = build_dummy_inputs(sess, seq_len)

    for _ in range(steps):
        sess.run(None, inputs)

    profile_path = sess.end_profiling()
    print(f"[1/4] optimized graph 存到: {optimized_path}")
    print(f"[1/4] profiling json 存到:  {profile_path}")
    return optimized_path, profile_path


def find_memcpy_nodes(optimized_path):
    """打開優化後的圖，抓出所有 Memcpy node。"""
    g = onnx.load(optimized_path)
    memcpy_nodes = [n for n in g.graph.node if "Memcpy" in n.op_type]
    print(f"\n[2/4] 圖裡總共 {len(g.graph.node)} 個 node，"
          f"其中 {len(memcpy_nodes)} 個是 Memcpy")
    return g, memcpy_nodes


def find_producer(graph, tensor_name):
    """誰的 output 是這個 tensor_name。"""
    for n in graph.node:
        if tensor_name in n.output:
            return n
    return None


def find_consumers(graph, tensor_name):
    """誰的 input 用到這個 tensor_name（可能不只一個）。"""
    return [n for n in graph.node if tensor_name in n.input]


def trace_causal_chain(g, memcpy_nodes):
    """對每個 Memcpy node，印出上游 producer + 下游 consumer。"""
    print("\n[3/4] 因果鏈追蹤（往上游找 producer、往下游找 consumer）：\n")
    print(f"{'方向':<16}{'Memcpy node':<22}{'搬運的 tensor':<45}{'上游 producer':<20}下游 consumer")
    print("-" * 140)

    for n in memcpy_nodes:
        tensor_name = n.input[0]
        producer = find_producer(g.graph, tensor_name)
        producer_desc = f"{producer.op_type}({producer.name})" if producer else "(graph input / initializer)"

        # Memcpy 的 output 才是下游真正會用到的 tensor
        out_tensor = n.output[0] if n.output else tensor_name
        consumers = find_consumers(g.graph, out_tensor)
        consumer_desc = ", ".join(f"{c.op_type}({c.name})" for c in consumers) or "(graph output)"

        print(f"{n.op_type:<16}{n.name:<22}{tensor_name:<45}{producer_desc:<20}{consumer_desc}")


def summarize_profiling(profile_path):
    """統計 profiling json 裡 Memcpy 相關事件的耗時佔比。"""
    with open(profile_path) as f:
        events = json.load(f)

    total_us = 0
    memcpy_us = 0
    memcpy_by_name = {}

    for e in events:
        dur = e.get("dur", 0)
        name = e.get("name", "")
        cat = e.get("cat", "")
        if cat != "Node":
            continue
        total_us += dur
        if "Memcpy" in name:
            memcpy_us += dur
            memcpy_by_name[name] = memcpy_by_name.get(name, 0) + dur

    print(f"\n[4/4] Profiling 統計：")
    print(f"  所有 node 總耗時: {total_us} us")
    print(f"  Memcpy 總耗時:   {memcpy_us} us")
    if total_us > 0:
        print(f"  Memcpy 佔比:     {memcpy_us / total_us * 100:.3f}%")

    if memcpy_by_name:
        print("\n  各 Memcpy node 耗時（跨所有 step 加總）：")
        for name, us in sorted(memcpy_by_name.items(), key=lambda x: -x[1]):
            print(f"    {name:<20} {us:>8} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="ONNX 模型路徑")
    ap.add_argument("--seq-len", type=int, default=8, help="dummy input 的序列長度")
    ap.add_argument("--steps", type=int, default=10, help="跑幾次 forward")
    ap.add_argument("--cpu", action="store_true", help="只用 CPUExecutionProvider（預設 CUDA+CPU）")
    args = ap.parse_args()

    providers = ["CPUExecutionProvider"] if args.cpu else [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    optimized_path, profile_path = run_and_profile(
        args.model, args.seq_len, args.steps, providers
    )
    g, memcpy_nodes = find_memcpy_nodes(optimized_path)

    if not memcpy_nodes:
        print("\n沒有找到任何 Memcpy node，這個圖從頭到尾都留在同一個 execution provider 上。")
        sys.exit(0)

    trace_causal_chain(g, memcpy_nodes)
    summarize_profiling(profile_path)


if __name__ == "__main__":
    main()