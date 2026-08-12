import time
import numpy as np
import onnxruntime as ort
import tiktoken

enc = tiktoken.get_encoding('gpt2')
sess = ort.InferenceSession("gpt2_step.onnx")


max_runs = 3
max_iters = 250

for run in range(max_runs):

    prompt = "Hello, I am a language model,"
    tokens = enc.encode(prompt)
    input_ids = np.array([tokens], dtype=np.int64)  # shape (1, T)
    timings = []

    for step in range(max_iters):
        t0 = time.perf_counter()
        ort_outs = sess.run(None, {'input_ids': input_ids})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000  # ms

        timings.append(dt)

        logits = ort_outs[0]  # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :]  # 只取最後一個位置的機率分布
        next_token = np.argmax(next_token_logits)  # 先用 greedy,避免 sampling 的隨機性干擾量測

        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

        print(f"step {step:2d} | seq_len={input_ids.shape[1]:3d} | {dt:7.2f} ms")

    print(f"\n runs {run} | 第一步耗時: {timings[0]:.2f} ms")
    print(f"runs {run} | 最後一步耗時: {timings[-1]:.2f} ms")
    print(f"runs {run} | 變慢倍率: {timings[-1]/timings[0]:.2f}x")