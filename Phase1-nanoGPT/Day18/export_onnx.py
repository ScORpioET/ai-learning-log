import torch
import torch.nn as nn
import tiktoken
from train_gpt2 import GPT, GPTConfig  # 直接 import 你原本的 class

# ---- 1. 載入 checkpoint ----
ckpt_path = "checkpoints/model_00400.pt"  # 換成你實際的檔名
checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

config = checkpoint['config']   # 直接拿訓練時存的真實 config，不要自己手動寫
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

# ---- 2. wrapper：只留 logits，甩掉 loss=None ----
class GPTForExport(nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        self.n_layer = gpt_model.config.n_layer

    def forward(self, idx, past_length, *past_kv_flat):
        past_key_value = [(past_kv_flat[2*i], past_kv_flat[2*i+1]) for i in range(self.n_layer)]
        logits, _, present_key_value = self.gpt(idx, past_key_value=past_key_value, use_cache=True, past_length=past_length)

        present_flat = []
        for k, v in present_key_value:
            present_flat.append(k)
            present_flat.append(v)

        return (logits, *present_flat)

wrapper = GPTForExport(model)
wrapper.eval()

B = 1
n_layer = config.n_layer
n_head = config.n_head
head_size = config.n_embd // config.n_head


# ---- 3. dummy input，決定 export 當下的固定 shape ----
idx_dummy = torch.randint(0, config.vocab_size, (B, 1), dtype=torch.long)
past_length_dummy = torch.tensor(0, dtype=torch.long)

past_kv_dummy = []
for i in range(n_layer):
    past_kv_dummy.append(torch.zeros(B, n_head, 0, head_size))
    past_kv_dummy.append(torch.zeros(B, n_head, 0, head_size))

dummy_input = (idx_dummy, past_length_dummy, *past_kv_dummy)

input_names = ['input_ids', 'past_length']
output_names = ['logits']
dynamic_axes = {
    'input_ids': {0: 'batch', 1: 'seq_len'},
    'logits': {0: 'batch', 1: 'seq_len'},
}

for i in range(n_layer):
    input_names += [f'past_key_{i}', f'past_value_{i}']
    output_names += [f'present_key_{i}', f'present_value_{i}']
    dynamic_axes[f'past_key_{i}'] = {0: 'batch', 2: 'past_len'}
    dynamic_axes[f'past_value_{i}'] = {0: 'batch', 2: 'past_len'}
    dynamic_axes[f'present_key_{i}'] = {0: 'batch', 2: 'present_len'}
    dynamic_axes[f'present_value_{i}'] = {0: 'batch', 2: 'present_len'}

# ---- 4. 真正 export ----
torch.onnx.export(
    wrapper,
    dummy_input,
    "gpt2_step_cache.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=18,
    dynamo=False
)
print("export 完成")

import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("gpt2_step.onnx")
print("providers:", sess.get_providers())

enc = tiktoken.get_encoding('gpt2')
prompt_text = "Hello, I am"
prompt_tokens = enc.encode(prompt_text)
num_new_tokens = 10

# ---- ONNX + cache 版本：prompt 一個一個餵，之後生成也一個一個餵 ----
past_kv = []
for i in range(n_layer):
    past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))
    past_kv.append(np.zeros((B, n_head, 0, head_size), dtype=np.float32))

past_length = 0
onnx_generated = list(prompt_tokens)

def run_one_step(token_id, past_kv, past_length):
    ort_inputs = {
        'input_ids': np.array([[token_id]], dtype=np.int64),
        'past_length': np.array(past_length, dtype=np.int64),
    }
    for i in range(n_layer):
        ort_inputs[f'past_key_{i}'] = past_kv[2*i]
        ort_inputs[f'past_value_{i}'] = past_kv[2*i+1]
    outputs = sess.run(None, ort_inputs)
    logits = outputs[0]
    new_past_kv = outputs[1:]
    return logits, list(new_past_kv)

for tok in prompt_tokens:
    logits, past_kv = run_one_step(tok, past_kv, past_length)
    past_length += 1

next_token = int(logits[0, -1, :].argmax())
onnx_generated.append(next_token)

for _ in range(num_new_tokens - 1):
    logits, past_kv = run_one_step(next_token, past_kv, past_length)
    past_length += 1
    next_token = int(logits[0, -1, :].argmax())
    onnx_generated.append(next_token)


# ---- 拿同一個 model，純 PyTorch 無 cache 版本，生成同樣長度，比對 ----
seq = torch.tensor([prompt_tokens], dtype=torch.long)
with torch.no_grad():
    for _ in range(num_new_tokens):
        logits_ref, _ = model(seq)
        next_tok = logits_ref[0, -1, :].argmax().item()
        seq = torch.cat([seq, torch.tensor([[next_tok]])], dim=1)

pytorch_generated = seq[0].tolist()
print("ONNX(帶 cache)生成:", enc.decode(onnx_generated))
print("PyTorch(無 cache)生成:", enc.decode(pytorch_generated))
print("ONNX == PyTorch:", onnx_generated == pytorch_generated)