from dataclasses import dataclass
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import os

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) 
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('inf'))
        # att = F.softmax(att, dim=1)
        # y = att @ v 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size,  bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):

        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}

        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
        print(f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = torch.tensor(enc.encode(text))

        n = int(0.9 * len(tokens))
        self.tokens = tokens[:n] if split == 'train' else tokens[n:]

        print(f'loaded {len(self.tokens)} tokens for {split}')
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


num_return_sequences = 5
max_length = 30

torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 #2**19 ~0.5M
B = 8
T = 1024
assert total_batch_size % (B * T) == 0, 'make sure total_batch_size is divisible by B * T'
grad_accum_steps = total_batch_size // (B * T)
print(f'total desired batch size: {total_batch_size}')
print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

train_loader = DataLoaderLite(B, T, 'train')
val_loader = DataLoaderLite(B, T, 'val')

torch.set_float32_matmul_precision('high')

enc = tiktoken.get_encoding("gpt2")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)

start_step = 0
ckpt_path = 'checkpoints/model_00050.pt'
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step'] + 1
    print(f'resumed from step {start_step}')
else:
    with open(log_file, "w") as f:
        pass

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = start_step + 1000
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    elif it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr) 




for step in range(start_step, max_steps):
    t0 = time.time()

    if step > 0 and (step+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f'validation loss: {val_loss_accum:.4f}')
        with open(log_file, "a") as f:
            f.write(f'{step} val {val_loss_accum.item():.4f}\n')
        model.train()

        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'val_loss': val_loss_accum.item(),
            'config': raw_model.config,   # 存架構設定,之後重建模型要用
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/model_{step+1:05d}.pt')

    optimizer.zero_grad(None)
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    t1 = time.time()
    dt = (t1 - t0)*1000
    token_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    print(f'step {step}, loss: {loss_accum.item():.6f}, norm:{norm:.4f}, dt: {dt}ms, tok/sec: {token_per_sec}')
    with open(log_file, "a") as f:   # "a" 是 append,接著寫、不會覆蓋前面的
        f.write(f'{step} train {loss_accum.item():.6f}\n')


num_return_sequences = 4
max_length = 64
tokens = enc.encode('We are')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

while xgen.size(1) < max_length:
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xgen) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        xgen = torch.cat((xgen, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f'sample {i}: {decoded}')