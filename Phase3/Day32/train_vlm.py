import os
import time
import json
import math
import wandb
import torch
import inspect
import tiktoken
import regex as re
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))

    def forward(self, x, past_key_value=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v)

        if past_key_value is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=(self.n_embd // self.n_head) ** -0.5)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=(self.n_embd // self.n_head) ** -0.5)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, present_key_value


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

    def forward(self, x, past_key_value=None):
        attn_out, present_key_value = self.attn(self.ln_1(x), past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    clip_vector_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            proj_layer = nn.Linear(config.clip_vector_size, config.n_embd, bias=False),
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

    def forward(self, idx, targets=None, image_feature=None, past_key_value=None, use_cache=False, past_length=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        if past_key_value is not None:
            pos = past_length + torch.arange(0, T, dtype=torch.long, device=idx.device)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        image_vector_emb = self.transformer.proj_layer(image_feature)

        tok_emb[:,0] = image_vector_emb[:]

        x = tok_emb + pos_emb
        if past_key_value is None:
            past_key_value = [None]*self.config.n_layer
        for i, block in enumerate(self.transformer.h):
            x, present_kev_value = block(x, past_key_value[i])
            past_key_value[i] = present_kev_value

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, past_key_value
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
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

class minbpe():

    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def get_stats(self, ids_split):
        counts = {}
        for ids in ids_split:
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids_split, pair, idx):
        new_ids_split = []
        for ids in ids_split:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(idx)
                    i+=2
                else:
                    new_ids.append(ids[i])
                    i+=1
            new_ids_split.append(new_ids)
        return new_ids_split

    def train(self, text, vocab_size):
        ids_split = re.findall(self.GPT4_SPLIT_PATTERN, text)
        ids_split = [ids.encode('utf-8') for ids in ids_split]
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids_split)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids_split = self.merge(ids_split, pair, idx)
            self.merges[pair] = idx

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens = [t.encode('utf-8') for t in tokens]

        while len(tokens):
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        return tokens

    def decode(self, ids_split):
        text = ''
        for ids in ids_split:
            tokens = b''.join(self.vocab[idx] for idx in ids)
            text += tokens.decode('utf-8', errors='replace')
        return text

class CaptionDataset(Dataset):
    def __init__(self, captions, features_path, tokenizer, base_vocab_size):
        self.image_token_id = base_vocab_size
        self.eos_token_id = base_vocab_size + 1
        self.total_vocab_size = base_vocab_size + 2

        self.captions = captions
        self.tokenizer = tokenizer

        cache = torch.load(features_path)
        self.feature_by_name = {
            name: cache['features'][i] for i, name in enumerate(cache['file_name'])
        }

        missing = [c['file_name'] for c in captions if c['file_name'] not in self.feature_by_name]
        if missing:
            self.captions = [c for c in captions if c['file_name'] in self.feature_by_name]

    def __len__(self):
        return len(self.captions)

    def _encode_flat(self, text):
        nested = self.tokenizer.encode(text)
        return [tid for chunk in nested for tid in chunk]

    def __getitem__(self, idx):
        row = self.captions[idx]
        image_feat = self.feature_by_name[row["file_name"]]

        cap_ids = self._encode_flat(row['caption'])
        seq = [self.image_token_id] + cap_ids + [self.eos_token_id]
        seq = torch.tensor(seq, dtype=torch.long)

        x = seq[:-1]
        y = seq[1:]

        return{
            'input_ids': x,
            'labels': y,
            'image_features': image_feat,
            'file_name': row['file_name']
        }

def collate_fn(batch):
    max_len = max(len(b['input_ids']) for b in batch)
    pad_token_id = 0
    ignore_index = -100

    input_ids, labels, image_feats, file_names = [], [], [], []

    for b in batch:
        L = len(b["input_ids"])
        pad_len = max_len - L

        ids = torch.cat([b["input_ids"],
                            torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        lbl = torch.cat([b["labels"],
                            torch.full((pad_len,), ignore_index, dtype=torch.long)])

        input_ids.append(ids)
        labels.append(lbl)
        image_feats.append(b["image_features"])
        file_names.append(b["file_name"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "image_features": torch.stack(image_feats),
        "file_name": file_names,
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):

    wandb.init(
        project="gpt2-nanogpt",
        config=dict(cfg.train) | dict(cfg.model) | dict(cfg.data),
    )

    seed = cfg.train.seed
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    B = cfg.data.B

    train_captions = []
    with open(cfg.data.captions_train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            train_captions.append(row)

    val_captions = []
    with open(cfg.data.captions_val_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            val_captions.append(row)

    tokenizer = minbpe()
    tokenizer.train(' '.join([train['caption'] for train in train_captions]), vocab_size=cfg.data.base_vocab_size)

    train_dataset = CaptionDataset(train_captions, cfg.data.features_train_path, tokenizer, cfg.data.base_vocab_size)
    val_dataset = CaptionDataset(val_captions, cfg.data.features_val_path, tokenizer, cfg.data.base_vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)

    model = GPT(GPTConfig(block_size=cfg.model.block_size, vocab_size=cfg.model.vocab_size,
                          n_layer=cfg.model.n_layer, n_head=cfg.model.n_head,
                          n_embd=cfg.model.n_embd, clip_vector_size=cfg.model.clip_vector_size))
    model.to(device)
    model = torch.compile(model)

    start_step = 0
    ckpt_path = cfg.train.ckpt_path
    ckpt_dir = cfg.train.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)      # <-- 補上,不然 torch.save 會因為資料夾不存在而報錯

    optimizer = model.configure_optimizers(weight_decay=cfg.train.weight_decay,
                                           learning_rate=cfg.train.max_lr, device=device)
    log_dir = cfg.train.log_dir
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

    max_lr = cfg.train.max_lr
    min_lr = max_lr * cfg.train.min_lr_ratio
    warmup_steps = cfg.train.warmup_steps
    steps_per_epoch = len(train_loader)
    num_epochs = cfg.train.epoch
    max_steps = start_step + steps_per_epoch * num_epochs

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        elif it > max_steps:
            return min_lr

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    step = start_step
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        t0 = time.time()
        loss_sum = 0.0
        norm_max = 0.0
        for batch in train_loader:

            optimizer.zero_grad(None)
            x, y, img_feat = batch['input_ids'], batch['labels'], batch['image_features']
            x, y, img_feat = x.to(device), y.to(device), img_feat.to(device)

            logits, loss = model(x, y, img_feat)

            loss.backward()

            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            norm_max = max(norm_max, torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            step += 1

            loss_sum += loss.detach().item()

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for batch in val_loader:
                x, y, img_feat = batch['input_ids'], batch['labels'], batch['image_features']
                x, y, img_feat = x.to(device), y.to(device), img_feat.to(device)
                logits, loss = model(x, y, img_feat)
                val_loss_sum += loss.detach().item()
        val_loss_avg = val_loss_sum / len(val_loader)
        print(f'validation loss: {val_loss_avg:.4f}')
        wandb.log({"val_loss": val_loss_avg, "epoch": epoch})
        model.train()

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            checkpoint = {
                    'model': raw_model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'config': raw_model.config,
                }
            torch.save(checkpoint, f'{ckpt_dir}/best_model.pt')

        t1 = time.time()
        dt = (t1 - t0)
        print(f'epoch {epoch}, loss: {loss_sum/len(train_loader):.6f}, norm_max:{norm_max:.4f}, dt: {dt}s')

    # ---- 訓練結束,載入 best checkpoint,實際生成幾句話看看 ----
    ckpt = torch.load(f'{ckpt_dir}/best_model.pt', map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(ckpt['model'])
    print(f"loaded best_model.pt, epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    image_token_id = train_dataset.image_token_id
    eos_token_id = train_dataset.eos_token_id

    @torch.no_grad()
    def generate_caption(image_feature, max_new_tokens=40):
        model.eval()
        idx = torch.tensor([[image_token_id]], dtype=torch.long, device=device)
        img_feat = image_feature.unsqueeze(0).to(device)  # (1, 512)

        for _ in range(max_new_tokens):
            logits, _ = model(idx, targets=None, image_feature=img_feat)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_token_id:
                break

        generated_ids = idx[0].tolist()
        clean_ids = [i for i in generated_ids if i not in (image_token_id, eos_token_id)]
        caption = tokenizer.decode([clean_ids])
        model.train()
        return caption

    num_samples = 5
    for i in range(num_samples):
        row = val_captions[i]
        img_feat = val_dataset.feature_by_name[row['file_name']]
        caption = generate_caption(img_feat)
        print(f"[{row['file_name']}]")
        print(f"  ground truth: {row['caption']}")
        print(f"  generated   : {caption}")
        print()


if __name__ == '__main__':
    train()