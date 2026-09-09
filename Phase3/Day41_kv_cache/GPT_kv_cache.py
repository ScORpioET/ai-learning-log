"""
KV cache 實驗專用的 GPT class 副本,獨立於 Phase3/Day36/GPT.py(正式pipeline用的
那份完全不動)。

跟原版唯一的差異在 forward() 裡 image token embedding 的覆蓋邏輯,原版:

    tok_emb = self.transformer.wte(idx)
    image_vector_emb = self.transformer.proj_layer(image_feature)
    tok_emb[:,0] = image_vector_emb[:]      # 無條件覆蓋"這次forward輸入序列"的第0個位置

在no-cache模式下,每一步都重新餵整個序列(從image token開始),所以idx[:,0]
永遠是image token,這行永遠正確。但如果直接把這個class套進KV cache的增量解碼
(每步只餵最新產生的1個token,past_key_value帶著之前的k/v),T=1的那個位置
就不再是"序列的第0個位置"而是"這一步新生成的token"——這行會錯誤地把它蓋成
image embedding,生成結果會整個跑掉。

這個問題是寫generate_with_cache()之前,讀forward()這行邏輯時先看出來的
(不是跑correctness check失敗後才回頭查),所以這份修過的檔案從一開始就沒有
這個bug,correctness_check.py沒有踩到它——但這是這個實驗裡最值得記錄的發現,
拿掉這段修正的話cache版本會直接壞掉,所以在這裡完整記錄問題成因跟修法,
不要因為"這次沒讓我踩雷"就假裝它不存在。修法:只有在past_key_value is None時
(=完整序列的prefill/no-cache這一步一定包含真正的position 0)才做這個覆蓋;
KV cache的增量解碼步(past_key_value is not None)跳過,image embedding已經
在prefill那一步的cache裡了,不需要每步重算。

這個修法對no-cache路徑完全没有行為改變(no-cache呼叫時past_key_value永遠是
None),只影響cache路徑,而且是修正一個bug而不是引入新行為。
"""
import inspect
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


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

        # 修法見檔案開頭docstring:只有在這次forward真的包含position 0(=prefill,
        # past_key_value is None)才把image embedding蓋上去;KV cache的增量解碼步
        # (past_key_value is not None)position 0早就在cache裡了,跳過。
        if past_key_value is None:
            image_vector_emb = self.transformer.proj_layer(image_feature)
            tok_emb[:, 0] = image_vector_emb[:]

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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}

        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
