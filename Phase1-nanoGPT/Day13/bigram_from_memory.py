import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 256
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2 

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # (B, T, hs)
        k = self.key(x) # (B, T, hs)
        wei = q @ k.transpose(-1, -2) * k.shape[-1] ** -0.5 # (B, T, T)
        mask = self.tril[:T:, :T:] # (B, T, T)
        wei = wei.masked_fill(mask==0, float='-inf') # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.drop(wei) # (B, T, T)
        v = self.value(x) # (B, T, C)
        output = wei @ v # (B, T, C)

        return output



class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(head_size * num_head, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        cat = []
        for head in self.heads:
            cat.append(head(x)) # (B, T, hs)
        output = torch.cat(cat, dim=-1)
        output = self.drop(self.proj(output))

        return output


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffwd(x)

class Block(nn.Module):

    def __init__(self, n_embd, num_head):
        super().__init__()
        head_size = n_embd // num_head
        self.sa = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        output = x + self.ffwd(self.ln2(x))

        return output


class Bigram(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):

        B, T = x.shape

        token_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T))
        emb = token_emb + pos_emb
        x = self.blocks(emb)
        x = self.drop(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self.forward(idx) # (B, T, C)

            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), -1)

        return idx
