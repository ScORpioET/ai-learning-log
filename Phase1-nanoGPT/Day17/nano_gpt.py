import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
n_embd = 32
n_head = 4
n_layer = 8
block_size = 256
batch_size = 32
dropout = 0.2
max_iter = 500

with open('input.txt', 'r') as f:
    data = f.read()

vocab = sorted(list(set(data)))
vocab_size = len(vocab)
stoi = {value:index for index, value in enumerate(vocab)}
itos = {index:value for index, value in enumerate(vocab)}
encode = lambda x: [stoi[s] for s in x]
decode = lambda x: ''.join(itos[i] for i in x)

n = int(0.9 * len(data))
train_data = torch.tensor(encode(data[:n]), dtype=torch.long, device=device)
val_data = torch.tensor(encode(data[n:]), dtype=torch.long, device=device)

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.queue = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size), 0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.queue(x) # (B, T, hs)
        k = self.key(x) # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, T)
        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T,  hs)

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)]) # 為甚麼這裡不用nn.Sequential()?
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(x)

        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        out = self.drop(x)
        return out

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tkn_embd = self.token_embedding(x) # (B, T, C)
        pos_embd = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        embd = tkn_embd + pos_embd
        wei = self.blocks(embd)
        wei = self.ln(wei)
        logits = self.lm_head(wei) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for i in range(max_new_tokens):

            logits, loss = self(idx[:,-block_size:])

            prob = F.softmax(logits[:,-1], dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)

            idx = torch.cat((idx, idx_next), dim=-1)

        return idx


def get_batch(split):

    if split == 'train':
        data = train_data
    else:
        data = val_data

    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])

    return x, y


if __name__ == '__main__':

    model = Bigram()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(max_iter):

        dataset, targets = get_batch('train')

        logits, loss = model(dataset, targets)

        opt.zero_grad(None)

        loss.backward()

        opt.step()

        print(f'step {i}： {loss:.4f}')
        
