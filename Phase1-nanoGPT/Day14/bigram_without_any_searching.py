import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda'
batch_size = 128
block_size = 256
n_embd = 384
dropout = 0.2
n_head = 8
n_layer = 16

max_iters = 5000
eval_interval = 500
eval_iters = 200

with open('input.txt', 'r') as f:
    data = f.read()

vocab = sorted(list(set(data)))
vocab_size = len(vocab)
stoi = {value:index for index, value in enumerate(vocab)}
itos = {index:value for index, value in enumerate(vocab)}
encode = lambda x : [stoi[s] for s in x]
decode = lambda x : ''.join([itos[i] for i in x])


n = int(0.9 * len(data))
train_set = torch.tensor(encode(data[:n])).to(device)
val_set = torch.tensor(encode(data[n:])).to(device)

def get_batch(split):

    dataset =  train_set if split == 'train' else val_set
    idx = torch.randint(0, len(dataset)-block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in idx])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in idx])

    return x, y



class head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # (B, T, hs)
        k = self.key(x) # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, T)  k.shape[-1]是背下來的
        mask = self.tril[:T,:T]
        wei = wei.masked_fill(mask==0, float('-inf')) # masked_fill是網路上查正確名字的
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) #  (B, T, hs)
        out = wei @ v # (B, T, hs)
        out = self.drop(out)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, head_size, n_head):
        super().__init__()
        self.heads = nn.ModuleList([head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        cat = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(cat)
        out = self.drop(x)
        return out


class FeedForward(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd)
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):

    def __init__(self, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size, n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        out = x + self.ffwd(self.ln2(x))
        return out

class bigram(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head) for _ in range(n_layer)])
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        t_embd = self.token_embd(x) # (B, T, C)
        p_embd = self.pos_embd(torch.arange(T).to(device)) # (T, C)
        wei = t_embd + p_embd # (B, T, C)

        wei = self.blocks(wei)
        wei = self.drop(wei)
        logits = self.lm_head(wei)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for i in range(max_new_tokens):
            x = idx[:, -block_size:] # (B, T)

            logits, loss = self(x)

            probs = F.softmax(logits[:, -1], dim=-1)
            next_idx = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_idx), dim=-1)

        return idx[-1, :]


if __name__ == '__main__':

    model = bigram(vocab_size)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros((eval_iters))
            for i in range(eval_iters):
                dataset, targets = get_batch(split)
                logits, loss = model(dataset, targets)

                losses[i] = loss
            out[split] = losses.mean().item()
        model.train()

        return out



        
    for i in range(max_iters):

        dataset, targets = get_batch('train')

        logits, loss = model(dataset, targets)

        if (i != 0 and i % eval_interval == 0) or i == max_iters-1:
            losses = estimate_loss()
            print(f'step {i}: trainig_loss:{losses["train"]:.4f} val_loss:{losses["val"]:.4f}')

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    context = torch.zeros((1,1), dtype=torch.long).to(device)
    print(decode(model.generate(context, 1000).tolist()))