import json
import hydra
import pickle
import random
from pathlib import Path
from PIL import Image
from omegaconf import DictConfig
from train_vlm import GPT, GPTConfig

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

root = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


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

    def save(self, path):
        state = {
            'merges': self.merges,
            'vocab': self.vocab
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)


    @classmethod
    def load(cls, path):

        with open(path, 'rb') as f:
            state = pickle.load(f)

        tok = cls()
        tok.merges = state['merges']
        tok.vocab = state['vocab']

        return tok



@hydra.main(version_base=None, config_path="config", config_name="config")
def test(cfg: DictConfig):

    model = GPT(GPTConfig(
        block_size=cfg.model.block_size,
        vocab_size=cfg.model.vocab_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        clip_vector_size=cfg.model.clip_vector_size,
    )).to(device)
    model.eval()

    checkpoint = torch.load(cfg.train.ckpt_path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(checkpoint['model'])
    model.eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    tokenizer = minbpe.load('tokenizer.pkl')

    file_names = []
    gt_captions = {}
    with open(cfg.data.captions_val_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            file_names.append(row["file_name"])
            gt_captions[row["file_name"]] = row["caption"]

    precomputed_raw = torch.load('clip_features_val.pt', map_location='cpu')
    precomputed = {
        name: precomputed_raw['features'][i]
        for i, name in enumerate(precomputed_raw['file_name'])
    }
    print(f"[debug] built lookup dict with {len(precomputed)} entries")
    print(f"[debug] file_names[0] example : {repr(file_names[0])}")
    print(f"[debug] precomputed key sample: {repr(list(precomputed.keys())[0])}")
    print() 



    n = 5
    samples = random.sample(range(len(file_names)), k=n)
    all_features = []

    for i in samples:
        name = file_names[i]
        img = Image.open(root / name).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt").to(device)

        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)  # (1, 512)

        if not isinstance(feat, torch.Tensor):
            vision_out = clip_model.vision_model(**inputs)
            pooled = vision_out.pooler_output
            feat = clip_model.visual_projection(pooled)

        if name in precomputed:
            old = precomputed[name].to(feat.device).view(-1)
            new = feat.view(-1)
            diff = (new - old).abs().max().item()
            cos = F.cosine_similarity(new.unsqueeze(0), old.unsqueeze(0)).item()
            print(f"[sanity] {name}")
            print(f"         max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")
        else:
            print(f"[warn] {name} not in precomputed features!")

        print(f"         GT: {gt_captions[name]}")
        print()

        all_features.append(feat.cpu())


    def generate(img_feat, max_new_tokens=40):

        image_token_id = 318
        eos_token_id = 319
        idx = torch.tensor([[image_token_id]], dtype=torch.long, device=device)
        for i in range(max_new_tokens):

            logits, loss = model(idx, None, img_feat.to(device))

            prob = F.softmax(logits[:,-1], dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=-1)

            if idx_next.item() == eos_token_id:
                break

        generated_ids = idx[0].tolist()
        clean_ids = [i for i in generated_ids if i not in (image_token_id, eos_token_id)]
        captions = tokenizer.decode([clean_ids])
        return captions


    for feat in all_features:

        print(generate(feat))




if __name__ == '__main__':
    test()