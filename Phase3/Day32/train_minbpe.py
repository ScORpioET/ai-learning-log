import json
import pickle
import regex as re

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


if __name__ == '__main__':

    train_captions = []
    with open('captions_train.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            train_captions.append(row)


    tokenizer = minbpe()
    tokenizer.train(' '.join([train['caption'] for train in train_captions]), vocab_size=318)


    tokenizer.save('tokenizer.pkl')