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
        # ids = text.encode('utf-8')
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



if __name__ == '__main__':


    with open('taylorswift.txt') as f:
        texts = f.read()

    tok = minbpe()
    tok.train(texts, vocab_size=300)
    print(tok.encode("안녕하세요 👋 (hello in Korean!)"))
    print(tok.decode(tok.encode("안녕하세요 👋 (hello in Korean!)")) == "안녕하세요 👋 (hello in Korean!)")
    # match the above for your own tokenizer, and also implement a train() function