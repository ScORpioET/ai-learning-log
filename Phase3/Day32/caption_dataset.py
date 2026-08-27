"""
把 captions_{split}.jsonl(規則模板生成的英文場景描述)+ clip_features_{split}.pt
(precompute_clip_features.py 算好的 CLIP pooled 圖片特徵)組成 Dataset,
搭配 collate_fn 給 DataLoader 用,訓練你 Phase 1 手刻的 GPT-2 decoder。

【設計對齊你 Day13-19 nanoGPT 系列的慣例,🟡 假設,不是核對過的事實】
沿用 x/y 錯開一位的訓練方式:
    seq = [IMAGE_TOKEN, cap_tok_1, cap_tok_2, ..., cap_tok_N, EOS_TOKEN]
    x   = seq[:-1]   # 拿去 forward 的輸入
    y   = seq[1:]    # 對齊的下一個 token,直接當 targets
這樣完全不需要 Qwen2-VL 那種 -100 mask「問題」的技巧——因為我們沒有獨立的
「問題」段落,整條序列從 image 到 caption 到 eos 都是有意義的預測目標:
IMAGE_TOKEN 的位置預測 caption 第一個字(「看到這張圖,該說的第一個詞是什麼」),
最後一個 caption token 的位置預測 EOS(「這句話講完了,該學會停」)。
只有 padding 補出來的部分要蓋掉,不要讓 loss 被無意義的 padding 稀釋。

如果你自己手刻的 GPT.forward(idx, targets) 不是這個「外部先錯位、targets
已經對齊」的介面,而是丟整段序列進去、內部自己做錯位,這支 Dataset 就要
跟著改(x 直接用完整 seq,不用先砍掉最後一個 token),麻煩對照你自己的
forward() 簽名確認一次。
"""
import json
import torch
from torch.utils.data import Dataset

# 兩個新的特殊 token,接在 tokenizer 原本的 vocab 後面。
# base_vocab_size 由呼叫端傳進來(=你的 minbpe tokenizer 訓練時的 vocab_size),
# 不在這支檔案裡寫死,因為不同次訓練 tokenizer 的 vocab_size 可能不一樣。


class CaptionDataset(Dataset):
    def __init__(self, captions_path, features_path, tokenizer, base_vocab_size):
        """
        captions_path: captions_{split}.jsonl 的路徑
        features_path: clip_features_{split}.pt 的路徑(precompute_clip_features.py 產出)
        tokenizer:     你的 minbpe 實例,必須已經 train() 過
        base_vocab_size: tokenizer 訓練時用的 vocab_size(還沒加 image/eos 之前)
        """
        self.image_token_id = base_vocab_size
        self.eos_token_id = base_vocab_size + 1
        self.total_vocab_size = base_vocab_size + 2

        self.tokenizer = tokenizer

        # 讀 caption
        self.captions = []
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.captions.append(row)

        # 讀預先算好的 CLIP 特徵,用 file_name 對照
        cache = torch.load(features_path)
        self.feature_by_name = {
            name: cache["features"][i] for i, name in enumerate(cache["file_name"])
        }

        # 過濾掉沒有對應圖片特徵的 caption(理論上不該發生,發生了要出聲)
        missing = [c["file_name"] for c in self.captions if c["file_name"] not in self.feature_by_name]
        if missing:
            print(f"[warn] {len(missing)} captions 沒有對應的 CLIP 特徵,已略過。"
                  f"例如: {missing[:3]}")
            self.captions = [c for c in self.captions if c["file_name"] in self.feature_by_name]

    def __len__(self):
        return len(self.captions)

    def _encode_flat(self, text):
        """minbpe.encode() 回傳的是巢狀清單(每個字詞片段各自一個子序列),
        這裡攤平成一維,才能接到 GPT 的輸入序列裡。"""
        nested = self.tokenizer.encode(text)
        return [tid for chunk in nested for tid in chunk]

    def __getitem__(self, idx):
        row = self.captions[idx]
        image_feat = self.feature_by_name[row["file_name"]]  # (512,)

        cap_ids = self._encode_flat(row["caption"])
        seq = [self.image_token_id] + cap_ids + [self.eos_token_id]
        seq = torch.tensor(seq, dtype=torch.long)

        x = seq[:-1]
        y = seq[1:]

        return {
            "input_ids": x,
            "labels": y,
            "image_feature": image_feat,
            "file_name": row["file_name"],
        }


def make_collate_fn(pad_token_id, ignore_index=-100):
    """
    回傳一個 collate_fn,把一個 batch 裡長度不一的樣本 pad 成同一長度。
    pad_token_id 建議直接傳 eos_token_id 重複用——padding 位置的 loss
    會被蓋成 ignore_index,pad 本身用哪個 id 填充不影響訓練結果,
    重複使用 eos 可以少加一個特殊 token。
    """
    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids, labels, attn_mask, image_feats, file_names = [], [], [], [], []

        for b in batch:
            L = len(b["input_ids"])
            pad_len = max_len - L

            ids = torch.cat([b["input_ids"],
                              torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            lbl = torch.cat([b["labels"],
                              torch.full((pad_len,), ignore_index, dtype=torch.long)])
            mask = torch.cat([torch.ones(L, dtype=torch.long),
                               torch.zeros(pad_len, dtype=torch.long)])

            input_ids.append(ids)
            labels.append(lbl)
            attn_mask.append(mask)
            image_feats.append(b["image_feature"])
            file_names.append(b["file_name"])

        return {
            "input_ids": torch.stack(input_ids),          # (B, T)
            "labels": torch.stack(labels),                # (B, T)
            "attention_mask": torch.stack(attn_mask),      # (B, T)
            "image_feature": torch.stack(image_feats),     # (B, 512)
            "file_name": file_names,                       # list[str],debug 用
        }

    return collate_fn


if __name__ == "__main__":
    # 語法/邏輯層面的假資料自我檢查,不依賴真實檔案。
    # 真正接你的資料跟 tokenizer 之前,先確認這支檔案本身邏輯沒問題。

    class FakeTokenizer:
        """模擬 minbpe 的 encode() 回傳巢狀清單這個特性。"""
        def encode(self, text):
            words = text.split()
            return [[hash(w) % 200 for _ in range(len(w) % 3 + 1)] for w in words]

    import tempfile, os

    tmpdir = tempfile.mkdtemp()
    captions_path = os.path.join(tmpdir, "captions.jsonl")
    features_path = os.path.join(tmpdir, "features.pt")

    rows = [
        {"file_name": "a.jpg", "caption": "nearby on the right there is a car."},
        {"file_name": "b.jpg", "caption": "at medium distance ahead there are two pedestrians."},
    ]
    with open(captions_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    torch.save({"file_name": ["a.jpg", "b.jpg"], "features": torch.randn(2, 512)}, features_path)

    tok = FakeTokenizer()
    ds = CaptionDataset(captions_path, features_path, tok, base_vocab_size=300)
    print(f"[test] dataset size = {len(ds)}")
    print(f"[test] image_token_id={ds.image_token_id}, eos_token_id={ds.eos_token_id}, "
          f"total_vocab_size={ds.total_vocab_size}")

    sample = ds[0]
    print(f"[test] sample x = {sample['input_ids'].tolist()}")
    print(f"[test] sample y = {sample['labels'].tolist()}")
    assert sample["input_ids"][0].item() == ds.image_token_id, "第一個 token 應該是 IMAGE_TOKEN"
    assert sample["labels"][-1].item() == ds.eos_token_id, "最後一個 target 應該是 EOS_TOKEN"

    collate_fn = make_collate_fn(pad_token_id=ds.eos_token_id)
    batch = collate_fn([ds[0], ds[1]])
    print(f"[test] batch input_ids shape = {tuple(batch['input_ids'].shape)}")
    print(f"[test] batch labels =\n{batch['labels']}")
    print(f"[test] batch attention_mask =\n{batch['attention_mask']}")
    assert batch["input_ids"].shape[0] == 2
    assert batch["image_feature"].shape == (2, 512)

    print("[test] all assertions passed")