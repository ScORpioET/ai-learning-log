"""
train_vlm.py 訓練時從不把 tokenizer 存成 tokenizer.pkl(只在記憶體訓練後直接拿去用),
但它的 minbpe.train() 是完全確定性的(dict 用插入順序、max()同分取第一個遇到的),
只要輸入文字順序跟訓練時一致,重跑一次 train() 就能重建出跟訓練時逐bit相同的 tokenizer。

train_vlm.py 第391-397行的載入邏輯:
    train_captions = [json.loads(line) for line in open(cfg.data.captions_train_path)]
    tokenizer = minbpe()
    tokenizer.train(' '.join([train['caption'] for train in train_captions]), vocab_size=cfg.data.base_vocab_size)

對應 best_model_full_capfix_reweight2x.pt 訓練時的 config 是
config/data/full_capfix.yaml: captions_train_path=captions_train_full_capfix.jsonl, base_vocab_size=318

這裡完全重放這段邏輯,重建出配對 best_model_full_capfix_reweight2x.pt 的 tokenizer。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Day36"))
from minbpe import minbpe  # noqa: E402

HERE = Path(__file__).parent
CAPTIONS_PATH = HERE / "captions_train_full_capfix.jsonl"
OUT_PATH = HERE / "tokenizer_full_capfix.pkl"
VOCAB_SIZE = 318


def main():
    train_captions = []
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            train_captions.append(json.loads(line))
    print(f"[info] 讀入 {len(train_captions)} 筆訓練caption(應為10241筆)")

    tokenizer = minbpe()
    tokenizer.train(' '.join([c['caption'] for c in train_captions]), vocab_size=VOCAB_SIZE)
    tokenizer.save(str(OUT_PATH))
    print(f"[done] 重建的tokenizer存到 {OUT_PATH}")

    # round-trip sanity check:用這個tokenizer對幾筆訓練caption原文編碼再解碼,
    # 應該要能完全還原原文(如果train()真的重現了訓練時的merges,這裡一定會過)
    print("\n=== round-trip sanity check(用訓練集本身的caption測試編碼/解碼是否還原) ===")
    for row in train_captions[:3]:
        original = row['caption']
        encoded = tokenizer.encode(original)
        flat = [tid for chunk in encoded for tid in chunk]
        decoded = tokenizer.decode([flat])
        match = "OK" if decoded == original else "MISMATCH"
        print(f"[{match}] original={original!r}")
        if match == "MISMATCH":
            print(f"         decoded ={decoded!r}")


if __name__ == "__main__":
    main()
