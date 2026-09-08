"""
跟 reconstruct_capfix_tokenizer.py 同一套邏輯,重建配對 best_model_rgb_full_reweight2x.pt
的tokenizer。對應 config/data/rgb_full.yaml: captions_train_path=captions_rgb_train_full.jsonl,
base_vocab_size=318。這份captions檔案mtime(9/2 16:59)剛好在checkpoint mtime(9/2 17:06)之前,
時序吻合。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Day36"))
from minbpe import minbpe  # noqa: E402

HERE = Path(__file__).parent
CAPTIONS_PATH = HERE / "captions_rgb_train_full.jsonl"
OUT_PATH = HERE / "tokenizer_rgb_full.pkl"
VOCAB_SIZE = 318


def main():
    train_captions = []
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            train_captions.append(json.loads(line))
    print(f"[info] 讀入 {len(train_captions)} 筆訓練caption(應為9656筆)")

    tokenizer = minbpe()
    tokenizer.train(' '.join([c['caption'] for c in train_captions]), vocab_size=VOCAB_SIZE)
    tokenizer.save(str(OUT_PATH))
    print(f"[done] 重建的tokenizer存到 {OUT_PATH}")

    print("\n=== round-trip sanity check ===")
    for row in train_captions[:3]:
        original = row['caption']
        encoded = tokenizer.encode(original)
        flat = [tid for chunk in encoded for tid in chunk]
        decoded = tokenizer.decode([flat])
        match = "OK" if decoded == original else "MISMATCH"
        print(f"[{match}] original={original!r}")


if __name__ == "__main__":
    main()
