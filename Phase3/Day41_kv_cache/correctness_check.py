"""
正確性驗證:同一checkpoint、同一批圖片、同一起始prompt(image token),
比較 generate_no_cache() 跟 generate_with_cache() 產生的token序列是否逐token一致。

主要比較用greedy decoding(argmax)——這樣不必擔心「兩邊呼叫random的時機/次數
是否對得上」這個額外變數,乾淨地只驗證cache邏輯本身對不對。額外跑一次
sampling(multinomial)、每張圖生成前都重設同一個seed,當作次要診斷:如果
greedy一致但sampling不一致,才需要去查是不是random呼叫的順序/次數兩邊對不上,
而不是cache邏輯本身有問題。

結果存成json,不管一致或不一致都如實記錄。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402

from common import (  # noqa: E402
    IMAGE_ROOT, TEST_IMAGES, load_gpt_model, load_clip, get_image_feature,
    generate_no_cache, generate_with_cache,
)

MAX_NEW_TOKENS = 60
OUT_PATH = Path(__file__).parent / "correctness_results.json"


def run_one(gpt_model, img_feat, mode, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    no_cache_seq = generate_no_cache(gpt_model, img_feat, max_new_tokens=MAX_NEW_TOKENS, mode=mode)
    if seed is not None:
        torch.manual_seed(seed)
    cache_seq = generate_with_cache(gpt_model, img_feat, max_new_tokens=MAX_NEW_TOKENS, mode=mode)
    return no_cache_seq, cache_seq


def main():
    device = "cpu"
    print("[info] loading checkpoint + CLIP on CPU for correctness check (determinism > speed here)")
    gpt_model = load_gpt_model(device)
    clip_model, processor = load_clip(device)

    results = []
    all_greedy_match = True

    for fn in TEST_IMAGES:
        img_path = IMAGE_ROOT / fn
        img_feat = get_image_feature(clip_model, processor, img_path, device)

        no_cache_greedy, cache_greedy = run_one(gpt_model, img_feat, mode="greedy")
        greedy_match = no_cache_greedy == cache_greedy
        all_greedy_match &= greedy_match

        no_cache_sample, cache_sample = run_one(gpt_model, img_feat, mode="sample", seed=1337)
        sample_match = no_cache_sample == cache_sample

        entry = {
            "image": fn,
            "greedy": {
                "match": greedy_match,
                "no_cache_tokens": no_cache_greedy,
                "cache_tokens": cache_greedy,
                "first_diverge_at": _first_diverge(no_cache_greedy, cache_greedy),
            },
            "sample_seed1337": {
                "match": sample_match,
                "no_cache_tokens": no_cache_sample,
                "cache_tokens": cache_sample,
                "first_diverge_at": _first_diverge(no_cache_sample, cache_sample),
            },
        }
        results.append(entry)

        print(f"\nimage: {fn}")
        print(f"  greedy match: {greedy_match}")
        if not greedy_match:
            print(f"    no_cache: {no_cache_greedy}")
            print(f"    cache   : {cache_greedy}")
        print(f"  sample(seed=1337) match: {sample_match}")
        if not sample_match:
            print(f"    no_cache: {no_cache_sample}")
            print(f"    cache   : {cache_sample}")

    OUT_PATH.write_text(json.dumps({
        "checkpoint": "best_model_full_capfix_reweight2x.pt",
        "max_new_tokens": MAX_NEW_TOKENS,
        "all_greedy_match": all_greedy_match,
        "results": results,
    }, indent=2))
    print(f"\n[done] all_greedy_match = {all_greedy_match}")
    print(f"[done] results written to {OUT_PATH}")

    if not all_greedy_match:
        print("[FAIL] greedy decoding diverged between cache and no-cache -- cache實作有bug,"
              "不要往下跑效能測試,先回頭查GPT_kv_cache.py。")
        sys.exit(1)


def _first_diverge(seq_a, seq_b):
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return i
    if len(seq_a) != len(seq_b):
        return min(len(seq_a), len(seq_b))
    return None


if __name__ == "__main__":
    main()
