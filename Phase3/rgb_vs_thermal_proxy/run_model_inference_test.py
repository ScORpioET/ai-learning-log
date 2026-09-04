"""
Jack 指出 val 的 RGB<->thermal 配對(Method A)還是有些影片配不到、有不匹配
疑慮,要求改用 test split——test 是同步雙鏡頭拍攝,100% frame 對應,已經
在 Day39(caption_fusion/build_test_pairs.py)查證過,不是部分配對。

這支腳本把 run_model_inference_val.py 的分批生成邏輯(run_domain_chunked,
直接 import 重用,不重寫)套用在 test split 全部 3749 組配對上
(Phase3/bright_region_hypothesis/full_test_pairs.json,上一個任務已經
建好、也已經跑過 YOLO 偵測,這裡直接重用那兩份 detections_*.jsonl,不用
重新掃)。

checkpoint 跟 val 那次同一組:
    thermal = best_model_full_capfix_reweight2x.pt
    rgb     = best_model_rgb_full_capfix_reweight2x.pt
GT 讀現成的 captions_test.jsonl(thermal,3493 筆——沒有動態物件的畫面
generate_captions.py 本來就會跳過,不是漏檔)/ captions_rgb_test.jsonl
(rgb,3749 筆全)。
"""
import json
import sys
from pathlib import Path

import torch

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
sys.path.insert(0, str(DAY32))
from train_vlm import GPT, GPTConfig, device  # noqa: E402,F401  同樣的 __main__ pickle 坑,見 run_model_inference_val.py 註解

sys.path.insert(0, str(Path(__file__).parent))
from run_model_inference_val import run_domain_chunked  # noqa: E402

TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
BRIGHT_DIR = Path.home() / "ai-transition-2026" / "Phase3" / "bright_region_hypothesis"
HERE = Path(__file__).parent

THERMAL_CKPT = DAY32 / "checkpoints" / "best_model_full_capfix_reweight2x.pt"
RGB_CKPT = DAY32 / "checkpoints" / "best_model_rgb_full_capfix_reweight2x.pt"


def main():
    pairs = json.load(open(BRIGHT_DIR / "full_test_pairs.json"))
    thermal_files = [p["thermal_file"] for p in pairs]
    rgb_files = [p["rgb_file"] for p in pairs]
    print(f"[info] {len(pairs)} 組樣本(test split,100% frame 對應), "
          f"thermal ckpt={THERMAL_CKPT.name}, rgb ckpt={RGB_CKPT.name}")

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    thermal_gen = run_domain_chunked(
        "thermal", TD / "video_thermal_test", thermal_files,
        DAY32 / "captions_train_full_capfix.jsonl", THERMAL_CKPT,
    )
    rgb_gen = run_domain_chunked(
        "rgb", TD / "video_rgb_test", rgb_files,
        DAY32 / "captions_rgb_train_full_capfix.jsonl", RGB_CKPT,
    )

    thermal_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
                  for l in open(TD / "captions_test.jsonl", encoding="utf-8")}
    rgb_gt = {json.loads(l)["file_name"]: json.loads(l)["caption"]
              for l in open(Path.home() / "ai-transition-2026" / "Phase3" / "rgb_validation" / "captions_rgb_test.jsonl", encoding="utf-8")}

    out = []
    for p in pairs:
        tf, rf = p["thermal_file"], p["rgb_file"]
        out.append({
            **p,
            "thermal_gt": thermal_gt.get(tf, ""), "rgb_gt": rgb_gt.get(rf, ""),
            "thermal_gen": thermal_gen[tf]["gen_caption"], "rgb_gen": rgb_gen[rf]["gen_caption"],
        })

    with open(HERE / "test_inference_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] test_inference_results.json written, {len(out)} records")


if __name__ == "__main__":
    main()
