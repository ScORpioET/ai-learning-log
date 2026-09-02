"""
evaluate_val.py — CLIP+GPT-2 VLM 在 val set (1097 筆) 上的量化評估

取代之前只肉眼看 5 筆生成結果的印象判斷。流程:
  1. 用 captions_train.jsonl 在本 process 裡重新訓練 minbpe tokenizer
     (minbpe 沒有 save/load,必須跟訓練時用同一份語料、同一個 vocab_size 重跑一次
     merge 過程,才能保證 token id 對得上 best_model.pt 的 embedding)
  2. 載入 best_model.pt(val loss 最低的那個 epoch,不是最後一輪)
  3. 對 val set 全部 1097 筆做 batch 生成(每個 step 對整個 batch forward 一次,
     用「已結束的樣本強制接 EOS」的方式處理不同樣本 EOS 時機不同步的問題)
  4. 算五類指標、輸出 CSV + stdout summary

雷點對齊 Day33 踩過的坑:
  - image_feature 一定要傳進 model.forward()
  - batch 的 key 是複數 image_features
  - eval 全程 model.eval() + torch.no_grad()
"""
import re
import csv
import json
import argparse
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from train_vlm import GPT, GPTConfig, minbpe, device  # noqa: F401
# GPTConfig 一定要 import 進來(即使沒有直接用到):checkpoint 裡 pickle 的是
# `__main__.GPTConfig`(train_vlm.py 當初以腳本方式執行,模組名稱是 __main__),
# 這裡把它引進本檔案的全域命名空間,torch.load(weights_only=False) 反序列化時
# 才能在「現在的 __main__」(也就是 evaluate_val.py 自己)裡找到這個 class。

# ---------------------------------------------------------------------------
# 路徑 / 超參數,對齊 train_vlm.py 與 config/data/shakespeare.yaml
# ---------------------------------------------------------------------------
CAPTIONS_TRAIN_PATH = "captions_train.jsonl"
CAPTIONS_VAL_PATH = "captions_val.jsonl"
FEATURES_VAL_PATH = "clip_features_rgb_val.pt"
CKPT_PATH = "checkpoints/best_model.pt"
OUT_CSV = "eval_val_results.csv"
# val split 原始 coco.json,拿裡面 image-level extra_info.hours 當「有沒有明確標日夜」
# 的判斷依據(對齊 generate_captions.py 的 scene_prefix() 讀法)
VAL_COCO_PATH = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_rgb_val" / "coco.json"

BASE_VOCAB_SIZE = 318
IMAGE_TOKEN_ID = BASE_VOCAB_SIZE        # 318
EOS_TOKEN_ID = BASE_VOCAB_SIZE + 1      # 319
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 2  # 320

MAX_NEW_TOKENS = 40   # 跟 train_vlm.py 裡 generate_caption 的預設一致
GEN_BATCH_SIZE = 128
SEED = 42

# ---------------------------------------------------------------------------
# 1. 物件類別關鍵字表
#    對齊 thermal_dataset/generate_captions.py v0.6 的 DYNAMIC_CLASSES /
#    plural 規則(IRREGULAR_PLURALS 只有 bus->buses,其餘 +s),
#    long-tail fallback 一律用 "object"。
# ---------------------------------------------------------------------------
EN_NAMES = ["pedestrian", "bicycle", "motorcycle", "car", "bus", "truck",
            "vehicle", "train", "skateboard", "stroller", "scooter", "object"]
IRREGULAR_PLURALS = {"bus": "buses"}


def plural_of(name):
    return IRREGULAR_PLURALS.get(name, name + "s")


CLASS_WORD_TO_CANON = {}
for _name in EN_NAMES:
    CLASS_WORD_TO_CANON[_name] = _name
    CLASS_WORD_TO_CANON[plural_of(_name)] = _name

CLASS_WORD_RES = [
    (re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE), canon)
    for word, canon in CLASS_WORD_TO_CANON.items()
]


def extract_classes(text):
    found = set()
    for pat, canon in CLASS_WORD_RES:
        if pat.search(text):
            found.add(canon)
    return found


# ---------------------------------------------------------------------------
# 2. 句型模板(對齊 generate_captions.py v0.6 的 scene_prefix() / build_caption())
# ---------------------------------------------------------------------------
DIST_PHRASES = ["nearby", "at medium distance", "in the distance"]
POS_PHRASES = ["on the left", "on the right", "ahead"]
WEATHER_WORDS = ["Cloudy", "Overcast", "Rainy", "Foggy", "Snowy"]

DIST_RE = "(?:" + "|".join(re.escape(p) for p in DIST_PHRASES) + ")"
POS_RE = "(?:" + "|".join(re.escape(p) for p in POS_PHRASES) + ")"


def _make_subject_res():
    singular_alts = [f"(?:a|an) {re.escape(n)}" for n in EN_NAMES]
    plural_alts = [re.escape(plural_of(n)) for n in EN_NAMES]
    return "(?:" + "|".join(singular_alts) + ")", "(?:" + "|".join(plural_alts) + ")"


SINGULAR_SUBJECT_RE, PLURAL_SUBJECT_RE = _make_subject_res()

CLAUSE_RE = (
    rf"(?:{DIST_RE} {POS_RE} there is {SINGULAR_SUBJECT_RE}"
    rf"|{DIST_RE} {POS_RE} there are two {PLURAL_SUBJECT_RE}"
    rf"|{DIST_RE} {POS_RE} there are several {PLURAL_SUBJECT_RE})"
)

PREFIX_ALTS = "|".join(WEATHER_WORDS)
PREFIX_RE = rf"(?:Night(?:, (?:{PREFIX_ALTS}))?|{PREFIX_ALTS}): "

TEMPLATE_RE = re.compile(
    rf"^(?:{PREFIX_RE})?{CLAUSE_RE}(?:; {CLAUSE_RE})?\.$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Day36:generate_captions.py v0.7+(class-first 重寫,2026-08-29 起)用的是完全
# 不同的句型——不再是「there is X」,改成 count_to_subject() + build_class_clause()
# 那套:「two cars, one nearby ahead」/「a car ahead」這種。
#
# 上面的 TEMPLATE_RE 是照 v0.6 的「there is X」句型寫的,舊版 captions_train.jsonl/
# captions_val.jsonl(best_model.pt 訓練用的那份,Aug 25 產出)是 v0.6 格式,
# 這個 regex 對它們是對的。但 Day35 的 captions_{train,val}_filtered.jsonl 是用
# 現在的 v0.9 腳本(基於 v0.7 class-first 邏輯)生的,句型完全不一樣,舊 regex
# 打上去必定 0% match——不是模型退化,是這個 regex 本來就沒涵蓋新句型。
# 新增 TEMPLATE_RE_V7,is_template_ok() 兩個 regex 都試,哪個對就算過。
# ---------------------------------------------------------------------------
COUNT_WORD_RE = "(?:two|three|several|many)"


def _make_subject_res_v7():
    singular_alts = [f"(?:a|an) {re.escape(n)}" for n in EN_NAMES]
    plural_alts = [re.escape(plural_of(n)) for n in EN_NAMES]
    singular = "(?:" + "|".join(singular_alts) + ")"
    plural = f"{COUNT_WORD_RE} (?:" + "|".join(plural_alts) + ")"
    return singular, plural


SINGULAR_SUBJECT_RE_V7, PLURAL_SUBJECT_RE_V7 = _make_subject_res_v7()

# distance_phrase_for() 在 v0.7 Level1 修正後,mid 距離不輸出任何字詞(直接省略),
# 只有 near("nearby")/far("in the distance") 兩個極端會出現在句子裡,"at medium
# distance" 這個字面不會再出現(這點跟 v0.6 的 TEMPLATE_RE 不一樣,不能沿用)。
DIST_RE_V7 = r"(?:nearby|in the distance)"

CLAUSE_RE_V7 = (
    rf"(?:{SINGULAR_SUBJECT_RE_V7}(?: {DIST_RE_V7})? {POS_RE}"          # count==1
    rf"|{PLURAL_SUBJECT_RE_V7}, the nearest {POS_RE}"                    # count>=2, style=nearest
    rf"|{PLURAL_SUBJECT_RE_V7}, one(?: {DIST_RE_V7})? {POS_RE})"         # count>=2, style=one
)

TEMPLATE_RE_V7 = re.compile(
    rf"^(?:{PREFIX_RE})?{CLAUSE_RE_V7}(?:; {CLAUSE_RE_V7})?\.$",
    re.IGNORECASE,
)


def is_template_ok(caption):
    if not caption or not caption[0].isupper():
        return False
    return TEMPLATE_RE.match(caption) is not None or TEMPLATE_RE_V7.match(caption) is not None


# ---------------------------------------------------------------------------
# 3. 日夜前綴
#
#    重要:generate_captions.py v0.6 的 scene_prefix() 只有在 hours=="night"
#    時才會加 "Night" 這個詞,白天完全沒有對應的 "Day" 字面 token——白天是用
#    「沒有前綴」表示,不是用 "Day:" 表示。實測 captions_val.jsonl 1097 筆裡
#    0 筆出現 "Day:",所以「生成句開頭是否是 Night: 或 Day:」這個判斷條件裡
#    Day: 永遠不會發生。這裡改成「有沒有 Night 前綴」的二元分類(有夜晚字樣
#    vs. 沒有),這才是資料集裡實際存在、可以拿來跟 GT 比對的訊號。
# ---------------------------------------------------------------------------
NIGHT_RE = re.compile(r"^Night[:,]", re.IGNORECASE)
VALID_PREFIX_RE = re.compile(rf"^{PREFIX_RE}", re.IGNORECASE)
LEADING_PREFIX_ATTEMPT_RE = re.compile(r"^([A-Za-z][A-Za-z ,]*):\s")


def has_night_prefix(caption):
    return bool(NIGHT_RE.match(caption))


def is_prefix_valid(caption):
    """合法定義:要嘛完全沒有前綴(多數白天/晴天樣本),要嘛前綴屬於已知的
    Night / weather 詞彙表(對齊 WEATHER_MAP 與 scene_prefix() 的組合方式)。
    """
    if not LEADING_PREFIX_ATTEMPT_RE.match(caption):
        return True
    return bool(VALID_PREFIX_RE.match(caption))


# ---------------------------------------------------------------------------
# 4. batch 生成:同一個 step 對整個 batch forward 一次算下一個 token。
#    已經遇到 EOS 的樣本,之後每一步都強制餵回 EOS_TOKEN_ID,
#    不會影響同一個 batch 裡其他樣本的輸出(batch 維度上樣本互相獨立)。
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_batch(model, image_feats, max_new_tokens=MAX_NEW_TOKENS):
    B = image_feats.size(0)
    idx = torch.full((B, 1), IMAGE_TOKEN_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    eos_step = torch.full((B,), -1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        logits, _ = model(idx, targets=None, image_feature=image_feats)
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_id = torch.where(finished, torch.full_like(next_id, EOS_TOKEN_ID), next_id)
        newly_finished = (~finished) & (next_id == EOS_TOKEN_ID)
        eos_step[newly_finished] = step
        finished = finished | newly_finished

        idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break

    return idx.cpu(), eos_step.cpu()


def decode_generated(tokenizer, idx_row, eos_step_i):
    tokens = idx_row[1:].tolist()  # 去掉開頭的 image token
    if eos_step_i >= 0:
        content_ids = tokens[:eos_step_i]
        eos_hit = True
    else:
        content_ids = tokens
        eos_hit = False
    content_ids = [t for t in content_ids if t not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    caption = tokenizer.decode([content_ids]) if content_ids else ""
    return caption, len(content_ids), eos_hit


def encode_flat(tokenizer, text):
    nested = tokenizer.encode(text)
    return [tid for chunk in nested for tid in chunk]


def load_hours_map(coco_path):
    """file_name -> extra_info.hours ('day'/'night'/'dawn/dusk'/None)。"""
    coco = json.load(open(coco_path, "r", encoding="utf-8"))
    return {im["file_name"]: im.get("extra_info", {}).get("hours") for im in coco["images"]}


def night_prf(rows):
    """Night 這個類別的 precision / recall / f1(正類 = 有 Night 前綴)。"""
    tp = sum(1 for r in rows if r["gt_night"] and r["gen_night"])
    fp = sum(1 for r in rows if r["gen_night"] and not r["gt_night"])
    fn = sum(1 for r in rows if r["gt_night"] and not r["gen_night"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] + (s[c] - s[f]) * (k - f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-captions", default=CAPTIONS_VAL_PATH,
                         help="val captions jsonl 路徑(換成 captions_val_v2.jsonl 可評估 long-tail bug 修好後的版本)")
    parser.add_argument("--train-captions", default=CAPTIONS_TRAIN_PATH,
                         help="train captions jsonl 路徑,重訓 tokenizer 用——一定要跟目標 checkpoint "
                              "訓練時用的 train captions 是同一份,不然 token id 對不上 embedding "
                              "(v3 model 要傳 captions_train_v3.jsonl)")
    parser.add_argument("--ckpt", default=CKPT_PATH, help="checkpoint 路徑(v3 model 傳 checkpoints_v3/best_model.pt)")
    parser.add_argument("--out", default=OUT_CSV, help="輸出 CSV 路徑")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)

    # --- 1. 重新訓練 tokenizer(必須跟 train_vlm.py 用同一份語料 + vocab_size) ---
    train_captions = []
    with open(args.train_captions, "r", encoding="utf-8") as f:
        for line in f:
            train_captions.append(json.loads(line))

    print(f"[1/5] 用 {len(train_captions)} 筆 train captions ({args.train_captions}) 重新訓練 tokenizer (vocab_size={BASE_VOCAB_SIZE})...")
    tokenizer = minbpe()
    tokenizer.train(" ".join(c["caption"] for c in train_captions), vocab_size=BASE_VOCAB_SIZE)

    # --- 2. 載入 val captions + val CLIP features ---
    val_captions = []
    with open(args.val_captions, "r", encoding="utf-8") as f:
        for line in f:
            val_captions.append(json.loads(line))

    cache = torch.load(FEATURES_VAL_PATH, map_location="cpu")
    feature_by_name = {name: cache["features"][i] for i, name in enumerate(cache["file_name"])}
    val_captions = [c for c in val_captions if c["file_name"] in feature_by_name]
    print(f"[2/5] val set: {len(val_captions)} 筆(有對應 CLIP feature),captions 檔案: {args.val_captions}")

    hours_map = load_hours_map(VAL_COCO_PATH)

    # --- 3. 載入 best_model.pt ---
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[3/5] 載入 {args.ckpt}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # --- 4. batch 生成 ---
    print(f"[4/5] 對 {len(val_captions)} 筆 val set 做 batch 生成 (batch_size={GEN_BATCH_SIZE}, max_new_tokens={MAX_NEW_TOKENS})...")
    rows = []
    with torch.no_grad():
        for start in range(0, len(val_captions), GEN_BATCH_SIZE):
            batch = val_captions[start:start + GEN_BATCH_SIZE]
            image_features = torch.stack([feature_by_name[c["file_name"]] for c in batch]).to(device)

            idx, eos_step = generate_batch(model, image_features, MAX_NEW_TOKENS)

            for i, row in enumerate(batch):
                gen_caption, gen_len, eos_hit = decode_generated(tokenizer, idx[i], int(eos_step[i]))
                rows.append({
                    "file_name": row["file_name"],
                    "gt_caption": row["caption"],
                    "gen_caption": gen_caption,
                    "gen_len": gen_len,
                    "eos_hit": eos_hit,
                })
            print(f"  ...{min(start + GEN_BATCH_SIZE, len(val_captions))}/{len(val_captions)}", end="\r")
    print()

    # --- 5. 算指標 ---
    print("[5/5] 計算指標中...")
    gt_lens = [len(encode_flat(tokenizer, r["gt_caption"])) for r in rows]

    sum_tp, sum_gen, sum_gt = 0, 0, 0
    for r, gt_len in zip(rows, gt_lens):
        gt_classes = extract_classes(r["gt_caption"])
        gen_classes = extract_classes(r["gen_caption"])
        tp = len(gt_classes & gen_classes)

        r["class_recall"] = (tp / len(gt_classes)) if gt_classes else (1.0 if not gen_classes else 0.0)
        r["class_precision"] = (tp / len(gen_classes)) if gen_classes else (1.0 if not gt_classes else 0.0)
        r["gt_night"] = has_night_prefix(r["gt_caption"])
        r["gen_night"] = has_night_prefix(r["gen_caption"])
        r["prefix_match"] = r["gt_night"] == r["gen_night"]
        r["prefix_valid"] = is_prefix_valid(r["gen_caption"])
        r["template_ok"] = is_template_ok(r["gen_caption"])
        r["gt_len"] = gt_len
        r["gt_hours"] = hours_map.get(r["file_name"])
        r["has_hours_label"] = r["gt_hours"] in ("day", "night", "dawn/dusk")

        sum_tp += tp
        sum_gen += len(gen_classes)
        sum_gt += len(gt_classes)

    n = len(rows)
    prefix_valid_rate = sum(r["prefix_valid"] for r in rows) / n
    prefix_match_rate = sum(r["prefix_match"] for r in rows) / n
    template_ok_rate = sum(r["template_ok"] for r in rows) / n

    micro_precision = sum_tp / sum_gen if sum_gen else 0.0
    micro_recall = sum_tp / sum_gt if sum_gt else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) else 0.0)

    gen_lens = [r["gen_len"] for r in rows]
    eos_hit_rate = sum(r["eos_hit"] for r in rows) / n

    # Night precision/recall/f1:全部樣本 vs. 只看 hours 有明確標注(排除 None)的樣本。
    # 動機:hours=None 目前被當「非 night」處理,會混進雜訊,拆開看才知道
    # Night 判斷本身的品質,還是被缺失標注拖累。
    night_p_all, night_r_all, night_f1_all = night_prf(rows)
    labeled_rows = [r for r in rows if r["has_hours_label"]]
    night_p_lab, night_r_lab, night_f1_lab = night_prf(labeled_rows)

    # --- 輸出 CSV ---
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "gt_caption", "gen_caption", "prefix_match",
                          "template_ok", "class_recall", "class_precision", "gen_len", "eos_hit",
                          "gt_hours", "has_hours_label"])
        for r in rows:
            writer.writerow([
                r["file_name"], r["gt_caption"], r["gen_caption"],
                int(r["prefix_match"]), int(r["template_ok"]),
                round(r["class_recall"], 4), round(r["class_precision"], 4),
                r["gen_len"], int(r["eos_hit"]),
                r["gt_hours"], int(r["has_hours_label"]),
            ])
    print(f"已寫出 {args.out} ({n} 列)\n")

    # --- stdout summary ---
    print("=" * 70)
    print("Val Set 量化評估 Summary (n = {})".format(n))
    print("=" * 70)
    print(f"[日夜前綴]")
    print(f"  合法前綴率 (prefix_valid_rate)  : {prefix_valid_rate:.4f}")
    print(f"  前綴匹配率 (prefix_match_rate)  : {prefix_match_rate:.4f}  (跟 GT 的 Night/非Night 是否一致)")
    print(f"  註: v0.6 生成規則沒有字面上的 'Day:' 前綴,白天/晴天是用「無前綴」表示,")
    print(f"      這裡的日夜判斷改成「是否有 Night 前綴」的二元比對。")
    print()
    print(f"[Night precision/recall/f1 —— 全部樣本 vs. 只看 hours 有明確標注的樣本]")
    print(f"  全部樣本 (n={n})           : precision={night_p_all:.4f}  recall={night_r_all:.4f}  f1={night_f1_all:.4f}")
    print(f"  hours 有標注 (n={len(labeled_rows)})     : precision={night_p_lab:.4f}  recall={night_r_lab:.4f}  f1={night_f1_lab:.4f}")
    print(f"  (hours=None 的 {n - len(labeled_rows)} 筆目前被當「非 night」處理,可能混雜訊)")
    print()
    print(f"[句型模板合規率]")
    print(f"  template_ok_rate                : {template_ok_rate:.4f}")
    print()
    print(f"[物件類別 (micro-averaged)]")
    print(f"  precision                       : {micro_precision:.4f}")
    print(f"  recall                          : {micro_recall:.4f}")
    print(f"  f1                               : {micro_f1:.4f}")
    print()
    print(f"[生成長度分布 (token 數)]")
    print(f"  gen  mean={statistics.mean(gen_lens):.2f}  median={statistics.median(gen_lens):.2f}  p95={percentile(gen_lens, 95):.2f}")
    print(f"  gt   mean={statistics.mean(gt_lens):.2f}  median={statistics.median(gt_lens):.2f}  p95={percentile(gt_lens, 95):.2f}")
    print()
    print(f"[EOS 命中率]")
    print(f"  eos_hit_rate                     : {eos_hit_rate:.4f}  (在 max_new_tokens={MAX_NEW_TOKENS} 內自然遇到 EOS 結束的比例)")
    print()

    verdict_bits = []
    if template_ok_rate >= 0.8:
        verdict_bits.append("句型模板學得穩")
    elif template_ok_rate >= 0.4:
        verdict_bits.append("句型模板學了一部分但還不穩定")
    else:
        verdict_bits.append("句型模板幾乎沒學起來")

    if micro_f1 >= 0.6:
        verdict_bits.append("物件類別辨識準確度尚可")
    elif micro_f1 >= 0.3:
        verdict_bits.append("物件類別辨識準確度普通")
    else:
        verdict_bits.append("物件類別幾乎對不上 GT")

    if eos_hit_rate >= 0.8:
        verdict_bits.append("EOS 收斂正常")
    else:
        verdict_bits.append(f"有 {(1 - eos_hit_rate) * 100:.1f}% 的生成句沒有自然結束(可能亂序或重複到撞 max_len)")

    print("結論: " + "、".join(verdict_bits) + "。")


if __name__ == "__main__":
    main()
