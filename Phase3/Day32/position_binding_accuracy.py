"""
Day36 加碼:Position-Class Binding Accuracy。

Jack 手動抓到的問題:aggregate class F1 只看「這張圖有沒有出現這個詞」,不看
「這個方位的詞對不對」——GT 說右邊是行人,模型可能生成右邊是車,F1 completely
miss 這種錯位(因為兩個類別可能都在句子裡出現過,precision/recall 照樣算對)。

做法:把 GT / 生成句子都 parse 成 [(position, distance, class), ...] segment
list,同位置(position)做逐一比對,分三桶:
  1. class-position 正確:同位置,class 一樣
  2. class-position 錯位:同位置都有物件,class 不一樣(Jack 抓到的這種)
  3. position 缺失/多餘:GT 有這個位置生成沒有,或反過來

兩個 checkpoint 的句型不同(best_model.pt 是 v0.6「there is X」,
best_model_filtered.pt 是 v0.7+ class-first),分別寫 parser,不能共用一份 regex。
"""
import re
import csv
import random
from collections import defaultdict, Counter

from evaluate_val import EN_NAMES, CLASS_WORD_TO_CANON, DIST_PHRASES, POS_PHRASES, PREFIX_RE

POS_RE = "(?:" + "|".join(re.escape(p) for p in POS_PHRASES) + ")"
DIST_RE_V6 = "(?:" + "|".join(re.escape(p) for p in DIST_PHRASES) + ")"
DIST_RE_V7 = r"(?:nearby|in the distance)"

CLASS_ALT = "(?:" + "|".join(re.escape(n) for n in sorted(CLASS_WORD_TO_CANON, key=len, reverse=True)) + ")"


def normalize_pos(pos):
    return pos.lower()


def normalize_dist(dist):
    return dist.lower() if dist else None


def canon_class(word):
    return CLASS_WORD_TO_CANON.get(word.lower().strip())


# ---------------------------------------------------------------------------
# v0.6 clause parser:「{dist} {pos} there is {a|an} {class}」
#                     「{dist} {pos} there are {two|several} {class_plural}」
# ---------------------------------------------------------------------------
CLAUSE_V6_SINGULAR = re.compile(
    rf"^({DIST_RE_V6}) ({POS_RE}) there is (?:a|an) ({CLASS_ALT})$", re.IGNORECASE)
CLAUSE_V6_PLURAL = re.compile(
    rf"^({DIST_RE_V6}) ({POS_RE}) there are (two|several) ({CLASS_ALT})$", re.IGNORECASE)


def parse_clause_v6(clause):
    m = CLAUSE_V6_SINGULAR.match(clause)
    if m:
        dist, pos, cls = m.groups()
        canon = canon_class(cls)
        if canon:
            return {"position": normalize_pos(pos), "distance": normalize_dist(dist), "class": canon, "count": "1"}
    m = CLAUSE_V6_PLURAL.match(clause)
    if m:
        dist, pos, count_word, cls = m.groups()
        canon = canon_class(cls)
        if canon:
            return {"position": normalize_pos(pos), "distance": normalize_dist(dist), "class": canon, "count": count_word}
    return None


# ---------------------------------------------------------------------------
# v0.7+ clause parser:count==1  「{a|an} {class} [{dist}] {pos}」
#                     count>=2  「{count_word} {class_plural}, the nearest {pos}」
#                     count>=2  「{count_word} {class_plural}, one [{dist}] {pos}」
# ---------------------------------------------------------------------------
CLAUSE_V7_SINGULAR = re.compile(
    rf"^(?:a|an) ({CLASS_ALT})(?: ({DIST_RE_V7}))? ({POS_RE})$", re.IGNORECASE)
CLAUSE_V7_NEAREST = re.compile(
    rf"^(two|three|several|many) ({CLASS_ALT}), the nearest ({POS_RE})$", re.IGNORECASE)
CLAUSE_V7_ONE = re.compile(
    rf"^(two|three|several|many) ({CLASS_ALT}), one(?: ({DIST_RE_V7}))? ({POS_RE})$", re.IGNORECASE)


def parse_clause_v7(clause):
    m = CLAUSE_V7_SINGULAR.match(clause)
    if m:
        cls, dist, pos = m.groups()
        canon = canon_class(cls)
        if canon:
            return {"position": normalize_pos(pos), "distance": normalize_dist(dist), "class": canon, "count": "1"}
    m = CLAUSE_V7_NEAREST.match(clause)
    if m:
        count_word, cls, pos = m.groups()
        canon = canon_class(cls)
        if canon:
            return {"position": normalize_pos(pos), "distance": None, "class": canon, "count": count_word}
    m = CLAUSE_V7_ONE.match(clause)
    if m:
        count_word, cls, dist, pos = m.groups()
        canon = canon_class(cls)
        if canon:
            return {"position": normalize_pos(pos), "distance": normalize_dist(dist), "class": canon, "count": count_word}
    return None


def parse_caption(caption, style):
    """回傳 (segments, unparsed_clauses)。style: 'v6' or 'v7'."""
    text = re.sub(rf"^{PREFIX_RE}", "", caption or "", flags=re.IGNORECASE).strip()
    text = text.rstrip(".").strip()
    if not text:
        return [], []
    clauses = [c.strip() for c in text.split(";") if c.strip()]
    parser = parse_clause_v6 if style == "v6" else parse_clause_v7
    segments, unparsed = [], []
    for c in clauses:
        seg = parser(c)
        if seg:
            segments.append(seg)
        else:
            unparsed.append(c)
    return segments, unparsed


# ---------------------------------------------------------------------------
# 位置比對:同 position(不強求 distance 一樣,Jack 講的是「方位」),逐一配對。
# ---------------------------------------------------------------------------
def compare_segments(gt_segs, gen_segs):
    gt_by_pos = defaultdict(list)
    gen_by_pos = defaultdict(list)
    for s in gt_segs:
        gt_by_pos[s["position"]].append(s["class"])
    for s in gen_segs:
        gen_by_pos[s["position"]].append(s["class"])

    correct, mismatched, missing, extra = 0, 0, 0, 0
    mismatch_pairs = []
    all_pos = set(gt_by_pos) | set(gen_by_pos)
    for pos in all_pos:
        gt_list = gt_by_pos.get(pos, [])
        gen_list = gen_by_pos.get(pos, [])
        n = min(len(gt_list), len(gen_list))
        for i in range(n):
            if gt_list[i] == gen_list[i]:
                correct += 1
            else:
                mismatched += 1
                mismatch_pairs.append((pos, gt_list[i], gen_list[i]))
        if len(gt_list) > n:
            missing += len(gt_list) - n
        if len(gen_list) > n:
            extra += len(gen_list) - n
    return correct, mismatched, missing, extra, mismatch_pairs


def run(csv_path, style, label):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_correct = total_mismatch = total_missing = total_extra = 0
    n_gt_unparsed_captions = 0
    n_gen_unparsed_captions = 0
    total_gt_clauses = total_gen_clauses = 0
    total_gt_unparsed_clauses = total_gen_unparsed_clauses = 0
    case_records = []

    for r in rows:
        gt_segs, gt_unparsed = parse_caption(r["gt_caption"], style)
        gen_segs, gen_unparsed = parse_caption(r["gen_caption"], style)

        total_gt_clauses += len(gt_segs) + len(gt_unparsed)
        total_gen_clauses += len(gen_segs) + len(gen_unparsed)
        total_gt_unparsed_clauses += len(gt_unparsed)
        total_gen_unparsed_clauses += len(gen_unparsed)
        if gt_unparsed:
            n_gt_unparsed_captions += 1
        if gen_unparsed:
            n_gen_unparsed_captions += 1

        correct, mismatched, missing, extra, mismatch_pairs = compare_segments(gt_segs, gen_segs)
        total_correct += correct
        total_mismatch += mismatched
        total_missing += missing
        total_extra += extra

        case_records.append({
            "file_name": r["image_id"], "gt_caption": r["gt_caption"], "gen_caption": r["gen_caption"],
            "correct": correct, "mismatched": mismatched, "missing": missing, "extra": extra,
            "mismatch_pairs": mismatch_pairs,
        })

    n = len(rows)
    total_position_matched = total_correct + total_mismatch
    binding_accuracy = total_correct / total_position_matched if total_position_matched else 0.0
    mismatch_rate = total_mismatch / total_position_matched if total_position_matched else 0.0

    result = {
        "label": label, "n": n,
        "total_correct": total_correct, "total_mismatch": total_mismatch,
        "total_missing": total_missing, "total_extra": total_extra,
        "total_position_matched": total_position_matched,
        "binding_accuracy": binding_accuracy, "mismatch_rate": mismatch_rate,
        "gt_clause_parse_rate": 1 - (total_gt_unparsed_clauses / total_gt_clauses if total_gt_clauses else 0),
        "gen_clause_parse_rate": 1 - (total_gen_unparsed_clauses / total_gen_clauses if total_gen_clauses else 0),
        "n_gt_unparsed_captions": n_gt_unparsed_captions,
        "n_gen_unparsed_captions": n_gen_unparsed_captions,
        "case_records": case_records,
    }
    return result


def main():
    random.seed(36)
    results = [
        run("eval_val_results_gt_full_day36.csv", "v6", "best_model.pt (GT full, v0.6 template)"),
        run("eval_val_results_filtered_day36.csv", "v7", "best_model_filtered.pt (GT filtered, v0.7+ template)"),
    ]

    lines = ["# Position-Class Binding Accuracy\n"]
    lines.append(
        "GT/生成句子都 parse 成 [(position, distance, class), ...],同 position 逐一比對 class 對不對。\n"
        "三桶:class-position 正確 / class-position 錯位(方位對、類別錯,Jack 抓到的這種)/ "
        "position 缺失或多餘(GT 有生成沒有,或反過來)。\n"
    )

    for res in results:
        lines.append(f"## {res['label']}\n")
        lines.append(f"- n = {res['n']} 筆")
        lines.append(f"- GT 句子 clause parse 成功率: {res['gt_clause_parse_rate']*100:.1f}%"
                      f"({res['n_gt_unparsed_captions']} 筆句子裡有至少 1 個 clause parse 不出來)")
        lines.append(f"- 生成句子 clause parse 成功率: {res['gen_clause_parse_rate']*100:.1f}%"
                      f"({res['n_gen_unparsed_captions']} 筆句子裡有至少 1 個 clause parse 不出來)")
        lines.append("")
        lines.append("| 桶 | 數量 | 佔比(相對 position-matched 或全部) |")
        lines.append("|---|---|---|")
        lines.append(f"| class-position 正確 | {res['total_correct']} | "
                      f"{100*res['total_correct']/res['total_position_matched']:.1f}% (相對同位置有配對的) |")
        lines.append(f"| class-position 錯位 | {res['total_mismatch']} | "
                      f"{100*res['total_mismatch']/res['total_position_matched']:.1f}% (相對同位置有配對的) |")
        lines.append(f"| position 缺失(GT 有生成沒有) | {res['total_missing']} | — |")
        lines.append(f"| position 多餘(生成有 GT 沒有) | {res['total_extra']} | — |")
        lines.append("")
        lines.append(f"**Position-Class Binding Accuracy = {res['binding_accuracy']*100:.1f}%** "
                      f"(同一個方位裡,GT 有物件、生成也有物件的情況下,類別講對的比例)")
        lines.append(f"**Class-Position 錯位率 = {res['mismatch_rate']*100:.1f}%**"
                      f"(同一個方位,兩邊都有物件,但類別不一樣——這是 Jack 抓到的那種問題)\n")

    # --- 抽 15 張具體案例 ---
    lines.append("## 15 張具體案例(含錯位 / 正確 / 缺失多餘各種情況)\n")

    all_cases = []
    for res in results:
        for c in res["case_records"]:
            c["_source"] = res["label"]
            all_cases.append(c)

    mismatch_cases = [c for c in all_cases if c["mismatched"] > 0]
    correct_only_cases = [c for c in all_cases if c["correct"] > 0 and c["mismatched"] == 0 and c["missing"] == 0 and c["extra"] == 0]
    missing_extra_cases = [c for c in all_cases if c["mismatched"] == 0 and (c["missing"] > 0 or c["extra"] > 0)]

    random.shuffle(mismatch_cases)
    random.shuffle(correct_only_cases)
    random.shuffle(missing_extra_cases)

    picked = mismatch_cases[:7] + correct_only_cases[:4] + missing_extra_cases[:4]
    picked = picked[:15]

    for i, c in enumerate(picked, 1):
        lines.append(f"### 案例 {i} [{c['_source']}] — {c['file_name']}")
        lines.append(f"- GT : {c['gt_caption']}")
        lines.append(f"- GEN: {c['gen_caption']}")
        lines.append(f"- correct={c['correct']}, mismatched={c['mismatched']}, missing={c['missing']}, extra={c['extra']}")
        if c["mismatch_pairs"]:
            for pos, gt_cls, gen_cls in c["mismatch_pairs"]:
                lines.append(f"  - ⚠️ 錯位: position={pos}, GT={gt_cls}, GEN={gen_cls}")
        lines.append("")

    with open("position_binding_accuracy.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[done] position_binding_accuracy.md written")
    for res in results:
        print(f"{res['label']}: binding_accuracy={res['binding_accuracy']*100:.1f}%  "
              f"mismatch_rate={res['mismatch_rate']*100:.1f}%  "
              f"gt_parse={res['gt_clause_parse_rate']*100:.1f}%  gen_parse={res['gen_clause_parse_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
