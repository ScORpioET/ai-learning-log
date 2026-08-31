"""
Day36 Task B:Missing position(GT 有講、生成沒講到同一個 position)依 (class,
distance) 分組,查是不是特定組合系統性被漏講,並對照這些組合在訓練資料裡的
出現頻率(是不是本來訓練樣本就少)。
"""
import csv
import json
from collections import defaultdict, Counter

from position_binding_accuracy import parse_caption, compare_segments


def missing_breakdown(csv_path, style):
    """回傳 (missing_counter, gt_total_counter):
    missing_counter[(class, distance)] = 這個組合在「GT 有、GEN 同 position 沒有」
    的情況下出現的次數;gt_total_counter 是這個組合在 GT 裡出現的總次數(當分母)。
    """
    missing_counter = Counter()
    gt_total_counter = Counter()

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_segs, _ = parse_caption(r["gt_caption"], style)
            gen_segs, _ = parse_caption(r["gen_caption"], style)

            gt_by_pos = defaultdict(list)
            gen_by_pos = defaultdict(list)
            for s in gt_segs:
                gt_by_pos[s["position"]].append(s)
            for s in gen_segs:
                gen_by_pos[s["position"]].append(s)

            for pos in set(gt_by_pos) | set(gen_by_pos):
                gt_list = gt_by_pos.get(pos, [])
                gen_list = gen_by_pos.get(pos, [])
                n = min(len(gt_list), len(gen_list))
                for seg in gt_list:
                    dist = seg["distance"] or "mid"
                    gt_total_counter[(seg["class"], dist)] += 1
                # leftover GT entries beyond n are "missing"
                for seg in gt_list[n:]:
                    dist = seg["distance"] or "mid"
                    missing_counter[(seg["class"], dist)] += 1

    return missing_counter, gt_total_counter


def train_frequency(train_path, style):
    counter = Counter()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            segs, _ = parse_caption(r["caption"], style)
            for s in segs:
                dist = s["distance"] or "mid"
                counter[(s["class"], dist)] += 1
    return counter


def run(label, eval_csv, train_path, style):
    missing_counter, gt_total_counter = missing_breakdown(eval_csv, style)
    train_counter = train_frequency(train_path, style)

    total_train = sum(train_counter.values())

    rows = []
    for combo in gt_total_counter:
        gt_n = gt_total_counter[combo]
        missing_n = missing_counter.get(combo, 0)
        miss_rate = missing_n / gt_n if gt_n else 0
        train_n = train_counter.get(combo, 0)
        train_pct = 100 * train_n / total_train if total_train else 0
        rows.append((combo, gt_n, missing_n, miss_rate, train_n, train_pct))

    rows.sort(key=lambda r: -r[3])  # miss_rate 由高到低
    return rows


def format_section(label, rows):
    lines = [f"## {label}\n"]
    lines.append("| (class, distance) | val GT 出現次數 | 漏講次數 | 漏講率 | train 出現次數 | train 佔比 |")
    lines.append("|---|---|---|---|---|---|")
    for combo, gt_n, missing_n, miss_rate, train_n, train_pct in rows:
        if gt_n < 5:  # 樣本太少的組合不列,避免雜訊
            continue
        cls, dist = combo
        lines.append(f"| {cls} / {dist} | {gt_n} | {missing_n} | {miss_rate*100:.1f}% | {train_n} | {train_pct:.2f}% |")
    return "\n".join(lines)


def main():
    rows_full = run("best_model.pt (GT full)",
                     "eval_val_results_gt_full_day36.csv", "captions_train.jsonl", "v6")
    rows_filtered = run("best_model_filtered.pt (GT filtered)",
                         "eval_val_results_filtered_day36.csv", "captions_train_filtered.jsonl", "v7")

    lines = ["# Task B: Missing Position 系統性根因分析\n"]
    lines.append(
        "「missing」= GT 在某個 position 有講物件,生成句子在同一個 position 完全"
        "沒提(不是講錯類別,是那個位置整個沒被生成句子命中)。依 (class, distance) "
        "分組,看是不是特定組合系統性被漏講,並對照這個組合在訓練資料裡出現的頻率。"
        "只列 val GT 出現次數 >= 5 的組合,避免樣本太少的雜訊。\n"
    )
    lines.append(format_section("best_model.pt (GT full)", rows_full))
    lines.append("")
    lines.append(format_section("best_model_filtered.pt (GT filtered)", rows_filtered))

    with open("task_b_missing_position_breakdown.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[done] task_b_missing_position_breakdown.md written")
    print("\ntop 5 missing combos (GT full):")
    for combo, gt_n, missing_n, miss_rate, train_n, train_pct in rows_full[:5]:
        if gt_n >= 5:
            print(f"  {combo}: gt_n={gt_n} missing={missing_n} rate={miss_rate*100:.1f}% train_n={train_n} train_pct={train_pct:.2f}%")
    print("\ntop 5 missing combos (GT filtered):")
    for combo, gt_n, missing_n, miss_rate, train_n, train_pct in rows_filtered[:5]:
        if gt_n >= 5:
            print(f"  {combo}: gt_n={gt_n} missing={missing_n} rate={miss_rate*100:.1f}% train_n={train_n} train_pct={train_pct:.2f}%")


if __name__ == "__main__":
    main()
