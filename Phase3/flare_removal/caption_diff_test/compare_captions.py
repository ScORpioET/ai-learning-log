"""
Step 3: 對 captions_before_after.json 的 20 組 before/after caption 做兩種比對:
  1. 原始文字比對(直接看兩句話是否逐字相同)。
  2. 結構化比對:用 Phase3/Day32/position_binding_accuracy.py 現成的
     parse_caption(caption, "v7") 把 before/after 各自拆成
     [{position, distance, class, count}, ...] 的 class 集合,比較兩邊
     class 集合(新增/刪除的 class)以及同一個 class 的 position 是否改變。
     這是 caption_fusion/fuse_captions.py 已經在用的同一套 parser,不重寫。

輸出:
  - comparison_table.csv:frame 名稱、group、diff_mean、before/after caption、
    結構化比對結果
  - comparison_table.md:同上內容的 markdown 表格版本
"""
import csv
import json
import sys
from pathlib import Path

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
sys.path.insert(0, str(DAY32))
from position_binding_accuracy import parse_caption  # noqa: E402

HERE = Path(__file__).parent


def structured_classes(caption):
    """caption -> {class: {position, distance, count}} (v7 parser)。回傳
    unparsed clause 清單一併帶出,方便人工複核 parser 失敗的狀況。"""
    segs, unparsed = parse_caption(caption, "v7")
    by_class = {}
    for s in segs:
        by_class[s["class"]] = {"position": s["position"], "distance": s["distance"], "count": s["count"]}
    return by_class, unparsed


def diff_structured(before_classes, after_classes):
    added = sorted(set(after_classes) - set(before_classes))
    removed = sorted(set(before_classes) - set(after_classes))
    position_changed = sorted(
        c for c in (set(before_classes) & set(after_classes))
        if before_classes[c]["position"] != after_classes[c]["position"]
    )
    count_changed = sorted(
        c for c in (set(before_classes) & set(after_classes))
        if before_classes[c]["position"] == after_classes[c]["position"]
        and before_classes[c]["count"] != after_classes[c]["count"]
    )
    has_diff = bool(added or removed or position_changed or count_changed)
    return {
        "added": added,
        "removed": removed,
        "position_changed": position_changed,
        "count_changed": count_changed,
        "has_diff": has_diff,
    }


def main():
    samples = json.load(open(HERE / "captions_before_after.json"))

    rows = []
    for s in samples:
        before_classes, before_unparsed = structured_classes(s["before_caption"])
        after_classes, after_unparsed = structured_classes(s["after_caption"])
        sd = diff_structured(before_classes, after_classes)
        text_same = s["before_caption"].strip() == s["after_caption"].strip()

        struct_summary_parts = []
        if sd["added"]:
            struct_summary_parts.append(f"+{','.join(sd['added'])}")
        if sd["removed"]:
            struct_summary_parts.append(f"-{','.join(sd['removed'])}")
        if sd["position_changed"]:
            struct_summary_parts.append(f"pos:{','.join(sd['position_changed'])}")
        if sd["count_changed"]:
            struct_summary_parts.append(f"count:{','.join(sd['count_changed'])}")
        struct_summary = "; ".join(struct_summary_parts) if struct_summary_parts else "(no change)"

        rows.append({
            "group": s["group"],
            "file_name": s["file_name"],
            "diff_mean": round(s["diff_mean"], 2),
            "diff_max": s["diff_max"],
            "before_caption": s["before_caption"],
            "after_caption": s["after_caption"],
            "text_identical": text_same,
            "structured_diff_summary": struct_summary,
            "structured_has_diff": sd["has_diff"],
            "before_unparsed": "; ".join(before_unparsed),
            "after_unparsed": "; ".join(after_unparsed),
        })

    # CSV
    csv_path = HERE / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_path = HERE / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| group | frame | diff_mean | before caption | after caption | text identical | structured diff |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['group']} | `{r['file_name']}` | {r['diff_mean']} | "
                f"{r['before_caption']} | {r['after_caption']} | "
                f"{'yes' if r['text_identical'] else 'no'} | {r['structured_diff_summary']} |\n"
            )

    # summary counts
    high = [r for r in rows if r["group"] == "high_diff"]
    low = [r for r in rows if r["group"] == "low_diff"]
    high_struct_diff = sum(r["structured_has_diff"] for r in high)
    low_struct_diff = sum(r["structured_has_diff"] for r in low)
    high_text_diff = sum(not r["text_identical"] for r in high)
    low_text_diff = sum(not r["text_identical"] for r in low)

    print(f"[summary] high_diff group: {high_text_diff}/10 text differs, {high_struct_diff}/10 structured differs")
    print(f"[summary] low_diff  group: {low_text_diff}/10 text differs, {low_struct_diff}/10 structured differs")
    print(f"\n[done] {csv_path}\n[done] {md_path}")


if __name__ == "__main__":
    main()
