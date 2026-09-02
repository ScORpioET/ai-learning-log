"""
Day38:重用 position_binding_accuracy.py 的 run() 函式(完全不改邏輯),
對三個 checkpoint 的 eval CSV 各跑一次,不用它原本寫死兩個特定檔名的
main(),自己指定三個 CSV 路徑。
"""
import json
from position_binding_accuracy import run

targets = [
    ("eval_val_results_exp2_reweight2x_rerun.csv", "v7", "thermal (full_v2 + reweight2x, 既有基準)"),
    ("eval_val_results_filtered_v2_reweight2x.csv", "v7", "thermal (filtered_v2 + reweight2x, 這次新的)"),
    ("eval_val_results_rgb_filtered_reweight2x.csv", "v7", "RGB (filtered + reweight2x, 這次新的)"),
]

results = {}
for csv_path, style, label in targets:
    res = run(csv_path, style, label)
    results[label] = res
    print(f"{label}: binding_accuracy={res['binding_accuracy']*100:.2f}%  "
          f"mismatch_rate={res['mismatch_rate']*100:.2f}%  "
          f"gt_parse={res['gt_clause_parse_rate']*100:.2f}%  "
          f"gen_parse={res['gen_clause_parse_rate']*100:.2f}%  "
          f"n={res['n']}")

with open("position_binding_day38.json", "w") as f:
    json.dump(
        {k: {kk: vv for kk, vv in v.items() if kk != "case_records"} for k, v in results.items()},
        f, indent=2,
    )
print("[done] position_binding_day38.json written")
