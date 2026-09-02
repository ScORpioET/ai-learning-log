"""
Day38 Phase 4:caption-completeness bug 修復前後對照。
重用 position_binding_accuracy.py 的 run() 函式(完全不改邏輯)。
"""
import json
from position_binding_accuracy import run

targets = [
    ("eval_val_results_exp2_reweight2x_rerun.csv", "v7", "thermal full+reweight2x (修復前, exp2 既有基準)"),
    ("eval_val_results_full_capfix_reweight2x.csv", "v7", "thermal full+reweight2x (修復後, capfix)"),
    ("eval_val_results_rgb_full_reweight2x.csv", "v7", "RGB full+reweight2x (修復前)"),
    ("eval_val_results_rgb_full_capfix_reweight2x.csv", "v7", "RGB full+reweight2x (修復後, capfix)"),
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

with open("position_binding_day38_capfix.json", "w") as f:
    json.dump(
        {k: {kk: vv for kk, vv in v.items() if kk != "case_records"} for k, v in results.items()},
        f, indent=2,
    )
print("[done] position_binding_day38_capfix.json written")
