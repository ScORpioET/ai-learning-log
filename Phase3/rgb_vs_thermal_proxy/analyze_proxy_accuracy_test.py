"""
同一套驗證邏輯(analyze_proxy_accuracy.py 的 frame_score()/winner()逐行重用,
不重寫),換成跑在 test split 全部 3749 組(100% frame 對應,Jack 要求換掉
val 的原因)。YOLO 偵測重用 bright_region_hypothesis 那次已經跑好的
detections_rgb_test.jsonl / detections_thermal_test.jsonl,不重新掃。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_proxy_accuracy import frame_score, winner  # noqa: E402

HERE = Path(__file__).parent
BRIGHT_DIR = Path.home() / "ai-transition-2026" / "Phase3" / "bright_region_hypothesis"


def load_detections(path):
    by_file = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_file[f"data/{r['file_name']}"] = r["detections"]
    return by_file


def main():
    records = json.load(open(HERE / "test_inference_results.json"))
    rgb_dets = load_detections(BRIGHT_DIR / "detections_rgb_test.jsonl")
    th_dets = load_detections(BRIGHT_DIR / "detections_thermal_test.jsonl")

    rows = []
    for r in records:
        rd = rgb_dets.get(r["rgb_file"], [])
        td = th_dets.get(r["thermal_file"], [])
        n_rgb, n_th = len(rd), len(td)
        conf_rgb = sum(d["conf"] for d in rd) / n_rgb if n_rgb else 0.0
        conf_th = sum(d["conf"] for d in td) / n_th if n_th else 0.0

        count_winner = winner(n_rgb, n_th)
        conf_winner = winner(conf_rgb, conf_th)
        combined_winner = count_winner if count_winner != "tie" else conf_winner

        rgb_score, rc, rm, rmi, rex = frame_score(r["rgb_gt"], r["rgb_gen"])
        th_score, tc, tm, tmi, tex = frame_score(r["thermal_gt"], r["thermal_gen"])
        real_winner = winner(rgb_score, th_score)

        rows.append({
            **r,
            "n_dets_rgb": n_rgb, "n_dets_thermal": n_th,
            "avg_conf_rgb": round(conf_rgb, 4), "avg_conf_thermal": round(conf_th, 4),
            "count_winner": count_winner, "conf_winner": conf_winner, "combined_winner": combined_winner,
            "rgb_score": round(rgb_score, 4), "thermal_score": round(th_score, 4),
            "rgb_breakdown": {"correct": rc, "mismatched": rm, "missing": rmi, "extra": rex},
            "thermal_breakdown": {"correct": tc, "mismatched": tm, "missing": tmi, "extra": tex},
            "real_winner": real_winner,
        })

    with open(HERE / "test_analysis_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    n_total = len(rows)
    non_tie = [r for r in rows if r["real_winner"] != "tie"]
    n_real_tie = n_total - len(non_tie)

    def agree_rate(key):
        agree = sum(1 for r in non_tie if r[key] == r["real_winner"])
        return agree, len(non_tie), round(agree / len(non_tie), 4) if non_tie else None

    combined_agree, n_nt, combined_rate = agree_rate("combined_winner")
    count_agree, _, count_rate = agree_rate("count_winner")
    conf_agree, _, conf_rate = agree_rate("conf_winner")

    n_real_rgb = sum(1 for r in non_tie if r["real_winner"] == "rgb")
    baseline_rate = round(n_real_rgb / n_nt, 4) if n_nt else None

    n_proxy_count_tie = sum(1 for r in rows if r["count_winner"] == "tie")
    n_proxy_combined_tie = sum(1 for r in rows if r["combined_winner"] == "tie")

    summary = {
        "n_total": n_total,
        "n_real_tie": n_real_tie, "real_tie_rate": round(n_real_tie / n_total, 4),
        "n_non_tie": n_nt,
        "n_real_rgb_win": n_real_rgb, "n_real_thermal_win": n_nt - n_real_rgb,
        "baseline_always_rgb_rate": baseline_rate,
        "combined_agree": combined_agree, "combined_agree_rate": combined_rate,
        "count_agree": count_agree, "count_agree_rate": count_rate,
        "conf_agree": conf_agree, "conf_agree_rate": conf_rate,
        "n_proxy_count_tie_of_total": n_proxy_count_tie,
        "n_proxy_combined_tie_of_total": n_proxy_combined_tie,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(HERE / "proxy_accuracy_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[done] test_analysis_rows.json + proxy_accuracy_summary_test.json written")


if __name__ == "__main__":
    main()
