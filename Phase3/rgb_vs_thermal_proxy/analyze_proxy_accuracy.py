"""
驗證「YOLO 偵測層級代理指標(數量、平均信心值)能不能預測哪個 domain 的
caption 比較準」。

代理指標(對每一幀):
    n_dets_rgb / avg_conf_rgb(RGB YOLO 偵測數量、平均信心值)
    n_dets_thermal / avg_conf_thermal(同上,thermal)
    count_winner:數量多的那邊贏,相等算 tie
    conf_winner:平均信心值高的那邊贏,相等算 tie
    combined_winner(起始規則):count_winner 為主,count 打平時看
        conf_winner,兩者都打平才算 tie
真實結果(對每一幀):
    用 Day36 position_binding_accuracy.py 的 parse_caption("v7") +
    compare_segments()(逐字重用,不改邏輯),對 RGB/thermal 各自的
    (gen vs gt)算 correct/mismatched/missing/extra,frame_score =
    correct/(correct+mismatched+missing+extra);GT 跟生成都完全沒有
    dynamic object(兩邊 segs 都是空)時,沒東西可以答錯,定義
    frame_score=1.0(不是 None,這樣才有明確定義,不會有除以 0 的
    未定義情況)。
    real_winner:frame_score 高的那邊贏,完全相等算 real_tie(這是這次
    唯一用的「打平」定義——沒有另外訂一個「差距很小」的模糊帶,理由是
    frame_score 本來就是小分母的比例數,能取到的值本來就疏,額外訂一個
    容忍帶會變成另一個需要交代理由的自由參數;完全相等已經是最不含糊的
    定義)。real_tie 的幀從「代理指標猜不猜得中」的準確率分母裡剔除,
    單獨報告比例,不稀釋準確率。

一致率:
    combined_agree_rate = P(combined_winner == real_winner | real_winner != tie)
    count_agree_rate    = P(count_winner == real_winner    | real_winner != tie)
    conf_agree_rate     = P(conf_winner == real_winner      | real_winner != tie)
    proxy 自己猜 tie,但 real 有明確贏家的情況,算「猜錯」(不算例外排除)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "Day32"))
from position_binding_accuracy import parse_caption, compare_segments  # noqa: E402

HERE = Path(__file__).parent


def load_detections(path):
    """run_yolo_inference.py 存的 file_name 沒有 'data/' 前綴,這裡補回去,
    跟 val_pairs.json 的 file_name 對齊(Day41 bright_region 任務踩過這個
    坑,這次一開始就修正)。"""
    by_file = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_file[f"data/{r['file_name']}"] = r["detections"]
    return by_file


def frame_score(gt_caption, gen_caption):
    gt_segs, _ = parse_caption(gt_caption, "v7")
    gen_segs, _ = parse_caption(gen_caption, "v7")
    correct, mismatched, missing, extra, _ = compare_segments(gt_segs, gen_segs)
    denom = correct + mismatched + missing + extra
    score = 1.0 if denom == 0 else correct / denom
    return score, correct, mismatched, missing, extra


def winner(a, b):
    if a > b:
        return "rgb"
    if b > a:
        return "thermal"
    return "tie"


def main():
    records = json.load(open(HERE / "val_inference_results.json"))
    rgb_dets = load_detections(HERE / "detections_rgb_val.jsonl")
    th_dets = load_detections(HERE / "detections_thermal_val.jsonl")

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

    with open(HERE / "val_analysis_rows.json", "w", encoding="utf-8") as f:
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

    # 額外揭露:proxy 自己也常常打平(count 相等),看看這種狀況出現多少次,
    # 不影響上面 agree_rate 的分母(那個分母只排除 real_tie),純粹提供脈絡。
    n_proxy_count_tie = sum(1 for r in rows if r["count_winner"] == "tie")
    n_proxy_combined_tie = sum(1 for r in rows if r["combined_winner"] == "tie")

    summary = {
        "n_total": n_total,
        "n_real_tie": n_real_tie, "real_tie_rate": round(n_real_tie / n_total, 4),
        "n_non_tie": n_nt,
        "combined_agree": combined_agree, "combined_agree_rate": combined_rate,
        "count_agree": count_agree, "count_agree_rate": count_rate,
        "conf_agree": conf_agree, "conf_agree_rate": conf_rate,
        "n_proxy_count_tie_of_total": n_proxy_count_tie,
        "n_proxy_combined_tie_of_total": n_proxy_combined_tie,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(HERE / "proxy_accuracy_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[done] val_analysis_rows.json + proxy_accuracy_summary.json written")


if __name__ == "__main__":
    main()
