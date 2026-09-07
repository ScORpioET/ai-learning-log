"""
用 severity_scores_v2.json:
1. 排除 halo_area,只用 laplacian_bbox + downstream_confidence_drop(有值的話,
   沒有就退回只用 laplacian_bbox)組一個「新嚴重度排序」:
   - laplacian 越低 = 對比度退化越嚴重 -> 用 (-laplacian) 排名
   - confidence_drop 越大 = 下游任務信心值掉越多 -> 用 (+drop) 排名
   兩者都轉成百分位排名後取平均,當作 new_severity_rank(0~1,越接近1越嚴重)
2. 對照原本用 halo_area_px 的排序,抓出兩份反轉清單:
   - halo_area 大(前25%)但 new_severity_rank 顯示不嚴重(後50%)
   - halo_area 小(後25%)但 new_severity_rank 顯示嚴重(前25%)
3. 交叉驗證:把線A的12張人工標記案例對進這個新排序,看排名對不對得起來。
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

LINE_A_SEVERE = {
    "video-GiDQGbWeWwtNTQEnG-frame-000618-f2ESyX6KKhcvwJkJa.jpg",
    "video-QQZ8wcAm8Y9EPufST-frame-003780-PW7Qp4mjzTFx3mTSh.jpg",
    "video-rXCGRrzyh98JMJk5v-frame-006431-yyyqnFhJiNKhBHZ3R.jpg",
    "video-7cJxWPFMvPdSiWASY-frame-003468-pDzAKf8tGhPdhmpb3.jpg",
    "video-QTT9nPmNSQceyBezK-frame-006008-cmF9Xmm4AFhdMdZDe.jpg",
    "video-dvZBYnphN2BwdMKBc-frame-000021-Fte9QvqiE6kenxHWP.jpg",
}
LINE_A_NOT_SEVERE = {
    "video-hXJxXmduG5Cjz9FBg-frame-001535-HfPEjYdx8M6iPhjfB.jpg",
    "video-WPJrL5MznEAgqaD47-frame-000253-ibNPeLcpWmf3niZtK.jpg",
    "video-BfJLkH7fCsS5Y9vQc-frame-002567-sEtfCzZswjRkyFqqS.jpg",
    "video-qdYxM3S6eGGsKTzwn-frame-000205-SPNuibkvDs2kqxojE.jpg",
    "video-vHLKy4kSYoaPYZoQo-frame-004474-kPX4nzEYpEXMtFodi.jpg",
    "video-23bsd9bsr962GdFBZ-frame-005935-e7u6popFNneGQhG6W.jpg",
}


def pct_rank(values):
    """回傳每個值的百分位排名(0~1),None 值保留 None"""
    idx_valid = [i for i, v in enumerate(values) if v is not None]
    valid_vals = [values[i] for i in idx_valid]
    order = np.argsort(valid_vals)
    ranks = np.empty(len(valid_vals))
    ranks[order] = np.arange(len(valid_vals)) / max(len(valid_vals) - 1, 1)
    out = [None] * len(values)
    for j, i in enumerate(idx_valid):
        out[i] = float(ranks[j])
    return out


def main():
    data = json.load(open(HERE / "severity_scores_v2.json"))

    laplacian = [r["laplacian_bbox"] for r in data]
    drop = [r["downstream_confidence_drop"] for r in data]
    halo = [r["halo_area_px"] for r in data]

    lap_rank_severity = pct_rank([-v if v is not None else None for v in laplacian])  # 越低越嚴重 -> 排名越高
    drop_rank_severity = pct_rank(drop)  # 越大越嚴重

    new_severity_rank = []
    for lr, dr in zip(lap_rank_severity, drop_rank_severity):
        vals = [v for v in (lr, dr) if v is not None]
        new_severity_rank.append(float(np.mean(vals)) if vals else None)

    for r, nsr in zip(data, new_severity_rank):
        r["new_severity_rank"] = round(nsr, 4) if nsr is not None else None

    halo_rank = pct_rank(halo)
    for r, hr in zip(data, halo_rank):
        r["halo_area_rank"] = round(hr, 4) if hr is not None else None

    with open(HERE / "severity_scores_v2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    valid = [r for r in data if r["new_severity_rank"] is not None and r["halo_area_rank"] is not None]
    print(f"[info] 有效比較樣本數: {len(valid)} / {len(data)}")

    # halo_area 大(前25%,rank>=0.75)但 new_severity 顯示不嚴重(後50%,rank<0.5)
    flip_overrated = [r for r in valid if r["halo_area_rank"] >= 0.75 and r["new_severity_rank"] < 0.5]
    # halo_area 小(後25%,rank<=0.25)但 new_severity 顯示嚴重(前25%,rank>=0.75)
    flip_underrated = [r for r in valid if r["halo_area_rank"] <= 0.25 and r["new_severity_rank"] >= 0.75]

    print(f"\n[halo_area 大但新指標判定不嚴重] {len(flip_overrated)} 張 ({100*len(flip_overrated)/len(valid):.1f}%)")
    for r in sorted(flip_overrated, key=lambda r: r["new_severity_rank"])[:15]:
        print(f"  {r['split']} {r['file_name']}  halo_rank={r['halo_area_rank']:.2f} new_rank={r['new_severity_rank']:.2f} "
              f"laplacian={r['laplacian_bbox']} drop={r['downstream_confidence_drop']}")

    print(f"\n[halo_area 小但新指標判定嚴重] {len(flip_underrated)} 張 ({100*len(flip_underrated)/len(valid):.1f}%)")
    for r in sorted(flip_underrated, key=lambda r: -r["new_severity_rank"])[:15]:
        print(f"  {r['split']} {r['file_name']}  halo_rank={r['halo_area_rank']:.2f} new_rank={r['new_severity_rank']:.2f} "
              f"laplacian={r['laplacian_bbox']} drop={r['downstream_confidence_drop']}")

    json.dump({"flip_overrated": flip_overrated, "flip_underrated": flip_underrated},
               open(HERE / "severity_reorder_flips_v2.json", "w"), ensure_ascii=False, indent=2)

    # 交叉驗證:線A的12張案例在新排序裡的百分位
    print("\n=== 交叉驗證:線A案例在新排序(new_severity_rank)裡的位置 ===")
    by_fn = {r["file_name"]: r for r in data}
    print("-- 線A判定「嚴重」的案例(希望 new_severity_rank 偏高)--")
    for fn in LINE_A_SEVERE:
        r = by_fn.get(fn)
        if r:
            print(f"  {fn}: new_severity_rank={r['new_severity_rank']}  halo_area_rank={r['halo_area_rank']}")
    print("-- 線A判定「不嚴重」的案例(希望 new_severity_rank 偏低)--")
    for fn in LINE_A_NOT_SEVERE:
        r = by_fn.get(fn)
        if r:
            print(f"  {fn}: new_severity_rank={r['new_severity_rank']}  halo_area_rank={r['halo_area_rank']}")

    print("\n[done]")


if __name__ == "__main__":
    main()
