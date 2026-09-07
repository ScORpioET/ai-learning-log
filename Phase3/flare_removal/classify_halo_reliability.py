"""
對 black_blob_summary_v3_keep.json(840 張,已經排除色塊雜訊之後的曝光
候選圖)逐張跑 radial_halo_profile.radial_profile_halo(),檢查每個 blob
36 個方向裡,有多少方向「真的」偵測到亮度掉回背景門檻以下(frac_reached_bg)。

如果一張圖裡有任何一個 blob 的 frac_reached_bg < 0.5(超過一半方向都沒
真的觸底,是用圖片邊界硬湊出來的),就把整張圖標記為「光暈範圍不可信」
——通常是霧夜/大範圍瀰漫散射光這種場景,背景本身不夠暗,radial profile
算不出有意義的邊界,不是核心曝光判斷錯誤,只是這個「光暈面積」指標對
這種場景不適用。

標記出來的圖片只搬移(不刪除)到 backup_halo_unreliable_v4/,可信的留在
原本的 orig/flare_blobs/radial_halo/radial_halo_bg80/radial_halo_bg100。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
import cv2  # noqa: E402

from radial_halo_profile import radial_profile_halo  # noqa: E402

HERE = Path(__file__).parent
FRAC_RELIABLE_THRESH = 0.5


def main():
    data = json.load(open(HERE / "black_blob_summary_v3_keep.json"))
    reliable, unreliable = [], []

    for i, row in enumerate(data):
        split, fn = row["split"], row["file_name"]
        img = cv2.imread(str(HERE / split / "flare" / fn))
        if img is None:
            continue

        fracs = []
        for b in row["blobs"]:
            halo = radial_profile_halo(img, tuple(b["centroid"]))
            fracs.append(halo["frac_reached_bg"])

        row2 = dict(row)
        row2["frac_reached_bg_per_blob"] = fracs
        if any(f < FRAC_RELIABLE_THRESH for f in fracs):
            unreliable.append(row2)
        else:
            reliable.append(row2)

        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(data)}", end="\r")
    print()

    print(f"[done] 可信(光暈面積數字可用):{len(reliable)} 張")
    print(f"[done] 不可信(標記為失敗,光暈面積不代表真實邊界):{len(unreliable)} 張")

    from collections import Counter
    print("可信 分split", Counter(r["split"] for r in reliable))
    print("不可信 分split", Counter(r["split"] for r in unreliable))

    json.dump(reliable, open(HERE / "black_blob_summary_v4_reliable.json", "w"), ensure_ascii=False, indent=2)
    json.dump(unreliable, open(HERE / "black_blob_summary_v4_unreliable.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
