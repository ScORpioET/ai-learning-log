"""
Day39:parser + 融合邏輯。

1. 用 position_binding_accuracy.py 現成的 parse_caption(caption, "v7")
   (完全不改邏輯,這支腳本本來就是為了同一份 v0.7+ class-first 模板寫的
   parser)把 thermal/RGB 各自的「model 生成」caption 拆成
   [{"position","distance","class","count"}, ...] 結構化 segment。

2. 融合規則(這次任務指定的):
   a. 同一個 class 兩邊都有:
      - position 一致 -> 合併成一個 clause,count 取兩邊「數量詞桶」較大的
        那個(1<2<3<several<many,誰的桶大就採誰的——這是沒有指定規則時
        我自己選的合併方式,不是查出來的事實,見下面 COUNT 桶說明)。
      - position 不一致 -> 生成兩個版本:RGB 優先版整句用 RGB 這個 class
        的 position/distance/count;thermal 優先版用 thermal 的。兩個
        版本裡,「其他沒有衝突的 class」內容不變。
   b. class 只有一邊有:直接照那一邊的 segment 收進融合清單,不然那個
      class 的資訊會憑空消失。
   c. 融合後的 class 清單轉成 generate_captions.py 的 build_caption() 能
      吃的 objects 格式,直接呼叫 build_caption()(逐行沿用,不重寫任何
      模板/排序邏輯)生成最終自然語言。build_caption() 內部本來就會用
      count 排序、只取 top-2 class,如果融合後超過 2 個 class,由它原本
      的邏輯決定留哪兩個,這裡不額外加規則。

COUNT 桶(count_to_subject() 的分界,查程式碼確認的事實):
  1 -> "a X" / 2 -> "two Xs" / 3 -> "three Xs" / 4-6 -> "several Xs" / 7+ -> "many Xs"
parser 只能拿回桶名(count>=2 時看到的是「several」這個詞本身,原始確切
數字已經在生成 caption 那一步被丟掉了),所以這裡融合/回填時桶名 ->
代表數字用 {"1":1,"two":2,"three":3,"several":4,"many":7},刻意取每個
桶的下界,能保證餵進 build_caption() 後 count_to_subject() 還原出同一個
桶名(而不是不小心跨到下一桶)。
"""
import json
import sys
from pathlib import Path

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
sys.path.insert(0, str(DAY32))

from position_binding_accuracy import parse_caption  # noqa: E402

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "thermal_dataset"))
import generate_captions as gc  # noqa: E402

POS_PHRASE_TO_CODE = {v: k for k, v in gc.POSITION_PHRASE.items()}
DIST_TEXT_TO_CODE = {None: "mid", "nearby": "near", "in the distance": "far"}
COUNT_WORD_TO_INT = {"1": 1, "two": 2, "three": 3, "several": 4, "many": 7}
COUNT_RANK = {"1": 0, "two": 1, "three": 2, "several": 3, "many": 4}


def segments_by_class(caption):
    """caption -> {en_name: segment}. 假設同一個 caption 裡每個 class 最多出現
    一次(現在的模板本來就是這樣,top-2 by class,不會同 class 講兩次)。"""
    segs, unparsed = parse_caption(caption, "v7")
    if unparsed:
        print(f"  [warn] 有 clause parse 不出來,略過:{unparsed}  (caption: {caption!r})")
    return {s["class"]: s for s in segs}


def seg_to_object(en_name, seg):
    """segment -> generate_captions.py build_caption() 吃的 objects tuple
    清單(一個 class 產生 count 個一模一樣的假 instance,靠數量本身把
    count_to_subject() 的桶還原回去,area 遞減只是為了讓 aggregate_by_class
    的「同 count 比 area」排序穩定,不影響結果)。"""
    n = COUNT_WORD_TO_INT[seg["count"].lower()]
    pos = POS_PHRASE_TO_CODE[seg["position"]]
    dist = DIST_TEXT_TO_CODE[seg["distance"]]
    return [(en_name, en_name, pos, dist, 1000 - i) for i in range(n)]


def fuse_one_class(en_name, t_seg, r_seg, priority):
    """priority: 'thermal' or 'rgb'。回傳這個 class 最終要用哪個 domain 的 segment。"""
    if t_seg is None:
        return r_seg
    if r_seg is None:
        return t_seg
    if t_seg["position"] == r_seg["position"]:
        # position 一致:count 桶取較大的那個(自己選的合併規則,見檔頭說明)
        bigger = t_seg if COUNT_RANK[t_seg["count"].lower()] >= COUNT_RANK[r_seg["count"].lower()] else r_seg
        return bigger
    # position 不一致:照這個版本的優先權domain 決定
    return r_seg if priority == "rgb" else t_seg


def build_fused_caption(thermal_gen, rgb_gen, image_id_base):
    t_classes = segments_by_class(thermal_gen)
    r_classes = segments_by_class(rgb_gen)

    all_class_names = set(t_classes) | set(r_classes)
    conflict_classes = {
        c for c in (set(t_classes) & set(r_classes))
        if t_classes[c]["position"] != r_classes[c]["position"]
    }
    has_conflict = bool(conflict_classes)

    # 兩個版本永遠都輸出(即使沒有 conflict、兩版字面相同也一樣)——
    # 使用者要求不要用系統自動替她/他選一版,「系統建議」只是加個標籤附理由,
    # 不取代兩個版本都給的原則。
    versions = ["rgb_priority", "thermal_priority"]
    results = {}
    for version in versions:
        priority = "rgb" if version == "rgb_priority" else "thermal"
        objects = []
        for en_name in all_class_names:
            t_seg = t_classes.get(en_name)
            r_seg = r_classes.get(en_name)
            seg = fuse_one_class(en_name, t_seg, r_seg, priority)
            objects.extend(seg_to_object(en_name, seg))
        objects.sort(key=lambda o: -o[4])
        caption = gc.build_caption(objects, {}, f"{image_id_base}-{version}")
        results[version] = {
            "caption": caption,
            "classes_used": sorted(all_class_names),
            "conflict_classes": sorted(conflict_classes),
        }
    return results, has_conflict


def main():
    records = json.load(open(Path(__file__).parent / "model_inference_results.json"))

    out = []
    for r in records:
        image_id_base = f"{r['thermal_video_id']}-{r['frame_index']}"
        print(f"\n=== {r['thermal_file']} <-> {r['rgb_file']} ===")
        print(f"  thermal_gen: {r['thermal_gen']}")
        print(f"  rgb_gen    : {r['rgb_gen']}")
        fused, has_conflict = build_fused_caption(r["thermal_gen"], r["rgb_gen"], image_id_base)
        for version, info in fused.items():
            print(f"  [{version}] classes={info['classes_used']} conflict={info['conflict_classes']}")
            print(f"    -> {info['caption']}")
        out.append({**r, "fused": fused, "has_conflict": has_conflict})

    with open(Path(__file__).parent / "fused_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[done] fused_results.json written")


if __name__ == "__main__":
    main()
