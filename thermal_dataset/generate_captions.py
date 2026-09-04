"""
FLIR ADAS v2 rule-based scene caption generator
Day32 - Week10 Task 1

用法:
    python generate_captions.py --split train --out captions_train.jsonl --sample 2000

先在小樣本上跑,人工檢查隨機抽出的 10 句輸出通不通順、資訊夠不夠、
詞彙會不會太雜,確認沒問題再不加 --sample 跑全量。

【v0.5 語言切換】輸出語言從中文改成英文——因為 Jack Phase 1 手刻的 GPT-2
byte-level BPE tokenizer 是用英文語料訓練的,merge 規則全部是英文文字的
統計分布,中文字元雖然技術上編碼得出來(byte-level BPE 有 byte-fallback),
但完全沒有學過對應的 merge,序列長度會暴增、對中文毫無效率可言。
改成英文才能跟 decoder 自己的 tokenizer 是同一個語言分布,這件事要趁
還沒開始 fine-tune 之前先改掉,不要等訓練完才發現要重來。
註解維持中文(給 Jack 看的),生成的句子本身改英文(給 decoder 吃的)。

【v0.7 top-N 截斷問題修正 + class-first 重寫】
v0.6 的 build_caption() 對 group_objects() 合併後的結果只取 top = merged[:2],
而 group_objects() 是用 (en_name, position, distance) 三維分組——位置切
left/right/ahead、距離切 near/mid/far,最多 9 組。同類別的物件只要散在
不同格子裡就不會合併,於是「畫面明顯 5 台車」常常在合併後被拆成 4-5 組,
top-2 一截,caption 只講得到 2 台。而「several」判斷是組內 count>=3,
散開的物件永遠合不到一組,several 幾乎不會出現。

v0.7 三層改動:
  Level 1 小 bug 修:IRREGULAR_PLURALS 補 person->people;mid 距離省略
    距離詞只留位置片語;occluded 過濾從無效的 "heavy" in occluded 子字串
    比對(資料集裡沒有任何 occluded 字串包含 "heavy",v0.6 這個過濾其實
    從未生效過)改成對實測到的 "70%_-_90%_occluded_(difficult_to_see)"
    精確比對。
  Level 2 build_caption 重寫:分組維度從三維砍成只依 en_name,類別總數
    (class_totals)當主軸決定要不要講、講幾隻,每類別最多附一個位置補述
    (取該類別裡 area 最大/最近的 instance)。
  Level 3 句型多樣化:模板 A(兩子句分號接)/ 模板 C(單類別時只有一子句)。
    模板 B(scene 後置)先跳過,避免弱化日夜前綴的訓練信號。
"""
SCRIPT_VERSION = "v0.10 (2026-09-02, fix build_caption() dropping 2nd class when top class count>=5, Day38 caption-completeness)"

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# 1. 類別分組
#
# DYNAMIC_CLASSES:會動、值得當事件主體生成句子的類別,value 是英文單數名詞。
# 這份清單是照你實測的 counts 決定的——只放你資料集裡真的有量的類別,
# COCO 80 類裡幾乎是 0 筆的(cat/laptop/pizza 這種)全部忽略不處理。
#
# 🟢 2026-08-25 Jack 實測 images_thermal_train/coco.json 完整 histogram:
#    car 73623 / person 50478 / sign 20770 / light 16198 / bike 7237 /
#    bus 2245 / other vehicle 1373 / motor 1116 / hydrant 1095 / truck 829 /
#    skateboard 29 / stroller 15 / scooter 15 / deer 8 / train 5 / dog 4
#    rider 完全沒出現(0 筆)。dog / rider 都不放進清單。
# ---------------------------------------------------------------------------

DYNAMIC_CLASSES = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "other vehicle": "vehicle",
    "train": "train",          # 實測 5 筆,一定會被下面的 long-tail 門檻收進 fallback
    "skateboard": "skateboard",  # 實測 29 筆,同上
    "stroller": "stroller",      # 實測 15 筆,同上
    "scooter": "scooter",        # 實測 15 筆,同上
}

# 不規則複數,沒列在這裡的一律用「+s」規則變化(car->cars, truck->trucks...)。
IRREGULAR_PLURALS = {
    "bus": "buses",
    "person": "people",  # v0.7 Level 1 修正:原本沒列,plural_of("person") 會回 "persons"
}

# 樣本數低於這個門檻的類別,不單獨命名,一律用通用詞帶過,
# 避免 decoder 學到幾乎沒看過幾次的生僻詞彙。
#
# v0.3 起改成在 main() 裡用當下這個 split 實際跑出來的 counts 動態判斷,
# 低於門檻的類別在生成句子時會被換成 LONG_TAIL_LABEL,不寫死清單,
# 這樣 train/val 兩個 split 各自的長尾類別即使不同,也會分別正確處理。
LONG_TAIL_THRESHOLD = 500
LONG_TAIL_LABEL = "object"

# 場景背景類別:不生成任何句子內容,只記錄「這張圖有沒有出現」當中繼資料。
# light/hydrant/sign 這三類數量都不小(light 快兩萬筆、sign近三萬筆),
# 在公路情境資料裡幾乎每張圖都會有,講出來對事件描述沒有分辨力,
# 反而會稀釋訓練資料的語意密度、讓句子讀起來不自然(v0 實測發現)。
STATIC_CONTEXT_CLASSES = {
    "light": "traffic light",
    "hydrant": "fire hydrant",
    "sign": "sign",
}

WEATHER_MAP = {
    "partly_cloudy": "cloudy",
    "cloudy": "overcast",
    "rain": "rainy",
    "fog": "foggy",
    "clear": None,  # 晴天不特別標註,句子更簡潔
}

# 🟢 2026-08-29 Jack 實測 images_thermal_train/val 兩個 coco.json 的
# extra_info.occluded 實際字串值,只有這 3 種(還有一批完全沒有這個欄位):
#   "no_(fully_visible)"
#   "1%_-_70%_occluded_(partially_occluded)"
#   "70%_-_90%_occluded_(difficult_to_see)"
# 沒有任何字串包含 "heavy"——v0.6 用 "heavy" in occluded 判斷,
# 這個過濾條件從頭到尾沒生效過。改成對「difficult_to_see」這個
# 最嚴重的實際存在檔位精確比對;partially_occluded 仍然保留,
# bbox 面積/位置對這個等級的遮蔽還算可信。
OCCLUDED_DIFFICULT = "70%_-_90%_occluded_(difficult_to_see)"

# ---------------------------------------------------------------------------
# Day35: --source yolo 分支用的 alias 對照表。
#
# Day35/outputs/task0_class_mapping.md 做完的三方對照(Day32 script 用的
# FLIR class name / FLIR ADAS 原始 taxonomy / COCO 80 類)結論:YOLO 偵測
# 出來的是 COCO class name(如 "bicycle"),要先轉回 FLIR 這邊慣用的
# class name(如 "bike"),才能直接沿用下面 DYNAMIC_CLASSES /
# STATIC_CONTEXT_CLASSES / long-tail 判斷這整套邏輯,不用另外重寫一份。
#
# 只收錄 Day35 run_yolo_inference.py KEEP_CLASSES 裡有的 11 類——
# GT 有但 COCO 偵測不到的 "other vehicle" / "stroller" / "scooter"
# 不在這裡,YOLO 版这三類必然不會出現在 caption 裡(pretrained 代價,
# 已記錄在 task0_class_mapping.md)。
# ⚠️ "stop sign" -> "sign" 是語意窄化(COCO 只有停車再開標誌,FLIR sign
# 涵蓋各種交通標誌),"bike"/"motor" 對應關係標記為待 Jack 確認,
# 詳細理由都在 task0_class_mapping.md。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Day35 Task 5/6: tiny-bbox 過濾門檻(--filter-tiny 開啟時生效)。
#
# 背景:Day35 加碼分析發現 GT 裡 82% 的框是 tiny(<0.5% 畫面面積),YOLO 對這批
# tiny 框的匹配率只有 24%,人眼在視覺化圖上也常常看不清楚,懷疑是訓練雜訊。
#
# Task 5 最初試過 per-class threshold(對每個 class 分別算出「保留 70%」所需
# 的門檻),但 skateboard/stroller/train 這幾類樣本數太小(n=5~32),門檻是
# 借用鄰近類別湊出來的,不夠可信。Task 6:Jack 決定改回**單一全域門檻**,
# 寧可保守少濾,也不要用不可信的 per-class 數字。
#
# Day36 v2 bug 修正:原本的 0.05 大於 compute_area_thresholds() 動態算出的
# far_thresh(train 0.0427% / val 0.0464%),數學上必然讓所有 far 距離物件
# 100% 被濾掉,導致訓練資料裡「一句話講兩個 position」的雙子句樣本從 94.7%
# 崩到 23.8%(見 Phase3/Day32/task_a_recall_gap_analysis.md)。改成 0.025%——
# 多值 sensitivity(0.015/0.02/0.025/0.03%)四個候選都通過「零框圖<5%、平均
# 框數合理、far 保留率不為 0」三條準則,0.025% 在其中 far 保留率排第二高
# (39.7%/47.2%)、比 0.015%/0.02% 濾得更接近 Day35 原本想濾掉的「人眼看不到
# 的 tiny 框」強度、又比 0.03% 更保守。詳細對照表見
# Phase3/Day32/threshold_sensitivity_v2.md。
#
# STATIC_CONTEXT_CLASSES(light/hydrant/sign)不影響 caption 文字本身(只影響
# has_static_context 這個 metadata 欄位,不會被學進 decoder),不列入這個過濾。
# ---------------------------------------------------------------------------
GLOBAL_MIN_AREA_PCT = 0.025


COCO_TO_FLIR_ALIAS = {
    "person": "person",
    "bicycle": "bike",
    "car": "car",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "skateboard": "skateboard",
    "traffic light": "light",
    "fire hydrant": "hydrant",
    "stop sign": "sign",
}


def plural_of(name):
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    return name + "s"


def indefinite_article(name):
    return "an" if name[0].lower() in "aeiou" else "a"


# ---------------------------------------------------------------------------
# 2. 位置 / 距離
#
# 位置用畫面左右三等分判斷(簡單起點,之後可以再細分)。
# 距離用 bbox 面積佔畫面比例的「資料集實測分位數」當門檻,
# 不是憑感覺猜的數字——先跑 compute_area_thresholds() 算出來再用。
# 兩者內部都用英文代碼("left"/"right"/"ahead"、"near"/"mid"/"far"),
# 實際句子用詞在 build_caption() 裡的 POSITION_PHRASE / DISTANCE_PHRASE 統一管理。
# ---------------------------------------------------------------------------

def position_label(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = x + w / 2
    ratio = cx / img_w
    if ratio < 1 / 3:
        return "left"
    elif ratio > 2 / 3:
        return "right"
    return "ahead"


def distance_label(bbox, img_w, img_h, near_thresh, far_thresh):
    """
    三個距離桶都要明確標詞,不要留白——v0.1 時中距離留白,
    跟近距離句子並列時讀起來像是在重複前一句,容易被誤會成合併 bug 沒修好。
    """
    x, y, w, h = bbox
    area_ratio = (w * h) / (img_w * img_h)
    if area_ratio >= near_thresh:
        return "near"
    elif area_ratio <= far_thresh:
        return "far"
    return "mid"


def compute_area_thresholds(coco, dynamic_cat_ids, near_pct=75, far_pct=25):
    """用資料集裡所有動態類別 bbox 的面積比例分布,取 25/75 百分位數當門檻。"""
    img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    ratios = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in dynamic_cat_ids:
            continue
        w_img, h_img = img_wh[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        ratios.append((w * h) / (w_img * h_img))
    ratios.sort()
    n = len(ratios)
    if n == 0:
        return 0.05, 0.005  # fallback,理論上不會走到這裡
    near = ratios[int(n * near_pct / 100)]
    far = ratios[int(n * far_pct / 100)]
    return near, far


# ---------------------------------------------------------------------------
# 3. 場景 metadata:image-level extra_info 裡有 hours(day/night)、
#    weather、scene(highway/city...)——這是意外發現的額外資產,
#    可以拿來豐富句子語境,不只靠 bbox。
# ---------------------------------------------------------------------------

def scene_prefix(image_extra_info):
    hours = image_extra_info.get("hours")
    weather = image_extra_info.get("weather")
    parts = []
    if hours == "night":
        parts.append("Night")
    weather_en = WEATHER_MAP.get(weather, weather)
    if weather_en:
        parts.append(weather_en.capitalize())
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 4. 主要生成邏輯
# ---------------------------------------------------------------------------

POSITION_PHRASE = {"left": "on the left", "right": "on the right", "ahead": "ahead"}
DISTANCE_PHRASE = {"near": "nearby", "mid": "at medium distance", "far": "in the distance"}


def distance_phrase_for(dist):
    """
    v0.7 Level 1 修正:mid 距離不再強制帶距離詞,只留位置片語。
    v0.6 三個桶都寫死一定要輸出,但「at medium distance」資訊量很低,
    反而讓近/中/遠三種距離讀起來一樣冗長。near/far 兩個極端保留
    (nearby / in the distance,這才是真的有分辨力的訊號),mid 直接省略。
    """
    if dist == "mid":
        return ""
    return DISTANCE_PHRASE[dist]


def count_to_subject(en_name, count):
    """
    v0.7 Level 2:類別總數 -> 主詞映射。1/2/3 各自明講,4-6 用 several,
    7+ 用 many,比 v0.6 只有 1/2/several 三檔更貼近「5 台車」這種畫面
    實際會出現的量級,不會被 several 一個詞把 4 台跟 9 台混在一起講。
    """
    if count == 1:
        return f"{indefinite_article(en_name)} {en_name}"
    elif count == 2:
        return f"two {plural_of(en_name)}"
    elif count == 3:
        return f"three {plural_of(en_name)}"
    elif count <= 6:
        return f"several {plural_of(en_name)}"
    else:
        return f"many {plural_of(en_name)}"


def aggregate_by_class(objects):
    """
    objects: [(cat_name, en_name, position, distance, area)]

    v0.7 Level 2 取代 v0.6 的 group_objects()。v0.6 用
    (en_name, position, distance) 三維分組,位置切 left/right/ahead、
    距離切 near/mid/far,最多 9 組——同類別物件只要散在不同格子裡就不會
    合併,build_caption() 對合併結果取 top-2 時常把同類別其他 instance
    全部丟掉。實測發現「畫面明顯 5 台車,caption 只講 2 台」就是這樣來的。

    改成只依 en_name 分組:每個類別算總數(class_totals),
    再從該類別所有 instance 裡找 area 最大(=最近)的那個,
    取它的 position/distance 當這個類別的位置補述(每類最多一個),
    不再逐一列舉「哪個位置有幾個」。

    回傳依「總數多者優先,同總數依最近距離(area 較大)優先」排序的
    [(en_name, {count, max_area, pos, dist, cat_name}), ...]。
    """
    class_data = {}
    for cat_name, en_name, pos, dist, area in objects:
        d = class_data.setdefault(en_name, {
            "cat_name": cat_name, "count": 0, "max_area": -1,
            "pos": pos, "dist": dist,
        })
        d["count"] += 1
        if area > d["max_area"]:
            d["max_area"] = area
            d["pos"] = pos
            d["dist"] = dist

    ranked = sorted(
        class_data.items(),
        key=lambda kv: (-kv[1]["count"], -kv[1]["max_area"]),
    )
    return ranked


def build_class_clause(en_name, info, rng):
    """
    每個類別最多一個位置補述,取該類別裡最近(area 最大)的 instance。

    count==1 時只有一個物件,「the nearest」語感很怪(沒有「最近」可比較),
    直接把距離/位置片語接在主詞後面,不用逗號補述。count>=2 才用逗號
    接補述子句,並在「the nearest X」/「one <距離> X」兩種說法之間
    隨機選一種(Level 3 句型多樣化),用呼叫端傳入、以 image_id 播種的
    rng 保證同一張圖每次重跑選到同一種說法。
    """
    subject = count_to_subject(en_name, info["count"])
    pos_phrase = POSITION_PHRASE[info["pos"]]
    dist_phrase = distance_phrase_for(info["dist"])

    if info["count"] == 1:
        parts = [subject]
        if dist_phrase:
            parts.append(dist_phrase)
        parts.append(pos_phrase)
        return " ".join(parts)

    style = rng.choice(("nearest", "one"))
    if style == "nearest":
        return f"{subject}, the nearest {pos_phrase}"
    if dist_phrase:
        return f"{subject}, one {dist_phrase} {pos_phrase}"
    return f"{subject}, one {pos_phrase}"


def build_caption(objects, image_extra_info, image_id):
    """
    objects: [(cat_name, en_name, position, distance, area)],
             已依單一物件 area 由大到小排序(合併前的原始清單)。

    v0.7 Level 2+3 重寫:類別總計(class_totals)為主軸決定要不要講、
    講幾隻,每個類別最近 instance 的位置只當補述,不再依位置/距離拆組。

    v0.10 Day38 caption-completeness bug 修正:v0.7 原本在最大類別
    count>=5 時把 ranked 砍到只剩 1 類(理由是「量大時寧可只完整講這一類,
    也不硬塞第二類」),但這個規則只看最大類別的數量,沒有排除「主類別
    數量多 + 副類別同時存在」的情況——只要畫面裡有 5 台以上車,即使同時
    有一台佔畫面 3% 的顯眼行人也會被整個丟掉,不是單純的 v0.6 top-2 截斷
    問題。全體掃描(Phase3/caption_completeness_bug.md)量到 84-88% 的
    多類別圖片至少漏講一個類別,絕大多數就是這個 count>=5 分支造成的。
    改成一律取 top-2(拿掉 count>=5 的特例),不再無條件丟棄第二類。

    v0.11(2026-09-04 Jack 指定,caption_fusion 專用路徑):句子改成不設
    類別數上限,合併後有幾個 class 就講幾個——拿掉下面原本的
    `ranked = ranked[:2]`。這是套用在 build_caption() 這個共用函式上的
    改動,train/val/test 的 GT caption 產出、RGB 版本都共用同一份程式碼,
    但這次只重跑了 caption_fusion 的 10 筆樣本,沒有重新產生整個
    dataset 的 captions_{split}.jsonl,所以既有的訓練資料/checkpoint
    不受影響;如果之後要真的拿掉上限重新產生全量 GT,需要另外評估對
    已訓練模型的影響。

    Level 3 模板:2 個子句時用分號接(模板 A);只剩 1 個子句時
    (畫面裡只有一個類別)自然變成模板 C,不需要另外判斷。模板 B
    (scene 後置)先跳過,不動日夜前綴的訓練信號。沒有任何動態物件的
    畫面直接跳過。
    """
    if not objects:
        return None

    ranked = aggregate_by_class(objects)

    rng = random.Random(image_id)
    clauses = [build_class_clause(en_name, info, rng) for en_name, info in ranked]

    prefix = scene_prefix(image_extra_info)
    sentence = "; ".join(clauses)
    if prefix:
        sentence = f"{prefix}: {sentence}"

    # 只把整句第一個字母大寫,分號接續的後續子句維持小寫(英文文法慣例),
    # 不要每個子句自己各自大寫開頭。
    sentence = sentence[0].upper() + sentence[1:]

    return sentence + "."


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def load_yolo_detections(detections_path):
    """Day35: detections_{split}.jsonl(Day35/run_yolo_inference.py 產出)讀成
    file_name -> record 的 dict,record 裡的 bbox 已經是 [x,y,w,h],跟 FLIR
    coco.json annotation 的 bbox 格式一致,可以直接餵給 position_label /
    distance_label,不用另外轉換。"""
    by_filename = {}
    with open(detections_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_filename[r["file_name"]] = r
    return by_filename


def main(split, out_path, sample_n=None, long_tail_ref_split=None,
         source="gt", detections_path=None, filter_tiny=False):
    # Day39:test split 的資料夾命名跟 train/val 不同(video_thermal_test,
    # 不是 images_thermal_test)——這個資料集是後來才拿到的獨立 held-out
    # 集合,原本的 train/val 抽取流程沒有替它取一致的資料夾名稱。這裡只加
    # 這一個特例,不改其他任何路徑邏輯。
    split_dir = "video_thermal_test" if split == "test" else f"images_thermal_{split}"
    root = Path.home() / "ai-transition-2026" / "thermal_dataset" / split_dir
    coco = json.load(open(root / "coco.json"))

    # Day35 --source yolo:偵測結果來自 YOLO,不是 coco.json 的 annotations,
    # 但影像清單、image-level extra_info(day/night、weather)、long-tail
    # 門檻、near/far 距離門檻——這些都還是要用這個 split 的 coco.json 算,
    # 跟 GT 版共用同一套,才能讓兩版 caption 可以互相比較。
    yolo_dets = None
    if source == "yolo":
        if not detections_path:
            raise ValueError("--source yolo 需要搭配 --detections <path>")
        yolo_dets = load_yolo_detections(detections_path)
        print(f"[info] source=yolo,讀入 {len(yolo_dets)} 筆偵測結果(from {detections_path}）")

    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    name2id = {c["name"]: c["id"] for c in coco["categories"]}
    dynamic_ids = {c["id"] for c in coco["categories"] if c["name"] in DYNAMIC_CLASSES}
    static_ids = {c["id"] for c in coco["categories"] if c["name"] in STATIC_CONTEXT_CLASSES}

    # long-tail 門檻:決定哪些 DYNAMIC_CLASSES 類別要收進 fallback。
    #
    # 🔴 2026-08-28 bug 修正:原本 train/val 各自用「這個 split 自己的」
    # annotation counts 判斷,導致同一個類別在樣本數大的 train 保留真名
    # (bicycle/bus/truck...),在樣本數小很多的 val 因為沒過門檻全被改叫
    # object——train/val 的 caption 詞彙表因此不一致,評估時看起來像模型在
    # 瞎講一堆 GT 沒有的類別,實際上是基準本身的 bug。用 --long-tail-ref-split
    # 指定一個「共用基準」的 split(預設沿用自己這個 split,向後相容;
    # 修 bug 時傳 "train")算 counts,讓所有 split 共用同一份 long_tail_names。
    ref_split = long_tail_ref_split or split
    if ref_split != split:
        ref_root = Path.home() / "ai-transition-2026" / "thermal_dataset" / f"images_thermal_{ref_split}"
        ref_coco = json.load(open(ref_root / "coco.json"))
        ref_id2name = {c["id"]: c["name"] for c in ref_coco["categories"]}
        cat_counts_by_name = Counter(ref_id2name[a["category_id"]] for a in ref_coco["annotations"])
        print(f"[info] long-tail 門檻改用「{ref_split}」split 的 counts 當基準(跨 split 共用,修正 long-tail bug)")
    else:
        cat_counts_by_name = Counter(id2name[a["category_id"]] for a in coco["annotations"])

    long_tail_names = {
        name for name in DYNAMIC_CLASSES
        if cat_counts_by_name.get(name, 0) < LONG_TAIL_THRESHOLD
    }
    if long_tail_names:
        detail = ", ".join(
            f"{name}({cat_counts_by_name.get(name, 0)})" for name in sorted(long_tail_names)
        )
        print(f"[info] long-tail 併入「{LONG_TAIL_LABEL}」: {detail}")

    near_thresh, far_thresh = compute_area_thresholds(coco, dynamic_ids)
    print(f"[info] near_thresh={near_thresh:.4f}, far_thresh={far_thresh:.4f}")

    if source == "yolo":
        print("[info] COCO_TO_FLIR_ALIAS (YOLO COCO class name -> Day32 FLIR class name):")
        for coco_name, flir_name in COCO_TO_FLIR_ALIAS.items():
            en_name = DYNAMIC_CLASSES.get(flir_name) or STATIC_CONTEXT_CLASSES.get(flir_name)
            tail = " [long-tail -> object]" if flir_name in long_tail_names else ""
            print(f"    {coco_name!r:16s} -> {flir_name!r:14s} -> caption word {en_name!r}{tail}")

    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    image_ids = list(img_meta.keys())
    if sample_n:
        random.seed(0)
        image_ids = random.sample(image_ids, min(sample_n, len(image_ids)))

    results = []
    n_missing_from_detections = 0
    for img_id in image_ids:
        im = img_meta[img_id]
        w_img, h_img = im["width"], im["height"]

        dyn_objs = []
        static_present = set()

        if source == "gt":
            anns = anns_by_img.get(img_id, [])
            for ann in anns:
                cat_name = id2name[ann["category_id"]]
                if ann["category_id"] in static_ids:
                    static_present.add(cat_name)
                    continue
                if ann["category_id"] not in dynamic_ids:
                    continue

                # 過濾嚴重遮蔽的物件,避免生成不可靠的描述
                occluded = ann.get("extra_info", {}).get("occluded", "") or ""
                if occluded == OCCLUDED_DIFFICULT:
                    continue

                x, y, w, h = ann["bbox"]
                area = w * h

                # Day35 Task 5/6:tiny-bbox 過濾(--filter-tiny 開啟時生效,
                # Task 6 改成單一全域門檻,不再依 class 查表)
                if filter_tiny:
                    area_pct = 100 * area / (w_img * h_img)
                    if area_pct < GLOBAL_MIN_AREA_PCT:
                        continue

                pos = position_label(ann["bbox"], w_img, h_img)
                dist = distance_label(ann["bbox"], w_img, h_img, near_thresh, far_thresh)

                if cat_name in long_tail_names:
                    en_name = LONG_TAIL_LABEL
                else:
                    en_name = DYNAMIC_CLASSES.get(cat_name, LONG_TAIL_LABEL)
                dyn_objs.append((cat_name, en_name, pos, dist, area))
        else:  # source == "yolo"
            # coco.json 的 file_name 帶 "data/" 前綴,detections jsonl 只存
            # bare filename(run_yolo_inference.py 用 img_path.name),用
            # Path(...).name 對齊,不要直接比對整個路徑字串。
            det_record = yolo_dets.get(Path(im["file_name"]).name)
            if det_record is None:
                n_missing_from_detections += 1
                det_record = {"detections": []}
            for det in det_record["detections"]:
                coco_name = det["class_name"]
                cat_name = COCO_TO_FLIR_ALIAS.get(coco_name)
                if cat_name is None:
                    continue  # 理論上不會發生,KEEP_CLASSES 跟這裡是同一份對照表
                if cat_name in STATIC_CONTEXT_CLASSES:
                    static_present.add(cat_name)
                    continue
                if cat_name not in DYNAMIC_CLASSES:
                    continue

                x, y, w, h = det["bbox"]
                area = w * h
                pos = position_label(det["bbox"], w_img, h_img)
                dist = distance_label(det["bbox"], w_img, h_img, near_thresh, far_thresh)

                if cat_name in long_tail_names:
                    en_name = LONG_TAIL_LABEL
                else:
                    en_name = DYNAMIC_CLASSES.get(cat_name, LONG_TAIL_LABEL)
                dyn_objs.append((cat_name, en_name, pos, dist, area))

        dyn_objs.sort(key=lambda o: -o[4])  # 面積大(近)的排前面

        caption = build_caption(dyn_objs, im.get("extra_info", {}), img_id)
        if caption is None:
            continue

        results.append({
            "file_name": im["file_name"],
            "caption": caption,
            "num_objects": len(dyn_objs),
            "has_static_context": sorted(static_present),  # 留著當中繼資料,不進句子
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if source == "yolo" and n_missing_from_detections:
        print(f"[warn] {n_missing_from_detections} 張圖在 detections jsonl 裡找不到,"
              f"當成 zero-detection 處理")
    print(f"[done] {len(results)} captions written to {out_path}")
    print("\n=== 隨機抽 10 句檢查 ===")
    for r in random.sample(results, min(10, len(results))):
        print(f"- {r['caption']}   ({r['file_name']})")


if __name__ == "__main__":
    print(f"[script version] {SCRIPT_VERSION}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None,
                         help="不指定時依 --split/--source 自動命名:"
                              "captions_{split}.jsonl(gt)或 captions_{split}_yolo.jsonl(yolo)")
    # 不打 --sample 就是真的全量;要小樣本測試才明確指定數字(例如 --sample 2000)。
    parser.add_argument("--sample", type=int, default=None,
                         help="先跑小樣本檢查品質(例如 --sample 2000),"
                              "確認沒問題後不要加這個參數,才會跑全量")
    parser.add_argument("--long-tail-ref-split", default=None, choices=["train", "val"],
                         help="long-tail 門檻用哪個 split 的 counts 當基準,"
                              "不指定就沿用自己這個 split(舊行為,向後相容)。"
                              "修 train/val 詞彙不一致的 bug 時,對 val 傳 train。")
    parser.add_argument("--source", default="gt", choices=["gt", "yolo"],
                         help="Day35:caption 的物件來源。gt(預設,向後相容)=沿用"
                              "coco.json annotations;yolo=改吃 Day35 run_yolo_inference.py "
                              "產出的 detections_{split}.jsonl(需搭配 --detections)。")
    parser.add_argument("--detections", default=None,
                         help="--source yolo 時,detections_{split}.jsonl 的路徑。")
    parser.add_argument("--filter-tiny", action="store_true",
                         help="Day35 Task 5:--source gt 時開啟 per-class tiny-bbox 過濾"
                              "(GLOBAL_MIN_AREA_PCT,見常數定義上方註解)。"
                              "對 --source yolo 無效(YOLO 版已決定不用,不動它)。")
    args = parser.parse_args()

    if args.filter_tiny and args.source != "gt":
        raise ValueError("--filter-tiny 只對 --source gt 有效")

    out_path = args.out
    if out_path is None:
        if args.source == "yolo":
            suffix = "_yolo"
        elif args.filter_tiny:
            suffix = "_filtered"
        else:
            suffix = ""
        out_path = f"captions_{args.split}{suffix}.jsonl"

    main(args.split, out_path, args.sample, args.long_tail_ref_split,
         source=args.source, detections_path=args.detections, filter_tiny=args.filter_tiny)