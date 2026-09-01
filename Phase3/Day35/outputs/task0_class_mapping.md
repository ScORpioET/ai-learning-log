# Task 0-B: class mapping 三方對照

## 三個來源

**1. Day32 generate_captions.py 實際用的 class name**(`thermal_dataset/generate_captions.py`)

```python
DYNAMIC_CLASSES = {
    "person": "pedestrian", "bike": "bicycle", "motor": "motorcycle",
    "car": "car", "bus": "bus", "truck": "truck",
    "other vehicle": "vehicle", "train": "train",
    "skateboard": "skateboard", "stroller": "stroller", "scooter": "scooter",
}
STATIC_CONTEXT_CLASSES = {
    "light": "traffic light", "hydrant": "fire hydrant", "sign": "sign",
}
```
(key = FLIR coco.json 的 category name,value = 生成句子用的英文詞;
train 實測 histogram 見 script comment:car 73623 / person 50478 / sign 20770 /
light 16198 / bike 7237 / bus 2245 / other vehicle 1373 / motor 1116 /
hydrant 1095 / truck 829 / skateboard 29 / stroller 15 / scooter 15 / deer 8 /
train 5 / dog 4,rider 0 筆)

**2. FLIR ADAS v2 原始 categories**(`images_thermal_{train,val}/coco.json`,
train/val 兩份完全一致,80 個 category):
person, bike, car, motor, airplane, bus, train, truck, boat, light, hydrant,
sign, parking meter, bench, bird, cat, dog, deer, sheep, cow, elephant, bear,
zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis,
snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
stroller, rider, scooter, vase, scissors, face, other vehicle, license plate

**3. COCO 80 類**(`YOLO('yolov8m.pt').names`):標準 COCO,含 bicycle / car /
motorcycle / bus / train / truck / traffic light / fire hydrant / stop sign /
skateboard 等。

## 三方對照表

| Day32 script FLIR key | Day32 en_name (別名) | FLIR taxonomy | COCO 對應 | id | YOLO 偵測? |
|---|---|---|---|---|---|
| person | pedestrian | person | person | 0 | ✓ |
| bike | bicycle | bike | bicycle | 1 | ✓ (見下方 flag) |
| car | car | car | car | 2 | ✓ |
| motor | motorcycle | motor | motorcycle | 3 | ✓ (見下方 flag) |
| bus | bus | bus | bus | 5 | ✓ |
| truck | truck | truck | truck | 7 | ✓ |
| other vehicle | vehicle | other vehicle | (無對應) | — | ✗ |
| train | train (long-tail→object) | train | train | 6 | ✓ |
| skateboard | skateboard (long-tail→object) | skateboard | skateboard | 36 | ✓ |
| stroller | stroller (long-tail→object) | stroller | (無對應) | — | ✗ |
| scooter | scooter (long-tail→object) | scooter | (無對應) | — | ✗ |
| light | traffic light (static) | light | traffic light | 9 | ✓ |
| hydrant | fire hydrant (static) | hydrant | fire hydrant | 10 | ✓ |
| sign | sign (static) | sign | stop sign | 11 | △ 語意變窄,見下方 flag |

## Q1: Day32 有用 + YOLO 能偵測 → 列入 KEEP_CLASSES
person, bike(bicycle), car, motor(motorcycle), bus, truck, train, skateboard,
light(traffic light), hydrant(fire hydrant), sign(stop sign)

## Q2: Day32 有用 + YOLO 偵測不到(FLIR 有 / COCO 沒有)→ YOLO 版必然缺失
- **other vehicle**(train 實測 1373 筆,量不小):COCO 80 類沒有對應的「其他車輛」
  通用類別,YOLO 版完全偵測不到,caption 也不會出現這個詞。
- **stroller**(15 筆,long-tail 本來就會被併成 "object"):COCO 沒有 stroller。
- **scooter**(15 筆,long-tail 本來就會被併成 "object"):COCO 沒有 scooter。

stroller/scooter 損失不大(本來就低於 long-tail 門檻 500,GT 版也講不出真名),
但 other vehicle 損失是實質的——GT 版 train 有 1373 筆會講出 "vehicle",
YOLO 版這個詞會完全消失。這是 pretrained COCO 模型的代價,寫進最終 summary。

## Q3: COCO 能偵測 + Day32 不用 → 不列入 KEEP_CLASSES
airplane, boat, parking meter, bench, bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase,
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove,
surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
→ 全部不列入 KEEP_CLASSES,避免 GT 沒有、YOLO 卻偵測出來的雜訊類別汙染下游 caption。

（附註:dog 在 FLIR taxonomy 裡有,COCO 也有,但 Day32 script 沒放進
DYNAMIC_CLASSES/STATIC_CONTEXT_CLASSES,train 實測只有 4 筆,所以照 Q3 規則不列入。）

## ⚠️ 語意模糊 / 待 Jack 確認的兩點

1. **bike / motor 是否真的分別對應 bicycle / motorcycle,不是同一物的兩種別名?**
   本地找不到 FLIR 官方 README(`thermal_dataset/` 下沒有 dataset card,coco.json
   的 `info` 欄位也沒有 description)。WebSearch 查到 FLIR ADAS v2(2022-01-19)
   官方擴充到 15 類的 category 清單裡明確同時列出 "bike" 和 "motorcycle" 為
   **兩個不同的類別**（來源:搜尋結果摘要引用 "bike, car, motorcycle, bus, train,
   truck, traffic light, fire hydrant, street sign, dog, skateboard, stroller,
   scooter, other vehicle" 15 類清單）,這與「bike=腳踏車、motor=機車,兩者不是
   同一物」的假設一致,也符合 ADAS 情境下需要分辨人力/動力兩輪車的常理。
   但這是二手摘要,不是我直接讀到 FLIR 官方 PDF/README 原文,信心不是 100%。
   **→ 已按 bike=bicycle(id 1)/ motor=motorcycle(id 3)分別對應執行,但這條
   標記為「待 Jack 確認」,如果之後看視覺化結果或有官方文件發現不對,這裡要重跑。**

2. **FLIR "sign"(20770 筆,量很大)只能對應到 COCO 的 "stop sign"**,COCO 80 類
   沒有通用的「路牌/標誌」類別。FLIR 的 sign 應該涵蓋各種交通標誌(限速、讓路、
   方向指示等),不是只有停車再開。這代表 YOLO 版偵測到的 sign 數量會遠低於 GT
   (只有真的長得像美式八角形 STOP 標誌的才會被 COCO stop-sign 偵測到),不是
   bug,是 pretrained COCO 類別粒度不夠細的已知限制。sign 是 STATIC_CONTEXT
   (不影響 caption 文字本身,只影響 has_static_context metadata),影響範圍比
   other vehicle 小,但仍要寫進 summary。

## 最終 KEEP_CLASSES(COCO id -> COCO name,給 run_yolo_inference.py 用)

```python
KEEP_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    36: "skateboard",
}
```

alias 對齊(COCO name -> Day32 en_name)留到 Task 3 generate_captions.py
`--source yolo` 那一層做,不在這裡改名——原因见 Task 3 段落。
