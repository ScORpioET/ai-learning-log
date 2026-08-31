# Position-Class Binding Accuracy

GT/生成句子都 parse 成 [(position, distance, class), ...],同 position 逐一比對 class 對不對。
三桶:class-position 正確 / class-position 錯位(方位對、類別錯,Jack 抓到的這種)/ position 缺失或多餘(GT 有生成沒有,或反過來)。

## best_model.pt (GT full, v0.6 template)

- n = 1097 筆
- GT 句子 clause parse 成功率: 100.0%(0 筆句子裡有至少 1 個 clause parse 不出來)
- 生成句子 clause parse 成功率: 99.4%(13 筆句子裡有至少 1 個 clause parse 不出來)

| 桶 | 數量 | 佔比(相對 position-matched 或全部) |
|---|---|---|
| class-position 正確 | 839 | 67.9% (相對同位置有配對的) |
| class-position 錯位 | 397 | 32.1% (相對同位置有配對的) |
| position 缺失(GT 有生成沒有) | 906 | — |
| position 多餘(生成有 GT 沒有) | 939 | — |

**Position-Class Binding Accuracy = 67.9%** (同一個方位裡,GT 有物件、生成也有物件的情況下,類別講對的比例)
**Class-Position 錯位率 = 32.1%**(同一個方位,兩邊都有物件,但類別不一樣——這是 Jack 抓到的那種問題)

**⚠️ 補充統計:position 缺失/多餘的量遠大於「同位置類別錯」**——missing(906)+extra(939)
合計 1845,是 mismatched(397)的 4.6 倍。換算:
- position recall(GT 講的位置,GEN 有沒有講到同一個位置——不論類別對不對) = 57.7%
- position precision(GEN 講的位置,有沒有對到 GT 講的位置) = 56.8%

也就是說:**模型「講錯位置」(整個 position 對不上,不是同位置但類別錯)比「同位置
講錯類別」更常見。** Jack 抓到的「同位置類別錯」問題確實存在且量不小(32.1% 錯位
率),但更大宗的問題是模型描述的空間結構本來就常常對不上 GT(只有約 57-58% 的
GT 位置陳述有被生成句子命中,無論類別對不對)。

## best_model_filtered.pt (GT filtered, v0.7+ template)

- n = 1088 筆
- GT 句子 clause parse 成功率: 100.0%(0 筆句子裡有至少 1 個 clause parse 不出來)
- 生成句子 clause parse 成功率: 98.9%(15 筆句子裡有至少 1 個 clause parse 不出來)

| 桶 | 數量 | 佔比(相對 position-matched 或全部) |
|---|---|---|
| class-position 正確 | 364 | 66.1% (相對同位置有配對的) |
| class-position 錯位 | 187 | 33.9% (相對同位置有配對的) |
| position 缺失(GT 有生成沒有) | 769 | — |
| position 多餘(生成有 GT 沒有) | 805 | — |

**Position-Class Binding Accuracy = 66.1%** (同一個方位裡,GT 有物件、生成也有物件的情況下,類別講對的比例)
**Class-Position 錯位率 = 33.9%**(同一個方位,兩邊都有物件,但類別不一樣——這是 Jack 抓到的那種問題)

**⚠️ 補充統計**:missing(769)+extra(805)合計 1574,是 mismatched(187)的 8.4 倍
(比 GT full 版的 4.6 倍更懸殊)。
- position recall = 41.7%
- position precision = 40.6%

filtered 版的 position recall/precision(41.7%/40.6%)明顯低於 GT full 版
(57.7%/56.8%)——推測跟 filtered captions 平均講的物件數變少(Task2 量到生成長度
從 21 token 降到 11.6 token)有關:句子變短、每句能命中的「位置」機會也變少,
分母結構不同,不是單純「filtered 模型比較爛」,但這點沒有進一步拆解驗證,
標「未確認」。

## 15 張具體案例(含錯位 / 正確 / 缺失多餘各種情況)

### 案例 1 [best_model.pt (GT full, v0.6 template)] — data/video-kBvcuDMtYv2Z4kmXi-frame-005107-LMqyE7tCh3CyAF5AG.jpg
- GT : Nearby on the right there is a car; nearby on the left there is an object.
- GEN: Nearby on the right there is a pedestrian; nearby ahead there is a car.
- correct=0, mismatched=1, missing=1, extra=1
  - ⚠️ 錯位: position=on the right, GT=car, GEN=pedestrian

### 案例 2 [best_model.pt (GT full, v0.6 template)] — data/video-WcRvuoSJHRxDoTmaT-frame-000600-6N23BJBuRRKyAgMfw.jpg
- GT : At medium distance ahead there is a car; at medium distance on the left there is a car.
- GEN: At medium distance on the left there are several motorcycles; at medium distance on the left there is a car.
- correct=0, mismatched=1, missing=1, extra=1
  - ⚠️ 錯位: position=on the left, GT=car, GEN=motorcycle

### 案例 3 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-JhYLiFCieHQHaY8o7-frame-003255-GvoLQh8z8KtnYLRu4.jpg
- GT : Night: a pedestrian on the left.
- GEN: Several cars, the nearest on the left.
- correct=0, mismatched=1, missing=0, extra=0
  - ⚠️ 錯位: position=on the left, GT=pedestrian, GEN=car

### 案例 4 [best_model.pt (GT full, v0.6 template)] — data/video-kBvcuDMtYv2Z4kmXi-frame-005575-iWKJ8ZJdJZrADfbCw.jpg
- GT : Nearby on the left there is an object; nearby on the left there is a car.
- GEN: Nearby on the left there is a bus; nearby ahead there is a car.
- correct=0, mismatched=1, missing=1, extra=1
  - ⚠️ 錯位: position=on the left, GT=object, GEN=bus

### 案例 5 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-kBvcuDMtYv2Z4kmXi-frame-013128-msrsNTCxuKhDAwxB5.jpg
- GT : Many pedestrians, the nearest ahead.
- GEN: Two cars, one nearby ahead.
- correct=0, mismatched=1, missing=0, extra=0
  - ⚠️ 錯位: position=ahead, GT=pedestrian, GEN=car

### 案例 6 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-k5bTJAiyEgHismN7Y-frame-006936-SDyxnfAzpe8qCukix.jpg
- GT : Several cars, the nearest on the right; two bicycles, the nearest on the left.
- GEN: Two buses, one on the left; a bus nearby ahead.
- correct=0, mismatched=1, missing=1, extra=1
  - ⚠️ 錯位: position=on the left, GT=bicycle, GEN=bus

### 案例 7 [best_model.pt (GT full, v0.6 template)] — data/video-JhYLiFCieHQHaY8o7-frame-005020-m5j9r2tAuLHSR8h6M.jpg
- GT : Night: nearby on the right there is a pedestrian; at medium distance on the right there is an object.
- GEN: Night: nearby ahead there is a pedestrian; nearby on the right there is a car.
- correct=0, mismatched=1, missing=1, extra=1
  - ⚠️ 錯位: position=on the right, GT=pedestrian, GEN=car

### 案例 8 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-k5bTJAiyEgHismN7Y-frame-002023-4jmSzonSPNSCmK2h4.jpg
- GT : Two cars, the nearest ahead; a pedestrian nearby on the left.
- GEN: Two pedestrians, one nearby on the left; a car nearby ahead.
- correct=2, mismatched=0, missing=0, extra=0

### 案例 9 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-YQpCvGJxowy9uhkCw-frame-004800-2amXNgZwPhgBvdaAv.jpg
- GT : A car nearby ahead.
- GEN: Many cars, one nearby ahead.
- correct=1, mismatched=0, missing=0, extra=0

### 案例 10 [best_model.pt (GT full, v0.6 template)] — data/video-Qk8msXvMopoYNDdco-frame-002180-p9WBoi2TtaRfiFxNC.jpg
- GT : Nearby on the right there is a car; nearby on the left there are several cars.
- GEN: Nearby on the left there is a car; nearby on the right there are two cars.
- correct=2, mismatched=0, missing=0, extra=0

### 案例 11 [best_model.pt (GT full, v0.6 template)] — data/video-57kWWRyeqqHs3Byei-frame-005492-zHZZC8KG4nMSS6hzL.jpg
- GT : Nearby on the right there is a car; nearby ahead there is a car.
- GEN: Nearby on the right there are two cars; nearby ahead there is a car.
- correct=2, mismatched=0, missing=0, extra=0

### 案例 12 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-kBvcuDMtYv2Z4kmXi-frame-010105-6dyDvCwFmpKGqAHSy.jpg
- GT : Many pedestrians, the nearest on the right.
- GEN: Many cars, the nearest ahead.
- correct=0, mismatched=0, missing=1, extra=1

### 案例 13 [best_model_filtered.pt (GT filtered, v0.7+ template)] — data/video-zp8ed5vPKfAJ2fKWh-frame-002043-usTBaskqpEbLoR8wW.jpg
- GT : Many pedestrians, the nearest on the right.
- GEN: Several cars, one nearby on the left.
- correct=0, mismatched=0, missing=1, extra=1

### 案例 14 [best_model.pt (GT full, v0.6 template)] — data/video-nMfT5vK8MfEEjQ44W-frame-005086-yfAZ4qGmZS34BKQ46.jpg
- GT : Nearby ahead there is a car; nearby on the left there are several cars.
- GEN: Nearby on the right there is a vehicle; nearby on the right there is a truck.
- correct=0, mismatched=0, missing=2, extra=2

### 案例 15 [best_model.pt (GT full, v0.6 template)] — data/video-AP7PvpujjZZGLnsJt-frame-002400-bmMDKoxLgabnmMhKT.jpg
- GT : Nearby on the right there is a car; nearby on the left there is a car.
- GEN: Nearby on the right there is a car; nearby ahead there are two cars.
- correct=1, mismatched=0, missing=1, extra=1

