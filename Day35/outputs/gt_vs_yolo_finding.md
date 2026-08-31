# GT vs YOLO 質性 Finding(30 張視覺化圖看完後)

範圍:val split,IoU>0.5 + 同 class 才算配對成功。原始統計見
[gt_vs_yolo_summary.md](gt_vs_yolo_summary.md),30 張圖在 `day35_gt_vs_yolo/`。

---

## Q1: YOLO 漏偵測的 GT 框,bbox size 分布怎樣?是不是主要集中在 tiny/small?

**是,而且非常集中。** 全 val 範圍(表 B,見 summary.md):

| size | GT 總數 | YOLO 匹配到 | 匹配率 |
|---|---|---|---|
| tiny (<0.5%) | 13625 | 3265 | 24.0% |
| small (0.5-2%) | 1978 | 1597 | 80.7% |
| medium (2-8%) | 864 | 769 | 89.0% |
| large (>8%) | 160 | 133 | 83.1% |

tiny 佔全部 GT 框的 82%(13625/16627),匹配率只有 24%;small/medium/large
三桶匹配率都在 80-89%。直方圖([gt_vs_yolo_size_hist.png](gt_vs_yolo_size_hist.png))
上紅色(漏掉)分布明顯集中在 0.5% 分界線左側,綠色(匹配到)集中在右側,肉眼可
直接印證這個 hypothesis 成立。

**3 張具體 image_id + 漏掉的 GT bbox 面積比:**

1. **image_id 5895**(`video-mKfYgxHA8ZZmXvw56-frame-000295`):GT 18 框(13 sign,
   5 car),YOLO 只出 5 框。漏掉的 13 個 sign 面積比全部落在 0.008%-0.323% 之間
   (13 個裡最大的才 0.32%),清一色 tiny。看視覺化圖,這些 sign 是同一根高速公路
   指示牌結構被拆成一串小格子分別標註,不是 13 個獨立標誌牌。
2. **image_id 5749**(`video-kBvcuDMtYv2Z4kmXi-frame-000298`):GT 36 框(20 person,
   11 sign,3 car,2 light),YOLO 只出 4 框。漏掉的 20 個 person 面積比 0.017%-0.123%,
   漏掉的 11 個 sign 0.011%-0.121%——全部 tiny。這是一張晚上市集街景(背景有摩天輪),
   密集人群在畫面遠端縮成一整排極小的點。
3. **image_id 5487**(`video-zp8ed5vPKfAJ2fKWh-frame-005911`):GT 29 框,漏掉的框裡
   大多數 <0.5%(sign 9 個都 tiny、person 4 個都 tiny),但有 **2 個例外**:一個 car
   3.45%(medium,fully_visible)、一個 car 1.96%(small,partially_occluded)。這兩個
   不是「太小看不到」被漏掉——查證後那個 3.45% 的 car 其實有被 YOLO 框到,只是被
   分類成「truck」不是「car」(同 class 才算配對,所以在這裡算漏偵測)。細節見 Q2。

**結論**:tiny bucket 漏偵測率(76%)遠高於其他 bucket(11-20%),支持「YOLO
沒偵測到的 GT 框、大部分是熱像上本來就很難辨識的極小物件(sign 結構被過度切分、
遠處密集人群縮成小點)」這個方向。但也有反例(見下面 Q3 的 image_id 5898),不是
100% 純粹的尺寸問題,也混了 Q2 講的 class 混淆。

---

## Q2: YOLO 過偵測的類別(truck/skateboard/train),實際上是什麼被誤判?

用程式對 YOLO 「truck」偵測結果、逐一找它跟哪個 GT 框 IoU 最高(不限同 class),
結果(val split,272 個 truck 偵測,IoU>0.3 才算有對到東西):

```
car: 223 個 (82%)
bus: 21 個 (7.7%)
other vehicle: 12 個 (4.4%)
truck: 10 個 (3.7%, 真的對到)
無任何 GT 重疊: 6 個 (2.2%)
```

**主結論(有實測數字支撐,不是猜的)**:YOLO 的「truck」誤判裡,82% 其實是
FLIR GT 標成「car」的物件——具體看視覺化圖,是廂型車、休旅車、皮卡這種 COCO
訓練分佈裡會被劃進「truck」但 FLIR 標註規範統一歸類成「car」的車型。這是
**taxonomy 粒度不一致**造成的系統性誤判,不是隨機雜訊。

**3 張具體 image_id 佐證:**

1. **image_id 3652**(`video-nMfT5vK8MfEEjQ44W-frame-005131`):GT 9 框(7 car,
   2 person),右側一台廂型休旅車 GT 標成「car #2」,YOLO 標成「truck 0.35」;
   左側一排停放的休旅車/箱型車,GT 全標「car」,YOLO 裡有 3-4 台被標成 truck
   (conf 0.35-0.99)。
2. **image_id 5488**(`video-zp8ed5vPKfAJ2fKWh-frame-005926`):GT 27 框裡完全沒有
   truck 這個類別(0 個),YOLO 卻標出 3 個 truck(conf 0.41-0.68),其中一個對到
   GT 的 bus,另外幾個疑似對到路邊的廂型車(GT 標 car)。
3. **image_id 5487**(`video-zp8ed5vPKfAJ2fKWh-frame-005911`):bottom-right 那台
   皮卡型車輛,GT 標「car #12」(fully_visible,面積 3.45%,不是 tiny),YOLO 標
   「truck 0.89」——同一個物理物件,兩邊只是叫法不同。

**train / skateboard 過偵測:未確認,信心低,懷疑是雜訊誤判。**
val split 裡 train 偵測 5 個、skateboard 1 個,樣本太少不足以下系統性結論。查
train 的 5 個偵測框跟任何 GT 物件的 IoU 都 <0.1(幾乎不重疊任何東西),而且框都
異常大(寬度佔畫面 40-58%)——**這幾個看起來像是背景建築物/道路結構被誤判成
train,不是真的有火車或哪個物件被錯叫成 train**,但這只是看框的位置形狀猜的,
沒有進一步驗證(例如沒去看那幾張圖片本體長什麼樣),標記「未確認」。skateboard
只有 1 個樣本,同樣未進一步驗證,不下結論。

---

## Q3: YOLO 漏偵測的 bike/motor/sign,是不是都在「難看見」的位置?還是有清楚可見的也偵測不到?

**不是全部都難看見——有明確的反例,這是真的模型弱點,不能全部歸咎給 GT 標註品質。**

**3 張具體 image_id 對照:**

1. **image_id 5898**(`video-mKfYgxHA8ZZmXvw56-frame-001180`)——**反例,重點看這張**:
   GT 有一個「motor」框,面積比高達 **12.97%**(large bucket,畫面左下角,近景,
   跟同一個 bbox 位置重疊的還有一個 person GT 框——騎士 + 機車)。肉眼直接看原圖
   可以清楚辨識出「一個人跨坐在有把手、輪子形狀的載具上」,不是隱約模糊的東西。
   YOLO 這張只偵測到 person,完全沒輸出 motorcycle。**這是實測到的真實模型弱點:
   YOLOv8m 在 COCO RGB 影像上學到的 motorcycle 外觀特徵(輪輻、車身反光、排氣管
   輪廓)在熱像的均勻灰階 blob 上對不太起來,即使物件夠大、夠近、夠清楚。**
2. **image_id 5895**(`video-mKfYgxHA8ZZmXvw56-frame-000295`):漏掉的 12 個 sign
   全是 tiny(0.008%-0.32%,細節見 Q1),這組屬於「本來就很小」,不算模型弱點,
   算 GT 標註粒度問題(見上方 Q1 對高速公路指示牌被拆成多個小格子的觀察)。
3. **image_id 5434**(`video-zp8ed5vPKfAJ2fKWh-frame-001803`):GT 有一個「bike」框,
   面積比 3.42%(small-medium 交界,不算 tiny),YOLO 沒偵測到。這是 bike 漏偵測
   裡面積最大的一個(除了純 tiny 的一大票之外),介於「勉強看得到」跟「小」之間,
   沒有像 5898 那個 motor 那麼戲劇性,但同樣不是純粹「太小」可以完全解釋的案例。

**結論**:sign 的漏偵測絕大多數確實是 tiny/GT 過度切分的問題(Q1 講的那套邏輯適用);
但 bike/motor 的漏偵測不能完全套用同一套解釋——像 image_id 5898 這種大、近、
清晰可辨的 motor 案例證明 YOLOv8m(COCO pretrained,沒 fine-tune)本身就對熱像
上的兩輪載具外觀辨識能力弱,這是真實的 domain gap 模型弱點,不是「GT 亂標、
其實看不到」可以開脫的。

---

## ⚠️ 待 Jack 確認 / 信心不足的地方(不要跳過)

1. **truck 誤判來源(Q2 主結論)**:82% 對到 GT car、7.7% 對到 GT bus 是實測算出來的
   數字(IoU>0.3 就算重疊,閾值是我選的,沒有特別驗證這個閾值多合理,只是想抓「大致
   在同個物理位置」),但我沒有真的去看那 223 個「重疊到 car」的偵測框,一張張確認
   它們是不是真的都是廂型車/休旅車這種車型(只挑了 3 張圖眼睛看)。如果 Jack 要拿
   這條當作論文/報告的核心論點,建議再多抽幾張人工核對。
2. **train/skateboard 過偵測解讀**:「背景建築物被誤判」是我看 bbox 尺寸/位置形狀
   猜的,没有真的把那 5 張圖叫出來看內容,信心低,標「未確認」。
3. **IoU 閾值 0.5(matching)/ 0.3(Q2 confusion 分析)**:matching 用 0.5 是延續
   task 指定值沒有換過;Q2 那段額外分析用 0.3 是我另外選的(想抓「大概在同個位置」
   的重疊,不是要求嚴格對齊),兩個用途不同、閾值也不同,沒有混用,但這個 0.3 是我
   自己選的,不是 Jack 指定或標準做法,如果覺得不合理可以換。
4. **本 finding 文件裡「肉眼看到 XX」的判斷全部是 Claude Code 主觀看圖的結果**,
   Jack 務必自己打開 `day35_gt_vs_yolo/` 30 張圖跟額外存的
   `day35_gt_vs_yolo/Q3_special_5898_missed_large_motor.png` 親眼確認一次,尤其
   image_id 5898 這張是整份 finding 裡最關鍵的反例,直接影響「GT 標註品質 vs
   模型弱點」這條線的結論方向。
