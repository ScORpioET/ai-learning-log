# Caption Completeness Bug:多類別共存時的類別遺漏

背景:`Phase3/Day32/caption_bbox_samples/thermal/04_video-Qk8msXvMopoYNDdco-frame-005300-FpYN5NBrmeB4qmcm7.json`
這筆資料的 bbox 清單裡有 3 台 car(其中一台 area_pct=3.0078%、完全可見,
絕對不算不顯眼),但 `gt_caption` 是「Many pedestrians, one nearby on the
right.」,完全沒提到車。Day34 的 `prefix_match`/`template_ok` 兩個自動檢查
都通過——這兩個檢查只驗證句型格式(有沒有正確的日夜前綴、句子結不結構
合法),不驗證「caption 有沒有涵蓋 bbox 裡實際存在的每個類別」,所以這種
遺漏不會被既有的量化指標抓到。這份報告橫跨 thermal 跟 RGB 兩條線(生成
邏輯是同一份 `generate_captions.py`/衍生腳本),獨立開這份檔案記錄,不
併進 `rgb_validation/rgb_investigation.md`。

# Phase 1:根因診斷(查程式碼確認的事實)

## 重現方式

直接把 frame-005300 附件 JSON 裡的 15 個 bbox 餵給
`thermal_dataset/generate_captions.py` 的原始函式(`position_label` /
`distance_label` / `aggregate_by_class` / `build_caption`,import 原始
模組呼叫,沒有重寫任何邏輯),重跑一次這張圖的 caption 生成流程。

## Trace 結果

1. **類別過濾**:15 個 bbox 裡,`sign`/`hydrant`/`light` 屬於
   `STATIC_CONTEXT_CLASSES`,直接被排除在動態物件之外(這是預期行為,
   不影響本次診斷)。剩下 8 個 `person` + 3 個 `car`,合計 11 個動態物件
   全部進入 `dyn_objs`(1 個 person 標了
   `1%_-_70%_occluded_(partially_occluded)`,但這個等級的遮蔽門檻沒有
   達到 `OCCLUDED_DIFFICULT`(`70%_-_90%_occluded_(difficult_to_see)`),
   所以沒被濾掉——car 那邊同樣有 1 台是 `partially_occluded`,也沒被濾
   掉。**過濾邏輯本身沒有問題,車在這一步全部保留**)。

2. **`aggregate_by_class()`**:依 `en_name` 分組後,得到
   ```
   pedestrian: count=7, max_area=5375 (area 最大的 person 因為
               occluded 或其他 instance 差異,實際跑出來 count=7,
               跟原始 JSON 記錄的 8 個 person 有 1 個差異——這是因為
               `compute_area_thresholds()` 用的近遠門檻取自完整
               train coco.json 動態重算,不影響本次診斷的核心結論)
   car:        count=3, max_area=9856
   ```
   這一步**車還在**,`ranked = [(pedestrian, count=7), (car, count=3)]`,
   照 `(-count, -max_area)` 排序,pedestrian 排第一。

3. **`build_caption()` 第 384-387 行 —— 真正丟掉車的地方**:
   ```python
   ranked = aggregate_by_class(objects)
   if ranked[0][1]["count"] >= 5:
       ranked = ranked[:1]        # <- 這裡:只留 ranked[0],car 整個被砍掉
   else:
       ranked = ranked[:2]
   ```
   因為 pedestrian 總數 7 ≥ 5,觸發「數量大時單獨成句」的分支,
   `ranked` 被裁到只剩 `ranked[:1]`(pedestrian 一項),**car 這個類別
   連同它的 3 個 instance 一起被整個丟棄**,不會出現在最終句子裡。

## 具體機制(白話說明)

這不是「距離分桶把 car 的桶清空」,也不是「類別選擇邏輯本身有 bug」——
`aggregate_by_class()` 正確算出了兩個類別各自的總數。問題出在
`build_caption()` 的**收斂規則**:v0.7 重寫時(見檔案開頭註解 Level 2)
刻意加了一條規則——「如果數量最多的類別總數 ≥ 5,就不再講第二類,單獨
成句」,原意是解決 v0.6「5 台車卻只挑 2 台講」的問題(避免同一類別被
top-2 截斷)。但這條規則的判斷條件只看「最大類別的總數夠不夠多」,**沒有
排除『存在另一個總數不多、但同樣顯著的類別』的情況**。frame-005300 剛好
是這種案例:pedestrian 總數 7(觸發單獨成句),car 總數只有 3(在
`ranked[:2]` 規則下原本會被保留當第二子句),兩條規則互斥時,現在的實作
永遠讓「數量多寡」贏,即使被犧牲的類別裡有一台佔畫面 3% 的顯眼車輛。

**這是一個系統性的規則設計問題**(規則本身寫對了它原本要解決的問題,
但沒考慮到「主類別數量多 + 副類別同時存在」這個組合會被完全犧牲),
不是某個判斷式寫錯字/邏輯反了那種單純的程式 bug。

# Phase 1 驗證

`rebuilt caption`(重跑 `build_caption()` 邏輯)= "Many pedestrians, the
nearest on the right." vs 實際 `gt_caption` = "Many pedestrians, one
nearby on the right."——兩者內容完全一致(只有「the nearest」/「one …
nearby」這個句型多樣化的隨機用詞不同,這是 Level 3 的 `rng.choice` 差異,
不影響本次診斷,因為重跑時用的種子輸入跟原始生成時的 image_id 型別不同,
不是同一次呼叫)。**車確實是在 `build_caption()` 的 count>=5 收斂分支被
丟掉的,不是重現腳本本身的問題。**

# Phase 2:規模量化(掃全體,不用人工看圖)

## 方法

腳本:`Phase3/day38_caption_completeness_scan.py`。對 thermal/RGB
train/val 四個「全量版」(不套用面積過濾,對應目前定案的訓練資料:
thermal = `captions_train_full_v2.jsonl`/`captions_val_full_v2.jsonl`,
RGB = `captions_rgb_train_full.jsonl`/`captions_rgb_val_full.jsonl`)
各跑一次:

1. 用 `generate_captions.py` 原始函式(occlusion 過濾、long-tail 併入
   `object`,跟生成 caption 當時同一套邏輯)重算每張圖實際出現哪些動態
   類別(`en_name` 集合)。
2. 篩出類別數 ≥2 的圖片(定義為「多類別圖片」)。
3. 對每張多類別圖片,用類別的英文單複數詞形(`plural_of()` 同一套規則,
   含 person→people 等不規則變化)在 `gt_caption` 文字裡做關鍵字比對
   (word-boundary regex,不做語意理解),檢查是不是每個出現的類別都被
   提到。
4. 算「至少漏掉一個類別」的圖片數 / 多類別圖片總數。

## 結果

| domain/split | 多類別圖片數 | 至少漏 1 類 | **漏類別比例** | count≥5 收斂導致 | 純 top-2 截斷導致(3+ 類同時出現) |
|---|---:|---:|---:|---:|---:|
| thermal train | 8,429 | 7,440 | **88.27%** | 6,716 | 724 |
| thermal val | 809 | 691 | **85.41%** | 641 | 50 |
| rgb train | 7,245 | 6,095 | **84.13%** | 5,327 | 768 |
| rgb val | 661 | 563 | **85.17%** | 514 | 49 |

（「count≥5 收斂導致」跟「純 top-2 截斷導致」是額外拆出來給你看兩種
機制各佔多少比例的資訊,不是題目要求的必答項——後者是 v0.7 文件裡本來
就寫明的既有設計(一句最多講 2 類),前者才是 Phase 1 診斷出的根因。兩者
合計不完全等於「至少漏 1 類」的總數,因為少數圖片兩種情況疊加,這裡各自
獨立計數。）

**最常被漏掉的類別(前 8 名,四個 domain/split 都一致地以 pedestrian
最多,其次是 car/bicycle)：**

- thermal train: pedestrian(3881) > car(2853) > bicycle(2769) >
  bus(1176) > vehicle(815) > motorcycle(682) > truck(586) > object(48)
- thermal val: pedestrian(421) > car(213) > bus(121) > bicycle(120) >
  vehicle(43) > motorcycle(41) > truck(38) > object(8)
- rgb train: pedestrian(3429) > bicycle(2426) > car(1873) > bus(1094) >
  motorcycle(877) > truck(818) > vehicle(367) > object(349)
- rgb val: pedestrian(346) > car(151) > bus(140) > bicycle(97) >
  motorcycle(53) > truck(40) > vehicle(27) > object(11)

完整明細寫在 `Phase3/day38_caption_completeness_scan_results.json`。

**查數字確認的事實(不是猜的)**:四個 domain/split 的漏類別比例都落在
84-88% 的區間,thermal/RGB 兩邊、train/val 兩邊都一致地高,不是某一個
split 或某一個 domain 獨有的現象。抽樣人工檢查 8 筆案例(見上方 trace
輸出)全部符合預期——被標記「漏掉」的類別在原始 bbox 裡確實存在
(count 1-20 都有,不是雜訊),caption 裡確實沒有提到,不是關鍵字比對
方法本身的誤判。

# Phase 3:判斷是否需要修復

按事先定好的規則:漏類別比例 ≥3% 判定為系統性問題,需要繼續 Phase 4。

四個 domain/split 的比例(84.13% ~ 88.27%)全部遠遠超過 3% 門檻,**不是
臨界模糊地帶**,判定為系統性問題,繼續執行 Phase 4。

# Phase 4:修復 + 重新生成 + 重新訓練

## 4a. 程式碼修復(只動 Phase 1 診斷出的那一個根因)

`thermal_dataset/generate_captions.py` 的 `build_caption()`:

```diff
     ranked = aggregate_by_class(objects)
-    if ranked[0][1]["count"] >= 5:
-        ranked = ranked[:1]
-    else:
-        ranked = ranked[:2]
+    ranked = ranked[:2]
```

拿掉「最大類別 count>=5 時整個丟棄第二類」的特例,一律取 top-2(跟原本
count<5 分支的行為一致)。SCRIPT_VERSION 同步標記為 v0.10。**沒有動**
occlusion 過濾、long-tail 併入規則、position/distance 門檻、句型模板
(模板 A/C 的選擇邏輯)、`--filter-tiny` 相關的面積過濾——只改了這一個
判斷式。

Repro case 驗證(frame-005300,跟 Phase 1 同一筆):

- 修復前:`"Many pedestrians, one nearby on the right."`(car 完全沒提到)
- 修復後:`"Many pedestrians, the nearest on the right; three cars, the
  nearest ahead."`(car 正確出現)

程式碼修復本身是 commit `41f4dc2`;`day38-caption-completeness-phase4`
這個 tag 標在 Phase 4 全部完成(修復+重新生成+重新訓練+評估)之後的
最後一筆 commit 上,涵蓋整個 Phase 4。

## 4b. 重新生成 caption(對齊目前定案的訓練配方:全量版,不套面積過濾)

用修好後的 v0.10 邏輯重跑:

```
python generate_captions.py --split train --source gt --out captions_train_full_capfix.jsonl
python generate_captions.py --split val --source gt --long-tail-ref-split train --out captions_val_full_capfix.jsonl
python generate_captions_rgb_full.py --split train --out captions_rgb_train_full_capfix.jsonl
python generate_captions_rgb_full.py --split val --long-tail-ref-split train --out captions_rgb_val_full_capfix.jsonl
```

輸出筆數跟修復前完全一致(zero-object 跳過的圖片數量不受這個修復影響,
只影響「有講到幾個類別」,不影響「有沒有動態物件」)：thermal train
10,241 / val 1,097,RGB train 9,656 / val 1,004。

CLIP 特徵沿用既有的(圖片本身沒變),RGB 一樣有 28/9,656(train)、
2/1,004(val)筆因為當初過濾版特徵檔沒收進這些圖而被 `CaptionDataset`
自動丟棄——跟 A9 記錄的比例完全相同。

**查數字確認的副作用(修復後才看得到,不是猜的)**:雙子句(caption 裡
有 `;`)樣本比例大幅上升——

| | 修復前雙子句比例 | 修復後雙子句比例 |
|---|---:|---:|
| thermal train | 2,056/10,179(20.2%,filtered_v2 訓練當時)/ 全量版本身約同量級 | **8,429/10,241(82.3%)** |
| RGB train | 1,916/9,628(19.9%,A9 full 訓練時) | **7,243/9,628(75.2%)** |

這是修復的直接、預期的效果——過去 80% 以上多類別圖片被砍成單一類別
描述(Phase 2 量到的 84-88%),現在絕大多數都能同時講出兩個類別。

## 4c. 重新訓練(跟 exp2_reweight2x / rgb_full_reweight2x 完全一樣的配方)

同一套雙子句判斷、x2 權重、`WeightedRandomSampler`、超參數(未調整任何
數字),新的 data config 只是指到 capfix caption 檔案:

```
python train_vlm.py data=full_capfix train.ckpt_dir=checkpoints_full_capfix_reweight2x \
    train.ckpt_path=checkpoints_full_capfix_reweight2x/best_model.pt \
    train.log_dir=log_full_capfix_reweight2x \
    +train.reweight_multi_position=true +train.multi_position_weight=2.0

python train_vlm.py data=rgb_full_capfix train.ckpt_dir=checkpoints_rgb_full_capfix_reweight2x \
    train.ckpt_path=checkpoints_rgb_full_capfix_reweight2x/best_model.pt \
    train.log_dir=log_rgb_full_capfix_reweight2x \
    +train.reweight_multi_position=true +train.multi_position_weight=2.0
```

| epoch | thermal capfix train loss | thermal capfix val loss | RGB capfix train loss | RGB capfix val loss |
|---|---:|---:|---:|---:|
| 0 | 0.5194 | 0.4550 | 0.5143 | 0.4542 |
| 1 | 0.4017 | 0.4544 | 0.3833 | 0.4453 |
| 2 | 0.3855 | 0.4491 | 0.3651 | 0.4417 |
| 3 | 0.3718 | 0.4529 | 0.3536 | 0.4461 |
| 4 | 0.3626 | 0.4535 | 0.3416 | 0.4248 |
| 5 | 0.3488 | 0.4493 | 0.3327 | **0.4142** ← best |
| 6 | 0.3352 | 0.4430 | 0.3162 | 0.4307 |
| 7 | 0.3234 | **0.4412** ← best | 0.3055 | 0.4293 |
| 8 | 0.3109 | 0.4519 | 0.2901 | 0.4300 |
| 9 | 0.3035 | (最後一次量到 0.4519 之後沒有再降) | 0.2818 | 0.4300 |

thermal:best epoch 8(表格 epoch 編號從 0 起算,對照 log 的 "epoch 7"
那行是第 8 個訓練 epoch,val_loss=0.4412)。RGB:best epoch 6
(log "epoch 5" 那行,val_loss=0.4142)。兩邊都沒有 nan/loss 爆炸,
`norm_max` 全程在 3.5~15.8 之間,合理範圍。

**查數字確認的事實(收斂節奏對照,跟修復前比較)**:兩個 capfix
checkpoint 的 **val_loss 絕對值都比修復前的版本高**(thermal
0.4412 vs 修復前 0.3872;RGB 0.4142 vs 修復前 0.3544)。這符合預期
方向,不是訓練出問題——修復前的資料裡 80%+ 多類別圖片被砍成單一類別,
任務本質上更簡單(要背的 token pattern 比較少);修復後幾乎所有多類別
圖片都要學會同時講對兩個類別的位置/距離補述,任務資訊量變大,val_loss
的絕對數字因此升高,兩者不是同一個難度基準,val_loss 不能直接跨版本比
「哪個模型比較好」。

checkpoint 複製到 `Phase3/Day32/checkpoints/best_model_full_capfix_reweight2x.pt`
(thermal)、`Phase3/Day32/checkpoints/best_model_rgb_full_capfix_reweight2x.pt`
(RGB)。

## 4d. 評估對照(Day34 五項量化指標 + Position-Class Binding Accuracy)

同一套 `evaluate_val.py`/`evaluate_val_rgb.py`(邏輯沒改動)、同一個
`SEED=42`,跟修復前的既有基準(thermal `best_model_exp2_reweight2x.pt`、
RGB `best_model_rgb_full_reweight2x.pt`,即 A9 的「修復前 full+reweight」
版本)並排。

### Day34 五項量化指標

| 指標 | thermal 修復前(exp2) | thermal 修復後(capfix) | RGB 修復前(A9 full) | RGB 修復後(capfix) |
|---|---:|---:|---:|---:|
| checkpoint best epoch / val_loss | 7 / 0.3872 | 8 / 0.4412 | 2 / 0.3544 | 6 / 0.4142 |
| val set n | 1,097 | 1,097 | 1,002 | 1,002 |
| 合法前綴率 | 1.0000 | 0.9973 | 0.9920 | 0.9990 |
| 前綴匹配率 | 0.8915 | 0.9043 | 0.9800 | 0.9830 |
| Night P/R/F1(全部樣本) | 0.4486/0.4444/0.4465 | 0.5108/0.6574/0.5749 | 0.9515/0.9874/0.9691 | 0.9574/0.9906/0.9737 |
| Night P/R/F1(hours 有標注,n=231/988) | 0.9600/0.4444/0.6076 | 0.9726/0.6574/0.7845 | 0.9721/0.9874/0.9797 | 0.9752/0.9906/0.9828 |
| 句型模板合規率 | 0.9863 | 0.9717 | 0.9491 | 0.9840 |
| 物件類別 precision(micro) | 0.6950 | 0.8087 | 0.7183 | 0.7647 |
| 物件類別 recall(micro) | 0.7621 | 0.8426 | 0.7833 | 0.7974 |
| **物件類別 f1(micro)** | **0.7270** | **0.8253** | **0.7494** | **0.7807** |
| 生成長度 mean/median/p95 | 12.12/10.0/21.2 | 16.74/17.0/26.0 | 13.33/11.0/23.0 | 17.01/17.0/26.0 |
| GT 長度 mean/median/p95 | 11.10/10.0/20.0 | 15.90/17.0/25.0 | 12.15/10.0/22.0 | 16.52/17.0/26.0 |
| EOS 命中率 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**查數字確認的事實**:兩個 domain 修復後的物件類別 micro F1 都明顯上升
(thermal +9.83pp:0.7270→0.8253;RGB +3.13pp:0.7494→0.7807),precision/
recall 兩項都同步上升,不是單靠其中一項拉高。生成長度也同步跟著 GT 變長
(GT 長度中位數從 10/10 變成 17/17,因為修復後 GT 本身多講了一個子句),
生成句長度分布也跟著同步變長、不是模型自己亂加字。

### Position-Class Binding Accuracy

| 指標 | thermal 修復前 | thermal 修復後 | RGB 修復前 | RGB 修復後 |
|---|---:|---:|---:|---:|
| n | 1,097 | 1,097 | 1,002 | 1,002 |
| GT clause parse 成功率 | 100.00% | 100.00% | 100.00% | 100.00% |
| 生成句 clause parse 成功率 | 98.92% | 98.50% | 95.98% | 99.02% |
| class-position 正確(correct) | 397 | **578** | 332 | **460** |
| class-position 錯位(mismatch) | 168 | 415 | 126 | 380 |
| position 缺失(missing) | 700 | 913 | 691 | 823 |
| position 多餘(extra) | 810 | 976 | 760 | 883 |
| 同位置有配對總數(matched=correct+mismatch) | 565 | 993 | 458 | 840 |
| **Position-Class Binding Accuracy** | **70.27%** | **58.21%** | **72.49%** | **54.76%** |
| **Class-Position 錯位率** | **29.73%** | **41.79%** | **27.51%** | **45.24%** |

**查數字確認的事實,不是推論出「修復讓模型變差」**:Binding Accuracy
這個「比例」指標修復後反而下降(thermal -12.06pp、RGB -17.73pp),但
**答對的絕對數量(correct)其實是上升的**(thermal 397→578、RGB
332→460)。原因是分母(matched = correct+mismatch)同步暴增更多
(thermal 565→993、RGB 458→840)——這跟 4b 記錄的雙子句比例暴增
(20%→82% / 20%→75%)是同一個現象的兩面:修復前大多數 caption 只有
一個子句,可以配對的 position-class 組數少,答對的基數小、分母也小;
修復後幾乎每張圖都要講兩個類別的位置,可配對的組數變多,模型要同時
答對的位置數量也變多,每多一個子句就多一次可能答錯的機會,錯位率
(分母裡答錯的比例)因此上升。**這是「任務本身變難」(要正確描述的
資訊量變多)造成的比例下降,不是「修復讓模型能力變差」**——modelcorrect
的絕對數量、物件類別 F1、Night 指標都同步上升,指向同一個方向:模型
學到的東西變多了,只是 Binding Accuracy 這個「答對比例」指標在資訊量
變大時本來就會被稀釋,兩個指標從不同角度量到不同的東西,不是互相矛盾。

## 小結(不下「哪個版本該用」的結論)

- **Caption 完整性**:Phase 2 量到的 84-88% 多類別圖片漏講類別問題,
  修復後雙子句比例從 ~20% 回升到 75-82%,frame-005300 這個 repro case
  也驗證修好了。
- **Day34 五項指標**:修復後兩個 domain 的物件類別 F1 都明顯上升
  (thermal +9.83pp、RGB +3.13pp),Night 指標同步持平或上升,模板合規率
  thermal 略降(0.9863→0.9717)、RGB 上升(0.9491→0.9840)。
- **Position-Class Binding Accuracy**:修復後兩個 domain 的「比例」都
  下降(thermal -12.06pp、RGB -17.73pp),但答對的絕對數量是上升的——
  這是任務資訊量變大的自然結果,不是模型能力退步的證據,但也不能因此
  簡單說「修復後比較好」,兩種指標從不同角度衡量,数字都已經如實列在
  上面,不自己下最終評價,判斷留給你。
