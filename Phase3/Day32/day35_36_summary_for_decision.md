# Day 35-36 全部 Finding 彙整——明天決策用一頁摘要

範圍:Day35(YOLO 偵測整合、GT 標註品質驗證、tiny-bbox threshold 篩選)+
Day36(GT-full vs GT-filtered decoder 對照、position-class binding 量化、
FLIR R-JPEG 溫度資料查證、課堂期間無人監督的根因分析與一次低風險實驗)。
所有原始分析文件路徑都附在對應段落,數字不摘要到失真。

---

## 一、已完成的決策(附理由)

### 1. YOLO class mapping(Day35 Task 0-B)
KEEP_CLASSES 11 類(person/bicycle/car/motorcycle/bus/train/truck/traffic
light/fire hydrant/stop sign/skateboard)。理由:Day32 script 有用 + COCO 有對應
才收錄,避免 GT 沒有、YOLO 卻亂偵測出來的雜訊類別汙染 caption。
`other vehicle`(train 1373 筆)/`stroller`/`scooter` 因 COCO 無對應類別,YOLO
版必然缺失。詳見 [Day35/outputs/task0_class_mapping.md](../../Day35/outputs/task0_class_mapping.md)。

### 2. YOLO vs GT 覆蓋度量化(Day35 加碼)
IoU>0.5 + 同 class 才算配對成功。全域配對率 34.7%(5764/16627)。**關鍵發現**:
tiny bbox(<0.5% 畫面,佔全部 GT 的 82%)匹配率只有 24.0%,small/medium/large
都在 80-89%——直接印證「GT 標的東西人眼/YOLO 都看不到」這條線。但也有反例
(image_id 5898,12.97% 大、近景的 motorcycle,YOLO 完全沒偵測到)——不能全部
歸咎 GT,YOLO 本身對熱像上兩輪載具辨識也有真實弱點。另外發現 truck 誤判 82%
其實是 GT car 被 COCO 拆出來(taxonomy 粒度不一致,不是隨機雜訊)。
詳見 [Day35/outputs/gt_vs_yolo_summary.md](../../Day35/outputs/gt_vs_yolo_summary.md)、
[gt_vs_yolo_finding.md](../../Day35/outputs/gt_vs_yolo_finding.md)。

### 3. 決定不用 YOLO 版訓練資料,改用 GT + tiny-bbox filter(Day35 Task 5/6)
Jack 拍板:decoder 訓練資料用 GT + tiny filter,避開 YOLO 的過偵測問題
(truck/skateboard/train 過偵測嚴重,見上方)。

- 先試 per-class threshold(每類分別抓「保住 70% 保留率」的門檻),但
  skateboard/stroller/train 樣本太少(n=5~32),門檻不可信(train 甚至算出 7.5%
  這種離譜數字),Jack 決定改用單一 **`GLOBAL_MIN_AREA_PCT = 0.05%`**。
- 5 張眼睛驗證:4/5 caption 跟 per-class 版完全一樣,1 張只差 1 個字(car 數量
  ±1),品質沒有因為改 global threshold 明顯變差。
詳見 [Day35/outputs/threshold_sensitivity.md](../../Day35/outputs/threshold_sensitivity.md)、
[threshold_comparison_task6.md](../../Day35/outputs/threshold_comparison_task6.md)。

### 4. FLIR ADAS v2 圖片沒有藏 radiometric 溫度資料(信心:高)
exiftool 全 metadata dump + binary FLIR 字串掃描 + JPEG segment marker 解析,
三種方法交叉驗證,10 張圖(val+train)全部只有純 JFIF header,無 APP1 segment,
無 EXIF,無 FLIR MakerNotes。結論:這批圖是 AGC 正規化的 8-bit 顯示用途 jpg,
Planck 方程式反推溫度這條路走不通,不用再花時間試。

### 5. Position-Class Binding Accuracy 量化(Day36)
Jack 手動抓到「方位對、類別錯」的問題,量化證實是系統性的,不是個案:

| | best_model.pt(GT full) | best_model_filtered.pt |
|---|---|---|
| class-position 正確 | 67.9% | 66.1% |
| class-position 錯位 | 32.1% | 33.9% |
| position recall(GT 位置有沒有被講到) | 57.7% | 41.7% |
| position precision | 56.8% | 40.6% |

**意外發現比錯位問題更大宗**:missing+extra(整個方位對不上,不只是類別錯)
是 mismatch 的 4.6 倍(GT full)/ 8.4 倍(filtered)。也就是說模型更常見的毛病
是「根本沒講對方位」,不是「方位對但類別錯」。詳見
[position_binding_accuracy.md](position_binding_accuracy.md)。

### 6. Position recall 落差根因(今晚 Task A/B,git commit `b421d08`/`46dd86a`)
- **精確機制找到**:`GLOBAL_MIN_AREA_PCT`(0.05%)剛好卡在 `far_thresh`
  (0.04%,`compute_area_thresholds()` 動態算出來的)之上,數學上必然讓所有
  far 距離物件 **100% 被濾掉**(near 100% 保留、mid 93.6% 保留、far 0.0% 保留)。
  這直接導致 filtered 版訓練資料裡「雙 position 句子」比例從 94.7% 崩到 23.8%。
- 額外查到 GT full 版現在還在用的 `captions_train.jsonl`/`captions_val.jsonl`
  (檔案時間戳記 8/25,早於 Day34 的 8/28 long-tail bug 修正)仍帶著舊 bug:
  「object」這個 long-tail fallback 詞在 train 裡 11 次全部是 nearby 距離,
  一次 medium/far 都沒有,但 val 裡 153 次分佈在三種距離——train/val 詞彙分布
  不一致,這個已知 bug 還沒套用到目前實際在跑的訓練檔案上。
- filtered 版的漏講「不能只用單一 (class,distance) 組合訓練樣本稀少解釋」——
  即使 train 佔比很大的組合(car/mid 31.96%、car/nearby 26.10%)漏講率仍有
  55-56%,比 GT full 版同量級組合(car/nearby, train 佔 52.82%,漏講率只
  36.0%)明顯更差,說明問題出在句型結構訓練訊號崩盤,不是單一組合頻率不足。
  詳見 [task_a_recall_gap_analysis.md](task_a_recall_gap_analysis.md)、
  [task_b_missing_position_breakdown.md](task_b_missing_position_breakdown.md)。

### 7. 今晚唯一一次低風險改善實驗:失敗,已 revert(git commit `134db22`→revert `2d15e3c`)
假設:把訓練資料裡「雙 position 句子」的抽樣機率乘 3 倍(WeightedRandomSampler),
補償 Task A 找到的訓練訊號流失。結果:**沒有改善,部分指標反而變差**
(binding accuracy 66.1%→59.4%,position precision 40.6%→36.8%,position
recall 幾乎持平 41.7%→41.1%)。已用 `git revert` 清掉 train_vlm.py 的改動,
checkpoint 檔案也刪除。**排除了一個假設**:單純調整抽樣權重無法繞過 Task A
找到的根本問題(threshold 跟 far_thresh 打架導致 far 距離資料整批消失),因為
問題出在資料內容本身沒有 far 距離的雙子句範例可以抽,不是抽樣機率不夠高。
詳見 [experiments_log.md](experiments_log.md) exp1 條目。

---

## 二、還沒拍板、需要 Jack 明天決定的事項

完整版另見 [pending_decisions.md](pending_decisions.md),這裡列重點:

1. **filtered vs full 到底哪個當主力 decoder**——filtered 版物件類別 F1
   (72.6%)、句型模板合規率(98.5%)都不遜於甚至略高於 full 版(70.8%/98.3%),
   但 position binding 全面較弱(recall 41.7% vs 57.7%)。到底看重「類別講得
   對不對」還是「方位講得對不對」,是這次要 Jack 拍板的核心取捨。
2. **要不要修 threshold 讓 far 距離物件留一些**——今晚找到 0.05% 卡在
   far_thresh(0.04%)之上是問題根源,但改 threshold 屬於「資料生成邏輯」,
   今晚工作範圍明確不允許碰(Task C 限定只能動 epoch/sampling/LR)。如果要修,
   選項包括:降低 threshold(讓部分 far 物件留下,但這會讓 Day35/36 已經驗證
   過的「保住 person/car 70% 保留率」的取捨重新洗牌)、或改用 per-position 而非
   全域統一的過濾邏輯。
3. **`captions_train.jsonl`/`captions_val.jsonl`(best_model.pt 用的 GT full
   訓練檔)要不要拿 Day34 修好的 long-tail 邏輯重新生成**——今晚查到這兩個檔案
   還是 8/25 的舊版,帶著已知 bug,如果 Jack 要繼續拿 GT full 當基準比較,
   這個資料本身有洞的事實要先確認要不要處理。
4. **要不要投入時間處理 position binding 缺陷,還是先進 ONNX/量化主線**——
   Jack 原本就列在 Day36 待確認清單裡的問題,今晚的分析讓「投入處理」這個選項
   多了明確的技術路徑(改 threshold 或重新設計句型結構),但也讓工作量看起來
   更大(牽涉資料生成邏輯,不是單純調參數就能解決),這個 trade-off 要 Jack
   決定。
5. **今晚只試了一種低風險改善方向(抽樣加權)就沒有再試第二種**——這是照任務
   指示「不要自己擴大實驗範圍」停下來的,不是判斷這條路已經走到頭。如果 Jack
   想繼續在允許的範圍內(epoch 數/LR schedule)嘗試,還有空間沒試過。

---

## 三、今晚 git 實驗軌跡摘要

```
day36-baseline (tag, commit a6f5a8f)
  → Task A: b421d08 (position recall 根因,找到 threshold 打架機制)
  → Task B: 46dd86a (missing position 根因,查到舊 long-tail bug 還在跑)
  → Task C exp1: 134db22 (reweight 抽樣,結果變差)
  → Task C exp1 revert: 2d15e3c (git revert 134db22)
  → Task C log: 4b5f82f (記錄 exp1 結果到 experiments_log.md)
```

沒有任何 checkpoint 覆蓋掉 `best_model.pt` 或 `best_model_filtered.pt`。
`checkpoints_exp1_reweight/` 已刪除(失敗實驗,不留檔案)。
