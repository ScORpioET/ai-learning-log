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

## 四、Day36 課堂 v2(第二晚):Phase 1 bug 修復 + Phase 2 改善實驗結果

**這是目前為止最重要的一段更新——上面第 6、7 點提到的兩個 bug 都已經修好,
第二晚 pending 清單裡的第 2、3 項已經有具體結果,不再只是「要不要修」的問題。**

### Phase 1:兩個 bug 修復結果(git commit `82023c6`/`e689686`/`115aa55`/`14a68d4`)

1. **`GLOBAL_MIN_AREA_PCT` 0.05%→0.025%**:far 距離物件保留率從 0% 回升到
   39.7%(train)/47.2%(val)。多值 sensitivity(0.015/0.02/0.025/0.03%)全部
   測過,見 `threshold_sensitivity_v2.md`。⚠️ 意外發現:雙子句 caption 比例
   沒有如預期上升(23.8%→20.2%,反而微降)——threshold 放寬讓每張圖物件數
   變多,連帶讓更多圖片的 car 數量衝過 `build_caption()` 的「count>=5 就
   collapse 成單子句」門檻,兩個機制互相抵銷。
2. **GT full 訓練資料重新套用 `--long-tail-ref-split train`**:確認根因是
   val 樣本數少(1144 張),bike/motor/bus/truck/other vehicle 這些類別在 val
   自己的 counts 都低於 500,舊版沒指定 ref split 時被錯誤折成「object」;
   train 用自己的 counts 判斷,全部遠高於 500 不會被折。修好後兩邊「object」
   出現比例相當(train 5 次/10241、val 1 次/1097),不再懸殊(修前 11 vs 153)。
   見 `task2_full_v2_regeneration.md`。

**修復前後 Position Binding 對照表(最重要的一張表)**:

| | v1 full(舊 bug 版) | v1 filtered(舊 bug 版) | v2 full(bug 修好) | v2 filtered(bug 修好) |
|---|---|---|---|---|
| GT 模板 | v0.6「there is X」 | v0.7+ class-first | v0.7+ class-first | v0.7+ class-first |
| binding accuracy | 67.9% | 66.1% | **70.9%** | **69.1%** |
| class-position 錯位率 | 32.1% | 33.9% | **29.1%** | **30.9%** |
| position recall | 57.7% | 41.7% | 41.6% | 41.3% |
| position precision | 56.8% | 40.6% | 41.4% | 39.5% |

⚠️ **重要方法論發現**:v1 full 用的訓練檔(8/25)是舊版 script 的 v0.6「there
is X」模板,重跑現在的 script 一定會變成 v0.7+「class-first」模板(沒有 flag
可以切回舊模板)——這代表 v1 full 的 position recall(57.7%)沒辦法直接跟
v2 full(41.6%)比較,不是同一把尺(v0.6 模板平均每句 1.95 個 position
segment,v0.7+ 只有 1.17 個,模板本身資訊密度不同)。**真正乾淨、可比較的是
v2 full vs v2 filtered(兩者現在共用同一套模板)**:binding accuracy 70.9%
vs 69.1%、position recall 41.6% vs 41.3%——v1 時期這個 gap 是 16.0pp,現在
只剩 0.3pp。**結論:兩個 bug 修完後,filtered 版在 position binding 上已經
追平 full 版,v1 觀察到的「filtered 版方位能力明顯較弱」主要是 bug 造成的,
不是 tiny-filter 訓練資料本身有害。** 詳見 `task4_v1_vs_v2_position_binding.md`。

### Phase 2:三個低風險改善實驗(以較優的 `best_model_full_v2.pt` 為基礎)

| exp | 改動 | position recall | binding accuracy | 結果 |
|---|---|---|---|---|
| baseline(full_v2) | — | 41.6% | 70.9% | — |
| exp2 | 雙子句抽樣加權 x2.0 | **44.7%(+3.1pp)** | 70.3%(-0.6pp) | **保留** |
| exp3 | epoch 10→16(建立在 exp2 上) | 41.9%(-2.8pp vs exp2) | 67.0%(-3.3pp) | revert(過擬合,val loss 在 ep7 後回升) |
| exp4 | 降峰值 LR 3e-4→1.5e-4 + 拉長 warmup(建立在 exp2 上) | 40.7%(-4.0pp vs exp2) | 61.2%(-9.1pp) | revert(更早過擬合,ep2 就觸底) |

exp2 過程有個誠實記錄的失誤:第一次跑 exp2 時忘記昨晚 exp1 的 `train_vlm.py`
改動已經被 revert,參數沒有真的生效(整組重跑一模一樣的訓練),發現後整組
清掉重跑,詳見 `experiments_log.md`。

**目前最佳版本**:`checkpoints/best_model_exp2_reweight2x.pt`
(= full_v2 資料 + 雙子句抽樣加權 x2,epoch7,val_loss=0.3872,position
recall=44.7%,binding accuracy=70.3%)。exp3/exp4 都確認過擬合更早發生,
**目前 10 epoch、max_lr=3e-4、warmup=50 的組合已經接近這個模型規模/資料量
下的合理設定,沒有找到更好的 epoch 數或 LR schedule。**

---

## 五、還沒拍板、需要 Jack 明天決定的事項

完整版另見 [pending_decisions.md](pending_decisions.md),這裡列重點(比第一晚
版本少兩項——原本的第 2、3 項今晚已經有具體修復結果,不再是空白選項):

1. **filtered vs full 到底哪個當主力 decoder**——bug 修好後兩者在 position
   binding 上已經幾乎打平(v2 full 70.9%/41.6% vs v2 filtered 69.1%/41.3%),
   物件類別 F1 filtered 略低(73.63% vs 74.45%)。**跟第一晚比,這個決策的
   急迫性降低了**——不再是「filtered 版方位能力明顯較弱」這種明確劣勢,
   比較像是「兩者伯仲之間,選哪個都可以」。但 Task C 的 exp2(雙子句加權)是
   建立在 full_v2 上做的,如果要選 filtered 當主力,同樣的 exp2 改善有沒有
   遷移過去還沒驗證,需要 Jack 決定要不要另外花時間跑。
2. **exp2(位居目前最佳)要不要正式取代 full_v2 成為新基準**——它在 position
   recall 上有實質改善(+3.1pp),但 binding accuracy 略降(-0.6pp),是不是
   接受這個 trade-off,還是要再找別的方向,需要 Jack 判斷。
3. **要不要投入更多時間處理 position binding,還是先進 ONNX/量化主線**——
   兩晚下來 position binding 從「filtered 版有明顯缺陷」進步到「兩版打平、
   還能再擠出 +3pp」,報酬遞減的訊號已經出現(exp3/exp4 都沒有進一步幫助),
   這個優先順序要 Jack 決定要不要繼續投入。
4. **今晚 Phase 2 只在允許範圍內(epoch/sampling/LR)試了 3 個方向就都測完
   了**——三個方向都跑完,沒有再自己發明新方向,如果 Jack 想繼續在允許範圍內
   嘗試(例如不同的加權倍數如 1.5x/2.5x,或更細緻的 LR warmup 調整),還有
   空間但今晚沒有做窮舉。

---

## 六、兩晚 git 實驗軌跡摘要

**第一晚(commit `a6f5a8f` tag `day36-baseline` 之後)**:
```
Task A: b421d08 (position recall 根因,找到 threshold 打架機制)
Task B: 46dd86a (missing position 根因,查到舊 long-tail bug 還在跑)
Task C exp1: 134db22 → revert 2d15e3c → log 4b5f82f (reweight x3,結果變差,已排除)
Task D: 94a578d (finding 彙整 + pending decisions)
```

**第二晚(commit `94a578d` tag `day36-course2-baseline` 之後)**:
```
Phase 1 Task 1: 82023c6 (修 threshold vs far_thresh 衝突,0.05%→0.025%)
Phase 1 Task 2: e689686 (修 GT full long-tail ref-split bug)
Phase 1 Task 3: 115aa55 (乾淨重訓 full_v2/filtered_v2)
Phase 1 Task 4: 14a68d4 (乾淨 position binding 對照,找到模板混淆變因)
Phase 1 log: a7a2a47 之前的部分(見上方 Phase 1 段落)
Phase 2 exp2: 9e30fda (雙子句加權x2,保留,position recall +3.1pp)
Phase 2 exp2/3 log: a7a2a47
Phase 2 exp4 log: e1f9d09 (降LR,revert)
```

沒有任何 checkpoint 覆蓋掉 `best_model.pt`、`best_model_filtered.pt`、
`best_model_full_v2.pt`、`best_model_filtered_v2.pt`。所有失敗實驗的
checkpoint(`checkpoints_exp1_reweight/`、`checkpoints_exp2_reweight2x/`
第一次無效跑、`checkpoints_exp3_epoch16/`、`checkpoints_exp4_lr/`)都已刪除。
