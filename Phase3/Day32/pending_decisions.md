# Pending Decisions —— 需要 Jack 判斷的事項(Day35-36 累積)

不會自己決定,只列選項跟取捨。完整脈絡見
[day35_36_summary_for_decision.md](day35_36_summary_for_decision.md)。

## 1. filtered vs full 到底哪個當主力 decoder(最優先)

| | best_model.pt(GT full) | best_model_filtered.pt |
|---|---|---|
| 物件類別 F1 | 70.8% | **72.6%** |
| 句型模板合規率 | 98.27% | **98.53%** |
| Position-Class Binding Accuracy | **67.9%** | 66.1% |
| Position recall | **57.7%** | 41.7% |
| val_loss(best epoch,不完全可比,見下方) | 0.3548 | 0.3993 |

取捨:filtered 版類別/模板略好,但 position binding(尤其 recall)明顯較弱。
val_loss 數字不建議直接拿來當決策依據——baseline 的 loss curve 震盪劇烈
(epoch-to-epoch 平均變動 0.032,std 0.053),epoch4 的 0.3548 疑似noisy dip;
filtered 版曲線平滑很多(std 0.012)。兩版 tokenizer 各自重訓,cross-entropy
數值本來就不是同尺度。

## 2. 要不要修 tiny-bbox threshold 讓 far 距離物件留一些

今晚(Task A)找到精確機制:`GLOBAL_MIN_AREA_PCT=0.05%` 卡在
`compute_area_thresholds()` 動態算出來的 `far_thresh`(0.04%)之上,數學上
必然讓所有 far 距離物件 100% 被濾掉,直接導致訓練資料裡雙 position 句子比例
從 94.7% 崩到 23.8%。

選項:
- (a) 降低 threshold,讓部分 far 物件留下——但這會讓 Day35 Task 5/6 已經
  驗證過的「保住 person/car 70% 保留率」的取捨重新洗牌,需要重跑 threshold
  sensitivity 分析
- (b) 改用 per-position(而非全域統一)的過濾邏輯,但 per-class threshold
  已經在 Task 5 試過、因為長尾類別樣本太少不可信而被 Jack 否決,per-position
  可能有類似問題
- (c) 維持現狀,接受 filtered 版 far 距離描述能力結構性缺失

今晚的低風險實驗範圍(只能動 epoch/sampling/LR)明確不允許碰這個,選項 a/b
都屬於「資料生成邏輯」層級的改動,需要 Jack 判斷要不要投入。

## 3. `captions_train.jsonl`/`captions_val.jsonl`(GT full 訓練檔)要不要重新生成

今晚(Task B)查到這兩個檔案時間戳記是 8/25,早於 Day34 的 long-tail bug 修正
(8/28)。實測:「object」這個 long-tail fallback 詞在 train 裡只出現在
nearby 距離(11 次),但 val 裡分佈在三種距離(153 次)——這個已知 bug 還沒
套用到 best_model.pt 實際訓練用的檔案上。如果 Jack 要繼續拿 GT full 當比較
基準,建議用 `--long-tail-ref-split train` 重新生成 val captions,確保
train/val 詞彙分布一致再比。

## 4. 要不要投入時間處理 position binding 缺陷,還是先進 ONNX/量化主線

Day36 原本就列出的待確認項目。今晚的分析(Task A/B)讓「投入處理」這個選項
有了明確技術路徑(修 threshold 或重新設計句型結構),但也顯示工作量比原本
想像的大(牽涉資料生成邏輯,不是單純調參數/訓練程序能解決——今晚試的抽樣
加權方向已驗證無效,見下一項)。這個優先順序要 Jack 決定。

## 5. 低風險改善方向只試了一種(抽樣加權),還沒試完

今晚 Task C 只跑了一次實驗(訓練資料抽樣加權,把雙 position 句子權重乘 3),
結果沒有改善(部分指標反而變差),已 revert。**這是照任務指示「不要自己擴大
實驗範圍」主動停下來的,不是判斷這整個方向(epoch/sampling/LR 微調)已經沒
希望。** 如果 Jack 想在允許範圍內繼續試(例如調整 epoch 數到 15-20、換 LR
schedule、用更溫和的加權倍數如 1.5x 而不是 3x),還有空間沒試過,只是今晚
沒有繼續往下做主觀判斷要不要試,交給 Jack 決定方向。
