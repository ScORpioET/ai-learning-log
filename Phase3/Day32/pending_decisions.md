# Pending Decisions —— 需要 Jack 判斷的事項(Day35-36 累積,含第二晚更新)

不會自己決定,只列選項跟取捨。完整脈絡見
[day35_36_summary_for_decision.md](day35_36_summary_for_decision.md)。

**✅ 第一晚原本的第 2、3 項(threshold bug / long-tail bug)第二晚都已經修好、
有具體結果,從這份清單移除,移到 summary 文件的「已完成」段落。**

## 1. filtered vs full 到底哪個當主力 decoder(優先度下降,但還是要選)

第二晚 bug 修好後的數字:

| | best_model_full_v2.pt | best_model_filtered_v2.pt |
|---|---|---|
| 物件類別 F1 | **74.45%** | 73.63% |
| Position-Class Binding Accuracy | **70.9%** | 69.1% |
| Position recall | **41.6%** | 41.3% |

跟第一晚比,gap 幾乎消失(第一晚 recall 差 16.0pp,現在只差 0.3pp)。full_v2
現在每項指標都略優於 filtered_v2,但差距都在 1-2pp 內,不是壓倒性的。

**但這裡有個新的複雜度**:Phase 2 的 exp2(雙子句抽樣加權,position recall
+3.1pp 的改善)是建立在 `full_v2` 上做的,還沒有在 `filtered_v2` 上試過同樣
的改動有沒有效。如果 Jack 想選 filtered 當主力,建議先確認 exp2 這個改善能不能
遷移過去,不然兩邊比較基礎不對等(一個有加權改善、一個沒有)。

## 2. exp2(目前最佳版本)要不要正式取代 full_v2 成為新基準

`checkpoints/best_model_exp2_reweight2x.pt`:position recall 44.7%(+3.1pp
vs full_v2)、binding accuracy 70.3%(-0.6pp vs full_v2)。是小幅淨正,但不是
全面提升。選項:
- (a) 接受這個 trade-off,exp2 變成新的 full 版基準
- (b) 覺得 -0.6pp binding accuracy 不能接受,維持用 full_v2
- (c) 兩個都留著,依下游任務(比較看重 recall 還是 accuracy)決定用哪個

## 3. 要不要修 tiny-bbox threshold 讓 far 距離物件保留率更高

第二晚已經從 0.05% 修到 0.025%(far 保留率 0%→39.7%/47.2%),但這個修正
意外沒有讓訓練資料裡的雙子句比例上升(23.8%→20.2%,因為 threshold 放寬也讓
更多圖片的 car 數量衝過 `build_caption()` 的「count>=5 就 collapse」門檻)。
如果 Jack 覺得 0.025% 還不夠、想繼續調低,或想同時處理這個 collapse 門檻
(例如把 count>=5 的門檻拉高,讓雙子句在物件多的圖片也能保留),這些都超出
Phase 1 的修復範圍,需要另外決定要不要做。

## 4. 要不要投入更多時間處理 position binding,還是先進 ONNX/量化主線

兩晚下來的進展:第一晚發現 filtered 版 position binding 明顯較弱(recall 差
16.0pp)→第二晚修好兩個 bug,gap 幾乎消失(只差 0.3pp)→Phase 2 三個改善
方向裡只有一個(適度抽樣加權)有效,且只帶來 +3.1pp,另外兩個方向(加長
epoch、降低 LR)都因為過擬合更早發生而失敗。**報酬遞減的訊號已經出現**——
繼續在「訓練程序」這個層級摳,可能很難再擠出大幅進步,如果要繼續大幅提升
position binding,可能需要動到 Phase 1/2 範圍外的東西(句型模板重新設計、
資料生成邏輯的 collapse 規則等)。這個優先順序(繼續投入 vs 轉去 ONNX/量化)
要 Jack 決定。

## 5. Phase 2 三個方向都跑完了,沒有窮舉所有可能

今晚在允許範圍內(epoch/sampling/LR)試了三個方向(抽樣加權 x2、epoch
10→16、降低峰值 LR),一個保留兩個 revert,都有記錄在
[experiments_log.md](experiments_log.md)。**沒有再自己發明第四個方向**,
如果 Jack 想繼續試(例如加權倍數 1.5x/2.5x 之間找更精細的甜蜜點,或不同的
warmup steps 單獨測試而不跟降 LR 綁在一起),還有空間,但今晚沒有做窮舉,
是主動停下來等 Jack 決定要不要繼續。

## 6. exp3/exp4 都顯示模型在這個資料量級下很容易過擬合(<10 epoch 就達到最佳)

三次獨立訓練(full_v2、exp2、exp3、exp4)的 best epoch 都落在 epoch 2-8 之間,
從沒有超過 epoch 8。如果之後要換更大的訓練資料量或不同資料集,這個「容易
過擬合」的現象值得留意,可能需要更強的正則化(weight decay 目前是 0.1,或
其他 dropout 之類的手段,這些都超出今晚 Task C 允許碰的範圍,不在今晚驗證
範圍內)。
