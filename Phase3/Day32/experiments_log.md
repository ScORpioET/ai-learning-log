# Day 36 課堂期間實驗紀錄(append-only,不要覆寫舊條目)

baseline commit: `a6f5a8f` (tag `day36-baseline`)

## exp1 - 2026-08-31 15:45 - commit `134db22`(已 revert,見 `2d15e3c`)

假設: Task A 量到 filtered 版訓練資料裡「雙 position 句子」從 94.7%(GT full)
崩到 23.8%,猜測用 WeightedRandomSampler 把雙子句樣本(含 "; " 的 caption)
抽樣機率乘 3 倍,能部分補償這個訓練訊號流失,改善 position recall/binding
accuracy。

改動: `train_vlm.py` 加一個 opt-in flag `+train.reweight_multi_position=true`
(預設 False,不影響既有行為),搭配 `+train.multi_position_weight=3.0`。
只改 DataLoader 的 sampler,不動 dataset 內容、模型架構、tokenizer。用
`data=filtered` 資料、epoch=10(跟 Task 6 的 filtered baseline 完全同設定,
只有 sampler 這一個變數),存到 `checkpoints_exp1_reweight/best_model.pt`,
複製一份到 `checkpoints/best_model_exp1_reweight.pt` 做 position binding 分析。

結果(position binding 相關指標,不是只看 loss):

| 指標 | filtered baseline(best_model_filtered.pt) | exp1 reweight |
|---|---|---|
| binding accuracy | 66.1% | **59.4%(變差)** |
| class-position 錯位率 | 33.9% | **40.6%(變差)** |
| position recall | 41.7% | 41.1%(持平,幾乎沒差) |
| position precision | 40.6% | **36.8%(變差)** |
| val_loss(best epoch) | 0.3993 (ep6) | 0.4222 (ep1) |

loss curve 本身健康(無 NaN/爆炸,train loss 穩定下降 0.557→0.306),不是訓練
失敗,是這個方向對 position binding 沒有幫助,甚至讓 precision/binding accuracy
變差(推測:3x 過度加權雙子句樣本,犧牲了單子句樣本的訓練佔比,單子句樣本
本來就佔多數且是模型基本功,過度傾斜反而讓整體品質下降)。

結論: **revert**。已用 `git revert 134db22`(產生 commit `2d15e3c`)清掉
`train_vlm.py` 的改動,checkpoint 檔案(`checkpoints_exp1_reweight/`、
`checkpoints/best_model_exp1_reweight.pt`)、eval CSV 都手動刪除(這些本來就
不進 git)。**這個方向被排除:「單純加權多 position 樣本抽樣機率」不能解決
position binding 問題,問題根源(Task A 找到的 tiny filter 跟 far_thresh 打架)
沒有從訓練程序這一層可以繞過去,需要回頭動 threshold 本身或資料生成邏輯——
但這兩個都超出今晚允許的改動範圍(只能動 epoch/sampling/LR,不能動資料生成),
留給 Jack 明天判斷。**
