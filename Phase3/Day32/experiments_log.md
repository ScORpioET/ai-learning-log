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

---

## Phase 1 (Day36 v2) - 2026-08-31 - commits `82023c6`/`e689686`/`115aa55`

**Task 1**:修 `GLOBAL_MIN_AREA_PCT` 跟 `far_thresh` 衝突。0.05%→0.025%(多值
sensitivity 0.015/0.02/0.025/0.03% 都測過,見 `threshold_sensitivity_v2.md`)。
far 保留率從 0% 回升到 39.7%(train)/47.2%(val)。意外發現:雙子句 caption
比例沒有如預期上升,反而 23.8%→20.2% 微降——threshold 放寬後每張圖存活物件數
變多,更多圖片的主類別(car)數量衝過 `build_caption()` 的「count>=5 就 collapse
成單子句」門檻,兩個獨立機制互相抵銷。threshold 修正本身是對的(far 不再是
0%),但這個中間指標沒有直接證明訓練效果會變好,要看 Task 4 才知道。

**Task 2**:重生 GT full 訓練資料,套用 `--long-tail-ref-split train`。查證
根因:val 樣本數少(1144 張),bike/motor/bus/truck/other vehicle 這些類別的
val 自己 counts 都低於 500,舊版沒有用 train 當基準時全部被錯誤折成
「object」;train 用 train 自己的 counts 判斷,全部遠高於 500,不會被折。
修正後(train/val 都用 train 的 counts 判斷)兩邊真名 vocabulary 一致
(object 出現次數 train 5 次 / val 1 次,比例相當;修正前是 train 11 次 vs
val 153 次,懸殊)。

**Task 3**:乾淨重訓兩版。`best_model_full_v2.pt`(epoch 8, val_loss=0.3760)、
`best_model_filtered_v2.pt`(epoch 6, val_loss=0.3856)。兩條 loss curve 都
平滑收斂,無 NaN/爆炸,比 v1 的 full 版(震盪劇烈,epoch-to-epoch std=0.053)
穩定很多。

**Task 4**:乾淨 position binding 對照,發現一個重要的分析工具混淆變因——
v1 full 版用的是舊 v0.6「there is X」句型(平均每句 1.95 個 position
segment),v2 full 版因為重新跑現在的 script 自動變成 v0.7+「class-first」
句型(平均每句只有 1.17 個 position segment,模板本身資訊密度較低,不是
資料或模型變差)。這導致 v1 full 的 position recall(57.7%)沒辦法直接跟
v2 full(41.6%)比較——不是同一把尺。**唯一乾淨的對照組是 v2 full vs v2
filtered(兩者現在用同一套 v0.7+ 模板)**:binding accuracy 70.9% vs 69.1%、
position recall 41.6% vs 41.3%(v1 時期這個 gap 是 16.0pp,現在只剩
0.3pp)。**結論:Task 1/2 兩個 bug 修完後,filtered 版跟 full 版在 position
binding 上已經打平,v1 觀察到的「filtered 版 position 能力明顯較弱」主要是
bug 造成的,不是 tiny-filter 訓練資料本身有害。**

Phase 1 全部保留(沒有 revert 的理由,兩個 bug 都確認修好、對照乾淨),詳見
`task_a_recall_gap_analysis.md`(bug 診斷)、`threshold_sensitivity_v2.md`、
`task2_full_v2_regeneration.md`、`task4_v1_vs_v2_position_binding.md`。
