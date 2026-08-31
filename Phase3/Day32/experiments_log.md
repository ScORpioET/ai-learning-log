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

---

## Phase 2 - 2026-08-31 16:4x - Phase 1 較優的 checkpoint(best_model_full_v2.pt,
所有指標都優於 filtered_v2)為基礎,依序嘗試 exp2/exp3/exp4

### exp2 - commit `9e30fda`(先跑錯一次,已清乾淨重跑,見下方 ⚠️)

假設: 用比昨晚失敗的 x3 更溫和的 x2 倍數,重新加權雙 position 句子抽樣機率。

**⚠️ 過程犯的錯,誠實記錄**:第一次跑 exp2 時忘記昨晚 exp1 的 `train_vlm.py`
改動已經被 `git revert` 掉了,直接下 `+train.reweight_multi_position=true`
參數,Hydra 沒有報錯(`+` 前綴允許新增任意 key),但程式碼裡根本沒有任何地方
讀取這個 key,等於整個實驗跑成一次跟 `full_v2` 一模一樣的 shuffle=True 訓練
(兩條 loss curve 幾乎一致:0.519871 vs 0.519731 epoch0 train loss,只有浮點/
CUDA 非決定性造成的極小差異)。**發現後整組清掉(checkpoint、eval csv、log
目錄)重新來,把 reweight 邏輯重新加回 `train_vlm.py`,用同樣的 x2.0 倍數
重跑一次,這次日誌裡有印出 `reweight_multi_position=True: 1713/10241 筆
雙子句樣本,權重 x2.0` 確認邏輯真的有生效。**

改動(重跑後,實際生效版本): `data=full_v2` + `+train.reweight_multi_position=true`
+ `+train.multi_position_weight=2.0`,epoch=10(跟 full_v2 baseline 一致)。

結果:

| 指標 | full_v2(baseline) | exp2(reweight x2,真的生效) |
|---|---|---|
| binding accuracy | 70.9% | 70.3%(-0.6pp) |
| mismatch rate | 29.1% | 29.7%(+0.6pp) |
| **position recall** | 41.6% | **44.7%(+3.1pp)** |
| position precision | 41.4% | 41.1%(-0.3pp) |

結論: **保留**。position recall(Task A/B 鎖定的核心問題指標)真的有提升,
其他指標小幅下降但幅度不大,net positive。往下一個 exp 疊加。

### exp3 - 增加 epoch 數(10→16),建立在 exp2 的 reweight x2 設定上

假設: 訓練更久,position binding 有沒有隨之改善,同時觀察過擬合訊號。

結果:val loss 在 epoch 7(0.3827)觸底,之後一路回升到 epoch 15(0.4176)——
**明確過擬合訊號**(train loss 持續降到 0.270,val loss 卻回升),epoch 13
還出現 gradient norm 異常尖峰(119.46,平常都在 5-25 之間)。`train_vlm.py`
的 checkpoint 機制只存 val_loss 最低的那次,所以實際存下來的是 epoch7
(0.3827),不是 epoch16 訓練完的最終權重。

評估這個 epoch7 checkpoint(注意:雖然跟 exp2 一樣是 epoch7,但因為
`train.epoch=16` 改變了 cosine LR schedule 的總步數分母,兩者的 LR 曲線
形狀不同,不是完全一樣的訓練軌跡):

| 指標 | exp2(epoch10 budget,best@ep7) | exp3(epoch16 budget,best@ep7) |
|---|---|---|
| binding accuracy | 70.3% | 67.0%(-3.3pp) |
| mismatch rate | 29.7% | 33.0%(+3.3pp) |
| position recall | 44.7% | 41.9%(-2.8pp) |
| position precision | 41.1% | 40.4%(-0.7pp) |

結論: **revert**。全部指標都比 exp2 差,不只是「訓練更久沒幫助」,是「把
epoch 預算設定到 16 這件事本身」透過 LR schedule 改變讓早期訓練軌跡變差了。
checkpoint、eval csv、log 目錄都已刪除(沒有 code 改動需要 git revert,只有
hydra override,不影響 `train_vlm.py`)。**確認 Day33 提過的過擬合訊號在這個
模型上是真的,固定在 10 epoch 附近(best 通常落在 epoch 6-8)是合理的訓練
長度,不需要也不應該加長。**

### exp4 - 調整 LR schedule(降低峰值 LR + 拉長 warmup),建立在 exp2 上

假設: 現有 schedule 已經是 warmup+cosine(不是任務原本預期的「固定 LR 或簡單
decay」),所以測的是同一種 schedule 形狀下,峰值 LR 溫和一點會不會更穩定
(exp3 觀察到 gradient norm 偶爾飆到 119,懷疑峰值 LR 3e-4 可能偏高)。

改動: `train.max_lr=1.5e-4`(原本 3e-4 的一半) + `train.warmup_steps=150`
(原本 50 的三倍),其餘同 exp2(`data=full_v2` + reweight x2 + epoch=10)。

結果:gradient norm 確實變穩定(6-17 區間,沒有 exp3 那種尖峰),但 **val loss
在 epoch 2(0.3896)就觸底,之後一路狂升到 epoch 9(0.4602)**——比 exp3
更早、更明顯的過擬合,可能是峰值 LR 降太多、加上模型在少數幾步內就把訓練集
記住了。存下來的 checkpoint 是 epoch2 那個:

| 指標 | exp2 | exp4(降 LR + 拉長 warmup) |
|---|---|---|
| binding accuracy | 70.3% | 61.2%(-9.1pp,明顯變差) |
| mismatch rate | 29.7% | 38.8%(+9.1pp) |
| position recall | 44.7% | 40.7%(-4.0pp) |
| position precision | 41.1% | 39.2%(-1.9pp) |

結論: **revert**。全部指標明顯變差,checkpoint/eval csv/log 目錄都已刪除。
**這個方向被排除:目前的 warmup+cosine schedule(max_lr=3e-4,warmup=50)已經
是這個資料量級/模型大小下還算合理的設定,降低峰值 LR 沒有換來更穩定的收斂,
反而讓模型更早陷入過擬合。**

---

## Phase 2 exp 總結:exp2(reweight x2)保留,exp3(epoch16)/exp4(降LR)都
revert。目前最佳版本 = exp2 checkpoint(`checkpoints/best_model_exp2_reweight2x.pt`,
epoch7, val_loss=0.3872, position recall=44.7%, binding accuracy=70.3%)。
三個 exp 都跑完,沒有再嘗試第四個方向,照任務指示停在這裡進 Phase 3。
