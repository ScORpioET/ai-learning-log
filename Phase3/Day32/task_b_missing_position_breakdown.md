# Task B: Missing Position 系統性根因分析

## ⚠️ 開頭先講一個查出來的具體 bug,比籠統的「長尾類別訓練不足」更精確

`best_model.pt (GT full)` 表格裡 miss rate 最高的第一名是 `object / at medium
distance`(gt_n=31,漏講 21 次,67.7%,但 train_n 顯示 0)。乍看是「這個組合
train 完全沒出現過」,但 train_n=0 太極端,值得先查一次是不是分析工具的問題
還是資料本身真的這樣。追下去查到:

```
train object distance distribution: Counter({'nearby': 11})
val   object distance distribution: Counter({'nearby': 121, 'at medium distance': 31, 'in the distance': 1})
```

**`captions_train.jsonl` 裡「object」這個 long-tail fallback 詞,11 次全部都是
`nearby` 距離,一次 `medium`/`far` 都沒有;但 `captions_val.jsonl` 裡「object」
在三種距離都有出現(121/31/1)。** 查了檔案時間戳記,`captions_train.jsonl` 跟
`captions_val.jsonl` 都是 **8/25 17:00** 產生的,比 generate_captions.py 註解裡
記載的 long-tail bug 修正(8/28)還早——**這代表 best_model.pt 實際訓練用的
GT captions,現在還是帶著 Day34 記錄過的那個「train/val 各自用自己的 long-tail
門檻,導致兩邊 object 分布不一致」的舊 bug,沒有拿修好之後的版本重新生成過。**

這不是本次分析新發現的 bug 種類,是已知 bug 還沒套用到目前實際在跑的訓練檔案
上。**model 在 val 遇到「object 在中距離/遠距離」的情況,幾乎是 zero-shot——
訓練資料裡幾乎沒看過這種組合,漏講率 67.7% 完全合理,不是模型學不好,是資料
本身有洞。**


「missing」= GT 在某個 position 有講物件,生成句子在同一個 position 完全沒提(不是講錯類別,是那個位置整個沒被生成句子命中)。依 (class, distance) 分組,看是不是特定組合系統性被漏講,並對照這個組合在訓練資料裡出現的頻率。只列 val GT 出現次數 >= 5 的組合,避免樣本太少的雜訊。

## best_model.pt (GT full)

| (class, distance) | val GT 出現次數 | 漏講次數 | 漏講率 | train 出現次數 | train 佔比 |
|---|---|---|---|---|---|
| object / at medium distance | 31 | 21 | 67.7% | 0 | 0.00% |
| pedestrian / in the distance | 21 | 13 | 61.9% | 156 | 0.78% |
| car / in the distance | 33 | 20 | 60.6% | 361 | 1.81% |
| car / at medium distance | 280 | 149 | 53.2% | 2498 | 12.52% |
| object / nearby | 121 | 62 | 51.2% | 11 | 0.06% |
| pedestrian / at medium distance | 182 | 91 | 50.0% | 1540 | 7.72% |
| pedestrian / nearby | 213 | 95 | 44.6% | 2429 | 12.17% |
| car / nearby | 1260 | 454 | 36.0% | 10539 | 52.82% |

## best_model_filtered.pt (GT filtered)

| (class, distance) | val GT 出現次數 | 漏講次數 | 漏講率 | train 出現次數 | train 佔比 |
|---|---|---|---|---|---|
| vehicle / nearby | 6 | 5 | 83.3% | 97 | 0.78% |
| bus / mid | 6 | 5 | 83.3% | 44 | 0.35% |
| bicycle / mid | 19 | 14 | 73.7% | 262 | 2.10% |
| vehicle / mid | 6 | 4 | 66.7% | 48 | 0.38% |
| pedestrian / mid | 284 | 179 | 63.0% | 3086 | 24.72% |
| bicycle / nearby | 8 | 5 | 62.5% | 155 | 1.24% |
| bus / nearby | 10 | 6 | 60.0% | 102 | 0.82% |
| pedestrian / nearby | 89 | 53 | 59.6% | 1199 | 9.60% |
| car / mid | 484 | 272 | 56.2% | 3990 | 31.96% |
| car / nearby | 398 | 219 | 55.0% | 3258 | 26.10% |

## 結論

1. **GT full 版**:漏講率大致跟 train 出現頻率成反比,符合直覺——`car / in
   the distance`(train 只佔 1.81%)漏講率 60.6%,`car / nearby`(train 佔
   52.82%,最大宗組合)漏講率只有 36.0%。個案裡最極端的 `object / at medium
   distance` 更是直接對應到上面查到的 train/val long-tail 分布不一致 bug。
   **這個版本的漏講,大部分可以用「訓練樣本稀少」解釋。**
2. **GT filtered 版**:漏講率普遍偏高(55-83%),**即使是 train 佔比很大的組合
   也一樣慘**——`car / mid` train 佔 31.96%(僅次於全版最大宗的 car/nearby)、
   `car / nearby` train 佔 26.10%,漏講率卻仍然有 56.2%/55.0%,比 GT full 版
   同量級的 car/nearby(train 佔 52.82%、漏講率只 36.0%)明顯更差。**這代表
   filtered 版的漏講不能只用「這個 (class, distance) 組合訓練樣本太少」解釋
   ——就算樣本量體不小,還是漏講得更兇,呼應 Task A 的發現:filtered 版真正
   的問題是「雙 position 句型結構」本身的訓練訊號崩盤(94.7%→23.8%),不是
   單一 (class, distance) 組合的頻率問題,是更底層的句型學習能力受損。**
3. Task A + Task B 合起來看:**tiny filter 造成的傷害是複合的**——不只是移除
   了 far 距離的 GT 標註(Task A),連帶讓「一句話講兩個 position」這個句型
   結構本身的訓練訊號大幅減少,使得就算某個 (class, distance) 組合在訓練資料
   裡量還算充足,模型仍然更容易把它漏掉(Task B)。
