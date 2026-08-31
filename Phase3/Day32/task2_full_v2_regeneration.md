# Task 2: 重生 GT full 訓練資料,套用 Day34 long-tail fix

## Bug 確認
`best_model.pt` 實際用的 `captions_train.jsonl`/`captions_val.jsonl` 檔案時間戳記
8/25 17:00,早於 generate_captions.py 註解記載的 long-tail bug 修正(8/28)。
驗證(v6 template parser,對照舊檔案):

```
train object distance distribution: {'nearby': 11}
val   object distance distribution: {'nearby': 121, 'at medium distance': 31, 'in the distance': 1}
```

根因:舊版 `main()` 在沒有 `--long-tail-ref-split` 參數時,long-tail 門檻用
「自己這個 split 的 counts」判斷。val 只有 1144 張圖,bike(170)/motor(55)/
bus(179)/truck(46)/other vehicle(63)這些類別在 val 自己的樣本數都低於
`LONG_TAIL_THRESHOLD=500`,於是被折成「object」;但同樣這些類別在 train(bike
7237/motor 1116/bus 2245/truck 829/other vehicle 1373)全部遠高於 500,不會被
折。所以「val 把 bike/motor/bus/truck/other vehicle 全部叫 object,train 卻用
真名」,才是 val 的 object 用量(153 次)遠高於 train(11 次)的真正原因。

## 修正方式
用 `--long-tail-ref-split train`,讓 val 的 long-tail 門檻判斷改成用 train 的
counts 當基準(這個參數在 Day34 已經加進 `generate_captions.py`,只是
`captions_val.jsonl` 這個實際檔案沒有拿它重新生成過)。

```
python generate_captions.py --split train --source gt --out captions_train_full_v2.jsonl
python generate_captions.py --split val --source gt --long-tail-ref-split train --out captions_val_full_v2.jsonl
```

## 驗證修正後 train/val 詞彙分布是否一致

⚠️ 順帶發現:現在的 `generate_captions.py`(v0.9)不管 `--filter-tiny` 開不開,
`--source gt` 輸出的都是 v0.7+「class-first」句型(`"Two cars, one nearby
ahead"`),不是舊 `captions_train.jsonl`(8/25 產生)那種 v0.6「there is X」
句型——這代表**新生成的 v2 full 版,跟 v2 filtered 版現在共用同一套句型模板**,
比 v1(full 用 v0.6、filtered 用 v0.7,兩者句型結構本身就不同)更適合拿來做
apples-to-apples 對照,消除了一個原本存在的混淆變因。

用 v0.7 parser 重新統計(不是舊的 v0.6 parser):

| class | train_v2 出現次數 | val_v2 出現次數 |
|---|---|---|
| car | 6950 | 834 |
| pedestrian | 4308 | 394 |
| bicycle | 309 | 15 |
| vehicle | 126 | 10 |
| truck | 111 | 4 |
| bus | 97 | 3 |
| motorcycle | 48 | 4 |
| object | 5 | 1 |

**bike/motor/bus/truck/vehicle 現在在 train/val 兩邊都用真名,不再被 val 錯誤
折成 object。「object」在兩邊都只出現個位數次(train 5 次、佔 0.05%;val 1
次、佔 0.09%),比例大致相當,不再有 v1 那種「train 11 次 vs val 153 次」的
懸殊落差。修正確認生效。**
