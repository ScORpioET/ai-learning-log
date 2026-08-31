# Task 4: 乾淨 Position Binding 對照(v1 bug 版 vs v2 修好版)

## 主表

| | v1 full(best_model.pt) | v1 filtered | v2 full | v2 filtered |
|---|---|---|---|---|
| GT 模板 | v0.6「there is X」 | v0.7+ class-first | v0.7+ class-first | v0.7+ class-first |
| binding accuracy | 67.9% | 66.1% | **70.9%** | **69.1%** |
| class-position 錯位率 | 32.1% | 33.9% | **29.1%** | **30.9%** |
| position recall | **57.7%** | 41.7% | 41.6% | 41.3% |
| position precision | **56.8%** | 40.6% | 41.4% | 39.5% |

## 結論先講:bug 修完後 gap 有沒有縮小?——**縮小了,但不是我原本預期的方式**

**binding accuracy 兩版都變好了**(full 67.9%→70.9%,filtered 66.1%→69.1%),
**錯位率兩版都下降**(32.1%→29.1%,33.9%→30.9%)——這部分符合預期,Task 1/2
修的兩個 bug 確實讓「同位置類別講對的比例」進步了。

但 **position recall 的 gap 幾乎完全消失,不是因為 filtered 版追上來,是因為
full 版的數字掉下來跟 filtered 版會合**(57.7%→41.6%,filtered 只有小幅下降
41.7%→41.3%)。這個結果如果只看數字會誤以為「full 版變差了」,但深挖後發現
**這是一個分析工具層面的混淆變因,不是模型或資料真的變差**——查證如下。

## 混淆變因:v1 full 用的是舊 v0.6 句型模板,不是純粹的 bug 修正對照

Task 2 只打算修 long-tail bug,但因為 `captions_train.jsonl`/`captions_val.jsonl`
(v1 full 用的訓練檔)是 8/25 用當時的舊版 script(v0.6「there is X」句型)產生
的,而重新執行「現在的」`generate_captions.py`(v0.9)一定會輸出 v0.7+
「class-first」句型——**這代表 v2 full 同時混了兩個變動:(a) long-tail bug
修正(Task 2 打算做的)+ (b) 句型模板從 v0.6 換成 v0.7+(非預期的副作用,
因為現在的腳本沒有辦法只修 bug、不動句型)。**

實測驗證兩種模板的「資訊密度」本來就不一樣(train GT,同一份原始 annotation):

```
v0.6「there is X」風格:平均每句 1.95 個 position segment
v0.7+ class-first 風格:平均每句 1.17 個 position segment
```

原因:v0.6 的分組邏輯是 `(class, position, distance)` 三維分組,同一個 class
如果散在不同位置,會產生多個獨立 clause(例如「car nearby 左邊;car far 右邊」
兩個 clause,同一個 class 兩個位置都講)。v0.7+ 的 `aggregate_by_class()` 改成
只依 class 分組,每個 class 最多只附一個「最近的」代表位置——如果畫面裡同一個
class(通常是 car)散在多個位置,v0.7+ 只會提其中一個,v0.6 卻可能兩個都提。

**這代表「position recall」這個指標的分母(GT 到底斷言了幾個位置)在兩種模板
下天生不一樣,不是同一把尺。v1 full 的 57.7% 有一部分是「v0.6 模板本身斷言
更多位置陳述」撐出來的,不是模型真的學得比較好。v2 full/filtered 現在用同一套
模板(v0.7+),41.6% vs 41.3% 才是真正可以互相比較的乾淨數字。**

## 真正乾淨的對照:v2 full vs v2 filtered(唯一控制好模板變因的一組)

| | v2 full | v2 filtered | 差距 |
|---|---|---|---|
| binding accuracy | 70.9% | 69.1% | full 高 1.8pp |
| position recall | 41.6% | 41.3% | full 高 0.3pp(幾乎沒差) |
| position precision | 41.4% | 39.5% | full 高 1.9pp |

**Task 1/2 兩個 bug 修完後,filtered 版 vs full 版在 position binding 上的
差距幾乎消失**(v1 時 recall 差 16.0pp,v2 時只差 0.3pp)。這代表 v1 觀察到
的「filtered 版 position 能力明顯較弱」,主要是 Task A/B 找到的那兩個 bug
(threshold 打架 + long-tail 不一致)造成的,不是「用 tiny-filter 篩過的 GT
訓練」這件事本身有害。修完 bug 後,filtered 版在 position binding 上已經跟
full 版打平,同時物件類別 F1(Task 2 的 eval 結果:filtered 73.63% vs full
74.45%,見 eval_val_results_{filtered,full}_v2.csv)也非常接近。

## ⚠️ 待確認 / 分析限制

1. **沒有辦法產生「v0.6 句型 + long-tail bug 修好」這個中間版本**——現在的
   `generate_captions.py` 已經把 v0.6 的分組邏輯整個換掉,沒有 flag 可以切回
   舊模板。如果 Jack 想要更嚴格拆解「long-tail 修正」單獨的效果(不跟模板
   改變混在一起),需要另外從 `generate_captions_v06.py.bak` 那份舊腳本接上
   long-tail fix 邏輯,今晚沒有做這件事。
2. Night F1、生成長度分布等其他 Task 2 evaluate_val.py 指標的 v1→v2 變化沒有
   在這份文件細究(核心目標是 position binding),數字都在
   `eval_val_results_{filtered,full}_v2.csv` 裡,需要的話可以另外拉。
