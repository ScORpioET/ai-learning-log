# Task A: Position Recall 落差根因分析

Task 6 發現 filtered 版(best_model_filtered.pt)position recall(41.7%)明顯低於
GT full 版(57.7%),當時只在 Day35 的 finding 文件裡標「未確認」的猜測——跟
caption 變短有關。這裡用資料統計實測驗證。

**結論先講:假設成立,而且比原本猜測的更清楚——不是「caption 變短」這種模糊
說法,是可以精確定位到「tiny-bbox filter 在數學上把 far 距離物件整批清空」
這個具體機制。**

## 1. 訓練資料裡「單句提到幾個 position」的分布

用 `position_binding_accuracy.py` 的 parser 對 train set 兩版 captions 逐句 parse
(`generate_captions.py` 的 `build_caption()` 設計上最多取 2 個 class 進句子——
`ranked[:2]`,所以理論上限就是 2,不會有 3+ 這個桶,這是模板本身的機制限制,
不是 GT full/filtered 版才有的差異):

| clause 數 | GT full (train, n=10250) | GT filtered (train, n=10086) |
|---|---|---|
| 1 個 position | 546 (5.3%) | **7687 (76.2%)** |
| 2 個 position | 9704 (94.7%) | **2399 (23.8%)** |

**filtered 版訓練資料裡,「雙 position」樣本從 94.7% 崩到 23.8%,少了將近 71
個百分點。** 這代表模型在 filtered 資料上幾乎沒看過「一句話講兩個不同方位」的
範例,訓練訊號嚴重不足,難怪生成時經常只講一個方位(或方位亂猜),直接對應
Task 6 量到的 position recall 崩盤。

## 2. 是哪個具體機制造成這個崩盤?——tiny filter 把 far 距離整批清空

`generate_captions.py` 的 `compute_area_thresholds()` 用資料集實測 25/75 百分位數
算距離門檻(train split):

```
near_thresh = 0.47%(area_ratio >= 這個值 → "near")
far_thresh  = 0.04%(area_ratio <= 這個值 → "far")
```

Day35/36 決定的 tiny-bbox 全域過濾門檻是 `GLOBAL_MIN_AREA_PCT = 0.05%`。

**問題就在這裡:0.05% > far_thresh(0.04%)。** 這不是巧合造成的輕微偏差,是
數學上的必然結果——任何被分類成「far」的物件,依定義 area_ratio <= 0.04%,
一定小於過濾門檻 0.05%,**100% 會被濾掉,沒有例外**。實測驗證:

| distance bucket | 過濾前總數 | 過濾後剩下 | 保留率 |
|---|---|---|---|
| near | 34262 | 34262 | 100.0% |
| mid | 68440 | 64069 | 93.6% |
| **far** | 34263 | **0** | **0.0%** |

train split 裡三分之一的動態物件標註屬於 far 距離(34263/136965),tiny filter
把這整批(0%不留)全部清掉。GT full 版的 caption 裡,常見的雙子句結構是
「一個 near 物件 + 一個 far 物件」(near_thresh/far_thresh 各佔資料的 25%,
兩者常常同時出現在同一張圖、被 `aggregate_by_class()` 排進句子的兩個 class),
far 那個子句消失,句子就從「兩個 position」塌縮成「一個 position」——這正好
解釋了第 1 節看到的分布崩盤。

## 3. 額外查到:position(左/中/右)本身的過濾率也不均勻

不是 Jack 原本問的「哪個 position 被濾掉比例特別高」的字面意思(那個問題原本
問的是 far/near 這種距離,不是方位),但既然可能造成混淆,順便查了左右中三個
方位各自的保留率:

| position(左右中) | 過濾前總數 | 過濾後剩下 | 保留率 |
|---|---|---|---|
| left | 33693 | 29044 | 86.2% |
| **ahead** | 73974 | 43566 | **58.9%** |
| right | 29298 | 25721 | 87.8% |

`ahead`(畫面中央三分之一)的保留率明顯比左右低很多。合理解釋:這批資料是
公路前視視角,畫面中央通常是消失點方向的遠處車流(bbox 天生小),左右兩側
較常是近距離的路邊物件(停放車輛、路人)。這是資料集本身的透視幾何特性,
不是 filter 設計的問題,但兩者疊加後,filtered 版對「ahead」方位的訓練訊號
被砍得比左右兩側更嚴重——如果之後要看 position binding 是不是 ahead 特別弱,
這條可以當延伸線索(這裡沒有進一步拆解驗證,只給第一手數字)。

## 結論

1. **假設成立,而且找到精確機制**:不是籠統的「caption 變短了所以比較難學」,
   是 **`GLOBAL_MIN_AREA_PCT`(0.05%)剛好卡在 far distance 門檻(0.04%)之上**,
   數學上必然讓所有 far 物件全部消失,直接導致雙子句訓練樣本從 94.7% 崩到
   23.8%。
2. 這不是 filtered 版「訓得比較差」,是 Task 5/6 選 threshold 時,沒有意識到
   這個值跟 `compute_area_thresholds()` 動態算出來的 far_thresh 離得這麼近,
   兩個獨立設計的門檻互相打架,產生了非預期的加乘效果。
3. **如果 Jack 想保留雙物件描述能力,threshold 要嘛明顯低於 far_thresh(讓部分
   far 物件留下),要嘛接受 far 距離描述在 filtered 版裡結構性消失**——這是
   個要 Jack 決定的取捨,不是分析可以自己下的判斷,已列入
   `pending_decisions.md`。
