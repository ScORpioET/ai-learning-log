# Binary Presence Sanity Check 結論

任務範圍:只用線A既有的12張人工標記樣本(6張「誤選/不嚴重」+ 6張「確實嚴重」),
驗證 Laplacian、下游YOLO信心值(avg)、信心值落差(drop)這三個指標能不能當
「曝光是否存在」的**二元**判別訊號(不是排序,是「有沒有」)。不擴大範圍、
不跑934張全量、不建立陰性對照組、不決定門檻值去篩選既有清單。

## 資料來源

合併兩份既有輸出(依 Jack 確認的方式):
- `human_severity` 標籤 <- `severity_line_a.csv`(權威來源,`severity_line_a_manual_compare.py`
  裡的 `SELECTED` 名單本身就是分組依據)
- `laplacian_bbox` / `downstream_confidence_avg` / `downstream_confidence_drop` 數值
  <- `severity_scores_v2.json`(`severity_line_a.csv` 沒有落差欄位,只有單一
  `downstream_confidence`)
- 兩份的 `laplacian_bbox` 逐筆核對完全一致(用 assert 驗證過),沒有矛盾

`downstream_confidence_drop` 只有 11/12 張有值——太陽正中央那張
(`video-dvZBYnphN2BwdMKBc-frame-000021`)YOLO 在曝光幀完全沒偵測到任何物件,
落差算不出來(標成 N/A,不是 0)。

## 產出檔案

- `binary_presence_scatter_v1.png` — laplacian_bbox (x) vs downstream_confidence_drop (y) 散佈圖,兩組上色
- `binary_presence_strip_v1.png` — 三個指標各自的 box+strip plot,兩組分開
- `binary_presence_stats_v1.json` — 完整統計數字

## 統計表(min / 25% / median / 75% / max)

| 指標 | 組別 | min | 25% | median | 75% | max | n |
|---|---|---|---|---|---|---|---|
| laplacian_bbox | not_severe | 7.3 | 8.78 | 11.5 | 21.58 | 84.5 | 6 |
| laplacian_bbox | severe | 5.1 | 9.73 | 16.1 | 94.1 | 236.4 | 6 |
| downstream_confidence_avg | not_severe | 0.408 | 0.555 | 0.577 | 0.598 | 0.729 | 6 |
| downstream_confidence_avg | severe | 0.48 | 0.538 | 0.605 | 0.62 | 0.874 | 5* |
| downstream_confidence_drop | not_severe | -0.174 | -0.08 | 0.033 | 0.078 | 0.171 | 6 |
| downstream_confidence_drop | severe | -0.274 | -0.026 | 0.057 | 0.063 | 0.107 | 5* |

\* severe 組其中一張(太陽正中央,無 YOLO 偵測)這兩欄都是 N/A,故 n=5 不是 6。

## 能不能分開:具體數字結論

**三個指標都不能乾淨分開這12張樣本,且重疊程度都很高:**

1. **laplacian_bbox**:兩組範圍重疊區間 `[7.3, 84.5]`。**12張裡有9張(75%)落在這個重疊區間內**——not_severe組全部6張都在裡面、severe組6張裡有3張也在裡面。也就是說,不管門檻線設在7.3~84.5之間哪個位置,not_severe組至少會有樣本被門檻線誤判,severe組的3張也會被誤判。唯一分得開的只有severe組另外3張(5.1、94.1以上的118.8跟236.4)——但這3張裡有1張(5.1,太陽那張)反而是laplacian**最低**的,方向跟「嚴重->laplacian高」的假設相反。

2. **downstream_confidence_avg**:兩組重疊區間 `[0.48, 0.73]`。**11張裡有9張(82%)落在重疊區間內**(not_severe組5/6、severe組4/5)。分得開的只有not_severe組1張(0.729,最高值,方向反而跟「不嚴重->信心值低」的假設相反)跟severe組1張(0.874,最高值,方向正確)。

3. **downstream_confidence_drop**:兩組重疊區間 `[-0.17, 0.11]`。**11張裡有9張(82%)落在重疊區間內**(not_severe組5/6、severe組4/5)。分得開的只有not_severe組1張(-0.174,最負)跟severe組1張(-0.274,更負,方向反而是「嚴重的信心值落差比不嚴重的還負」,也就是嚴重的那張下游信心值反而沒掉、甚至比乾淨參考幀還高)。

**結論:沒有任何一個指標,能用單一門檻線把這12張樣本的「有/無曝光現象」乾淨分開。** 三個指標的重疊比例都落在75%~82%之間,且僅存的幾個「分得開」的邊界樣本裡,有多筆的方向跟原本假設(嚴重->laplacian高/confidence低/drop大)是相反的,不是單純「訊號弱但方向對」,而是訊號本身在這12張小樣本上不穩定。

## 對照前一輪(排序驗證)結論

這個結果跟前一輪「halo_area/laplacian/confidence_drop 不能拿來排嚴重度排序」的結論一致且更進一步:**不僅排不出「多嚴重」,連最弱化的「有沒有曝光」二元判斷,這三個指標在12張樣本上都做不到乾淨分類。** 沒有找到證據支持這三個指標可以在任何形式(排序或二元)下替代/補強現有的候選篩選邏輯。

本輪只做了訊號存在性檢查,沒有做任何門檻決策,不影響現有840/1,994張清單。
