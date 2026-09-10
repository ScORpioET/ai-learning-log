# Saturation/Brightness Sanity Check 結論

任務範圍:延續上一輪 binary_presence sanity check,只用同一批12張線A人工標記樣本
(6張「誤選/不嚴重」+ 6張「確實嚴重」),換兩個更貼近曝光物理本質的指標——
**飽和像素比例**、**核心平均亮度**——重新驗證能不能當「曝光是否存在」的
二元判別訊號。不擴大範圍、不跑934張全量、不建陰性對照組、不決定門檻篩選清單、
不重跑或修改上一輪的 Laplacian/信心值輸出。

## 核心區域定義(沿用,未重新定義)

跟 `laplacian_bbox` 完全相同的核心裁切區域(`severity_line_a_manual_compare.py`/
`severity_line_b_full_scores.py` 共用邏輯):`black_blob_summary_matched_after_param_v2.json`
裡最大黑核 blob 的 bbox,換算回原圖解析度後,再往外擴張 `0.5*max(bbox寬,高)`。
兩個新指標都在這個區域的**灰階圖**(`cv2.cvtColor(..., COLOR_BGR2GRAY)`)上計算:

- **saturation_ratio** = 灰階值 ≥ 250 的像素數 / 區域總像素數(用轉灰階後的單一亮度值判斷,不是逐channel)
- **core_mean_brightness** = 區域內灰階像素值的算術平均數

## 產出檔案

- `saturation_brightness_scatter_v1.png` — saturation_ratio (x) vs core_mean_brightness (y)
- `saturation_brightness_strip_v1.png` — 兩指標各自 box+strip plot
- `saturation_brightness_stats_v1.json` — 完整統計數字

## 統計表

| 指標 | 組別 | min | 25% | median | 75% | max | n |
|---|---|---|---|---|---|---|---|
| saturation_ratio | not_severe | 0.0026 | 0.1597 | 0.2032 | 0.2451 | 0.2826 | 6 |
| saturation_ratio | severe | 0.1881 | 0.3671 | 0.3747 | 0.3976 | 0.4072 | 6 |
| core_mean_brightness | not_severe | 134.91 | 205.57 | 213.36 | 219.48 | 221.47 | 6 |
| core_mean_brightness | severe | 200.56 | 214.65 | 221.36 | 224.47 | 233.19 | 6 |

## 能不能分開:具體數字結論

**跟上一輪(Laplacian/信心值,重疊75-82%)相比,明顯改善,但仍不是乾淨分開:**

1. **saturation_ratio**:兩組重疊區間 `[0.1881, 0.2826]`。**12張裡有5張(42%)落在重疊區間內**——not_severe組6張裡有4張、severe組6張裡只有1張。相較上一輪三個指標(75-82%重疊),這個指標的重疊比例明顯更低,而且方向穩定一致(severe組整體都比not_severe組高,唯一的例外是severe組裡最低的那張0.1881,剛好卡進not_severe組的上緣)。

2. **core_mean_brightness**:兩組重疊區間 `[200.56, 221.47]`。**12張裡有8張(67%)落在重疊區間內**——not_severe組5張、severe組3張。這個指標的區分力比 saturation_ratio 差,重疊比例接近上一輪的 laplacian_bbox/confidence 系列(75-82%),核心平均亮度本身對「有沒有嚴重曝光」的鑑別力有限——推測原因是不管嚴不嚴重,只要核心區域本身就是亮部(街燈、車燈等光源),平均亮度都容易偏高,飽和比例才是真正拉開差距的地方。

3. **太陽正中央那張的檢查(這次沒有反方向異常)**:`saturation_ratio=0.4052`、`core_mean_brightness=225.43`,兩個指標都落在 severe 組**前83百分位**(6張裡排第5高),屬於「確實嚴重」組的合理高值範圍,**沒有像上一輪 laplacian_bbox 那樣出現數值最低、方向相反的異常**。這印證了背景推論:太陽核心是大範圍平滑漸層飽和,laplacian(局部對比度/邊緣偵測)天生量不到這種現象,但飽和比例、平均亮度這種「直接量有多亮/多少比例死白」的指標,方向是穩定符合直覺的。

## 與上一輪的對照

| 指標 | 重疊比例 | 太陽幀方向 |
|---|---|---|
| laplacian_bbox(上一輪) | 9/12 = 75% | 反方向(最低值之一) |
| downstream_confidence_avg(上一輪) | 9/11 = 82% | N/A(無偵測) |
| downstream_confidence_drop(上一輪) | 9/11 = 82% | N/A(無偵測) |
| **saturation_ratio(本輪)** | **5/12 = 42%** | **正確方向(severe組前83百分位)** |
| core_mean_brightness(本輪) | 8/12 = 67% | 正確方向(severe組前83百分位) |

**saturation_ratio 是目前四輪測試下來,鑑別力最好、方向最穩定的單一指標**——雖然42%的重疊比例仍不算「乾淨分開」,不足以單獨拿來做二元判斷,但已經比其他四個指標都好一截,值得作為後續(如果要繼續往這個方向做)優先考慮的候選指標。這輪只做訊號存在性檢查,沒有做任何門檻決策,不影響現有840/1,994張清單。
