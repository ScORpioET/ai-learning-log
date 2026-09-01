# Task 6: 統一 global threshold(0.05%)取代 per-class threshold

`generate_captions.py` 的 `--filter-tiny` 邏輯改成單一常數
`GLOBAL_MIN_AREA_PCT = 0.05`,不再依 class 查表。per-class 版本(Task 5)已存到
[day35_filter_check_per_class_task5/](day35_filter_check_per_class_task5/)當歷史對照,
沒有刪除。

## 保留率對照表(per-class vs global-0.05%)

| split | 版本 | 零框圖% | 平均/圖 | 整體保留% | person% | car% | bike% | motor% | truck% |
|---|---|---|---|---|---|---|---|---|---|
| val | per-class(Task5) | 5.2% | 7.69 | 72.5% | 68.0% | 74.8% | 84.1% | 80.0% | 76.1% |
| val | global-0.05(Task6) | 4.9% | 7.74 | 73.0% | **61.3%** | 78.9% | 94.7% | 96.4% | 91.3% |
| train | per-class(Task5) | 6.1% | 9.03 | 70.8% | 70.8% | 70.5% | 69.9% | 69.9% | 70.2% |
| train | global-0.05(Task6) | 6.0% | 9.15 | 71.8% | **64.0%** | 74.7% | 84.1% | 93.7% | 92.5% |

**整體統計(零框圖%、平均框數、全體保留%)兩版幾乎沒差**——這是巧合,不是
「0.05% 剛好等於 per-class 的平均效果」,而是 person/car(數量最大的兩類)一升一降
互相抵銷:person 保留率下降(per-class 特別把 person 門檻壓到 0.04% 保住 70%,
global 0.05% 比這個嚴,person 保留率掉到 61-64%),car/bike/motor/truck 保留率則
上升(這幾類 per-class 門檻本來就比 0.05% 高,例如 motor 0.21%、bike 0.10%,
global 反而對它們更寬鬆)。

## ⚠️ Flag:person 保留率在 global-0.05% 下明顯低於原本 70% 目標

**沒有跌破 30% panic 門檻(還在 61-64%),但確定低於 Task 5 設定的 70% 目標**,
是這次改動裡最主要的犧牲。這是 global threshold 的必然取捨——person 天生
bbox 面積比小,任何比 0.04% 高的全域門檻都會讓 person 保留率跌破 70%,Jack
選擇接受這個犧牲(換取「不用 per-class 這種對 skateboard/stroller/train 不可信
的湊數字」),文件裡如實記錄,不隱瞞。

**次要 flag(樣本太小,參考用)**:skateboard 在 train 只剩 17.2%(5/29 筆),
但這類反正會被 long-tail 邏輯併成 "object",對最終 caption 詞彙影響很小,
不特別處理。

## 5 張眼睛驗證圖:新舊 caption 差異

沿用 Task 5 同一批 5 個 image_id(674 / 2077 / 5452 / 5471 / 5720),新圖存在
[day35_filter_check/](day35_filter_check/),舊版(per-class)保留在
[day35_filter_check_per_class_task5/](day35_filter_check_per_class_task5/)。

| image_id | FULL(無過濾) | PER-CLASS(Task5) | GLOBAL-0.05%(Task6) | 有沒有差 |
|---|---|---|---|---|
| 674 | Night: several pedestrians, one ahead; three cars, the nearest ahead. | Night: several pedestrians, one ahead; a car ahead. | Night: several pedestrians, one ahead; a car ahead. | **一樣** |
| 2077 | Many cars, one nearby on the right. | Many cars, one nearby on the right. | Many cars, one nearby on the right. | **一樣** |
| 5452 | Many pedestrians, the nearest on the right. | Many pedestrians, the nearest on the right. | Many pedestrians, the nearest on the right. | **一樣** |
| 5471 | Several cars, the nearest ahead. | Two cars, the nearest ahead. | **Three** cars, the nearest ahead. | **不一樣** |
| 5720 | Many cars, one nearby on the right. | Several cars, one nearby on the right. | Several cars, one nearby on the right. | **一樣** |

**5 張裡 4 張完全一樣,1 張(5471)差 1 個字**(two→three cars)——查證是該圖裡
有一台車面積比落在 0.05%~0.06% 之間,car 的 per-class 門檻(0.06%)比 global
門檻(0.05%)嚴一點,把它濾掉了;global 版本保留了它。肉眼看那台車(視覺化圖
裡最右邊那個框)是清楚可辨識的小車,不是雜訊,global 版本的判斷沒有明顯比較差。

**結論:5 張樣本裡兩版差異極小,filtered caption 品質沒有因為改成 global threshold
而明顯變差**;主要的實質犧牲是統計層面 person 保留率從 ~70% 降到 ~62%,不是這
5 張圖能看出來的東西(5452 那張 person 很多的圖剛好兩版效果一樣,因為所有留下來
的 person 框面積都遠高於兩個門檻,真正被多濾掉的 person 都是更極端 tiny 的,樣本
沒抽到)。
