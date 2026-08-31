# GT vs YOLO 覆蓋度彙總(val split,IoU>0.5 + 同 class 才算配對成功)

- 全 val 1144 張圖
- 平均每張 GT 14.53 個框(僅算 KEEP_CLASSES 11 類範圍內)
- 平均每張 YOLO 6.16 個框
- 全域配對率(matched / gt_total)= 5764/16627 = 34.7%

## 依類別 precision / recall / F1

(precision = matched/yolo_total,即 YOLO 框裡有幾成真的對到 GT;
recall = matched/gt_total,即 GT 框裡有幾成被 YOLO 抓到)

| class | gt_total | yolo_total | matched | precision | recall | F1 |
|---|---|---|---|---|---|---|
| bicycle | 170 | 65 | 51 | 78.5% | 30.0% | 43.4% |
| bus | 179 | 83 | 51 | 61.4% | 28.5% | 38.9% |
| car | 7133 | 4352 | 3816 | 87.7% | 53.5% | 66.5% |
| fire hydrant | 94 | 27 | 17 | 63.0% | 18.1% | 28.1% |
| motorcycle | 55 | 12 | 8 | 66.7% | 14.5% | 23.9% |
| person | 4470 | 1936 | 1680 | 86.8% | 37.6% | 52.5% |
| skateboard | 3 | 1 | 0 | 0.0% | 0.0% | 0.0% |
| stop sign | 2472 | 72 | 41 | 56.9% | 1.7% | 3.2% |
| traffic light | 2005 | 219 | 92 | 42.0% | 4.6% | 8.3% |
| train | 0 | 5 | 0 | 0.0% | 0.0% | 0.0% |
| truck | 46 | 272 | 8 | 2.9% | 17.4% | 5.0% |

## GT 有、YOLO 缺最嚴重的 top 3 類(recall 最低,且 gt_total>=20 才列入避免小樣本雜訊)

- **stop sign**: recall 1.7% (GT 2472 個,只配對到 41 個)
- **traffic light**: recall 4.6% (GT 2005 個,只配對到 92 個)
- **motorcycle**: recall 14.5% (GT 55 個,只配對到 8 個)

## YOLO 有、GT 缺最嚴重的 top 3 類(precision 最低,且 yolo_total>=20 才列入)

- **truck**: precision 2.9% (YOLO 偵測 272 個,只有 8 個對得到 GT)
- **traffic light**: precision 42.0% (YOLO 偵測 219 個,只有 92 個對得到 GT)
- **stop sign**: precision 56.9% (YOLO 偵測 72 個,只有 41 個對得到 GT)

## 表 A: GT bbox size 分布(依 class,area_ratio = w*h / (img_w*img_h))

- tiny: <0.5% / small: 0.5-2% / medium: 2-8% / large: >8%

| class | tiny | small | medium | large | total |
|---|---|---|---|---|---|
| bicycle | 112 | 52 | 6 | 0 | 170 |
| bus | 106 | 37 | 24 | 12 | 179 |
| car | 4912 | 1383 | 704 | 134 | 7133 |
| fire hydrant | 94 | 0 | 0 | 0 | 94 |
| motorcycle | 27 | 19 | 8 | 1 | 55 |
| person | 3981 | 359 | 117 | 13 | 4470 |
| skateboard | 3 | 0 | 0 | 0 | 3 |
| stop sign | 2385 | 85 | 2 | 0 | 2472 |
| traffic light | 1974 | 31 | 0 | 0 | 2005 |
| train | 0 | 0 | 0 | 0 | 0 |
| truck | 31 | 12 | 3 | 0 | 46 |

## 表 B: YOLO 匹配率(依 size bucket,全 class 彙總)

| size | GT 總數 | YOLO 匹配到 | 匹配率 |
|---|---|---|---|
| tiny | 13625 | 3265 | 24.0% |
| small | 1978 | 1597 | 80.7% |
| medium | 864 | 769 | 89.0% |
| large | 160 | 133 | 83.1% |
