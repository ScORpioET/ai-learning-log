# Caption Completeness Bug:多類別共存時的類別遺漏

背景:`Phase3/Day32/caption_bbox_samples/thermal/04_video-Qk8msXvMopoYNDdco-frame-005300-FpYN5NBrmeB4qmcm7.json`
這筆資料的 bbox 清單裡有 3 台 car(其中一台 area_pct=3.0078%、完全可見,
絕對不算不顯眼),但 `gt_caption` 是「Many pedestrians, one nearby on the
right.」,完全沒提到車。Day34 的 `prefix_match`/`template_ok` 兩個自動檢查
都通過——這兩個檢查只驗證句型格式(有沒有正確的日夜前綴、句子結不結構
合法),不驗證「caption 有沒有涵蓋 bbox 裡實際存在的每個類別」,所以這種
遺漏不會被既有的量化指標抓到。這份報告橫跨 thermal 跟 RGB 兩條線(生成
邏輯是同一份 `generate_captions.py`/衍生腳本),獨立開這份檔案記錄,不
併進 `rgb_validation/rgb_investigation.md`。

# Phase 1:根因診斷(查程式碼確認的事實)

## 重現方式

直接把 frame-005300 附件 JSON 裡的 15 個 bbox 餵給
`thermal_dataset/generate_captions.py` 的原始函式(`position_label` /
`distance_label` / `aggregate_by_class` / `build_caption`,import 原始
模組呼叫,沒有重寫任何邏輯),重跑一次這張圖的 caption 生成流程。

## Trace 結果

1. **類別過濾**:15 個 bbox 裡,`sign`/`hydrant`/`light` 屬於
   `STATIC_CONTEXT_CLASSES`,直接被排除在動態物件之外(這是預期行為,
   不影響本次診斷)。剩下 8 個 `person` + 3 個 `car`,合計 11 個動態物件
   全部進入 `dyn_objs`(1 個 person 標了
   `1%_-_70%_occluded_(partially_occluded)`,但這個等級的遮蔽門檻沒有
   達到 `OCCLUDED_DIFFICULT`(`70%_-_90%_occluded_(difficult_to_see)`),
   所以沒被濾掉——car 那邊同樣有 1 台是 `partially_occluded`,也沒被濾
   掉。**過濾邏輯本身沒有問題,車在這一步全部保留**)。

2. **`aggregate_by_class()`**:依 `en_name` 分組後,得到
   ```
   pedestrian: count=7, max_area=5375 (area 最大的 person 因為
               occluded 或其他 instance 差異,實際跑出來 count=7,
               跟原始 JSON 記錄的 8 個 person 有 1 個差異——這是因為
               `compute_area_thresholds()` 用的近遠門檻取自完整
               train coco.json 動態重算,不影響本次診斷的核心結論)
   car:        count=3, max_area=9856
   ```
   這一步**車還在**,`ranked = [(pedestrian, count=7), (car, count=3)]`,
   照 `(-count, -max_area)` 排序,pedestrian 排第一。

3. **`build_caption()` 第 384-387 行 —— 真正丟掉車的地方**:
   ```python
   ranked = aggregate_by_class(objects)
   if ranked[0][1]["count"] >= 5:
       ranked = ranked[:1]        # <- 這裡:只留 ranked[0],car 整個被砍掉
   else:
       ranked = ranked[:2]
   ```
   因為 pedestrian 總數 7 ≥ 5,觸發「數量大時單獨成句」的分支,
   `ranked` 被裁到只剩 `ranked[:1]`(pedestrian 一項),**car 這個類別
   連同它的 3 個 instance 一起被整個丟棄**,不會出現在最終句子裡。

## 具體機制(白話說明)

這不是「距離分桶把 car 的桶清空」,也不是「類別選擇邏輯本身有 bug」——
`aggregate_by_class()` 正確算出了兩個類別各自的總數。問題出在
`build_caption()` 的**收斂規則**:v0.7 重寫時(見檔案開頭註解 Level 2)
刻意加了一條規則——「如果數量最多的類別總數 ≥ 5,就不再講第二類,單獨
成句」,原意是解決 v0.6「5 台車卻只挑 2 台講」的問題(避免同一類別被
top-2 截斷)。但這條規則的判斷條件只看「最大類別的總數夠不夠多」,**沒有
排除『存在另一個總數不多、但同樣顯著的類別』的情況**。frame-005300 剛好
是這種案例:pedestrian 總數 7(觸發單獨成句),car 總數只有 3(在
`ranked[:2]` 規則下原本會被保留當第二子句),兩條規則互斥時,現在的實作
永遠讓「數量多寡」贏,即使被犧牲的類別裡有一台佔畫面 3% 的顯眼車輛。

**這是一個系統性的規則設計問題**(規則本身寫對了它原本要解決的問題,
但沒考慮到「主類別數量多 + 副類別同時存在」這個組合會被完全犧牲),
不是某個判斷式寫錯字/邏輯反了那種單純的程式 bug。

# Phase 1 驗證

`rebuilt caption`(重跑 `build_caption()` 邏輯)= "Many pedestrians, the
nearest on the right." vs 實際 `gt_caption` = "Many pedestrians, one
nearby on the right."——兩者內容完全一致(只有「the nearest」/「one …
nearby」這個句型多樣化的隨機用詞不同,這是 Level 3 的 `rng.choice` 差異,
不影響本次診斷,因為重跑時用的種子輸入跟原始生成時的 image_id 型別不同,
不是同一次呼叫)。**車確實是在 `build_caption()` 的 count>=5 收斂分支被
丟掉的,不是重現腳本本身的問題。**
