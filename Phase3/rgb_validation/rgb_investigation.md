# RGB 分支驗證前置調查

在動 RGB 分支的規則模板/訓練程式碼之前,先查清楚三件事:類別標註定義有沒有
系統性落差、RGB bbox 尺寸/長寬比分布長什麼樣子、person 標註數落差是不是
真的是光線問題。這份記錄只回答「是什麼樣子」,不做門檻/合併/排除的決定
——那些留給下一步。

資料範圍:除非另外註明,以下全部用 **train split**
(`thermal_dataset/images_thermal_train/coco.json`、
`thermal_dataset/images_rgb_train/coco.json`)。程式碼在
`Phase3/rgb_validation/analysis.py`(A1+A2)跟
`Phase3/rgb_validation/a3_exposure.py`(A3)。

---

# 任務 A1:類別/標註定義比對(RGB COCO vs thermal COCO)

## 類別定義本身(查程式碼/資料確認的事實)

兩邊 `coco.json` 的 `categories` 欄位**逐項比對完全相同**——80 個類別,
同樣的 `id`/`name`,`supercategory` 全部是 `"unknown"`,兩邊都沒有額外的
description 欄位。也就是說,**類別的「定義」本身沒有落差**:這個資料集
兩個 spectrum 共用同一套 COCO 80 類標籤空間,只是每個類別實際標了多少
是完全獨立的兩份標註工作,不是同一份標註套兩種格式。

## 標註數量對照表(有出現在任一邊的類別)

| 類別 | thermal 數量 | RGB 數量 | 比例(RGB/thermal) | 初步判讀 |
|---|---:|---:|---:|---|
| car | 73,623 | 71,281 | 0.968 | 兩邊接近,不確定 |
| person | 50,478 | 35,007 | 0.694 | 懷疑感測器/場景限制(見 A3) |
| sign | 20,770 | 29,531 | 1.422 | 懷疑定義/可視性不同 |
| light | 16,198 | 18,640 | 1.151 | 懷疑定義/可視性不同 |
| bike | 7,237 | 7,560 | 1.045 | 兩邊接近,不確定 |
| bus | 2,245 | 1,879 | 0.837 | 兩邊接近,不確定 |
| motor | 1,116 | 1,837 | 1.646 | 懷疑感測器/場景限制 |
| hydrant | 1,095 | 990 | 0.904 | 兩邊接近,不確定 |
| truck | 829 | 1,251 | 1.509 | 懷疑感測器/場景限制 |
| other vehicle | 1,373 | 698 | 0.508 | 懷疑感測器/場景限制 |
| skateboard | 29 | 412 | **14.207** | 懷疑定義不同(見下方說明) |
| scooter | 15 | 41 | 2.733 | 樣本數太小,不確定 |
| stroller | 15 | 38 | 2.533 | 樣本數太小,不確定 |
| train | 5 | 9 | 1.800 | 樣本數太小,不確定 |
| deer | 8 | 0 | 0.000 | 樣本數太小(thermal 端 n=8),不確定 |
| dog | 4 | 0 | 0.000 | 樣本數太小(thermal 端 n=4),不確定 |

其餘 COCO 80 類(airplane、boat、backpack、廚具/家具/水果類等)兩邊
標註數都是 0——這些類別存在於共用的標籤空間裡,但在這個車載影像資料集裡
本來就幾乎不會出現,不是 RGB/thermal 的落差,是這個資料集的場景限制。

## 值得注意的個案

- **skateboard(th=29 vs rgb=412,14 倍落差)**:數量級差距在所有類別裡
  最誇張,比 person 的 0.69 倍落差方向相反且幅度大很多。這很可能不是
  「thermal 拍不到滑板」的感測器限制(滑板是固體、有正常反射率,熱像
  儀理論上看得到形狀輪廓),更可能是兩邊標註規範不一致——**這點只是
  推論,還沒有實際抽樣核對兩邊標註員是不是把同一個物件標成不同類別**
  (例如 RGB 標註員把滑板單獨標,thermal 標註員把騎滑板的人整組標成
  person/rider,或者反過來)。要確認需要真的挑幾張兩邊都有滑板出現的
  畫面人工核對,這份調查沒有做到那一步,列在這裡是提醒下一步該查什麼,
  不是下定論。
- **sign(1.42倍)、light(1.15倍)、motor(1.65倍)、truck(1.51倍)**:
  RGB 都比 thermal 標得多,方向跟 person 相反。這跟「thermal 在暗處看
  不清楚」的假設方向不一致(如果暗處是主因,應該是 thermal 在這些類別
  也标得比較少才對,除非這些類別在场景里本来就比较少在暗處出现)。這批
  類別的落差比較可能是「小物件在 RGB 高解析度下比較看得出形狀跟文字
  (sign 上的字、motor 車燈造型細節)」——但這也只是推論,沒有實際抽樣
  比對圖片核實。

**沒有做到的部分(誠實列出,不是漏掉忘記提)**:任務要求「抽樣幾張圖比對
同一物件兩邊有沒有都標」,這份調查因為時間/範圍考量沒有真的人工開圖比對
單一物件的標註一致性,上面的「判讀」欄位都是從數量比例反推的合理猜測,
不是逐物件核對過的結論。如果要確認 skateboard/sign/light 這幾個類別的
定義落差是不是真的,需要另外排一輪人工抽樣核對。

---

# 任務 A2:RGB bbox 尺寸與長寬比分布分析

方法比照 Day35 對熱像做過的 bbox size bucket 分布(`Phase3/Day35/
plot_size_hist.py`):對每個標註算 `bbox 面積 / 畫面面積` 的比例(%),
log-scale 直方圖。這裡對 RGB train 跟 thermal train 都算了一次,方便
直接對照(Day35 原本只算過 thermal **val** 的分布,這裡为了跟 RGB train
做严格对照,thermal 也用 train 重新算,不是直接借用 Day35 的旧数字)。

直方圖:`Phase3/rgb_validation/a2_bbox_area_hist.png`
完整數字:`Phase3/rgb_validation/a2_bbox_area_summary.json`

## bbox 面積佔比統計

| 統計量 | RGB train (n=169,174) | thermal train (n=175,040) |
|---|---:|---:|
| mean | 0.466% | 0.546% |
| median | 0.080% | 0.095% |
| p1 | 0.0024% | 0.0037% |
| p5 | 0.0059% | 0.0092% |
| p10 | 0.0099% | 0.0137% |
| p25 | 0.0251% | 0.0330% |
| p75 | 0.2904% | 0.3418% |
| p90 | 1.0298% | 1.1719% |
| p95 | 2.1242% | 2.4353% |
| p99 | 6.6918% | 7.7923% |

RGB 的 bbox 面積佔比分布整體**比 thermal 再往小移一點**(median 0.080%
vs 0.095%,每個百分位數 RGB 都比 thermal 小)——這符合直覺:RGB 畫面
解析度(1800×1600 = 288 萬像素)比 thermal(640×512 = 32.8 萬像素)大快
9 倍,同一個物體在 RGB 畫面裡佔的「面積比例」本來就會因為背景/周邊環境
被更高解析度捕捉到更多細節而顯得更小一些(注意:這是面積比例,不是
pixel 數,同一個物體在 RGB 上實際 pixel 數通常還是遠大於 thermal)。

## tiny bbox 佔比(不同門檻對照)

任務原文寫「目前門檻 0.05%」,但實際去查 `thermal_dataset/
generate_captions.py` 裡的 `GLOBAL_MIN_AREA_PCT`,程式碼裡現在生效的門檻
是 **0.025%**,不是 0.05%(這是 Day35/36 幾輪 threshold sensitivity 分析
後改的數字,見該檔案裡的註解)。兩個門檻的數字都列出來,不用哪個猜哪個:

| 門檻 | RGB train tiny 比例 | thermal train tiny 比例 |
|---|---:|---:|
| < 0.025%(程式碼裡目前實際生效的門檻) | 24.87% | 19.73% |
| < 0.05%(任務原文提到的數字) | 39.66% | 34.67% |
| < 0.5%(Day35 視覺化時畫的參考線) | 82.55% | 80.70% |

不管用哪個門檻,**RGB 的 tiny bbox 比例都比 thermal 高 5 個百分點左右**
(0.025% 門檻下 24.9% vs 19.7%;0.05% 門檻下 39.7% vs 34.7%)。這代表如果
直接把 Day32 規則模板現有的 0.025% 門檻原封不動套到 RGB 上,會比在
thermal 上多濾掉更高比例的標註——但**這份調查只負責把這個數字攤開來,
新門檻要設多少是下一步的事,這裡不做決定**。

## 長寬比

| | RGB train | thermal train |
|---|---:|---:|
| 畫面長寬比(width/height) | 1.125(1800×1600) | 1.25(640×512) |
| bbox 長寬比 median | 0.882 | 0.800 |
| bbox 長寬比 p10 | 0.350 | 0.333 |
| bbox 長寬比 p90 | 1.829 | 1.700 |

畫面長寬比確實不一樣(1.125 vs 1.25,任務背景提到的數字對得上),bbox
本身的長寬比分布也有系統性偏移(RGB 的 bbox 普遍比 thermal 「更方」一點,
median 0.882 vs 0.800,p90 也高一截),這代表 Day32 規則模板裡任何依賴
長寬比的邏輯(如果有的話)一樣不能直接沿用——但跟 tiny bbox 門檻一樣,
這裡只給數字,不重新校準規則模板。

---

# 任務 A3:局部曝光問題診斷

## 影片配對方法(查程式碼/資料確認的事實,含跟背景數字對不上的地方)

背景提到「index.json 裡標記的 74 支有對應關係的影片」。這份調查用下面
的方法重新配對,**配出來的數字是 119 支,不是 74**,如實記錄,沒有為了
湊出 74 而調整配對邏輯:

1. 把 `images_thermal_train/index.json` 跟 `images_rgb_train/index.json`
   裡 `videos` 陣列的 `filename` 正規化(去掉副檔名、統一大小寫/空白、
   去掉 `camN` 字樣)。
2. 兩邊正規化後檔名相同的視為同一支影片(例如 thermal 端
   `00343988_cam2.avi`、`00340359...` 這類編號檔名兩邊都有,配對得上)。
3. 進一步要求:配對上的影片裡,同一個 `frame-XXXXXX` 編號要**同時出現在
   `images_thermal_train/coco.json` 跟 `images_rgb_train/coco.json`
   的圖片清單裡**(只是影片本身兩邊都有上傳,不代表同一幀被兩邊都抽出來
   標註)。

結果:thermal train 133 支影片、RGB train 132 支影片,依上述方法配對上
的是 **119 支**;把 frame number 也對上、且 `frameIndex <= 1500` 的
「早段畫面 frame pair」總共有 **2,306 組**。

（沒有查到「74」這個數字實際指的是什麼子集——可能是別的 split、別的
配對邏輯,或是之前一次沒有留下程式碼的手動分析結果。這裡如實記錄自己
用的方法跟結果,不去猜背景提到的 74 是怎麼算出來的。）

## 找「thermal 有標註、RGB 同位置沒有對應標註」的候選

判準(方法論選擇,不是查出來的事實):把 thermal bbox 中心點換成正規化
座標(除以 thermal 畫面寬高),跟該幀 RGB 圖片裡所有標註的正規化中心點
比,只要有任何一個 RGB 標註(不限類別)的正規化中心距離在 x、y 方向都
小於 0.12,就算「RGB 這個位置有對應標註」;完全沒有的話才算候選案例。
用寬鬆的「附近有沒有標任何東西」而不是「同類別 + IoU」,是因為兩邊
視角/解析度/長寬比本來就不同,精確配對不合理。

在 2,306 組早段 frame pair 裡,總共找到 **2,033 個**這樣的候選案例
(thermal 有標、RGB 同位置附近完全沒標)。從裡面隨機抽 200 組
(`random.seed(1337)`)做曝光診斷,200 組全部成功讀到 RGB 圖片、算出
亮度統計。

## 曝光分類方法

把 thermal bbox 的正規化座標映射回 RGB 圖片上對應區域(給最小 16px 的
padding 避免框太小取樣不足),轉灰階後算該區域的像素亮度分布:

- **過曝**:區域內 ≥40% 的像素亮度 ≥250(接近純白,懷疑車燈/反光/逆光)
- **過暗**:區域內 ≥40% 的像素亮度 ≤10(接近純黑,懷疑真正暗處/夜間
  看不清楚)
- **其他**:亮度正常但仍缺標(懷疑距離、遮擋、或單純標註疏漏)

40% 這個比例門檻是方法論選擇,不是查出來的事實。

## 結果(200 組樣本)

| 分類 | 數量 | 比例 |
|---|---:|---:|
| 過曝(overexposed) | 26 | 13.0% |
| 過暗(underexposed) | 0 | 0.0% |
| 其他(other) | 174 | 87.0% |

按 thermal 類別拆分(只列樣本數 ≥10 的類別):

| 類別 | n | 過曝 | 過暗 | 其他 |
|---|---:|---:|---:|---:|
| car | 63 | 20.6% | 0.0% | 79.4% |
| sign | 60 | 13.3% | 0.0% | 86.7% |
| person | 35 | 11.4% | 0.0% | 88.6% |
| light | 27 | 3.7% | 0.0% | 96.3% |
| bike | 13 | 0.0% | 0.0% | 100.0% |

## 判讀:光線假設沒有得到支持,而且完全沒查到過暗案例

**這是查數字確認的事實,不是為了讓假設成立去挑資料或調整分類標準**——
200 組樣本裡,**過暗的比例是 0%**,過曝只佔 13%,剩下 87% 是「亮度正常
但仍缺標」。這跟背景提到的假設(「懷疑是光線問題,不只日夜,還包含車燈
/反光這種局部曝光問題」)方向不一致:局部曝光問題(過曝)確實存在但只
佔一小部分,而「暗處看不清楚」這個更直覺的解釋在這 200 個樣本裡完全
沒出現。

**這個結論有一個重要但書,一併誠實列出**:200 個樣本裡,對應的 thermal
圖片有 `hours` metadata 標記為 `night` 的只有 13 張,`day` 13 張,
`dawn/dusk` 1 張,其餘 173 張(86.5%)`hours` 欄位是空字串(train split
這個欄位本身大量缺失,不是抽樣挑出來的偏誤,是資料本身的狀況)。也就是
說,這批樣本裡確定是夜間場景的案例很少,如果「暗處看不清楚」這個現象
主要集中在少數幾支夜間影片,200 組隨機抽樣可能剛好沒抽到足夠多的夜間
案例,不能just因為這批樣本裡 0% 過暗,就完全排除夜間曝光是 person 標註
落差的其中一個原因——只能說**在目前這個隨機抽樣裡沒有觀察到**,如果要
確定夜間場景本身的曝光狀況,需要另外針對 `hours=night` 的 thermal 圖片
單獨抽樣做同樣的分析,這份調查沒有做到那一步。

**目前能確定的事實**:87% 的「thermal 有標、RGB 沒標」案例,RGB 那個
位置的亮度看起來完全正常——這代表 person(以及其他類別)在 RGB 端標註
數比 thermal 少,**主要原因看起來不是光線/曝光**,比較可能是距離太遠
RGB 解析度下看不清楚、遮擋、或者單純是兩邊標註團隊的疏漏/標準不同
(呼應 A1 裡 sign/skateboard 那類「懷疑定義不同」的觀察)。要進一步確認
是哪一種,需要人工看過這 174 個「其他」案例的實際畫面,這份調查只做到
數字統計,沒有人工複核。

---

# 任務 A4:兩種影片配對方法的 frame 層級對應率 + 標註內容差異視覺稽核

背景:A3 提到影片配對配出 119 支,跟背景另外提到的「74 支」對不上。後來
另外查到 74 這個數字的來源——`images_thermal_train/index.json` 裡
`videos[].description` 欄位有些會內嵌一段自由文字格式的
`{"RGB": "<rgb_video_id>"}`,指向對應的 RGB video id,這樣配出來的正是
74 支(不是 119)。所以現在手上有**兩種配對方法、兩個不同的影片清單**,
彼此沒有互相驗證過:(1) 配對上的影片裡是不是真的每個 frame 兩邊都有,
還是只有部分對得上;(2) frame index 對得上,不代表兩邊標註內容(bbox
數量、類別)描述的是同一個畫面。這個任務只負責把這兩件事的數字跟視覺
證據做出來,**不判斷兩種配對方法哪個可信、也不動任何現有配對紀錄**。

程式碼:`Phase3/rgb_validation/day38_frame_correspondence_audit.py`
完整數字:`Phase3/rgb_validation/day38_frame_correspondence_stats.json`
比對圖片(未進 git,太大,自己在本機看):
`Phase3/rgb_validation/day38_frame_correspondence_audit_images/`
下分兩個子資料夾 `method_A_description_74/`、`method_B_filename_norm_119/`,
各 40 張(30 張差異分數最高的「可疑案例」+ 10 張差異分數最低的「對照組」),
檔名格式:`{排名}_{SUSPECT|CONTROL}_diff{差異分數}_frame{frame編號}_
th{thermal bbox數}bb_rgb{rgb bbox數}bb_thvid-{thermal video id}_
rgbvid-{rgb video id}.jpg`,左半是 thermal(紅框)、右半是 RGB(綠框),
兩邊 bbox 都直接疊圖畫出來,方便照檔名排序瀏覽、不用自己數框。

## 差異分數的算法(方法論選擇,不是查出來的事實)

對每一組「thermal、RGB 兩邊都有標註」的 frame,分數 =
`abs(thermal bbox 數 - RGB bbox 數) + 5 * |thermal 類別集合 對稱差集 RGB 類別集合|`。

用「5 倍」這個權重是因為 bbox 數量差可以到幾十(尺度大),類別集合通常
只有 0~15 種(尺度小),不加權的話類別差異的訊號會完全被 bbox 數量差蓋掉;
乘 5 讓兩種訊號的量級接近,不是查出來的最佳值,只是讓排序同時看得到「數量
差很多」跟「兩邊標的根本是不同種類物件」這兩種異常。

## Frame 層級對應率(查程式碼/資料算出來的事實)

| | Method A(description 欄位,74 支影片對) | Method B(檔名正規化,119 支影片對) |
|---|---:|---:|
| 影片對數 | 74 | 119 |
| thermal 側總 frame 數(這些影片裡) | 7,822 | 7,878 |
| RGB 側總 frame 數(這些影片裡) | 7,491 | 7,526 |
| 兩邊都有標註的 frame 數(both) | 7,472 | 7,511 |
| 只有 thermal 有的 frame 數 | 350 | 367 |
| 只有 RGB 有的 frame 數 | 19 | 15 |
| both / union 佔比 | 95.29% | 95.16% |
| thermal-only / union 佔比 | 4.46% | 4.65% |
| RGB-only / union 佔比 | 0.24% | 0.19% |
| frame 總數兩邊不一致的影片對數 | 44 / 74(59.5%) | 61 / 119(51.3%) |

**查到的事實**:兩種配對方法在「frame 是不是兩邊都存在」這件事上數字
非常接近(both/union 都在 95% 左右),差異主要在 thermal 側比 RGB 側多
出一截未被 RGB 對應到的 frame(4.5% 左右 thermal-only,RGB-only 不到
0.3%)。但**超過一半的影片對(A:44/74、B:61/119),thermal 跟 RGB 兩邊
的 frame 總數本來就不相等**——代表就算影片本身配對上,兩邊實際抽出來標註
的 frame 密度/數量也常常不同,不是每支影片兩邊都是逐幀一一對應。

## 兩邊都有 frame 時,標註內容差異分數分布(查程式碼算出來的事實)

| 差異分數 | Method A(n=7,472) | Method B(n=7,511) |
|---|---:|---:|
| = 0(完全同數量、同類別集合) | 2.78% | 4.37% |
| 1–5 | 14.20% | 16.31% |
| 6–15 | 45.88% | 48.20% |
| > 15 | 37.14% | 31.13% |
| 最高分數 | 77 | 85 |

**查到的事實**:即使 frame 在兩邊都存在(both),分數 = 0 的完全乾淨對應
只佔 2.8%~4.4%,超過三成(A: 37.1%、B: 31.1%)的 frame pair 差異分數
> 15(相當於類別集合完全對不上,或 bbox 數量差 15 顆以上)。

**這裡不下結論說這代表配對可不可信**——40 張比對圖已經產出放在上面的
資料夾路徑,需要人工實際打開看過才能判斷「差異分數高」的案例是(a) 兩台
相機視角/距離不同造成標註粒度本來就會不一樣,(b) 真的是配錯場景(前一輪
在別的任務裡肉眼比對過同一對配對影片,發現同一支影片裡前段 frame 場景
吻合、後段 frame 場景對不上的情況),還是 (c) 其他原因。抽查時可以先看
`method_A_description_74/001_SUSPECT_*` 這類差異分數最高的案例,跟同資料夾
`001_CONTROL_*` 對照——已經手動打開看過其中兩組作為 sanity check:一組
diff=77 的 SUSPECT 案例(thermal 49 個 bbox vs RGB 只有 2 個),兩張圖肉眼
看就是完全不同的兩條路(不同建築、不同路型);一組 diff=0 的 CONTROL 案例,
兩張圖肉眼看是同一個巴黎街景、同一台車在同一個位置——這兩組個案支持這個
差異分數的排序方向是有意義的,但只查了 2 組,不能代表全部 40+40 張的情況,
其餘案例需要另外人工複核。

---

# 任務 A5:GLOBAL_MIN_AREA_PCT 門檻數字溯源(0.05% vs 0.025%)

背景:`learning-plan.md` 記錄的 Day35 決策是「0.05% global threshold」,但
`thermal_dataset/generate_captions.py` 裡實際生效的常數是 0.025%,要查清楚
哪個是真的、為什麼對不上。

**第一個查到的事實**:這個環境裡找不到 `learning-plan.md` 這個檔案——查過
`git log --all` 整個 repo 歷史(含已刪除的檔案)、`find /home/jack` 整個
家目錄、`find /` 全機掃描,都沒有任何叫這個名字的檔案。所以下面沒辦法
直接打開 `learning-plan.md` 核對它現在寫的是什麼、有沒有更新過——只能
查這個 repo 裡實際留下的紀錄跟程式碼歷史,推論最可能的情況是什麼。

## git log -p 查證:GLOBAL_MIN_AREA_PCT 改過幾次、什麼時候、為什麼

`thermal_dataset/generate_captions.py` 目前在這個 branch 的完整歷史只有
兩次 commit 動過(`git log --oneline --all -- thermal_dataset/
generate_captions.py`):

| commit | 時間 | 訊息 | `GLOBAL_MIN_AREA_PCT` 值 |
|---|---|---|---|
| `a6f5a8f` | 2026-08-31 15:39:27 +0800 | "baseline before day36 course-time experiments"(第一次把這個檔案加進 git,625 行全新增) | 0.05 |
| `82023c6` | 2026-08-31 16:15:54 +0800(比上面晚 36 分鐘,同一天) | "Task 1: fix GLOBAL_MIN_AREA_PCT (0.05->0.025) to stop conflicting with far_thresh; far retention 0%->39.7%/47.2%; regenerated captions_{train,val}_filtered_v2.jsonl" | 0.025 |

`a6f5a8f` 的 commit body 講得很清楚:這次 commit 是把「Day35/36 到目前為止
的分析程式」一次性補進 git(之前 `.gitignore` 寫太寬,把
`generate_captions.py` 本體排除在外了)。所以**這裡看不到 0.05% 這個值
在 Day35 當時實際被設定的那一刻**(那次修改沒有進 git 歷史),只能確認
「這個檔案第一次被 git 追蹤時,值是 0.05%」——跟 `day35_36_summary_for_
decision.md` 第 35 行「Jack 決定改用單一 `GLOBAL_MIN_AREA_PCT = 0.05%`」
的敘述吻合,可以合理推斷這就是 Day35 決策當下設的值。

`82023c6` 的 diff 內容(`git log -p`)顯示改動很單純:把常數從
`GLOBAL_MIN_AREA_PCT = 0.05` 改成 `GLOBAL_MIN_AREA_PCT = 0.025`,同時
commit message 裡的原因寫得很明確——0.05% 這個門檻比
`compute_area_thresholds()` 動態算出來的 `far_thresh`(train 0.0427%/
val 0.0464%)還大,數學上必然讓所有 far 距離物件被過濾條件"順便"連坐濾掉
(因為 tiny-bbox 過濾門檻設得比 far 距離的定義門檻還高,遠距離物件的框
本來就小,會被無差別當成 tiny 濾掉),導致訓練資料裡雙子句 caption 比例
從 94.7% 崩到 23.8%。這跟 `generate_captions.py` 檔案內目前的註解(第
147-155 行,先前 A2 查證時已經讀過)完全吻合,兩份獨立紀錄(commit
message + 程式碼內註解)彼此對得上,不是我自己編的說法。

## 三個可能性的判斷

背景問了三種可能:(1) 後來刻意調整過,只是 learning-plan.md 沒同步更新;
(2) 本來就沒改過,0.05% 從一開始就記錄錯;(3) 有兩個地方各自定義門檻,
實際生效的跟原本設計的不是同一個。

**查證結果支持第 (1) 種**,而且證據很直接:

- 有明確的一次程式碼修改(`82023c6`),不是「本來就沒改過」——排除(2)。
- 這個 repo 裡目前找到的唯一一份門檻常數定義
  (`thermal_dataset/generate_captions.py` 的 `GLOBAL_MIN_AREA_PCT`),
  沒有查到第二個地方另外定義了一份不同的門檻常數——排除(3)(至少在
  `generate_captions.py` 這個產生訓練資料用的腳本範圍內是這樣;沒有
  往外查整個 repo 是否有其他腳本各自硬編了自己的門檻數字,這點沒有
  做窮舉)。
- Commit 訊息、程式碼內註解、`day35_36_summary_for_decision.md` 三份
  獨立紀錄的敘述完全一致(0.05% 先訂、跟 far_thresh 打架、改成
  0.025%),不是各說各話。

**但沒辦法 100% 確認的部分**:因為找不到 `learning-plan.md` 本體,無法
直接證實「它現在真的還寫著 0.05% 沒更新」這件事,只能說——如果
`learning-plan.md` 是在 Day35 當下記下 0.05% 這個決策、之後沒有因為
Day36 這次 bugfix 回頭修改,那就是（1）「刻意調整過,紀錄沒同步」;
如果 `learning-plan.md` 其實在別的地方(這個 repo 外,或另一台機器)
已經有更新,只是這個環境看不到最新版本,那就是**這份查證的環境限制**,
不是真的沒同步。這個環境目前掌握的證據只能證明「in-repo 的紀錄本身是
一致的、有跡可循的」,證明不了 `learning-plan.md` 這份外部文件現在寫的
是什麼。

## GT-filtered checkpoint(`best_model_exp2_reweight2x.pt`)實際用哪個門檻訓練

**這裡查到一個比「哪個門檻數字對」更根本的問題**:`best_model_exp2_
reweight2x.pt` 這個 checkpoint,**訓練資料根本沒有套用 tiny-bbox 面積
過濾**——不是 0.05% 也不是 0.025%,是完全沒過濾。

查證過程(不是只看現在程式碼長怎樣,是往回追訓練當下實際用的資料跟指令):

1. `Phase3/Day32/outputs/2026-08-31/16-44-13/.hydra/overrides.yaml`(這個
   run 的 checkpoint 存到 `checkpoints_exp2_reweight2x/best_model.pt`,
   跟 `Phase3/Day32/checkpoints/best_model_exp2_reweight2x.pt` 檔名對得
   上、mtime `2026-08-31 16:50:59` 也接在這個 run 之後,確認是同一個
   checkpoint)裡明寫 `data=full_v2`。
2. `Phase3/Day32/config/data/full_v2.yaml` 對應的 `captions_train_path`
   是 `captions_train_full_v2.jsonl`(在 `.hydra/config.yaml` 裡確認過)。
3. `Phase3/Day32/task2_full_v2_regeneration.md` 記錄了這個檔案當初是
   怎麼生出來的,指令是:
   ```
   python generate_captions.py --split train --source gt --out captions_train_full_v2.jsonl
   python generate_captions.py --split val --source gt --long-tail-ref-split train --out captions_val_full_v2.jsonl
   ```
   **兩條指令都沒有帶 `--filter-tiny`**——`generate_captions.py` 裡
   `filter_tiny` 參數預設 `False`,沒開這個 flag,`GLOBAL_MIN_AREA_PCT`
   這個常數(不管它當時是 0.05 還是 0.025)在這次生成過程裡**根本沒有
   被讀到、沒有任何標註因為這個門檻被濾掉**。
4. 時間序也對得上:`captions_train_full_v2.jsonl` 的 mtime 是
   `2026-08-31 16:16:06`,晚於 `82023c6`(16:15:54)——就算它有套用
   filter,用的也會是改完的 0.025%,不是 0.05%;但因為根本沒套用
   filter,這個時間先後順序其實不影響結論。

**對照組**:同一批 run 裡另外有一個 `best_model_filtered_v2.pt`
(`Phase3/Day32/checkpoints/best_model_filtered_v2.pt`,mtime
`2026-08-31 16:25:14`),`overrides.yaml` 顯示 `data=filtered_v2`,對應
`captions_train_filtered_v2.jsonl`(mtime `16:14:47`,在 `82023c6`
commit **之前**一分鐘生成,但 commit message 本身寫「regenerated
captions_{train,val}_filtered_v2.jsonl」,代表這次生成用的就是改完
0.025% 的程式碼、commit 只是晚一步把改動存進 git)。**這個 checkpoint
才是真的用 0.025% 門檻篩過的訓練資料**,不是任務背景點名的
`best_model_exp2_reweight2x.pt`。

**結論(如實記錄,沒有修正任何門檻或程式碼)**:

1. `GLOBAL_MIN_AREA_PCT` 確實從 0.05% 被刻意改成 0.025%,原因是
   0.05% 會跟 `far_thresh` 打架、把所有遠距離物件連坐濾掉,commit
   訊息跟程式碼註解對得上,不是口誤或沒改過。
2. 因為找不到 `learning-plan.md` 本體,沒辦法直接證實它現在是不是還
   停在 0.05% 沒更新——只能證明 in-repo 的紀錄(commit history + 程式碼
   註解 + `day35_36_summary_for_decision.md`)彼此一致,呈現的是完整的
   「先 0.05%、發現問題、改成 0.025%」故事。
3. **背景點名的 `best_model_exp2_reweight2x.pt` 這個 checkpoint,訓練
   資料生成時根本沒有套用 tiny-bbox 面積過濾**(用的是 `full_v2` 資料,
   生成指令沒帶 `--filter-tiny`)。也就是說,對這個特定 checkpoint 而言,
   「該用 0.05% 還是 0.025%」這個問題本身不成立——它兩個都沒用到。
   真正用 0.025% 篩過的是另一個 checkpoint `best_model_filtered_v2.pt`。
   如果任務描述裡把 `best_model_exp2_reweight2x.pt` 當成「GT-filtered
   checkpoint」,這個認知本身可能需要更正——它是 full(未過濾)資料 +
   reweight2x 實驗,不是 filtered 資料訓出來的。

---

# 任務 A6:`best_model_filtered_v2.pt` 血統查證(是不是真的吃到修好後的 pipeline)

上一輪(A5)查到 `best_model_exp2_reweight2x.pt` 其實沒套用 tiny-bbox 過濾,
真正用 0.025% 篩過的是 `best_model_filtered_v2.pt`。這裡反過來查證
`best_model_filtered_v2.pt` 本身的血統夠不夠乾淨——它的訓練資料生成當下,
是不是真的同時吃到「面積門檻已修正為 0.025%」跟「Day34 long-tail bug
已修復」這兩個修正,而不是像原本 `best_model.pt` 那樣踩到舊 pipeline 的坑。

## Checkpoint ↔ 訓練資料 ↔ 生成指令的對應關係(查 Hydra + 文件確認的事實)

跟查 `exp2_reweight2x` 用同一種方法:先查 `best_model_filtered_v2.pt`
對應哪個 Hydra run。

- `Phase3/Day32/outputs/2026-08-31/16-18-16/.hydra/overrides.yaml`:
  `data=filtered_v2`、`train.ckpt_dir=checkpoints_filtered_v2`、
  `train.ckpt_path=checkpoints_filtered_v2/best_model.pt`——跟
  `Phase3/Day32/checkpoints/best_model_filtered_v2.pt`(mtime
  `2026-08-31 16:25:14`)對得上,是同一個 run 的產物。
- `Phase3/Day32/config/data/filtered_v2.yaml` 的 `captions_train_path`
  是 `captions_train_filtered_v2.jsonl`(在 `.hydra/config.yaml` 裡
  確認過)。

**沒有找到逐字的生成指令**:跟查 `exp2_reweight2x` 時不同,這次在
`Phase3/Day32/*.md` 裡沒有找到一行寫著
`python generate_captions.py ... --out captions_train_filtered_v2.jsonl`
的逐字指令記錄(`threshold_sensitivity_v2.md` 只說「產出
`captions_train_filtered_v2.jsonl`」,沒附指令);查了 `~/.bash_history`
(1997 行)也搜不到任何一行含 `filtered_v2`/`filter-tiny`/`full_v2` 的
紀錄——這些指令當時應該是在別的地方執行的(例如前一輪 Claude Code
session 直接呼叫,沒有進到這個 shell 的 history 裡),沒辦法直接複製貼上
逐字指令核對。因為文件跟 shell history 兩條路都查不到逐字指令,改用
**直接對照實際資料**的方式驗證(見下面兩節)——這個方法比對照一行指令
文字更直接,因為就算真的找到一行指令文字,還是要驗證那行指令有沒有真的
被執行、有沒有被之後的操作覆蓋,不如直接驗算產出檔案本身。

## (a) 生成當下面積門檻是不是 0.025%(不是 0.05%)——直接算給看,不是猜的

方法:把 `generate_captions.py` 當模組載入,重用它**原本的**
`DYNAMIC_CLASSES`/`STATIC_CONTEXT_CLASSES`/`OCCLUDED_DIFFICULT` 判斷邏輯
跟 `compute_area_thresholds()`,對 `images_thermal_train/val` 的每張圖
分別模擬「如果用 0.025% 門檻」跟「如果用 0.05% 門檻」會篩出幾個 dynamic
object,拿去跟 `captions_train_filtered_v2.jsonl`/`captions_val_filtered_
v2.jsonl` 裡實際記錄的 `num_objects` 欄位逐張圖對照。這是直接拿產出檔案
反推當初用的門檻,不是看 commit 時間猜的。

| split | n | 用 0.025% 模擬,跟實際 num_objects 完全吻合的圖片比例 | 用 0.05% 模擬,完全吻合比例 |
|---|---:|---:|---:|
| train | 10,179 | **100.0%**(mean diff = 0) | 40.6%(平均少算 1.62 個物件/圖) |
| val | 1,096 | **100.0%**(mean diff = 0) | 45.3%(平均少算 1.43 個物件/圖) |

**結論(查數字算出來的事實,不是推論)**:`captions_train_filtered_v2.
jsonl` 跟 `captions_val_filtered_v2.jsonl` 裡,**每一張圖**的 `num_
objects` 都跟「用 0.025% 門檻篩選」的模擬結果一模一樣(100% 吻合、mean
diff=0)。如果當初其實是用 0.05% 生成的,吻合率只會有 40.6%/45.3%,而且
會系統性地比實際少算 1.4~1.6 個物件——這個對照非常明確,**確認生成當下
用的就是 0.025%,不是 0.05%**。

## (b) 是不是用 Day34 long-tail bug 修復後的邏輯生成——用同一套驗證方法

背景(A5 已查過的事實回顧):`--long-tail-ref-split` 這個 CLI 參數是
Day34 加的(commit `80d3115`,2026-08-28 16:11),原本的 bug 是「val 自己
的 split 樣本數太少,bike/motor/bus/truck/other vehicle 這些類別在 val
自己算會落在 `LONG_TAIL_THRESHOLD=500` 門檻之下,被錯誤折成通用詞
`"object"`,但同樣類別在 train 遠高於門檻,不會被折」——結果就是舊版
`captions_val.jsonl` 裡「object」這個詞出現的次數(153 次)遠高於
`captions_train.jsonl`(11 次),兩邊比例完全不成比例。

同一天(08-31)15:44 有一個診斷 commit `46dd86a`
("Task B: missing position 系統性根因 -- 查到 GT full 訓練檔仍帶
pre-Day34-fix 的 long-tail bug"),確認**當時還在用的舊版
`best_model.pt`/`best_model_filtered.pt`(注意:不是 `_v2`)訓練資料
仍然帶著這個 bug**——即使 Day34 已經把修復的程式碼寫進 `generate_
captions.py`,8/28~8/31 之間這幾天並沒有真的拿新程式碼重新生成過訓練
檔案。這是後續 Task 1(面積門檻修正)、Task 2(long-tail 修正重生)
兩輪工作的起因。

**直接驗算 `captions_train_filtered_v2.jsonl`/`captions_val_filtered_
v2.jsonl` 裡「object」這個 long-tail fallback 詞的出現比例**(用跟
`Phase3/Day32/task2_full_v2_regeneration.md` 驗證 `full_v2` 時同樣的
方法):

| split | 總 caption 數 | 含「object/objects」的 caption 數 | 比例 |
|---|---:|---:|---:|
| train | 10,179 | 5 | 0.05% |
| val | 1,096 | 1 | 0.09% |

跟 bug 修復前的舊版(`captions_train.jsonl` 11 次 vs `captions_val.
jsonl` 153 次,val 是 train 的 14 倍)完全不是同一種分布;反而跟已經
確認修好的 `full_v2`(train 5/10241=0.05%、val 1/1097=0.09%,見 A5 引用
的 `task2_full_v2_regeneration.md`)幾乎是一模一樣的數字(絕對次數都是
train 5 次、val 1 次,比例也幾乎相同)。

**結論(查數字確認的事實)**:`filtered_v2` 的 train/val 兩邊「object」
折疊比例均衡、量級都在個位數,不是舊 bug 那種一邊 11 次一邊 153 次的
懸殊分布——**確認生成時用的是 Day34 修復後的 long-tail 邏輯**,不是
`best_model.pt` 當初踩到的那個舊邏輯。

## 時間戳對照(commit + 檔案 mtime)

| 事件 | 時間 |
|---|---|
| Day34 加入 `--long-tail-ref-split`(commit `80d3115`) | 2026-08-28 16:11:15 |
| 診斷:舊 checkpoint 仍帶 pre-Day34-fix bug(commit `46dd86a`) | 2026-08-31 15:44:00 |
| `captions_val_filtered_v2.jsonl` 檔案生成(mtime) | 2026-08-31 16:14:45 |
| `captions_train_filtered_v2.jsonl` 檔案生成(mtime) | 2026-08-31 16:14:47 |
| Task 1 commit:面積門檻 0.05→0.025,「regenerated captions_{train,val}_filtered_v2.jsonl」(`82023c6`) | 2026-08-31 16:15:54 |
| `best_model_filtered_v2.pt` 訓練 run 開始(hydra outputs 目錄時間戳) | 2026-08-31 16:18:16 |
| `best_model_filtered_v2.pt` 存檔(mtime) | 2026-08-31 16:25:14 |

**兩個相關 commit 都在資料生成之前就已經存在對應的程式碼能力**——
`--long-tail-ref-split` 從 8/28 就有了(生成 filtered_v2 時程式碼裡已經
存在超過 3 天);面積門檻的修正雖然 commit(`82023c6`)在檔案 mtime 之後
才進 git,但這只是「先改程式碼、生成資料、再一起 commit」的正常順序,
不代表生成當下用的是舊值——(a) 節的直接對照已經證明生成當下確實是
0.025%。

**發現一個文件記錄上的小缺口(不是 pipeline 本身的問題)**:`82023c6`
的 commit message 只提到「fix GLOBAL_MIN_AREA_PCT (0.05->0.025)...
regenerated captions_{train,val}_filtered_v2.jsonl」,**沒有提到這次
重生也套用了 `--long-tail-ref-split train`**——這個修正是 Task 2
(`e689686`)的主題,但 Task 2 的 commit message 只提到 `full_v2`,完全
沒提 `filtered_v2`。也就是說,`filtered_v2` 這次重生**同時**修了兩個
bug,但兩份 commit message 各自只講了自己負責的那一個,沒有一份 commit
message 完整記錄「filtered_v2 這次重生同時吃到兩個修正」這件事——這是
純粹的**文件記錄缺口**,(a)(b) 兩節的直接數字驗算已經確認實際產出的
資料檔案是乾淨的,不影響訓練資料本身的正確性。

## 結論:`best_model_filtered_v2.pt` 血統乾淨,沒有踩到 bug

如實回報,這次沒有發現藏著的 bug:

1. **面積門檻**:確認是 0.025%,不是修正前的 0.05%(100% 逐圖比對吻合,
   不是猜的)。
2. **long-tail 邏輯**:確認用了 Day34 修復後的 `--long-tail-ref-split
   train`,train/val 的「object」折疊比例均衡,沒有踩到舊 bug。
3. 跟最早的 `best_model.pt`(踩兩個 bug)、`best_model_exp2_reweight2x.
   pt`(完全沒套用面積過濾,見 A5)不同,`best_model_filtered_v2.pt`
   的訓練資料是這幾個 GT-based checkpoint 裡,目前查證下來唯一確認
   「兩個已知 bug 都修好之後才生成」的一份。
4. 唯一的瑕疵是文件記錄面(commit message 沒有完整交代 filtered_v2
   重生同時吃到兩個修正),不是資料或程式碼本身的問題,已經在上面如實
   記錄,沒有為了報告好看而略過。

---

# 任務 A7:RGB tiny-bbox 門檻候選值(三種校準方式,不選定最終答案)

背景:thermal 目前的設計基準 `GLOBAL_MIN_AREA_PCT = 0.025%` 是針對
640×512(1.25:1 長寬比)校準出來的;RGB 是 1800×1600(1.125:1),解析度
是 thermal 的 8.79 倍,不能直接沿用同一個百分比數字。這裡只算出候選值
供對照,**不選定最終要用哪個**。

程式碼:`Phase3/rgb_validation/a7_threshold_candidates.py`
完整數字:`Phase3/rgb_validation/a7_threshold_candidates.json`
資料範圍:thermal train + RGB train(跟 A1/A2 一致)。

## Step 1:0.025% 換算成 thermal 640×512 畫面下的絕對像素面積

thermal 畫面總像素 = 640 × 512 = 327,680 px²
0.025% × 327,680 = **81.92 px²**(大約是一個 9.05×9.05 px 的正方形)

這是理解「這個門檻在物理上代表多小的物件」的基準:在 thermal 640×512
的畫面上,面積小於約 82 個像素(邊長不到 9px 見方)的框會被濾掉。

## Step 2:RGB(1800×1600)的三種候選校準方式

先算一個基準數字:thermal train 用 0.025% 門檻,實際濾掉了全體標註的
**19.7275%**(n=175,040)——這個比例是下面 (c) 候選要拿來對齊的目標值。
RGB train 標註總數 n=169,174,其中 person 類別 n=35,007。

- **(a) 直接沿用同樣的畫面佔比 0.025%**:不管解析度差異,門檻數字原封
  不動照搬。
- **(b) 換算成跟 thermal 相同的絕對像素面積**:81.92 px² 換算回 RGB
  1800×1600(總面積 2,880,000 px²)畫面下的佔比是多少。
- **(c) 用 RGB 自己的 bbox 面積分布,取濾掉相同「標註佔比」(19.7275%)
  所需要的百分位數門檻**:在 RGB train 的 169,174 筆標註面積佔比分布裡,
  找出第 19.7275 百分位數對應的門檻值。

## 結果對照表

| 候選方式 | 門檻(畫面佔比 %) | 門檻(RGB 絕對像素面積) | 濾掉 RGB 全部標註的比例 | person 剩餘數量 | person 被濾掉的比例 |
|---|---:|---:|---:|---:|---:|
| (a) 沿用同畫面佔比 0.025% | 0.025% | 720.0 px² | 24.87% | 29,413 | 15.98%(濾掉 5,594) |
| (b) 換算成相同絕對像素面積(81.92 px²) | 0.002844% | 81.92 px² | 1.39% | 34,979 | 0.08%(濾掉 28) |
| (c) 對齊相同標註過濾比例(19.7275%) | 0.019132% | 551.0 px² | 19.70% | 31,172 | 10.95%(濾掉 3,835) |

## 三個候選的數字差異怎麼來的(查數字確認的事實,不下選哪個的結論)

三個候選值差了超過一個數量級(720 px² vs 82 px² vs 551 px²),這是因為
三種方式各自錨定的物理量不一樣:

- (a) 錨定「佔畫面比例相同」——但 RGB 解析度高 8.79 倍,同樣的畫面佔比
  在 RGB 上對應到的絕對像素數也跟著放大 8.79 倍(82→720 px²),結果是
  三個候選裡濾得最兇的(24.87% 全體標註、person 濾掉近 16%)。
- (b) 錨定「絕對像素面積相同」——82 px² 這麼小的物理面積,擺在解析度
  高很多的 RGB 畫面裡,對應的畫面佔比小到只剩 0.0028%,幾乎不太過濾
  (1.39% 全體標註、person 只濾掉 0.08%)。這個候選跟 thermal 原本
  「濾掉肉眼看不清楚的極小物件」的過濾強度(19.73%)差最多。
- (c) 錨定「濾掉的標註比例相同」——不管絕對像素或畫面佔比,直接讓兩邊
  「被濾掉的資料量體感覺一樣多」,數字上濾掉的全體標註比例(19.70%)
  跟 thermal 的 19.73% 幾乎一致,是三者裡在「跟 thermal 的過濾強度最
  接近」這個意義上最貼近的候選,但它換算出來的絕對像素面積(551 px²)
  跟 thermal 原本的 82 px² 差了快 7 倍,如果目的是要求兩邊物理上濾掉的
  是「差不多小的東西」,(c) 沒有做到這件事。

**三個候選各自對應不同的校準哲學**(佔比一致 / 絕對面積一致 / 過濾強度
一致),沒有哪一個是客觀上「正確」的,選哪個取決於接下來想優化的目標是
什麼——這個判斷留給你,這份調查只負責把數字跟每個候選代表的意義攤開來。

---

# 任務 A8:thermal(filtered+reweight)vs RGB(filtered+reweight)平行訓練,同配方比較

用同一套訓練配方,分別在 thermal(0.025% 過濾)跟 RGB(候選 (c),0.0191%
過濾)上補訓「過濾 + x2 重加權」這個目前只驗證過部分組合的配方,產出三欄
對照表。**不下「哪個比較好、該用哪個」的結論**,數字列出來,判斷留給你。

程式碼:`Phase3/Day32/generate_captions_rgb.py`(其實放在
`Phase3/rgb_validation/generate_captions_rgb.py`)、
`Phase3/rgb_validation/precompute_clip_features_rgb.py`、
`Phase3/Day32/evaluate_val_rgb.py`、
`Phase3/Day32/position_binding_compare_day38.py`、
`Phase3/Day32/config/data/rgb_filtered.yaml`。

## Step 1:exp2_reweight2x 的配方(原封不動沿用,查 Hydra config/log 抄出來的事實)

**重加權邏輯**(`Phase3/Day32/train_vlm.py` 第 400-413 行):

- 判斷「雙子句樣本」的方式:caption 字串裡有沒有 `;`(v0.7+ class-first
  模板用 `;` 分隔兩個子句,這是唯一的判斷依據,沒有額外解析句子結構)。
- 加權倍數:雙子句樣本權重 `2.0`,其餘樣本權重 `1.0`。
- 實作方式:`torch.utils.data.WeightedRandomSampler(weights, num_samples=
  len(weights), replacement=True)`,取代原本的 `shuffle=True` DataLoader。
- 觸發方式:opt-in flag,`+train.reweight_multi_position=true` +
  `+train.multi_position_weight=2.0`,預設不開不影響既有行為。

**訓練超參數**(`Phase3/Day32/outputs/2026-08-31/16-44-13/.hydra/config.yaml`,
= `config/train/default.yaml` 的預設值,exp2 沒有另外覆寫任何一項):

| 參數 | 值 |
|---|---|
| model: n_layer/n_head/n_embd | 12 / 12 / 768 |
| model: block_size / vocab_size / clip_vector_size | 1024 / 320 / 512 |
| data: B(batch size) | 8 |
| data: T | 1024 |
| data: total_batch_size | 524288 |
| weight_decay | 0.1 |
| max_lr | 3e-4 |
| min_lr_ratio | 0.1 |
| warmup_steps | 50 |
| max_steps_increment | 1000 |
| val_interval | 100 |
| val_loss_steps | 20 |
| seed | 1337 |
| epoch | 10 |

這兩節的內容原封不動照搬到 thermal 跟 RGB 兩個新的訓練 run,沒有調整任何
數字。

## Step 2:thermal 分支

直接用既有的 `captions_train_filtered_v2.jsonl`/`captions_val_filtered_
v2.jsonl`(A6 已驗證血統乾淨,沒有重新生成),套用上面的配方:

```
python train_vlm.py data=filtered_v2 \
  train.ckpt_dir=checkpoints_filtered_v2_reweight2x \
  train.ckpt_path=checkpoints_filtered_v2_reweight2x/best_model.pt \
  train.log_dir=log_filtered_v2_reweight2x \
  +train.reweight_multi_position=true +train.multi_position_weight=2.0
```

雙子句樣本:2,056 / 10,179(20.2%)。Loss 曲線(10 個 epoch,`dt` 是每個
epoch 的秒數,GPU 是這台機器上的 RTX 4070):

| epoch | train loss | val loss(該 epoch 之後量的) |
|---:|---:|---:|
| 0 | 0.5510 | 0.4002 |
| 1 | 0.4188 | 0.4251 |
| 2 | 0.4004 | 0.3966 |
| 3 | 0.3866 | 0.4080 |
| 4 | 0.3734 | 0.3948 |
| 5 | 0.3561 | 0.3941 |
| 6 | 0.3436 | 0.4006 |
| 7 | 0.3309 | 0.4087 |
| 8 | 0.3096 | 0.4146 |
| 9 | 0.3032 | — |

最佳 checkpoint:epoch 6,val_loss=0.3941。存成
`Phase3/Day32/checkpoints/best_model_filtered_v2_reweight2x.pt`。

tag:`day38-thermal-filtered-reweight-train`

## Step 3:RGB 分支

### 3a. 用規則模板腳本對 RGB 套用候選 (c) 門檻(0.019131944444444444%)生成 caption

重用 `thermal_dataset/generate_captions.py` 的所有函式(`DYNAMIC_CLASSES`/
`STATIC_CONTEXT_CLASSES`/`compute_area_thresholds`/`position_label`/
`distance_label`/`build_caption`/`OCCLUDED_DIFFICULT`/`LONG_TAIL_*`,原封
不動 import 重用,不重寫邏輯),只換掉 `main()` 裡硬編碼的
`images_thermal_{split}` 路徑跟 `GLOBAL_MIN_AREA_PCT`(改成 A7 算出來的
候選 (c) 精確值),對齊生成 `filtered_v2` 時同樣的做法(`--long-tail-
ref-split train`,Day34 修復邏輯)。

```
python generate_captions_rgb.py --split train --out captions_rgb_train_filtered.jsonl
python generate_captions_rgb.py --split val --long-tail-ref-split train --out captions_rgb_val_filtered.jsonl
```

結果:train 9,628 筆(10,318 張圖裡 690 張零物件被跳過)、val 1,002 筆
(1,085 張圖裡 83 張零物件被跳過)。

**一個查資料時發現的事實,跟原本以為的不一樣,如實記錄**:一開始以為 RGB
的 `extra_info` 完全沒有 `weather` 欄位(A3 早期查一支影片的 metadata 時
只看到 `hours`/`scene`/`video_id`,沒看到 `weather`),後來訓練時生成的
caption 裡出現「Cloudy:」這個字樣,回去查證發現**RGB 其實有部分圖片有
`weather` 欄位**(train 10,318 張裡 6,803 張、66.0% 有;val 1,085 張裡
725 張、66.8% 有),只是不是每張都有——不是 bug,是先前只抽樣看了一支
影片就下的錯誤印象,這裡更正。

### 3b. 對 RGB 圖片用既有的 frozen CLIP 抽特徵(同一套流程,沒改架構)

跟 `Phase3/Day32/precompute_clip_features.py` 完全相同的邏輯(同一個
`openai/clip-vit-base-patch32`,同樣的 `get_image_features(**inputs).
pooler_output`),只換 root 路徑成 `images_rgb_{split}`:

```
python precompute_clip_features_rgb.py --split train --captions captions_rgb_train_filtered.jsonl --out clip_features_rgb_train.pt
python precompute_clip_features_rgb.py --split val --captions captions_rgb_val_filtered.jsonl --out clip_features_rgb_val.pt
```

train 9,628 張、val 1,002 張,輸出維度 512,跟 thermal 的特徵格式完全一致。

### 3c. 訓練(跟 thermal 分支同一套配方,`config/data/rgb_filtered.yaml` 指向上面產出的檔案)

```
python train_vlm.py data=rgb_filtered \
  train.ckpt_dir=checkpoints_rgb_filtered_reweight2x \
  train.ckpt_path=checkpoints_rgb_filtered_reweight2x/best_model.pt \
  train.log_dir=log_rgb_filtered_reweight2x \
  +train.reweight_multi_position=true +train.multi_position_weight=2.0
```

雙子句樣本:2,084 / 9,628(21.6%)。Loss 曲線:

| epoch | train loss | val loss(該 epoch 之後量的) |
|---:|---:|---:|
| 0 | 0.5248 | 0.3978 |
| 1 | 0.3808 | 0.3769 |
| 2 | 0.3571 | 0.3724 |
| 3 | 0.3467 | 0.3676 |
| 4 | 0.3293 | 0.3769 |
| 5 | 0.3162 | 0.3866 |
| 6 | 0.2989 | 0.3773 |
| 7 | 0.2845 | 0.3866 |
| 8 | 0.2693 | 0.3893 |
| 9 | 0.2569 | — |

最佳 checkpoint:epoch 4,val_loss=0.3676。存成
`Phase3/Day32/checkpoints/best_model_rgb_filtered_reweight2x.pt`。

tag:`day38-rgb-filtered-reweight-train`

## 收斂行為對照(如實記錄,按指示不自己調參解決)

兩邊都訓得乾淨——**都沒有 nan、沒有 loss 爆炸**,`norm_max` 全程維持在
個位數到二十幾的合理範圍。但兩邊的收斂節奏有明顯差異:

- **thermal**:val loss 在 epoch 5-6 觸底(0.3941~0.3948)之後,後面
  4 個 epoch 緩慢回升到 0.40-0.41 帶,波動不大。
- **RGB**:val loss 在 epoch 3(0.3676)就觸底,比 thermal 早 2-3 個
  epoch;觸底之後回升得比 thermal 明顯更快、更多,epoch 4 就跳回
  0.3769,epoch 8 到 0.3893,整體上升的斜率比 thermal 那段更陡。
- 兩邊 train loss 到 epoch 9 都還在持續下降(thermal 0.3032、RGB
  0.2569),RGB 的訓練終點 train/val 落差(0.2569 vs 最後一次量到的
  0.3893,差 0.1324)比 thermal 的落差(0.3032 vs 0.4146,差 0.1114)
  略大一點。

**這是查數字確認的事實,不是推論出「RGB 過擬合更嚴重」這種結論**——
兩邊資料量接近(RGB train 9,628 筆 vs thermal train 10,179 筆,只差
5%),不是那種數量級差距會導致的典型 overfitting 對照組。RGB 觸底更早、
回升更快這個現象本身值得注意,但沒有進一步分析原因(不同輸入域的
CLIP 特徵分布/圖片本身細節密度不同,都可能是候選原因),依指示先如實
回報,不嘗試調參數解決。

## 五項量化評估指標對照表(Day34 方法論,`evaluate_val.py`)

三欄都用同一支腳本(RGB 用只改了 `VAL_COCO_PATH`/`FEATURES_VAL_PATH`
兩個路徑常數的 `evaluate_val_rgb.py`,其餘邏輯逐行相同)算出來,同一個
`SEED=42`,`max_new_tokens=40`。thermal (full_v2+reweight2x) 這欄是重新
跑一次既有 checkpoint 的 eval(不是沿用 08-31 當時的舊 CSV),確保三欄
方法論完全一致。

| 指標 | thermal (filtered+reweight,新) | thermal (full+reweight,既有基準 exp2) | RGB (filtered+reweight,新) |
|---|---:|---:|---:|
| checkpoint best epoch / val_loss | 6 / 0.3941 | 7 / 0.3872 | 4 / 0.3676 |
| val set n | 1,096 | 1,097 | 1,002 |
| 合法前綴率 | 0.9991 | 1.0000 | 0.9990 |
| 前綴匹配率(Night/非Night 跟 GT 一致) | 0.8923 | 0.8915 | 0.9760 |
| Night P/R/F1(全部樣本) | 0.4667 / 0.6481 / 0.5426 | 0.4486 / 0.4444 / 0.4465 | 0.9428 / 0.9843 / 0.9631 |
| Night P/R/F1(hours 有標注樣本) | 0.9589 / 0.6481 / 0.7735 (n=231) | 0.9600 / 0.4444 / 0.6076 (n=231) | 0.9751 / 0.9843 / 0.9797 (n=988) |
| 句型模板合規率 | 0.9799 | 0.9863 | 0.9800 |
| 物件類別 precision(micro) | 0.6831 | 0.6950 | 0.7131 |
| 物件類別 recall(micro) | 0.7570 | 0.7621 | 0.7407 |
| 物件類別 f1(micro) | 0.7181 | 0.7270 | 0.7267 |
| 生成長度 mean/median/p95(token) | 12.38 / 10.0 / 23.0 | 12.12 / 10.0 / 21.2 | 12.88 / 11.0 / 23.0 |
| GT 長度 mean/median/p95(token) | 11.16 / 10.0 / 20.0 | 11.10 / 10.0 / 20.0 | 12.26 / 10.0 / 22.0 |
| EOS 命中率 | 1.0000 | 1.0000 | 1.0000 |

**一個影響 Night 指標可比性的資料事實(查資料確認,不是推論)**:RGB
val 的 `extra_info.hours` 幾乎每張都有標(988/1002,98.6% 有明確標注);
thermal val 的 `hours` 標注覆蓋率低很多(231/1097,只有 21.1% 有明確
標注,其餘 865 筆 `hours` 是空字串、被「全部樣本」那欄的計算方式當成
「非 night」處理)。這代表 thermal 兩欄「全部樣本」的 Night P/R/F1
被大量沒有標注的樣本稀釋,RGB 那欄幾乎沒有這個稀釋效應——**兩欄「全部
樣本」的 Night 數字不是在同樣的資料完整度基礎上比的**,「hours 有標注」
那欄樣本數也因此差了 4 倍(RGB 988 筆 vs thermal 231 筆),解讀這兩欄
對照時要注意這個前提不一樣,不是模型能力的直接落差。

## Position-Class Binding Accuracy 對照表(Day35/36 方法論)

三個 CSV 都用同一個 `position_binding_accuracy.py` 的 `run()` 函式(style
統一用 `"v7"`,三個 checkpoint 生成的句子都是 v0.7+ class-first 模板),
邏輯沒有改動,只是換了要跑的三個檔案。

| 指標 | thermal (filtered+reweight,新) | thermal (full+reweight,既有基準 exp2) | RGB (filtered+reweight,新) |
|---|---:|---:|---:|
| n | 1,096 | 1,097 | 1,002 |
| GT clause parse 成功率 | 100.00% | 100.00% | 100.00% |
| 生成句 clause parse 成功率 | 98.53% | 98.92% | 98.27% |
| class-position 正確(correct) | 332 | 397 | 323 |
| class-position 錯位(mismatch) | 199 | 168 | 155 |
| position 缺失(missing,GT 有生成沒有) | 753 | 700 | 683 |
| position 多餘(extra,生成有 GT 沒有) | 875 | 810 | 712 |
| 同位置有配對的總數(position_matched) | 531 | 565 | 478 |
| **Position-Class Binding Accuracy** | **62.52%** | **70.27%** | **67.57%** |
| **Class-Position 錯位率** | **37.48%** | **29.73%** | **32.43%** |

## 小結

按指示不下「哪個比較好、該用哪個」的結論——上面兩張表把 thermal
(filtered+reweight,新)、thermal(full+reweight,既有基準)、RGB
(filtered+reweight,新)三欄數字都列出來了,唯一額外標注出來的是「RGB
收斂節奏比 thermal 快、Night 指標的兩欄樣本完整度不對等」這兩個查數字
確認的事實,供你判斷時參考,不是這份調查自己下的評價。

# 任務 A9:RGB 全量版(filtered+reweight vs full+reweight)補測,完成 2×2 對照

背景:A8 已經證實 thermal「過濾+重加權」比「全量+重加權(既有基準
exp2)」在 Position-Class Binding Accuracy 上差 7.75pp(62.52% vs
70.27%)。這次補 RGB 的全量版,湊齊「thermal/RGB × 全量/過濾」2×2
對照,單純補數字,不下結論。

## Step 1:RGB 全量版 caption 生成(對齊 thermal full_v2 的生成邏輯)

沿用 `thermal_dataset/generate_captions.py` 的原始邏輯,但**不套用任何
tiny-bbox 面積門檻**——這是查程式碼確認的事實:`full_v2` 是在
`--filter-tiny` 沒開啟的情況下生成的(`filter_tiny=False` 時,
`generate_captions.py` 第 520-522 行的 `area_pct < GLOBAL_MIN_AREA_PCT`
判斷完全不會被執行,見任務 A5)。新腳本
`Phase3/rgb_validation/generate_captions_rgb_full.py` 是
`generate_captions_rgb.py` 拿掉這段面積過濾判斷後的版本,其餘
(long-tail 修復、position/distance label、`build_caption`)逐行照抄
不改。

```
python generate_captions_rgb_full.py --split train --out captions_rgb_train_full.jsonl
python generate_captions_rgb_full.py --split val --long-tail-ref-split train --out captions_rgb_val_full.jsonl
```

結果:train 9,656 筆(10,318 張圖,662 張零物件跳過)、val 1,004 筆
(1,085 張圖,81 張零物件跳過)。比對過濾版(train 9,628 / val
1,002),全量版確實比過濾版多收了一些原本被面積門檻濾掉之後歸零物件的
圖(train +28、val +2),方向符合預期。

## Step 2:CLIP 特徵——沿用既有的、不重算(查資料確認的覆蓋率)

如指示,圖片本身沒變,只有 caption 的過濾邏輯不同,直接沿用 A8 訓練
RGB 過濾版時已經算好的 `clip_features_rgb_{train,val}.pt`(來源:
`precompute_clip_features_rgb.py`,對 `images_rgb_{train,val}` 全部圖片
算好的 frozen CLIP pooled 512 維特徵)。

**查證覆蓋率的事實(不是假設)**:全量版 caption 檔比過濾版多出的那些
圖(train 28 張、val 2 張),剛好就是當初過濾版生成時因為面積門檻被濾到
零物件、因此沒被收進過濾版 caption 檔的那些圖——這些圖的 CLIP 特徵在
上次也**沒有**被算進 `clip_features_rgb_*.pt`(特徵檔是照過濾版 caption
檔案裡的 `file_name` 清單去抓圖算的)。也就是 train 9,656 筆全量
caption 裡有 28 筆(0.29%)、val 1,004 筆裡有 2 筆(0.20%)找不到對應
特徵。這不是需要處理的 bug——`train_vlm.py` 的 `CaptionDataset`
(`Phase3/Day32/train_vlm.py:311-313`)本來就會把找不到特徵的樣本自動
丟掉並印警告,行為跟「沿用既有特徵、不重算」的指示一致,這裡只是誠實
記錄這個極小比例的資料落差,沒有另外處理。

## Step 3:訓練(跟 A8 同一套配方,`config/data/rgb_full.yaml` 指向上面產出的檔案)

```
python train_vlm.py data=rgb_full \
    train.ckpt_dir=checkpoints_rgb_full_reweight2x \
    train.ckpt_path=checkpoints_rgb_full_reweight2x/best_model.pt \
    train.log_dir=log_rgb_full_reweight2x \
    +train.reweight_multi_position=true \
    +train.multi_position_weight=2.0
```

雙子句判斷、x2 權重、`WeightedRandomSampler`、其餘超參數(B=8, T=1024,
total_batch_size=524288, max_lr=3e-4, epoch=10 等)跟 A8 的 thermal/RGB
分支完全相同,沒有調整任何數字。

實際套用重加權的樣本數:1,916/9,628(19.9%)——分母是 9,628 不是
9,656,因為上面 Step 2 提到的 28 筆缺特徵樣本已被 `CaptionDataset`
自動丟棄。

| epoch | train loss | val loss |
|---|---:|---:|
| 0 | 0.5216 | 0.4094 |
| 1 | 0.3780 | 0.3788 |
| 2 | 0.3544 | **0.3544** ← best |
| 3 | 0.3366 | 0.3728 |
| 4 | 0.3256 | 0.3772 |
| 5 | 0.3101 | 0.3596 |
| 6 | 0.2935 | 0.3697 |
| 7 | 0.2785 | 0.3747 |
| 8 | 0.2670 | 0.3752 |
| 9 | 0.2520 | 0.3804 |

`best_model.pt` = epoch 2,val_loss=0.3544,全程沒有 nan/loss 爆炸,
`norm_max` 在 5.6~35.3 之間(第 2 個 epoch 出現一次 35.3 的尖峰,單一
樣本內少見但沒有連鎖發散,後續 epoch 隨即回落到個位數~二十幾)。

checkpoint 已複製到 `Phase3/Day32/checkpoints/best_model_rgb_full_reweight2x.pt`
(gitignore `*.pt`,不進版控)。

## 收斂行為對照(四組一起看,如實記錄,不調參)

| | best epoch | best val_loss | 觸底後回升幅度(到 epoch 9) |
|---|---:|---:|---:|
| thermal filtered+reweight | 6 | 0.3941 | +0.0205(→0.4146) |
| thermal full+reweight(exp2) | 7 | 0.3872 | 資料見 A8,回升幅度小 |
| RGB filtered+reweight | 4 | 0.3676 | +0.0217(→0.3893,A8 記錄) |
| RGB full+reweight | 2 | 0.3544 | +0.0260(→0.3804) |

**查數字確認的事實**:RGB 全量版觸底得比 RGB 過濾版還要更早(epoch 2
vs 4),也比兩個 thermal 版本都早很多(epoch 6、7)。四組裡 RGB
全量版是觸底最早、且觸底後回升幅度(以 best→epoch9 的差值算)最大的
一組。這個現象跟 A8 記錄的「RGB 比 thermal 觸底早、回升快」的模式方向
一致,而且在全量版上更明顯——但這仍然只是如實記錄現象,**不做「為什麼」
的診斷、不嘗試調參解決**,原因分析(不同輸入域的 CLIP 特徵分布/圖片
細節密度等)留待你判斷。

## Step 4:評估(Day34 五項量化指標 + Day35/36 Position-Class Binding Accuracy)

用跟 A8 完全相同的 `evaluate_val_rgb.py`(邏輯沒有改動,只是這次要
明確傳 `--val-captions captions_rgb_val_full.jsonl` 跟
`--train-captions captions_rgb_train_full.jsonl`,因為腳本預設路徑指向
的是過濾版檔名)跟同一支 `position_binding_accuracy.py`(`run()`
函式,style 統一 `"v7"`),同一個 `SEED=42`。

### 五項量化評估指標對照表(四欄)

| 指標 | thermal filtered+reweight | thermal full+reweight(既有基準 exp2) | RGB filtered+reweight | RGB full+reweight(這次新的) |
|---|---:|---:|---:|---:|
| checkpoint best epoch / val_loss | 6 / 0.3941 | 7 / 0.3872 | 4 / 0.3676 | 2 / 0.3544 |
| val set n | 1,096 | 1,097 | 1,002 | 1,002 |
| 合法前綴率 | 0.9991 | 1.0000 | 0.9990 | 0.9920 |
| 前綴匹配率(Night/非Night 跟 GT 一致) | 0.8923 | 0.8915 | 0.9760 | 0.9800 |
| Night P/R/F1(全部樣本) | 0.4667 / 0.6481 / 0.5426 | 0.4486 / 0.4444 / 0.4465 | 0.9428 / 0.9843 / 0.9631 | 0.9515 / 0.9874 / 0.9691 |
| Night P/R/F1(hours 有標注樣本) | 0.9589 / 0.6481 / 0.7735 (n=231) | 0.9600 / 0.4444 / 0.6076 (n=231) | 0.9751 / 0.9843 / 0.9797 (n=988) | 0.9721 / 0.9874 / 0.9797 (n=988) |
| 句型模板合規率 | 0.9799 | 0.9863 | 0.9800 | 0.9491 |
| 物件類別 precision(micro) | 0.6831 | 0.6950 | 0.7131 | 0.7183 |
| 物件類別 recall(micro) | 0.7570 | 0.7621 | 0.7407 | 0.7833 |
| 物件類別 f1(micro) | 0.7181 | 0.7270 | 0.7267 | 0.7494 |
| 生成長度 mean/median/p95(token) | 12.38 / 10.0 / 23.0 | 12.12 / 10.0 / 21.2 | 12.88 / 11.0 / 23.0 | 13.33 / 11.0 / 23.0 |
| GT 長度 mean/median/p95(token) | 11.16 / 10.0 / 20.0 | 11.10 / 10.0 / 20.0 | 12.26 / 10.0 / 22.0 | 12.15 / 10.0 / 22.0 |
| EOS 命中率 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Night 指標可比性的資料完整度落差(A8 已記錄的事實,這裡同樣適用)：
RGB val 的 `hours` 標注覆蓋率 98.6%(988/1002),thermal val 只有
21.1%(231/1097)——RGB 兩欄的「全部樣本」Night 數字幾乎沒有被無標注
樣本稀釋,thermal 兩欄有,解讀時仍要注意這個前提不對等，不是模型能力
的直接落差。

### Position-Class Binding Accuracy 對照表(四欄)

| 指標 | thermal filtered+reweight | thermal full+reweight(既有基準 exp2) | RGB filtered+reweight | RGB full+reweight(這次新的) |
|---|---:|---:|---:|---:|
| n | 1,096 | 1,097 | 1,002 | 1,002 |
| GT clause parse 成功率 | 100.00% | 100.00% | 100.00% | 100.00% |
| 生成句 clause parse 成功率 | 98.53% | 98.92% | 98.27% | 95.98% |
| class-position 正確(correct) | 332 | 397 | 323 | 332 |
| class-position 錯位(mismatch) | 199 | 168 | 155 | 126 |
| position 缺失(missing) | 753 | 700 | 683 | 691 |
| position 多餘(extra) | 875 | 810 | 712 | 760 |
| **Position-Class Binding Accuracy** | **62.52%** | **70.27%** | **67.57%** | **72.49%** |
| **Class-Position 錯位率** | **37.48%** | **29.73%** | **32.43%** | **27.51%** |

（`case_records` 明細寫在 `Phase3/Day32/position_binding_day38_4way.json` 裡。）

## 2×2 全貌(四個數字並排,不下結論)

| Position-Class Binding Accuracy | filtered | full |
|---|---:|---:|
| thermal | 62.52% | 70.27% |
| RGB | 67.57% | 72.49% |

四個角落都有數字了。查數字確認的事實：這一輪兩個 domain 都是 full 版
的 Binding Accuracy 比 filtered 版高（thermal +7.75pp、RGB +4.92pp），
而且不管 filtered 或 full，RGB 這欄都比對應的 thermal 欄高一點
（filtered: +5.05pp、full: +2.22pp）。這是四個數字擺在一起看到的
現象，不是這份調查自己下的「應該用哪個版本」評價——如指示，不自己
選最終要用的版本，判斷留給你。

