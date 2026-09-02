# CLIP Vision INT8 量化實驗記錄

比對方式:同一批 200 張 holdout 熱像圖片(`random.seed(1337)`,排除 calibration 用過的 500 張),
INT8 vs FP32 輸出的 cosine similarity,`profile_engine.py` / `eval_accuracy.py`。

## Step 0 — baseline (`calibration_method="max"`)

tag: `baseline-broken-int8`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    output_path="clip_vision.int8.onnx",
)
```

結果 (`outputs_logs/step0_profile.log`):

- cosine sim mean = 0.547456
- cosine sim min  = 0.479761
- cosine sim std  = 0.024778

額外觀察(非本次量化精度問題,但值得記錄):INT8 模型在這台機器的 CPU
onnxruntime 上比 FP32 慢約 11 倍(median 28.5ms → 333.3ms,加速比 0.09x)。

## Step 1 — 只改 calibration_method="entropy"

tag: `exp1-entropy`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="entropy",
    output_path="clip_vision.int8.step1_entropy.onnx",
)
```

結果 (`outputs_logs/step1_eval.log`):

- cosine sim mean = 0.523660
- cosine sim min  = 0.459029
- cosine sim std  = 0.026371

**結論:沒效,甚至比 max 略差**(mean 0.547 → 0.524)。calibration_method
不是主因,mean 遠低於 0.9,依計畫進入 Step 2。

## Step 2 — 在 Step 1 基礎上加 op_types_to_exclude=["LayerNormalization", "Softmax"]

tag: `exp2-exclude-layernorm`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="entropy",
    op_types_to_exclude=["LayerNormalization", "Softmax"],
    output_path="clip_vision.int8.step2_exclude_ln_softmax.onnx",
)
```

結果 (`outputs_logs/step2_eval.log`):

- cosine sim mean = 0.524144
- cosine sim min  = 0.458979
- cosine sim std  = 0.026613

**結論:幾乎沒效**,跟 Step 1 幾乎一樣(mean 0.523660 → 0.524144,誤差範圍內)。
量化過程的 log 顯示原因:

- Quantized node 數只從 171 降到 170 ——排除 LayerNorm/Softmax 實際上只少
  quantize 了 1 個節點,不是我們以為的「一大批數值敏感層被排除」。
- `clip_vision.onnx` 的計算圖裡本來就有 12 個 Softmax 節點、26 個
  LayerNormalization 節點(用 onnx 直接數的),但 modelopt 在兩次執行中
  log 出的 "Quantizable op types" 從頭到尾都**不包含 Softmax**
  (`['MatMul', 'Add', 'Mul', 'Gemm', 'Conv', 'LayerNormalization']`)——
  代表 Softmax 在這個版本的 modelopt INT8 quantizer 裡本來就不會被量化,
  我們排除它是無效操作(exclude 一個從未被 include 的 op type)。
- LayerNormalization 雖然在 quantizable 清單裡,但 log 顯示
  `Found 1 Conv->LayerNorm patterns to quantize`——26 個 LayerNorm 節點裡
  只有 1 個符合它內部的 Conv→LayerNorm fusion pattern 而被量化,其餘 25 個
  本來就維持 FP32。所以「排除 LayerNorm」對這張圖幾乎是排除了一個本來就
  沒被動到的東西。

進 Step 3:兩個假設都不成立,回報異常訊息,不再自行猜測其他參數。

## Step 3 — 觀察到的異常訊息(照計畫,不再自行猜參數亂試)

按照計畫,Step 2 沒把 mean 拉到 0.9 以上,所以這裡列出量化過程中觀察到、
值得討論的異常,不再自己加新參數重跑。

1. **`Found 0 MHA (QK_AV) Patterns`**(兩次執行都一樣)
   modelopt 有專門辨識 Multi-Head-Attention QK^T→Softmax→AV 這種 pattern
   的邏輯,辨識到的話量化器才會用比較安全的方式處理 attention 裡的
   matmul(例如 `mha_accumulation_dtype` 這個參數就是設計給這種 pattern
   用的,但完全沒被觸發)。這次匯出的 `clip_vision.onnx` 裡沒有被認出任何
   一個 MHA pattern,推測是 `torch.onnx.export` 把 CLIP attention 展開成
   一般的 MatMul/Transpose/Softmax/Reshape 節點,不是 modelopt 認得的固定
   結構。結果是:attention 裡的 QK^T 和 AV 這兩個 matmul 很可能被當成一般
   MatMul 節點做了樸素的 per-tensor INT8 量化,沒有任何針對 attention
   數值範圍的特殊處理。這是目前觀察到,跟「精度崩潰」最直接相關的異常。

2. **`Quantizable op types` 從頭到尾不含 Softmax**
   代表這版 modelopt 的 INT8 quantizer 預設就不量化 Softmax(這點跟一開始
   的假設 2 相反——不是「Softmax 被無腦量化了要排除」,而是它本來就沒被
   量化,問題不在這裡)。

3. **`Found 1 Conv->LayerNorm patterns to quantize`,其餘 25 個 LayerNorm
   維持 FP32**
   同理,LayerNorm 幾乎沒被量化,op_types_to_exclude 對它幾乎是無效操作。

4. **`Failed to enable ORT with CUDA EP` / `TensorRT EP`**(cuDNN 用不到,
   兩次執行都出現)
   calibration 跟最終驗證都 fallback 回 CPUExecutionProvider。這不會直接
   造成 INT8 數值錯誤,但如果之後想用 CUDA/TensorRT EP 校正或部署,環境
   裡缺 `libcudnn_adv*.so*`,需要另外處理(裝 `nvidia-cudnn-cu12` 或把
   cuDNN 路徑加進 `LD_LIBRARY_PATH`)。

5. **INT8 模型在 CPU onnxruntime 上比 FP32 慢 ~11 倍**(baseline 就有,
   三次量化結果應該都一樣,沒有另外重測)。這通常代表這個 QDQ 格式的
   INT8 模型是準備給有 INT8 kernel 支援的 EP(如 TensorRT)吃的,在純
   CPU EP 上會退化成「先 dequant 成 fp32 再算」,反而更慢——跟精度問題
   無關,但如果目的是要用 INT8 換效能,現在的路線在 CPU 上也達不到目的。

**沒有再嘗試的參數**:`dq_only`、`block_size`、`disable_mha_qdq`、
`autotune_*` 等等都還沒試過,但這些屬於「繼續猜參數」,照要求先停在這裡
回報,交給你決定要不要往這個方向繼續,還是要先確認 MHA pattern 辨識失敗
這件事。

---

# Day38:路線 A——針對「MHA pattern 辨識失敗」關掉 attention QDQ

假設:既然 modelopt 認不出這份 export 出來的 CLIP attention pattern、
沒辦法用它自己的保護機制處理,那能不能直接關掉 attention 部分的 QDQ
插入,讓崩潰的根源不被量化。

判定標準:cosine similarity mean 回到 0.95+ 才算解決。

## exp5 — disable_mha_qdq=True(其餘跟 baseline 一致:calibration_method="max",無 op exclude)

tag: `day38-exp5-disable-mha-qdq`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    disable_mha_qdq=True,
    output_path="clip_vision.int8.exp5_disable_mha_qdq.onnx",
)
```

結果 (`outputs_logs/exp5_eval.log`):

- cosine sim mean = 0.555079
- cosine sim min  = 0.487913
- cosine sim std  = 0.024990

**結論:沒解決。** mean 0.547456 → 0.555079,幾乎沒變(在 std ~0.025 的
雜訊範圍內),離 0.95 目標非常遠。

量化 log 裡的異常(`outputs_logs/exp5_quantize.log`):

- log 明確印出 `Disabling QDQ for all MHA nodes`,代表 `disable_mha_qdq`
  這個參數確實被觸發、執行了「關閉 MHA QDQ」這個動作。
- 但同一次執行仍然印出 `Found 0 MHA (QK_AV) Patterns`——也就是說,
  modelopt 一開始就沒辨識出任何 MHA pattern,`disable_mha_qdq` 這個開關
  是作用在「它認得的 MHA 節點」上,而這份圖裡它認得的 MHA 節點數量是 0。
  這跟 exp2 裡「排除一個從沒被量化的 op type」是類似的情況:操作本身有
  被執行,但可能沒有實際節點可以作用。
- 量化節點數:`Total number of quantized nodes: 147`(exp5,
  calibration_method=max + disable_mha_qdq)。跟 exp1
  (`calibration_method=entropy`,無 op exclude)的 171 個相比少了 24 個,
  跟 exp2(entropy + exclude LayerNorm/Softmax)的 170 個也不同。有變化,
  但精度分數幾乎沒有跟著變,所以這個節點數差異看起來跟精度問題無關
  (或至少不是主導因素)——如實記錄這個現象,不下定論。
- 新增觀察:log 裡出現 `Converting float32 tensors to fp16`,是這次執行
  才出現的訊息(exp1、exp2 的 log 裡沒有)。目前不確定觸發條件是什麼,
  記錄下來供之後比對。

**mean < 0.9,依計畫進 exp6。**

## exp6 — dq_only=True(其餘跟 baseline 一致:calibration_method="max",無 op exclude)

tag: `day38-exp6-dq-only`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    dq_only=True,
    output_path="clip_vision.int8.exp6_dq_only.onnx",
)
```

結果 (`outputs_logs/exp6_eval.log`):

- cosine sim mean = 0.548602
- cosine sim min  = 0.482383
- cosine sim std  = 0.024888

**結論:沒解決。** mean 0.547456 → 0.548602,幾乎沒變,離 0.95 目標非常遠。

量化 log 裡的異常(`outputs_logs/exp6_quantize.log`):

- `Found 0 MHA (QK_AV) Patterns` 依然出現——這是三次(exp1/exp2/exp5/exp6)
  唯一每次都一樣的訊息,再次確認 modelopt 從頭到尾沒認出這份 export 的
  attention 結構。
- `dq_only=True` 有確實生效:log 印出
  `Converting model with QDQ nodes to DQ only model`、
  `Removed 73 Q nodes and redundant cast nodes`,量化節點數變成 171
  (跟 exp1 的 171 一樣)。這代表 dq_only 只是改變「QDQ 節點怎麼放」的
  形式(拿掉多餘的 Quantize 節點,只留 Dequantize),不是改變「哪些節點
  被量化」的邏輯,精度分數沒動在預期之內。

**mean < 0.9,依計畫進 exp7(手動排除 attention 裡實際的 MatMul 節點)。**

## exp7 — 手動用 nodes_to_exclude 排除 attention 裡實際的 24 個 MatMul 節點

先用 onnx 直接檢查 `clip_vision.onnx` 的計算圖(不是用 Netron 用猜的),
確認每一層 attention 的 QK^T 和 AV 各是哪個 MatMul 節點:

- 找每個 `Softmax` node 的輸入來源節點 → 12 層的 QK^T matmul:
  `node_MatMul_61, node_MatMul_117, node_MatMul_172, node_MatMul_227,
  node_MatMul_282, node_MatMul_337, node_MatMul_392, node_MatMul_447,
  node_MatMul_502, node_MatMul_557, node_MatMul_612, node_MatMul_667`
- attn_weights @ V 的 12 個 matmul,節點名稱剛好被 export 器命名成
  `node_scaled_dot_product_attention`(未編號的第一個)到
  `node_scaled_dot_product_attention_11`,op_type 確認就是 `MatMul`。

因為這些節點的 `op_type` 跟其他 72 個非 attention 的 MatMul 一樣都是
`MatMul`,`op_types_to_exclude` 會連專案投影層的 MatMul 一起排掉,所以改用
`nodes_to_exclude`(用節點名稱排除)只針對這 24 個節點。

tag: `day38-exp7-exclude-attn-matmul`

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    nodes_to_exclude=[
        "node_MatMul_61", "node_MatMul_117", "node_MatMul_172", "node_MatMul_227",
        "node_MatMul_282", "node_MatMul_337", "node_MatMul_392", "node_MatMul_447",
        "node_MatMul_502", "node_MatMul_557", "node_MatMul_612", "node_MatMul_667",
        "node_scaled_dot_product_attention", "node_scaled_dot_product_attention_1",
        "node_scaled_dot_product_attention_2", "node_scaled_dot_product_attention_3",
        "node_scaled_dot_product_attention_4", "node_scaled_dot_product_attention_5",
        "node_scaled_dot_product_attention_6", "node_scaled_dot_product_attention_7",
        "node_scaled_dot_product_attention_8", "node_scaled_dot_product_attention_9",
        "node_scaled_dot_product_attention_10", "node_scaled_dot_product_attention_11",
    ],
    output_path="clip_vision.int8.exp7_exclude_attn_matmul.onnx",
)
```

結果 (`outputs_logs/exp7_eval.log`):

- cosine sim mean = 0.555079
- cosine sim min  = 0.487913
- cosine sim std  = 0.024990

**結論:沒解決。而且這組數字跟 exp5(disable_mha_qdq=True)到小數點後六位
完全一樣**(mean 0.555079/min 0.487913,兩邊一模一樣)。代表手動排除的這
24 個 MatMul,跟 `disable_mha_qdq=True` 實際上排除的節點集合是同一批——
`disable_mha_qdq` 內部應該就是用類似方式(認節點命名/結構)在關閉這些
matmul 的 QDQ,只是它自己 log 出來的 `Found 0 MHA (QK_AV) Patterns` 講的
是另一個(沒作用的)pattern 偵測邏輯,兩者不是同一段程式碼路徑。

量化 log(`outputs_logs/exp7_quantize.log`)跟 exp5 幾乎一致:一樣是
`Found 0 MHA (QK_AV) Patterns`、一樣 147 個量化節點、一樣有
`Converting float32 tensors to fp16`,這些都跟 exp5 對得上,不是新異常。

**exp5/exp6/exp7 三個都做完,沒有一個把 cosine similarity 拉回 0.95+
(甚至連 0.7 都沒到)。照計畫在這裡停下,不再猜 `block_size`、
`autotune_*` 等其他參數。**

### Day38 路線 A 總表

| 實驗 | 改動 | mean | min | std |
|---|---|---|---|---|
| baseline (Day36) | calibration_method=max | 0.547456 | 0.479761 | 0.024778 |
| exp1 | calibration_method=entropy | 0.523660 | 0.459029 | 0.026371 |
| exp2 | entropy + exclude LayerNorm/Softmax | 0.524144 | 0.458979 | 0.026613 |
| exp5 | max + disable_mha_qdq=True | 0.555079 | 0.487913 | 0.024990 |
| exp6 | max + dq_only=True | 0.548602 | 0.482383 | 0.024888 |
| exp7 | max + nodes_to_exclude(24 個 attention MatMul) | 0.555079 | 0.487913 | 0.024990 |

五個實驗的 mean 全部卡在 0.52～0.56 之間,沒有一個接近 0.9,更不用說 0.95
的目標。是否要換路線,由你判斷。

---

# Day38 延伸:定位真正根因

exp5 跟 exp7 數字完全一致,證明「排除 attention 量化」這個操作本身有確實
執行,但沒解決精度問題 → attention matmul 量化被證偽不是主因,不再往這裡
查。以下兩條路徑改用 Day25-28 的節點級因果鏈方法論定位。

## exp8 — 校準資料 vs holdout 前處理一致性檢查(純程式碼比對,沒有重跑量化)

**這節的結論全部是「查程式碼確認的事實」,不是從數字推論。**

比對兩份程式碼實際呼叫的前處理路徑:

- Calibration(`test.py` 第 12、26-29 行):
  `processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)` →
  `img = Image.open(img_path).convert("RGB")` →
  `processor(images=[img], return_tensors="pt")` →
  `inputs["pixel_values"].numpy()`
- Holdout 驗證(`eval_accuracy.py` 第 13、28-30 行,`profile_engine.py`
  同樣寫法):
  `processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)` →
  `img = Image.open(img_path).convert("RGB")` →
  `processor(images=[img], return_tensors="pt")` →
  `inputs["pixel_values"].numpy()`

兩邊**沒有各自寫一份前處理函式/class**——都是直接呼叫同一個
`transformers.CLIPProcessor`(同一個 model name
`"openai/clip-vit-base-patch32"`,兩份檔案裡字串常數逐字相同),呼叫方式
逐行相同。搜尋整個 `Phase3/Day36/` 目錄沒有找到任何 local
`preprocessor_config.json` 或其他 override 這個 processor 設定的檔案,
`config/` 目錄裡的內容是 GPT decoder 訓練用的 hydra config,跟 CLIP
processor 無關。

實際印出這個 processor 當下解析出來的前處理參數(用同一支
`openai/clip-vit-base-patch32` 跑出來的真實數值,不是文件上的預設值):

| 項目 | 數值 |
|---|---|
| resize | shortest_edge=224,resample=bicubic(PIL 3) |
| center crop | 224×224 |
| rescale | 先除以 255(rescale_factor=1/255) |
| normalize mean | (0.48145466, 0.4578275, 0.40821073) |
| normalize std | (0.26862954, 0.26130258, 0.27577711) |
| channel order | RGB(兩邊都先 `.convert("RGB")` 才丟進 processor) |
| 輸出 dtype / shape | float32,(1, 3, 224, 224) |
| 輸出數值範圍(實測) | 約 -1.78 ~ 2.15(normalize 後,不是 0-1 或 0-255) |

因為兩邊呼叫的是同一個 `processor` 物件邏輯(同一個 class、同一個
pretrained config、同一個呼叫方式),上面這張表對 calibration 跟 holdout
兩邊完全適用,不存在「兩邊各自維護一份、可能不一致」的風險。

**結論:沒有發現前處理不一致。** calibration 跟 holdout 用的是同一支
`CLIPProcessor`、同一份設定、同一種呼叫方式,resize/normalize/channel
order/數值範圍/dtype 全部相同。這條假設查完沒有發現問題,不是「差異量級
不夠解釋崩潰」,而是**根本沒有差異**。

tag: `day38-exp8-preprocessing-check`(這輪沒有新的 onnx 產物,commit 只
包含這份紀錄)

進 exp9:前處理排除後,回到 Day25-28 的逐層 bisection 找根因。

## exp9/exp10 — 12 層 ViT block 逐層 bisection(第一輪:前 6 層 vs 後 6 層)

**這節的節點切分方式是「查程式碼確認的事實」**:用 onnx 直接讀
`clip_vision.onnx` 的計算圖,以每層 transformer block 的 pre-attn
LayerNormalization(`layer_norm_{2i+1}`,i=0..11)當作該層在 graph 節點
列表裡的起始邊界,依 topological 順序切出 12 段,每段固定 35 個節點
(最後一段 36 個,含 post_layernorm 前的邊界),總數 421(12×35+1)+
embeddings 前處理 9 個節點 = 430,跟 `clip_vision.onnx` 總節點數對得上。
不是用 op 名稱猜的,是直接照 graph 拓樸切出來的。

**下面的「哪一半貢獻比較多誤差」是從數字推論**,不是直接查出來的事實。

### exp9 — 只量化前 6 層(layer 0-5),後 6 層(layer 6-11)整層排除量化

tag: `day38-exp9-bisect-first6`

quantize 節點範圍:量化 layer 0-5 共 210 個節點(quantizable 子集,實際
`Total number of quantized nodes: 87`);layer 6-11 全部節點(210 個)
放進 `nodes_to_exclude` 排除。

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    nodes_to_exclude=<layer 6-11 全部節點名稱>,
    output_path="clip_vision.int8.exp9_quantize_front6.onnx",
)
```

結果 (`outputs_logs/exp9_eval.log`):

- cosine sim mean = 0.633780
- cosine sim min  = 0.538963
- cosine sim std  = 0.043417

### exp10 — 只量化後 6 層(layer 6-11),前 6 層(layer 0-5)整層排除量化

tag: `day38-exp10-bisect-last6`

quantize 節點範圍:量化 layer 6-11(實際 `Total number of quantized
nodes: 44`);layer 0-5 全部節點放進 `nodes_to_exclude` 排除。

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    nodes_to_exclude=<layer 0-5 全部節點名稱>,
    output_path="clip_vision.int8.exp10_quantize_back6.onnx",
)
```

結果 (`outputs_logs/exp10_eval.log`):

- cosine sim mean = 0.759286
- cosine sim min  = 0.663037
- cosine sim std  = 0.039732

### 第一輪小結

| 實驗 | 量化範圍 | mean | min | std |
|---|---|---|---|---|
| baseline | 全 12 層都量化 | 0.547456 | 0.479761 | 0.024778 |
| exp9 | 只量化 layer 0-5(前 6 層) | 0.633780 | 0.538963 | 0.043417 |
| exp10 | 只量化 layer 6-11(後 6 層) | 0.759286 | 0.663037 | 0.039732 |

兩組都各自比 baseline(全部量化)好,符合「量化的節點變少,誤差變小」的
方向。但兩組之間差距明顯:只量化前 6 層(exp9, mean 0.634)比只量化後
6 層(exp10, mean 0.759)還要差——**從數字推論**,前 6 層(layer 0-5)
單獨量化時貢獻的誤差,看起來比後 6 層(layer 6-11)單獨量化時大。std
也是 exp9(0.043)比 exp10(0.040)大一些,前半段誤差看起來波動也比較大。

依計畫,對「比較差的那組」繼續往下二分——也就是繼續切前 6 層
(layer 0-5),分成前 3(layer 0-2)跟後 3(layer 3-5)。

## exp11/exp12 — 第二輪 bisection:layer 0-5 裡再切前 3 / 後 3

節點切分方式跟上一輪一樣(用 pre-attn LayerNormalization 邊界,查程式碼
確認的事實),只是這輪把其餘 9 層全部排除,只留 3 層量化。

### exp11 — 只量化 layer 0-2,其餘(3-11)排除量化

tag: `day38-exp11-bisect-layer0to2`

`Total number of quantized nodes: 45`

結果 (`outputs_logs/exp11_eval.log`):

- cosine sim mean = 0.679399
- cosine sim min  = 0.581472
- cosine sim std  = 0.038937

### exp12 — 只量化 layer 3-5,其餘(0-2,6-11)排除量化

tag: `day38-exp12-bisect-layer3to5`

`Total number of quantized nodes: 23`

結果 (`outputs_logs/exp12_eval.log`):

- cosine sim mean = 0.813260
- cosine sim min  = 0.701619
- cosine sim std  = 0.042004

### 第二輪小結(從數字推論)

| 實驗 | 量化範圍 | mean | min |
|---|---|---|---|
| exp11 | 只量化 layer 0-2 | 0.679399 | 0.581472 |
| exp12 | 只量化 layer 3-5 | 0.813260 | 0.701619 |

layer 0-2 單獨量化(exp11, mean 0.679)比 layer 3-5 單獨量化(exp12,
mean 0.813)差更多——跟第一輪的方向一致(越前面的層,量化後掉的精度
越多)。continue 往 layer 0-2 裡繼續切:layer 0 單獨 vs layer 1-2。

## exp13/exp14 — 第三輪 bisection:layer 0-2 裡再切 layer 0 / layer 1-2

### exp13 — 只量化 layer 0,其餘(1-11)排除量化

tag: `day38-exp13-bisect-layer0`

`Total number of quantized nodes: 17`

結果 (`outputs_logs/exp13_eval.log`):

- cosine sim mean = 0.841682
- cosine sim min  = 0.737047
- cosine sim std  = 0.033980

### exp14 — 只量化 layer 1-2,其餘(0,3-11)排除量化

tag: `day38-exp14-bisect-layer1to2`

`Total number of quantized nodes: 16`

結果 (`outputs_logs/exp14_eval.log`):

- cosine sim mean = 0.808206
- cosine sim min  = 0.684352
- cosine sim std  = 0.040841

### 第三輪出現的異常現象——bisection 方法論的假設可能不成立

**這一段是從數字推論,不是查出來的事實,而且是這輪 bisection 裡最值得
在繼續切之前先回報的東西。**

| 實驗 | 量化範圍 | mean |
|---|---|---|
| exp13 | 只量化 layer 0 | 0.841682 |
| exp14 | 只量化 layer 1-2 | 0.808206 |
| exp11 | 量化 layer 0-2(= layer0 + layer1-2 合起來) | 0.679399 |

layer 0 單獨量化(0.842)跟 layer 1-2 單獨量化(0.808)**兩個分開測都
遠比合在一起測(layer 0-2 一起量化,0.679)好**——而且 exp11 的 0.679
甚至比 exp13、exp14 兩個「單獨的更差那個」(exp14 的 0.808)還要差很多。
這不是簡單的「兩邊誤差相加」可以解釋的:如果誤差是每層獨立、線性疊加的,
合起來的分數應該落在兩個單獨分數之間或接近較差的那個,而不是比兩個都
明顯更差。

同樣的現象在第一輪也看得到:exp9(前 6 層一起量化,mean 0.634)比
exp11(其中 layer 0-2,mean 0.679)、exp12(其中 layer 3-5,mean 0.813)
單獨測都差。

**推論(不是定論)**:多層一起量化時,誤差可能不是每層獨立貢獻、線性
加總,而是有交互作用——量化層 A 產生的誤差,可能會被下一個「也被量化」
的層 B 放大或用非線性方式合成,而不是單純沿著殘差流累加。如果這個推論
成立,原本 bisection 想找「哪一層是元兇」這個方法論的前提(誤差可以歸因
到單一或少數幾層)可能本身就不成立——問題可能是「連續多層一起量化」這件
事本身,而不是某一層特別敏感。

**在這裡停下,不繼續切 layer 1 vs layer 2**,先回報這個現象,因為如果
交互作用的推論是對的,繼續切下去很可能只會一直看到「合起來比分開差」
的同一個模式,對定位問題沒有幫助,值得先跟你確認要不要換方法。

## exp15/exp16 — 驗證「相鄰量化誤差複合」假設:layer 0+6(不相鄰)對照組

判讀標準(你設定的):

- exp15(layer 0+6)接近「layer 0 單獨」「layer 6 單獨」兩者較差的那個,
  或明顯優於 exp14(layer 1-2 相鄰,0.808)→ 支持「相鄰誤差複合」
- exp15 跟 exp14 差不多爛或更爛 → 跟相鄰與否無關,複合推論不成立

### exp16 — 先補上 layer 6 單獨量化的數字(公平對照用)

tag: `day38-exp16-layer6-alone`

只量化 layer 6,其餘(0-5, 7-11)排除量化。`Total number of quantized
nodes: 10`。

結果 (`outputs_logs/exp16_eval.log`):

- cosine sim mean = 0.846152
- cosine sim min  = 0.732661
- cosine sim std  = 0.037728

### exp15 — 只量化 layer 0 跟 layer 6(不相鄰,中間 layer 1-5 維持 FP32)

tag: `day38-exp15-layer0-6-nonadjacent`

其餘(1,2,3,4,5,7,8,9,10,11)排除量化。`Total number of quantized
nodes: 31`。

```
quantize(
    onnx_path="clip_vision.onnx",
    quantize_mode="int8",
    calibration_data={"pixel_values": calib_array},
    calibration_method="max",
    nodes_to_exclude=<layer 1,2,3,4,5,7,8,9,10,11 全部節點>,
    output_path="clip_vision.int8.exp15_quantize_layer0and6.onnx",
)
```

結果 (`outputs_logs/exp15_eval.log`):

- cosine sim mean = 0.641599
- cosine sim min  = 0.544478
- cosine sim std  = 0.037424

### 對照表與判讀(從數字推論,不是查出來的事實)

| 實驗 | 量化範圍 | 是否相鄰 | mean |
|---|---|---|---|
| exp13 | layer 0 單獨 | — | 0.841682 |
| exp16 | layer 6 單獨 | — | 0.846152 |
| exp14 | layer 1-2(2 層,相鄰) | 相鄰 | 0.808206 |
| exp15 | layer 0+6(2 層,不相鄰) | 不相鄰 | 0.641599 |

layer 0 單獨(0.842)跟 layer 6 單獨(0.846)幾乎一樣好,兩個都是目前測過
「單層量化」裡數一數二好的。但把這兩個「各自都很無害」的層放在一起量化
(exp15),結果反而是 0.642——**比相鄰的 layer 1-2 一起量化(exp14,
0.808)還要差很多**,甚至比 layer 0-2 三層一起量化(exp11, 0.679)還差。

依你設定的判讀標準:exp15 的數字不是「接近兩者較差的那個」(0.846 附近),
也沒有「明顯優於 exp14」,而是比 exp14 明顯更爛——**落入「跟相鄰與否
無關」這一類判讀**。而且結果比原本推論的方向更極端:不是「不相鄰所以
複合效果消失、退化成接近較差單層」,而是不相鄰的組合反而比相鄰的組合
更差。

**結論(推論,非定論):「相鄰量化誤差沿殘差流複合」這個假設不成立**——
如果成立,非相鄰的 layer 0+6 應該比相鄰的 layer 1-2 更不容易複合、分數
應該更好或至少持平,但實測是不相鄰的反而更差。同時也不是單純「量化層數
增加就等比例惡化」可以解釋(exp14 跟 exp15 都是量化 2 層,分數差了
0.166,層數相同分數卻差很多)。看起來比較像是**特定層的組合有各自的
交互作用**,不是可以用「相鄰/不相鄰」或「量化了幾層」這種簡單規則預測
的——但這只是這四個數字目前能看出的方向,樣本數(4 組)還很小,不足以
再往下細分機制。

**照計畫在這裡停下,不往 mixed precision 方案設計走,機制怎麼解讀、
下一步怎麼走由你判斷。**

---

# Day38 續:FP16 轉換 + INT8 收尾嘗試 + decoder 對照組

## exp17 — clip_vision.onnx 轉 FP16,驗證是否繞開 INT8 的精度崩潰

**API 查證(查程式碼確認的事實,不是查文件猜的)**:本機環境裡
`onnxruntime.transformers.float16.convert_float_to_float16` 跟
`onnxconverter_common.float16.convert_float_to_float16` 兩個都裝了,用
`inspect.signature()` 實際印出來確認參數:

```
onnxruntime.transformers.float16.convert_float_to_float16(
    model, min_positive_val=5.96e-08, max_finite_val=65504.0,
    keep_io_types=False, disable_shape_infer=False, op_block_list=None,
    node_block_list=None, force_fp16_initializers=False,
    force_fp16_inputs=None, use_bfloat16_as_blocked_nodes_dtype=False,
)
```

用的是 `onnxruntime.transformers.float16` 這支(裝在本機 onnxruntime
1.22.0 裡,不是額外裝的套件),`keep_io_types=True`(輸入輸出保持 float32,
方便直接用 `eval_accuracy.py` 同一套前處理/後處理比對,不用改介面),其餘
用預設值。輸出成新檔案 `clip_vision.fp16.onnx`(沒有覆蓋 FP32 原始檔)。

```python
model = onnx.load("clip_vision.onnx")
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "clip_vision.fp16.onnx", save_as_external_data=True,
          location="clip_vision.fp16.onnx.data")
```

用跟 exp1-16 完全相同的 200 張 holdout(同一個 seed=1337、同一套
`CLIPProcessor.from_pretrained(...)` 前處理,exp8 已經查過這是共用同一份
呼叫路徑)。

結果 (`outputs_logs/exp17_eval.log`):

- cosine sim mean = 0.999999
- cosine sim min  = 0.999997
- cosine sim max  = 1.000000
- cosine sim std  = 0.000000

**結論:FP16 完全沒有 INT8 那種精度崩潰。** 幾乎是數值上的無損轉換,200
張圖裡最差的一張也還有 0.999997,跟 INT8 系列實驗卡在 0.55～0.85 差了
一個量級。這跟背景假設一致:FP16 是連續浮點縮短尾數,不像 INT8 需要
per-tensor/per-channel 的 min-max calibration 跟離散化,不會遇到 INT8
在這個模型上一直踩到的 scale/校準/attention pattern 辨識問題。

### 延遲對照(CPU EP + CUDA EP,都有跑,環境裡本來就有 GPU)

`nvidia-smi -L` 確認機器上有一張 RTX 4070;
`onnxruntime.get_available_providers()` 回傳
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`。
TensorRT EP 沿用先前 exp5-16 遇到的問題(缺 `libcudnn_adv*.so*`)沒有再
另外排除,這裡只測**CUDA EP**(不是 TensorRT EP),用固定亂數輸入跑
100 次(20 次 warmup),跟先前 profile_engine.py 的做法一致。

結果 (`outputs_logs/exp17_eval.log` 下半部):

| 版本 | CPU EP median (ms) | CUDA EP median (ms) |
|---|---|---|
| FP32 | 29.905 | 3.836 |
| FP16 | 56.202 | 3.359 |
| INT8(exp5 disable_mha_qdq) | 339.466 | 12.470 |

**異常/發現(查數字確認的事實,不是推論)**:

- FP16 在 **CPU EP 上比 FP32 慢約 1.9 倍**(56.2ms vs 29.9ms)。CPU 沒有
  原生 FP16 SIMD 執行路徑,onnxruntime CPU EP 對 FP16 運算通常會內部轉回
  FP32 算,所以比 FP32 慢是預期內的現象,跟 INT8 QDQ 在 CPU 上慢 11 倍
  是類似性質的問題(格式不是為這個 EP 設計的),但慢的倍率小很多。
- FP16 在 **CUDA EP 上比 FP32 快一點**(3.36ms vs 3.84ms,約 1.14 倍),
  符合預期——GPU 有原生 FP16 tensor core 支援。
- INT8(exp5 版)在 **CUDA EP 上還是比 FP32/FP16 慢**(12.47ms,比 FP32
  的 3.84ms 慢約 3.25 倍)。這代表這份 QDQ 格式的 INT8 模型,就算換到
  CUDA EP(不是 CPU EP)一樣沒有速度優勢——因為 QDQ 格式的 INT8 tensor
  core 加速通常要靠 TensorRT EP 才有,純 CUDA EP 執行 QDQ 圖時一樣要做
  逐節點 dequant/requant,開銷比原生 FP32/FP16 matmul 還大。跟先前
  baseline 那則「INT8 在 CPU 上比 FP32 慢 11 倍」的觀察是同一個根本原因,
  只是在 GPU CUDA EP 上倍率沒那麼誇張(3.25x vs 11x)。

tag: `day38-exp17-fp16-conversion`

## exp18 — 查 modelopt 本機版本有沒有 activation smoothing/equalization(SmoothQuant/AWQ 類)選項

**這節全部是查程式碼原始碼確認的事實,不是查網路文件、也不是猜的。**

先找本機安裝的 modelopt 套件路徑,對 `modelopt/onnx/quantization/` 底下
`grep -rli "smooth|equaliz|awq"`,結果命中:

- `quantization/int4.py`:定義了 `AWQClipHelper`、`AWQLiteHelper`、
  `_quantize_awq_clip`、`run_awq_scale_search_per_node` 等——AWQ(把量化
  難度從 activation 轉移到 weight 的其中一種演算法)在這個版本裡**確實
  存在**,不是完全沒實作。
- `quantization/quantize.py`:往上一層的 `quantize()` 主函式裡看
  `quantize_mode` 怎麼分派(第 681-758 行左右):

  ```python
  if quantize_mode in ["fp8", "int8"]:
      quantize_func = quantize_int8 if quantize_mode == "int8" else quantize_fp8
      onnx_model = quantize_func(
          onnx_path=onnx_path,
          calibration_method=calibration_method or "entropy",
          ...  # 沒有任何 awq/smooth 相關參數傳進去
      )
  elif "int4" in quantize_mode:
      onnx_model = quantize_int4(
          onnx_path=onnx_path,
          calibration_method=calibration_method or "awq_clip",
          ...
      )
  ```

  AWQ 相關的邏輯只掛在 `quantize_mode` 包含 `"int4"` 的分支(`quantize_int4`)
  底下,`int8`/`fp8` 分支呼叫的是 `quantize_int8`/`quantize_fp8`,預設
  `calibration_method` 是 `"entropy"`,不會走到任何 AWQ/smoothing 程式碼。
- 再進一步查 `quantization/int8.py` 裡 `calibration_method` 實際怎麼用
  (第 121、152、283 行),確認只有 `"entropy"` 這個分支邏輯跟隱含的
  `"max"`,完全沒有 smoothing/equalization/awq 相關字串。

**結論:查證後確認本機版本的 INT8 量化路徑不支援 activation
smoothing/equalization(SmoothQuant/AWQ 類)這種選項。** AWQ 在這個版本
的 modelopt 裡是存在的,但只綁定給 `int4`(weight-only 量化)用,`int8`
路徑完全用不到它,不是「這個功能不存在」,是「這個功能存在但沒有接到
int8 量化流程裡」——這條路不通,不用硬找替代方案。

tag: `day38-exp18-smoothing-check`(這輪沒有新的量化產物,commit 只包含
這份查證紀錄)

## exp19 — 對 gpt.onnx(decoder)跑一次跟 exp1 一樣設定的 baseline INT8 量化

**目的**:當一個對照組。如果 GPT decoder 的 INT8 精度掉得比 ViT 溫和,
就能佐證「這是 ViT/CLIP attention-in-vision 特有的問題」,不是所有
transformer 架構套這套流程都會這麼脆弱。

### 資料準備(跟 clip vision 系列方法論一致)

用 `captions_val.jsonl`(1097 筆,`config/data/*.yaml` 裡設定的
`captions_val_path`)取代 clip vision 用的 holdout 圖片集,一樣
`random.seed(1337)`、抽 500 筆當 calibration、剩下取前 200 筆當 holdout。
每筆資料:

- 圖片先過 `clip_vision.onnx`(FP32)得到 `img_feat`(1, 512)
- caption 文字用 `minbpe.load("tokenizer.pkl")` tokenize,前面接
  `image_token_id=318`、後面接 `eos_token_id=319`(數值跟 `config/data`
  的 `base_vocab_size=318`、`special_token_size=2` 對得上,`GPT.py` 第
  128 行 `tok_emb[:,0] = image_vector_emb[:]` 確認第 0 個 token 位置會被
  換成圖片向量,不是普通文字 token)
- 序列 `seq = [318] + caption_tokens + [319]`,`idx = seq[:-1]`,
  `targets = seq[1:]`(標準 next-token 訓練/驗證用的位移方式)

**API 踩坑(查程式碼確認的事實)**:decoder 的 `input_ids` 每筆長度不同
(caption 長度不一樣),不能像 clip vision 那樣疊成單一固定 shape 的
array,所以改用 `calibration_data_reader` 逐筆餵。第一次用標準
`onnxruntime.quantization.calibrate.CalibrationDataReader` 介面(只實作
`get_next()`)下去跑,直接噴:

```
AttributeError: 'GPTCalibrationReader' object has no attribute 'get_first'
```

追進 `modelopt/onnx/quantization/graph_utils.py`
(`find_nodes_from_mha_to_exclude` → `get_extended_model_outputs`)確認它
會呼叫 `calibration_data_reader.get_first()`——這不是標準 onnxruntime
介面的一部分,是 modelopt 自己在
`modelopt/onnx/quantization/calib_utils.py` 的
`CalibrationDataProvider`/`RandomDataProvider` 額外加的方法。照那份原始
碼補上 `get_first()` 跟 `rewind()`(entropy calibration 通常要不只一輪
掃描,rewind 也一併補,不是猜的)後就能正常跑。

### 量化設定(跟 exp1 一致:calibration_method="entropy",無額外排除)

tag: `day38-exp19-gpt-decoder-baseline`

```
quantize(
    onnx_path="gpt.onnx",
    quantize_mode="int8",
    calibration_data_reader=reader,
    calibration_method="entropy",
    output_path="gpt.int8.exp19_baseline.onnx",
)
```

**量化 log 裡的一個重大對照事實(查 log 確認,不是推論)**:

```
Found 25 layer norm partitions
Found 12 MHA (QK_AV) Patterns
Total number of quantizable nodes: 50
Final number of nodes to quantize: 48
```

`gpt.onnx` 的 12 層 attention block,modelopt **全部 12 個都成功辨識成
MHA (QK_AV) pattern**——跟 `clip_vision.onnx` 系列實驗(exp1 到 exp16)
每次都印出 `Found 0 MHA (QK_AV) Patterns` 形成鮮明對比。這代表 modelopt
的 attention 保護機制(`mha_accumulation_dtype` 之類)在 GPT decoder 上
是真的有生效的,在 CLIP vision 上完全沒生效——這兩個模型的 attention
匯出方式顯然不一樣(`GPT.py` 用的是 `F.scaled_dot_product_attention`
直接呼叫 torch 內建的 SDPA,`CLIPModel`〔HuggingFace transformers 套件〕
內部的 attention 實作/匯出方式跟這個不同,才會讓 modelopt 認不出來)。

### 指標選擇:為什麼用 cross-entropy loss + top-1 一致率,不是 cosine similarity

CLIP vision 用 cosine similarity 是因為它的輸出(`img_feat`)是一個
**連續的 embedding 向量**,下游用途就是拿去算向量距離/相似度,cosine
similarity 直接對應這個用途。但 GPT decoder 的輸出是**每個位置的 vocab
分布(logits)**,下游用途是拿去 softmax 之後 greedy/取樣出下一個
token——真正決定生成結果的是「機率排序有沒有變」,不是「logit 向量的
方向有沒有偏」。logit 的 cosine similarity 是弱代理指標:softmax 是非
線性的,logit 量級小幅偏移可能翻動 argmax(而 cosine similarity 幾乎不
會反映出來),也可能 cosine similarity 掉很多但 softmax 之後排序完全
沒變。cross-entropy loss 直接是這個模型訓練時在最小化的量(`GPT.py`
第 141 行 `F.cross_entropy(...)`),能直接反映量化後模型「預測正確 caption
的能力」掉了多少;再搭配 top-1 token 一致率(INT8 跟 FP32 的 argmax 是否
一樣),直接量測「每個位置的生成決策有沒有被量化改變」,是更貼近這個
decoder 實際用途的指標。

結果 (`outputs_logs/exp19_eval.log`):

- FP32 cross-entropy loss: mean = 7.292551, std = 0.828097
- INT8 cross-entropy loss: mean = 7.292819, std = 0.836055
- loss 差值(INT8 − FP32): mean = 0.000267
- top-1 token 一致率(INT8 argmax == FP32 argmax): 0.984843
  (7667/7785 個位置)

（附註:FP32 loss 本身 mean≈7.29 nats 偏高,比 vocab_size=320 的均勻亂猜
基準 `ln(320)≈5.77` 還差,這代表這個 checkpoint 在這組 idx/targets 建構
方式下算出來的 loss 不算低——但這不影響這裡要驗證的東西,因為 INT8 跟
FP32 用的是完全一樣的資料建構方式,兩者的差值才是這裡真正要看的數字,
不受這個絕對值高低影響。）

**結論:GPT decoder 的 INT8 精度幾乎沒有掉。** loss 差值只有 0.000267
nats(幾乎是雜訊等級),top-1 token 一致率 98.48%,跟 CLIP vision
系列實驗(cosine similarity 卡在 0.55～0.85,連 disable_mha_qdq /
排除 attention 都救不回來)完全是不同量級的結果。**這佐證了背景假設:
INT8 精度崩潰看起來是 `clip_vision.onnx` 這個特定 export
(attention pattern 沒被 modelopt 認出來)的問題,不是「這整套 INT8
量化流程對 transformer 架構普遍都很脆弱」。**

## benchmark 收尾 — clip_vision FP32/FP16/INT8 延遲總表

這張表的數字全部來自 exp17 已經跑過的 latency benchmark(`outputs_logs/
exp17_eval.log`),這裡只是整理成一張表方便之後放 README/blog,沒有重測。
INT8 用目前測過最好的版本(exp5 的 `disable_mha_qdq=True`)。固定亂數
輸入,100 次 run(20 次 warmup),CPU EP 跟 CUDA EP(這台機器有 RTX 4070)
都測了。

| 精度 | 模型檔案 | CPU EP median (ms) | CUDA EP median (ms) | cosine sim vs FP32 |
|---|---|---|---|---|
| FP32 | `clip_vision.onnx` | 29.905 | 3.836 | 1.0(原始基準) |
| FP16 | `clip_vision.fp16.onnx` | 56.202 | 3.359 | 0.999999(exp17) |
| INT8 | `clip_vision.int8.exp5_disable_mha_qdq.onnx` | 339.466 | 12.470 | 0.555079(exp5) |

**先把數字備齊,不下部署建議的結論**(那是等你判斷完機制之後的事)——
從這張表能看到的純數字事實是:FP16 在 GPU(CUDA EP)上同時拿到「幾乎
無損的精度」跟「比 FP32 快一點」;INT8(這份 QDQ export)不管在 CPU 還是
GPU 上都比 FP32 慢,而且精度掉最多。
