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
