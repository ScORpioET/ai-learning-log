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
