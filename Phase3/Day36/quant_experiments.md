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
