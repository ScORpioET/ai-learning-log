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
