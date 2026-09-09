# TensorRT Execution Provider 加速實測(純好奇心實驗)

範圍:只讀既有的`Phase3/Day36`/`Phase3/Day39`ONNX產物,不修改任何正式部署路徑
(Docker demo、app.py、正式checkpoint全部沒動)。全部程式碼在這個新目錄底下。
這次沒有做任何「要不要採用」的建議或決定,純粹回答「差異有多大」。

## 0. 環境問題:兩份不相容的TensorRT共存,預設不會動

機器上同時有:
- 系統apt裝的 **TensorRT 11.1.0**(`/usr/lib/x86_64-linux-gnu/libnvinfer.so.11`)
- pip依賴帶進來的 **TensorRT 10.x**(`.venv/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.10`)

裝的`onnxruntime-gpu 1.22.0`的`TensorrtExecutionProvider`是對TensorRT 10.x的ABI
編譯的,系統版11.1完全不能用,預設直接報:

```
Failed to load library libonnxruntime_providers_tensorrt.so with error:
libnvinfer.so.10: cannot open shared object file: No such file or directory
```

解法:把`LD_LIBRARY_PATH`指到pip那份10.x的位置,排在系統路徑前面。這個實驗全部
腳本都要這樣執行:

```bash
LD_LIBRARY_PATH=<repo>/.venv/lib/python3.10/site-packages/tensorrt_libs python3 xxx.py
```

這不是這次實驗的分析目標,但如果沒先踩到、解掉,後面什麼都跑不了,值得記一筆。

## 1. 精度比對

### clip_vision.onnx(200張holdout圖片,跟Day36/eval_accuracy.py同一套seed=1337抽樣)

| 版本 | cosine sim mean | min | std |
|---|---:|---:|---:|
| TensorRT FP16 vs CUDA-FP32 | **0.999991** | 0.999941 | 0.000006 |
| TensorRT INT8 vs CUDA-FP32 | **0.543403** | 0.443584 | 0.041376 |
| (對照)CUDA EP INT8 vs CUDA-FP32(Day36 baseline,quant_experiments.md) | 0.547456 | 0.479761 | 0.024778 |

**結論:TensorRT FP16精度幾乎無損(跟CUDA EP FP16一樣安全)。TensorRT INT8完全
沒有解決這份QDQ模型原本在CUDA EP上就有的精度崩潰問題**——0.543 vs 0.547,兩個
數字在std範圍內幾乎相等,不是「换個EP解析方式,結果不一樣」,是同一個壞掉的
量化圖,不管哪個EP執行都一樣壞。這回答了user最初的好奇:**TensorRT自己的圖
解析/優化邏輯並沒有對這份QDQ模型的精度產生任何改善**,問題出在量化本身
(calibration/QDQ節點放置方式),不是CUDA EP的執行邏輯特別差。

### gpt.onnx(3張測試圖,整條pipeline跑到底,greedy decoding,token序列逐一比對)

img_feat固定用CUDA EP FP32的clip_vision.onnx抽(排除clip_vision這邊的變因),
只換GPT decoder的EP/精度:

| 圖片 | CUDA-FP32(基準) | TensorRT-FP16 | TensorRT-INT8 |
|---|---|---|---|
| frame-000000 | "Several cars, the nearest ahead; two pedestrians, one nearby on the right." | 一致 | 一致 |
| frame-000600 | "Night: a car on the left." | 一致 | 一致 |
| frame-000900 | "Night: several cars, the nearest on the left." | 一致 | 一致 |

**三張圖、三種EP/精度組合,產生的caption文字逐token完全一樣。** 這跟Day38
exp19發現的「GPT decoder的INT8量化本身就沒有像CLIP ViT那樣崩潰(top-1 token
一致率98.5%)」互相印證——GPT decoder在TensorRT EP上一樣穩,沒有因為換executor
而額外變差,也沒有變好(因為它本來就沒有壞)。

## 2. 意外狀況:INT8單獨開會build失敗,要跟FP16 flag一起開才行

`clip_vision.int8.onnx`跟`gpt.int8.exp19_baseline.onnx`(既有的QDQ INT8模型)
用`trt_int8_enable=True`單獨開,兩個模型都**直接build失敗**:

```
ERROR: ModelImporter.cpp:506 In function parseNode:
[6] Invalid Node - node_MatMul_630/MatMulAddFusion_bias_dq
IDequantizeLayer::setPrecision: Error Code 3: API Usage Error
(... A DequantizeLayer can only run in DataType::kINT8, DataType::kFP8,
DataType::kFP4, or DataType::kINT4 precision ...)
...
IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error
(fp16 precision has been set for at least one layer or layer output,
but fp16 is not configured in the builder.)
```

意思是:圖裡有些節點被TensorRT內部標成需要fp16精度,但builder沒有開fp16選項,
直接衝突失敗。**解法是把`trt_fp16_enable=True`也一起打開**,兩個模型都能build
成功——但這代表build出來的其實是一個**INT8+FP16混合精度engine**,不是使用者
以為的「純INT8」。而且就算加了fp16 flag「build成功」,底層log裡**一樣印出同一批
DequantizeLayer parser錯誤**,伴隨:

```
[W] Some nodes were not assigned to the preferred execution providers
```

也就是說,TensorRT並沒有真的把整張圖解析成功,是那些解析失敗的QDQ節點被partial
fallback回CPU EP執行,TRT只負責它解析得動的那部分。**這不是「TensorRT修好了
INT8」,是「TensorRT繞過了它解析不了的部分,湊合跑起來」**——跟精度比對裡
INT8結果沒有任何改善(0.543 vs 0.547)完全對得上:因為真正在計算INT8量化
矩陣乘法的那些節點,TensorRT根本沒有真的接管。

## 3. 效能比對(build時間 vs 穩態延遲分開報,median of 10 runs, warmup 3)

顯卡:NVIDIA GeForce RTX 4070(driver 610.88,CUDA 12.6)

### clip_vision.onnx(固定shape 1x3x224x224)

| 版本 | build時間 | 穩態median延遲 | vs CUDA-FP16 |
|---|---:|---:|---:|
| CUDA EP FP16(現行部署方式) | 1.49 s | 4.561 ms | 1.00x(基準) |
| TensorRT EP FP16 | 19.35 s | **1.527 ms** | **2.99x 更快** |
| TensorRT EP INT8(混合精度,見上節) | **262.4 s** | 11.222 ms | **0.41x 更慢** |

### gpt.onnx(T=20,見下方"為什麼只測一個長度"的說明)

| 版本 | build時間 | 穩態median延遲 | vs CUDA-FP16 |
|---|---:|---:|---:|
| CUDA EP FP16(現行部署方式) | 0.30 s | 4.073 ms | 1.00x(基準) |
| TensorRT EP FP16 | 12.18 s | 8.539 ms | **0.48x 更慢** |
| TensorRT EP INT8(混合精度) | 26.54 s | 11.731 ms | **0.35x 更慢** |

**如實記錄,不因為「理論上TensorRT該更快」而選擇性報告:**

- 只有 **clip_vision + TensorRT FP16** 這個組合真的比現行CUDA EP FP16快
  (2.99x)。
- **clip_vision + TensorRT INT8**、**gpt(不管FP16還是INT8)** 全部都比
  CUDA EP FP16基準**更慢**,不是打平,是明確變慢(0.35x~0.48x)。GPT decoder
  這個模型很小(12層、n_embd=768,batch=1),TensorRT engine執行的额外開銷
  (engine launch、context切換)在這個規模下可能蓋過了它kernel融合的優勢——
  這個推測沒有做進一步profiling驗證,誠實標成推測。
- **build時間全部都是CUDA EP的幾倍到幾百倍**:CUDA EP的"build"基本上只是
  session載入(0.3~1.5秒),TensorRT EP的engine編譯要12秒到**超過4分鐘**
  (clip_vision INT8混合精度那組,262秒)。這是每次啟動都要付的一次性成本
  (除非用`trt_engine_cache_enable`把build好的engine存到磁碟重複使用,這次
  沒有測快取重複載入的情境,只測了冷啟動的build時間)。

**為什麼gpt.onnx只測T=20一個長度**:TensorRT對動態shape的模型,如果沒有明確
指定shape範圍,每遇到一個新的seq_len就要重新build一次engine(逐token生成
場景下完全不可行)。這裡用`trt_profile_min/opt/max_shapes`明確告訴TensorRT
整個shape範圍(1~64),讓它一次build好一個涵蓋範圍的engine,已經驗證過這樣
不會每個長度都重build——但要測「不同T下穩態延遲怎麼變化」需要另外設計多組
測試,這次先用一個中庸的代表值(T=20,常見caption長度範圍內)得出結論,
沒有花時間测更多長度,不是忘記測。

## 4. 結論(僅限這次測試範圍,沒有做採用建議)

- **TensorRT FP16對clip_vision確實有感的加速(2.99x),精度幾乎無損
  (cosine sim 0.999991)**——如果只看這一個模型,理論上是有價值的。
- **TensorRT沒有解決INT8 QDQ模型原本就有的精度崩潰問題**,好奇心的答案是
  否定的:換個EP的圖解析邏輯,對這個特定量化圖沒有幫助,問題出在量化過程
  本身(這點在之前的Day36-38實驗已經花了很多篇幅查證)。
- **gpt.onnx在TensorRT上(不管FP16或INT8)反而比現行CUDA EP FP16慢**,這個
  模型規模下TensorRT沒有優勢。
- **TensorRT INT8實際上只能build成INT8+FP16混合精度engine**,單獨開INT8
  會直接失敗,而且就算"build成功"底層log顯示還是有節點解析失敗、partial
  fallback,不是乾淨的純INT8執行路徑。
- Build時間(尤其INT8的262秒)是實際部署要考慮的额外成本,這次只記錄數字,
  沒有評估要不要接受這個成本。
- **這次純粹是探索,沒有做任何「該不該整合進demo」的建議**,是否要往下走
  由Jack自己決定。

## 檔案

- `common.py` — 共用路徑設定跟環境檢查(LD_LIBRARY_PATH提醒)
- `accuracy_clip_vision.py` — clip_vision.onnx 200張holdout圖片cosine similarity比對
- `accuracy_gpt_caption.py` — gpt.onnx 整條pipeline caption生成比對
- `benchmark_latency.py` — build時間 vs 穩態延遲量測
- `*.json` — 實際跑出來的原始數字(git不追蹤,跟Day41 KV cache實驗同樣的處理方式,
  repo的`.gitignore`裡`*.json`是全域規則)
