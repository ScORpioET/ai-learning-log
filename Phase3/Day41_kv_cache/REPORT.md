# KV Cache vs No-Cache generate() 量化比較

範圍:PyTorch eager版本,`best_model_full_capfix_reweight2x.pt`。不動ONNX export/Docker
demo/正式checkpoint,全部程式碼在這個新目錄底下,舊的`Phase3/Day36/GPT.py`跟既有的
`generate()`呼叫路徑完全沒動。

## 0. 先講一個過程中發現、必須先修掉才能往下測的bug

原版`GPT.forward()`裡有這行,對每次forward輸入序列的**第0個位置**無條件蓋上image
embedding:

```python
tok_emb[:,0] = image_vector_emb[:]
```

no-cache模式下每步都重新餵「從image token開始的完整序列」,idx[:,0]永遠是image
token,這行永遠對。但如果原封不動套進KV cache的增量解碼(每步只餵最新生成的1個
token),T=1那個位置不再是「序列的第0個位置」而是「這一步新生成的token」——這行
會錯誤地把它蓋成image embedding。

這是讀`forward()`程式碼時先看出來的,不是先跑壞了才回頭查(所以下面的correctness
check從一開始就是against已修好的版本)。為了確認這個判斷/修法本身有意義、不是我
自己憑空想的假問題,另外跑了一次**故意保留原始bug**的對照測試(見下方),證實不修
的話cache版本會整個壞掉,不是「理論上有風險但實務上可能沒事」。

修法(`GPT_kv_cache.py`):只有在`past_key_value is None`(=這次forward真的包含
position 0,也就是no-cache的每一步,或KV cache的prefill那一步)才做這個覆蓋;
KV cache的增量解碼步跳過,image embedding已經在prefill那一步的cache裡了。這個
修法對no-cache路徑的行為**完全沒有改變**(no-cache呼叫時`past_key_value`恆為
`None`),純粹是補上cache路徑缺的條件判斷。

### 故意不修、驗證「這bug真的會炸」的對照測試

同一個checkpoint、同樣3張測試圖、greedy decoding,不修這行直接跑cache版本:

| 圖片 | no_cache(正常) | 不修bug的cache版本 |
|---|---|---|
| frame-000000 | `[318, 84, 315, 274, 44, 275, 285, 288, 59, ...]`(通順token序列,遇到319 EOS結束) | `[318, 84, 84, 78, 78, 78, 84, 84, 78, 78, 78, 78, 78, 78, 84, ...]`(從第2步開始就卡在幾個token反覆橫跳,60步都不會遇到EOS) |

三張圖全部同樣的退化模式:不修的話,cache版本從生成的第一個新token開始就完全跑掉,
不是「差一點點」,是整個生成邏輯失效。這證實了上面的推理不是紙上談兵。

## 1. 正確性驗證(greedy + sampling,逐token比對)

修完bug之後的`GPT_kv_cache.py`,跑`correctness_check.py`:同一checkpoint、同一批
3張validation圖片、同一起始prompt(image token),`generate_no_cache()` vs
`generate_with_cache()`,max_new_tokens=60(不提前因EOS停,取自然生成長度)。

| 圖片 | greedy 逐token比對 | sampling(seed=1337) 逐token比對 |
|---|---|---|
| video-JhYLiFCieHQHaY8o7-frame-000000 | **完全一致** | **完全一致** |
| video-JhYLiFCieHQHaY8o7-frame-000600 | **完全一致** | **完全一致** |
| video-JhYLiFCieHQHaY8o7-frame-000900 | **完全一致** | **完全一致** |

`all_greedy_match = True`(見`correctness_results.json`)。sampling版本用同一個
seed分別重設在no-cache/cache呼叫前,兩邊都一致——這條路徑没有額外的隨機性來源
需要排查。

**結論:修完bug後,cache版本產生的token序列跟現行no-cache版本逐token完全一致,
不是「差不多」或「品質接近」,是bit-exact相同的輸出。**

## 2. 效能量化比較

方法論比照`Phase1-nanoGPT/Day18`那批benchmark腳本:每個設定warmup 3次、正式計時
10次取median(不是單次數字);GPU計時前後都`torch.cuda.synchronize()`。生成長度
用`force_full_length=True`強制跑滿,不因為提早遇到EOS少算步數,確保cache/no-cache
兩邊在同一個長度設定下算的量是真的一樣多。

環境:CPU(WSL2,本機)/ GPU(RTX 4070,CUDA 12.6,torch 2.13+cu126)。

### CPU

| 生成長度 | no-cache median | cache median | no-cache 每token | cache 每token | speedup |
|---:|---:|---:|---:|---:|---:|
| 20 | 383.46 ms | 240.50 ms | 19.17 ms | 12.03 ms | **1.59x** |
| 50 | 1247.05 ms | 604.55 ms | 24.94 ms | 12.09 ms | **2.06x** |
| 100 | 4126.97 ms | 1346.05 ms | 41.27 ms | 13.46 ms | **3.07x** |

CPU上cache版本明顯更快,而且加速倍率隨長度增加而放大(1.59x→2.06x→3.07x),符合
「no-cache是O(T²)、cache是O(T)」的理論預期——長度越長,重算前面所有token的浪費
越明顯。

### GPU (CUDA, RTX 4070)

| 生成長度 | no-cache median | cache median | no-cache 每token | cache 每token | speedup |
|---:|---:|---:|---:|---:|---:|
| 20 | 117.42 ms | 126.83 ms | 5.87 ms | 6.34 ms | **0.93x** |
| 50 | 247.96 ms | 257.88 ms | 4.96 ms | 5.16 ms | **0.96x** |
| 100 | 551.13 ms | 557.89 ms | 5.51 ms | 5.58 ms | **0.99x** |

**這個結果不符合「cache理論上應該更快」的直覺預期,如實記錄:在這個模型規模
(12層、n_embd=768,batch=1)跟這個長度範圍(≤100 token)下,GPU上cache版本
反而略慢(或至多打平),不是更快。**

## 3. 記憶體

CUDA用`torch.cuda.max_memory_allocated()`量測(reset peak stats後單次呼叫的峰值);
CPU沒有量測——`resource.getrusage().ru_maxrss`是process-wide高水位,不會因為單次
函式呼叫重設,量出來的數字沒有意義,這次選擇不硬做,不是忘記做。

| 生成長度 | no-cache peak mem (GPU) | cache peak mem (GPU) |
|---:|---:|---:|
| 20 | 969.6 MB | 968.2 MB |
| 50 | 974.0 MB | 970.5 MB |
| 100 | 982.8 MB | 974.3 MB |

差異很小(<1%),而且大部分是CUDA context+模型權重的固定開銷,不是KV cache本身
的buffer。在這個長度範圍(≤100 token)、這個模型規模下,KV cache理論上該省的
attention矩陣記憶體(O(T²)→O(T))量級太小,量不出明顯差異。

## 4. 為什麼GPU上cache沒有優勢(意外發現,誠實記錄,不是為了讓結論好看而略過)

沒有進一步做profiling去confirm root cause,但可以合理推測:

- 這個GPT decoder本身很小(12層、n_embd=768),batch=1,序列長度只到100,
  GPU上單次forward的實際運算時間可能已經被kernel launch overhead主導,
  no-cache「重算全部」跟cache「只算最新1個token」在GPU上的wall-clock差異
  被launch overhead蓋過去了。
- cache版本每步都要做`torch.cat`把新的k/v接到past_key_value後面,這個操作
  在CPU上相對便宜,但在GPU上是額外的kernel + 記憶體搬移,可能抵銷掉省下的
  attention計算量。
- CPU上因為沒有launch overhead這個乾擾項,重算完整序列的計算量差異(O(T²)
  vs O(T))才會直接反映成wall-clock時間差。

這些是推測,沒有驗證,寫在這裡是為了誠實標注「不知道」的部分,不是結論。

## 5. 結論(僅限這次測試範圍,不代表最終採用建議)

- **正確性**:修掉一個關鍵bug(image embedding覆蓋邏輯)之後,cache版本跟
  no-cache版本逐token bit-exact相同,greedy跟seeded sampling都驗證過。
- **CPU效能**:cache版本明顯更快,且優勢隨生成長度放大(20 token 1.59x,
  100 token 3.07x)。
- **GPU效能**:在這個模型規模跟長度範圍內,cache版本沒有優勢,甚至略慢
  (0.93x~0.99x)。这跟理論預期不符,原因未深入profiling確認。
- **記憶體**:GPU上差異<1%,量不出明顯效果。
- 要不要在正式pipeline(目前是CPU demo + GPU demo都用no-cache)裡採用cache
  版本,取決於部署環境——如果主要跑在CPU上(目前CPU demo的情況),這個結果
  支持採用;如果主要跑在GPU上,這次的數字不支持只因為「理論上該更快」就換過去,
  需要先做進一步profiling(例如更長的生成長度、更大的模型、torch.compile、
  或CUDA Graphs)才能判斷cache在GPU上到底有沒有用。**這次只回答「差異有多大」,
  要不要採用是下一步的決定,這次沒有做。**

## 檔案

- `GPT_kv_cache.py` — 修過bug的GPT class副本(獨立於`Phase3/Day36/GPT.py`)
- `common.py` — 共用loader + `generate_no_cache()` / `generate_with_cache()`
- `correctness_check.py` — 正確性驗證腳本,輸出`correctness_results.json`
- `benchmark_latency.py` — 效能量測腳本,輸出`benchmark_results.json`
- `correctness_results.json` / `benchmark_results.json` — 實際跑出來的原始數字
