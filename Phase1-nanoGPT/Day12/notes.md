# Day 12 — Karpathy Ep 1 + bigram.py 通關筆記

## 1. Attention 的 sqrt(d_k) scaling

- Q, K 過 LayerNorm + Linear 後 entries 大致 N(0,1)
- dot product 是 d_k 個乘積相加、變異數 = d_k、標準差 = sqrt(d_k)
- 除以 sqrt(d_k) → 標準差拉回 1 → softmax 不飽和、梯度健康
- code 寫 `* k.shape[-1]**-0.5` = 乘 1/sqrt(d_k) = 除以標準差
- 用理論值不實測值、是為了省 runtime overhead
- 「Scaled Dot-Product Attention」的「Scaled」就是這一步

## 2. Causal mask 為什麼用 tril + -inf + softmax

- softmax 前 mask 而非後、才能天然歸一化（否則要 renormalize）
- 用 -inf 因為 exp(-inf) = 0、softmax 完該位置變乾淨的 0
- tril 下三角 = 只看過去、對應 causal 定義

## 3. Residual connection 三個功能

- (a) 學 delta 不學 full mapping、訓練變簡單、廢 layer 也不拖累
- (b) 梯度高速公路：`d(x + f(x))/dx = 1 + f'(x)`、「1」保底、梯度永遠有直達路徑
- (c) 保留原始資訊、attention 混太多也不怕

## 4. Pre-norm vs Post-norm

- 現代 GPT / LLaMA 用 pre-norm：`x + f(LN(x))`
- residual 路徑乾淨、梯度回傳無阻、深模型訓練穩、不需 warmup
- Post-norm（原始 Transformer）LN 卡在 residual 上、深模型難訓練

## 5. LayerNorm vs BatchNorm

- LN: 每個 sample 內、跨自己所有 feature 標準化
- BN: 每個 feature 內、跨 batch 標準化
- NLP 用 LN 三個原因：序列長度不固定、autoregressive 推論 batch=1、跨 token 平均沒物理意義
- CNN 用 BN 因為同 channel 跨圖片有物理意義

## 6. KV Cache（LLM 推論核心優化）

- Generate 時每個新 token 只需要算自己的 Q, K, V
- 之前 token 的 K, V 存 cache 裡、不重算
- 複雜度從 O(T²) 降到 O(T)、實測快 10-50 倍
- 是 LLM 部署 VRAM 大戶、有 PagedAttention / KV cache quantization / MQA 可壓
- 現代 framework（HF transformers、vLLM、TensorRT-LLM）內建、不用手刻

## 7. Communication vs Computation

- Attention = 通訊：token 之間互相看、混合資訊、跨 token
- FFN = 思考：每個 token 拿混完的向量、獨立做非線性變換、per-token
- 節奏：先開會（attention）、再回座位思考（FFN）、重複 n_layer 次
- FFN 佔 Transformer 大部分參數（4x 放大 + 縮回 = 2/3 參數量）

## 8. Decoder-only vs Encoder-Decoder

- Encoder-Decoder（原始 Transformer 2017）：翻譯用、encoder 讀 source、decoder 產 target
- Decoder-only（GPT / LLaMA）：續寫、self-attention 兼職 encoder 工作
- GPT 用 decoder-only 也能翻譯：prompt 讀進來時 self-attention 已經理解 = 隱含 encoder
- 現代都選 decoder-only：架構簡單、in-context learning、資料只要單語

## 9. Training 為什麼並行、快 100+ 倍

- 訓練時每個位置的正解都已知、causal mask 讓每位置預測互相獨立
- 一次 forward 拿到所有 T 位置的 loss、GPU 一次矩陣運算全部算完
- 對比 RNN 序列處理、Transformer 訓練效率壓倒性勝、這是取代 RNN 的主因
- GPU 有幾千 CUDA cores、大矩陣運算能用滿、小 for loop 大部分閒置

## 10. Embedding 量化的 trade-off

- Embedding 是 lookup、memory-bound、不吃 Tensor Core
- INT8 compute 沒收益（跟 Day 10 blog 的 yolov8n memory-bound 一樣邏輯）
- 但 memory 有收益：LLaMA-70B embed table = 524 MB (FP16)、量化能省一半
- 小 model 不划算、大 LLM 才要量化 embedding

## 11. GELU vs ReLU

- ReLU 硬砍負區、GELU 平滑 + 保留一點負區資訊
- Transformer 用 GELU：訓練更穩、避免 dead ReLU、實驗證明效果好
- LLaMA / Qwen 進化到 SwiGLU（更平滑 + 多一個 Linear gate）
