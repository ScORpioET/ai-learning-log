# Day 12 — 記憶點清單（面試常識 / PyTorch 慣例）

## PyTorch 慣例

- **register_buffer('name', tensor)** 三件事：跟 .to(device) 走 / 不被 optimizer 更新 / 進 state_dict
  - 判斷：跟 model 走但不被學 → buffer；跟 model 走且被學 → nn.Parameter
- **nn.Module 的 .to() 是 in-place**、tensor 的 .to() 不是（回傳新 tensor）
- **self.apply(fn)**：遞迴走過所有 submodule、每個都呼叫 fn 一次；配合 isinstance 判斷分流
- **nn.ModuleList vs 普通 list**：普通 list 導致 parameters() 抓不到、silent bug、model 學不到
- **nn.Sequential(*[...])** 的 `*` 是 Python unpack、把 list 拆成獨立參數
- **dtype=long 給 Embedding**：`nn.Embedding` 的 index 必須是 int64、int32 會爆
- **`set_to_none=True`** 是 zero_grad 的新版預設、省 memory
- **Buffer 建到「上限」、runtime 按「當下 T」切**（tril 的用法）

## 架構常識

- **Attention 內 Q/K/V Linear bias=False**：前面有 LayerNorm、bias 冗餘
- **FFN 中間層 4x n_embd**：Transformer 慣例、經驗值、佔大部分參數量
- **head_size × num_heads = n_embd**：concat 完剛好回 n_embd 的慣例
- **MHA 的 self.proj**：跨 head 資訊融合 + shape 對齊
- **兩個獨立 LN（ln1, ln2）**：不同 stage 的分佈不同、各自學 gamma/beta
- **ln_f（final LayerNorm）**：pre-norm 架構下 residual stream 從沒被 norm 過、head 前要收尾
- **std=0.02 初始化**：GPT-2 官方值、太大爆、太小學不動
- **_init_weights + self.apply**：模型初始化的標準 pattern

## Optimizer

- **AdamW = Adam + decoupled weight decay**、現代 LLM 標配
- **Adam vs SGD**：Adam 有 per-parameter adaptive lr + momentum + second moment、對 Transformer 各層梯度 scale 差異大的情況友善
- **SGD 不用於 Transformer**：要調 lr scheduler 調到死才勉強能訓練

## 名詞（面試會問要能講）

- **Autoregressive**：每步 output 是下步 input、LLM 推論慢的根本原因
- **In-context learning**：GPT 用 prompt 傳達 task、不用 fine-tune 就能做新 task
- **Weight tying**：Embedding table 和 lm_head 共享 weight（GPT-2 有、你的 bigram 沒）
- **Temperature**：softmax(logits / T)、T<1 更保守、T>1 更創意
- **Top-k / Top-p (nucleus)**：另外兩個 sampling strategy

## Sampling 策略對比

- **argmax**：決定性、輸出重複、遇到 t 永遠 h
- **multinomial**：機率採樣、有多樣性、也可能亂
- **加 temperature**：控制多樣性程度
- **加 top-k / top-p**：只從 top 幾個候選採樣、避免爛 token

## Training 兩階段

- **Pretraining**：海量文字、預測下一 token、需要幾千 GPU 幾個月、只有大公司做
- **Fine-tuning**：
  - **SFT**：指令 + 好回答的成對資料、教 model 聽話
  - **RLHF**：人類選好回答 → reward model → RL 調 base model、ChatGPT 的祕方
  - **LoRA / QLoRA**：只訓練 adapter、消費級 GPU 就能跑、你 Phase 1 Week 6 會做

## Optimized MHA（nanoGPT 版本、你版本是教學版）

- Q/K/V 合成一個 nn.Linear(n_embd, 3 * n_embd)、一次 matmul
- 用 view + transpose 把 num_heads 攤成 batch dim
- 一次 batched matmul 算完所有 heads
- 比 for loop 版本快 2-3 倍、GPU utilization 高

## 待實驗（有空玩）

- 把 ReLU 換成 GELU、看 loss 曲線差別
- 加上 _init_weights、看訓練前 500 步是否更平滑
- Uncomment 那行、生成 10000 個 token 存檔看品質