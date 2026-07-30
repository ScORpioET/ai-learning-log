# Day 13 — bigram.py 憑記憶重寫 gaps

範圍：Head → MultiHeadAttention → FeedForward → Block → Bigram（__init__ / forward / generate）
六個部分全部憑記憶重建完成，本檔記錄過程中抓到的問題，供明天複習用。

---

## 理解問題（觀念層，優先複習）

1. **FeedForward 該吃 n_embd，不是 head_size**
   attention 合併回 n_embd 之後才運作，不需要知道 head 的事。

2. **Block 需要多接收一個 n_head 參數**
   Block(n_embd, n_head)，內部自己算 head_size = n_embd // n_head，再往下傳給 MultiHeadAttention。

3. **Bigram 只需要 vocab_size，其他超參數不用傳**
   n_embd/n_head/n_layer 是全域變數，class 內部直接讀取即可；vocab_size 是資料決定的動態值才需要傳參數。
   （對照第 15 點：可重複使用的「積木」class（Head/MHA/Block/FeedForward）則要把參數都乾淨傳入，
   判斷原則不是「是不是全域變數」，是「這個 class 未來會不會被用不同參數值重複呼叫」。）

4. **tril 的 size 要用 block_size，不是 n_embd**
   attention score 形狀是 (T,T)，T 是 block_size（序列長度），跟 n_embd（向量維度）無關，兩個數字意義完全不同容易混用。

5. **forward() 裡的運算結果不能存進 self.，要用區域變數**
   __init__ 建的是「模組」（Linear），forward 裡算的是「這次輸入的結果」，角色不同。
   q = self.query(x)，不是 self.q = self.query(x)（會覆寫模組本身，第二次呼叫就壞掉）。

6. **對 Linear 模組本身做運算，而非它的輸出**
   一定要先 k = self.key(x)，再對 k（tensor）做 transpose / 矩陣乘法，不能對 self.key（模組）做。

7. **scale 符號反了：`** 0.5` 應該是 `** -0.5`**
   要「除以」sqrt(head_size)，不是「乘上」。呼應 Day 12 的 variance 隨維度變大要往下壓的推導。

8. **causal mask 不能用乘法，要用 masked_fill(-inf)**
   乘 0 之後 softmax 還是會給非零機率（exp(0)=1）；masked_fill(mask==0, float('-inf'))，exp(-inf)=0 才是真的遮住未來。

9. **softmax 要在最後一維（dim=-1），不是 dim=1**
   每個 query 對所有 key 的分數要加總為 1，是最後一維。

10. **pos_emb 要用 torch.arange(T)，不能拿 x 本身去查**
    x 裡裝的是「這個字是 vocab 第幾號」，跟「這個字排在句子第幾個位置」是完全不同的兩件事，
    就算兩句話字完全不同，「第 3 個位置」永遠是同一個數字 3。

11. **position_embedding 的表大小要用 block_size，不是 vocab_size**
    跟第 4 點同類混淆：vocab_size（有幾種字）vs block_size（序列多長），兩個都是單一數字容易想岔。

12. **lm_head 投影不管有沒有 targets 都要無條件執行，只有 loss 計算才是 if/else 分支**
    第一次寫成把 self.lm_head(...) 塞進 else 分支裡，導致 generate()（targets=None）時
    回傳的 logits 還停在 n_embd 維、根本沒投影到 vocab_size，後面 softmax 維度就整個錯。

13. **generate() 裡 logits[:, -1, :] 這個切法會降維**
    (B,T,C) → (B,C)，不是 (B,1,C)。切掉的維度會直接消失，不會保留成 1。

14. **LayerNorm 在 forward 當下防的是「數值分佈不穩」，不是「梯度」**
    forward pass 本身不算梯度、不更新參數，只是數值往前推。LayerNorm 防止數值爆走導致
    後面的 matmul / softmax 提早飽和。LayerNorm 位置真正影響梯度的地方在 backward：
    pre-norm（x + self.sa(self.ln1(x))）讓殘差相加的兩邊維持原始 x，
    梯度能沿加法直接、無阻礙地往回流（梯度高速公路）；
    post-norm（x + self.ln1(self.sa(x))，Jack 第一版寫的）會讓 LN 擋在殘差路徑上，
    每多一層 Block 就多一層阻礙，這是 post-norm 需要 warmup、訓練較不穩定的原因。

---

## 記憶/API 問題（語法層，密集複習修正）

15. **super().__init__() 反覆漏打**（Head ×2、Block ×1、Bigram ×1，今天至少漏 4 次）
    ⭐ 出現頻率最高的錯，明天第一件事：建立成反射動作，寫 class X(nn.Module): 後手指自動先打這行。

16. **打 class 骨架時手滑打成 def Head(...)**（第一輪）——純打字習慣問題。

17. **nn.Module 打成 nn.modules**（大小寫 + 多打 s）——兩個名字長得太像，容易混。

18. **masked_fill 是 tensor 的 method，不是 torch. 開頭的函式**
    wei.masked_fill(...) 對，torch.masked_fill(wei, ...) 錯。

19. **nn.Dropout(dropout) 有時打成 torch.dropout(dropout)**（第一輪對、第二輪重寫時錯）
    代表這個 API 還沒完全穩定記住，不是不會，是不穩定。

20. **參數名 head_size 打成 num_head**（第二輪重寫時新錯）
    跟 MultiHeadAttention 要用的 num_heads 長得太像、意思不同，容易混。

21. **打字錯字 flaot('-inf')，且誤用成關鍵字參數 flaot=...**
    masked_fill 第二個參數是位置參數，不用寫關鍵字。

22. **Linear 層輸出 shape 註解常寫成 (B,T,C)，應該是 (B,T,head_size)**（兩輪都犯）
    純註解錯誤不影響執行，但會誤導未來自己 debug。

23. **torch.multinomial 的 num_samples 是必填參數**，第一版漏掉直接會報錯。

24. **torch.cat 第一個參數要是 tuple/list，不能拆開寫成兩個獨立參數**
    要 torch.cat((idx, out), dim=-1)，不是 torch.cat(idx, out, -1)。

25. **算完新的 tensor 忘記存回原變數**（torch.cat 那行第一版沒有 idx = ...，算完就消失了）
    forward() 裡也犯過類似的：忘記 return，函式跑完等於回傳 None。

26. **targets 參數忘記給預設值 None**
    generate() 呼叫 self(idx_cond) 時只傳一個參數，沒有預設值會直接報錯。

27. **target / targets 大小寫或單複數打錯字**
    參數名叫 targets（有 s），內部用成 target.view(...)，執行時才會噴 NameError，Python 不會提前抓到。

---

## 已釐清的疑問（主動想通，不是被抓包，值得記一筆代表這是自己補上的理解）

- **nn.Linear 只對 tensor 最後一維做矩陣乘法**，前面的 B、T 維度不受影響、不會被攤平混在一起算。
  是「對每個 token 各自套用同一組權重」，不是「把整個 tensor 攤平成一條向量」。
- **post-norm vs pre-norm** 差在殘差相加的兩邊有沒有先被 LayerNorm 動過——自己寫錯一次、
  被問「為什麼」之後，能自己反推出「喔這是 post-norm」，代表機制是真懂的，不是背名詞。

---

## 明天優先順序

1. super().__init__() 反射動作（第 15 點）—— 5 分鐘內先確認能不能一次不漏地在四個 class 都寫對
2. pos_emb / lm_head 那兩個理解點（第 10、11、12 點）快速複述一次
3. 把今天寫對的六段組成一個完整檔案（今天沒做完的部分）：
   - import + hyperparameters 放最上面
   - 五個 class 的順序限制：想一下為什麼不能把 Bigram 寫在最前面
   - get_batch / estimate_loss / data loading（今天完全沒練到，明天要補）
   - training loop 順序
4. 組完跟原始 bigram.py diff，確認語法都對
5. 丟進 WSL2 實際跑一次，loss 曲線跟 Day 11 baseline 比對
