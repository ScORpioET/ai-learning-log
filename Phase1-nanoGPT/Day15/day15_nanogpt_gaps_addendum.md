## Day 15 追蹤(8/1)—— 組裝完整 training loop + generate() 復盤

範圍:把 Day14 沒修完的 bug 全部修完、組出完整可執行的 training loop、`generate()` 從無到有寫出來、超參數放大成 n_embd=384/n_head=8/n_layer=16 開始正式訓練(後決定不跑完,判斷合理:目標是驗證架構理解,不是成果)。

### 已經穩定、沒有再犯(值得記一筆,代表真的變成肌肉記憶了)

- **super().__init__()**:Day13 至少漏 4 次、Day14 限時預演又漏一次,**Day15 五個 class 全部沒漏**。
- **Block.forward 殘差連接**:Day13、Day14 都漏過,**Day15 一次寫對**。
- **`(i != 0 and i % eval_interval == 0) or i == max_iters-1`**:Day14 記錄過 and 優先於 or 導致條件式幾乎每個 iteration 都觸發,**Day15 重寫 training loop 時括號直接加對**。

### 理解問題(新出現,優先複習)

28. **2D tensor 跟 3D tensor 的單一 index 語意不同,不能套用同一直覺**:`idx`(shape `(B,T)`)用 `idx[-1]` 會保留第 1 維、結果是 `(T,)`——這是對的。但 `logits`(shape `(B,T,C)`)用 `logits[-1]`,選掉的是第 0 維(batch),留下 `(T,C)`——不是你要的,要寫成 `logits[:, -1, :]`。**先確認 tensor 現在是幾維,才能判斷少寫的 index 切掉了哪一維**。

29. **改完一個地方的 shape 邏輯,沒有回頭檢查呼叫端還在假設舊的 shape**:`generate()` 定案回傳 `idx[-1,:]`(已經是完整一維序列)之後,最後呼叫端還留著 `model.generate(context, 1000)[0]`——這個 `[0]` 是上一版邏輯的殘留,邏輯改了但呼叫端沒同步更新。

### 記憶/API 問題(語法層,組裝疏漏)

30. **`estimate_loss()` 只 `model.eval()` 沒有對應收尾 `model.train()`**——這不是不懂 eval 的作用,是「一組配對的 API 只寫了一半」。寫這類「切換狀態」的函式,養成反射:一開始寫 `model.eval()` 的當下,順手把 `return` 前的 `model.train()` 也一起打好。

31. **組裝時漏寫必要的東西:`get_batch()` 缺 `split` 參數、`max_iters` 忘記定義**——這兩個是「六段各自寫的時候都對,組裝成一個檔案時才會出現」的疏漏。組裝階段要多一個習慣:寫完呼叫端,回頭對一次函式簽名,確認每個必填參數都真的給了。

### 已釐清的疑問(主動想通,值得記一筆)

- `idx[-1]` 跟 `idx[:,-1]` 差在選的是哪個維度——Jack 主動追問「那我剛剛的 idx[-1] 不是也對?」,是精準的抓包(Claude 當時的引導把 2D 跟 3D 的情況混為一談,誤導去改成錯的版本)。

---

## Day 15 優先順序(下次複習用)

1. 「切換狀態」型函式(`model.eval()/train()`、`torch.no_grad()` 這類需要配對收尾的 API)寫完開頭就順手寫收尾
2. 動手改任何 tensor 的 shape / 切片邏輯之後,養成反射回頭掃一次所有呼叫端有沒有跟著同步
3. 2D vs 3D tensor 的單一 index 行為差異,找機會用假資料實測驗證一次
