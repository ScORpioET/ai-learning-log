# Markdown 速查表（Jack 版）

> 用你 INT8 blog 會用到的例子當範例，看原始碼學語法，看渲染後長什麼樣。
> VSCode 按 `Ctrl+Shift+V` 開 preview 邊寫邊看。

---

## 標題

```markdown
# 一級標題（整篇文章只有一個）
## 二級標題（每章用這個）
### 三級標題（章內小節）
```

渲染：

# 一級標題
## 二級標題
### 三級標題

---

## 段落與換行

段落之間**空一行**才會斷段。

```markdown
這是第一段。

這是第二段，中間有空一行。
沒空行的話會被視為同一段。
```

---

## 強調

```markdown
**粗體**用來標關鍵字
*斜體*（中文顯示不明顯，少用）
`inline code`用來包變數名、指令、檔名
~~刪除線~~
```

渲染：**粗體**用來標關鍵字、`inline code`用來包變數名。

---

## 程式碼區塊

前後用**三個反引號** ` ``` `，開頭那行加語言名做 syntax highlight：

````markdown
```python
def quantize_model(onnx_path):
    quantize_static(
        onnx_path,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
```
````

渲染：

```python
def quantize_model(onnx_path):
    quantize_static(
        onnx_path,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
```

**常用語言標記**：`python` / `bash` / `cpp` / `cmake` / `json` / `yaml`

Bash 範例：

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
nvcc --version
```

---

## 列表

**無序列表**用 dash：

```markdown
- INT8 甜區在 compute-bound 模型
- Toolchain 選擇決定 fusion 效率
  - ORT QDQ pattern 對不上 TRT fusion matcher
  - 換 modelopt 才對齊
```

渲染：

- INT8 甜區在 compute-bound 模型
- Toolchain 選擇決定 fusion 效率
  - ORT QDQ pattern 對不上 TRT fusion matcher
  - 換 modelopt 才對齊

**有序列表**用數字：

```markdown
1. PyTorch → ONNX
2. ONNX → 量化 QDQ
3. QDQ ONNX → TensorRT engine
```

---

## 引用（放金句超好用）

```markdown
> INT8 理論上該是 FP32 的 4 倍快，我實際跑起來卻慢了 3 倍。
```

渲染：

> INT8 理論上該是 FP32 的 4 倍快，我實際跑起來卻慢了 3 倍。

---

## 表格（benchmark 一定要用）

```markdown
| Model | FP32 | FP16 | INT8 modelopt | INT8 vs FP16 |
|---|---|---|---|---|
| yolov8n | 504.6 | 828.5 | 566.6 | 輸 32% |
| yolov8m | 184.5 | 368.0 | **386.6** | **贏 5%** |
```

渲染：

| Model | FP32 | FP16 | INT8 modelopt | INT8 vs FP16 |
|---|---|---|---|---|
| yolov8n | 504.6 | 828.5 | 566.6 | 輸 32% |
| yolov8m | 184.5 | 368.0 | **386.6** | **贏 5%** |

**對齊控制**（第二行）：

- `|:---|` 靠左
- `|---:|` 靠右
- `|:---:|` 置中

---

## 連結與圖片

```markdown
[TensorRT 官方文件](https://docs.nvidia.com/deeplearning/tensorrt/)
![INT8 benchmark 圖](./images/benchmark.png)
```

圖片路徑用**相對路徑**，之後上傳 Medium 或 GitHub Pages 都不會壞。

---

## 分隔線

三個 dash 獨立一行，用來分章節或段落中斷：

```markdown
---
```

---

## HTML 註解（寫作備忘用）

發文後**讀者看不到**，只有你自己寫作時看得到：

```markdown
<!-- Day 1 待寫 600-800 字 -->
<!-- TODO: 這裡要補一張 fusion pattern 對比圖 -->
```

你 blog 骨架裡的 `<!-- 待寫 -->` 就是這個。

---

## 你這篇 blog 會頻繁用到的 4 種

1. **`##` 二級標題** — 每章開頭
2. **``` 程式碼區塊 ``` 加語言名** — 貼 script、cmake、bash 指令
3. **表格** — benchmark 數據
4. **`> 引用`** — 放金句（「兩頭空」、「output_names 不是過濾器是命名器」）

其他都是 nice to have。

---

## 進階：Blog 常見小技巧

### 用 `<details>` 折疊長段落

```markdown
<details>
<summary>完整 cmake 指令（點開看）</summary>

```bash
cmake -G Ninja \
  -D WITH_GSTREAMER=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D CUDA_ARCH_BIN=8.9 \
  ..
```

</details>
```

適合把「不是每個讀者都想看的細節」收起來，保持文章可讀性。

### Emoji 適度用（不要過量）

```markdown
- ✅ 通關
- ❌ 失敗
- ⚠️ 注意
- ⭐ 推薦
```

技術 blog 用來標「成功/失敗/警告」很有幫助，但一段裡不要超過 2 個。

### 程式碼區塊裡標行號 highlight

Medium 不支援、GitHub 支援：

````markdown
```python {5,7-9}
```
````

（不是所有平台都吃，Medium 用得少）

---

## 你發布時的注意事項

**Medium**：
- 直接貼 Markdown 貼不進去，要用 [Medium's Markdown Editor](https://markdown-to-medium.surge.sh/) 之類的轉換工具
- 或用 Jekyll / Hugo 產出來的 HTML 貼進去
- 表格會渲染但沒有 syntax highlight —— 程式碼區塊會變成 monospace 灰底但沒顏色
- 引用 `>` 會變成很漂亮的大字引用

**iThome 鐵人賽**：
- 支援標準 Markdown
- 有內建 code highlight
- 每篇 30 天不能刪

**GitHub Pages（Jekyll）**：
- 支援最完整
- 也是你長期資產最佳選擇（自己 domain、SEO 你可以控）

---

## 現學現用：你 blog 第 1 章的骨架（可直接複製改寫）

```markdown
## 1. 第一次嘗試：ONNX Runtime PTQ

主流的 INT8 量化路徑是：

```
PyTorch → ONNX → ORT quantize_static → TensorRT
```

我第一次照這條路走，踩了三個坑。

### 坑 1：Non-zero zero_point

<!-- 這裡展開講你踩到什麼、怎麼發現、怎麼修 -->

### 坑 2：Bias Int32，TensorRT 不吃

<!-- 這裡講 strip_bias_qdq.py 剝 63 個節點 -->

### 坑 3：INT8 比 FP32 慢 3 倍 ⭐

這是整條 debug 之旅最有價值的一坑。

**症狀**：INT8 pure inference 5-7 ms、FP32 只要 3-5 ms。

> 我一開始不敢相信——量化不是應該加速嗎？跑了三次 benchmark 都一樣。

**診斷過程**：

1. 先確認不是量化實作錯 → 用官方 tutorial 重跑一次還是慢
2. 打開 TensorRT verbose log → 看到 fusion 沒發生
3. 對比 FP16 engine 的 layer 結構 → 找到 Q/DQ pattern 沒被 TRT fuse 掉

**根因**：ORT 的 QDQ pattern 跟 TensorRT 的 fusion matcher **設計就不對齊**。

**機制**——這就是我後來稱作「兩頭空」的失敗模式：

- Fusion 失敗 → Q/DQ 變成獨立 kernel
- Conv 找不到量化 pattern → fallback 回 FP32 執行
- 結果：多了 Q/DQ overhead，還沒享受到 INT8 加速

**結論**：不是實作錯，是 toolchain 選錯。ORT 是為 CPU/ORT execution provider 設計的 QDQ，NVIDIA 為 TRT 設計的量化 toolkit 是另外一套。

**工程 ROI 判斷**：與其繼續 hack ORT 的 QDQ pattern 讓 TRT 認得，不如直接換 NVIDIA 官方為 TRT 設計的 `nvidia-modelopt`。
```

---

**祝寫作順利！** 有卡住的語法或想加什麼效果再問我。
