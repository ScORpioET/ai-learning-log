# Thermal Caption Demo (v1.0)

CLIP + GPT caption 生成 demo,支援 thermal / RGB 兩個分支,CPU FP32 與 GPU FP16
可切換(偵測到GPU才會出現切換鈕),附YOLO偵測疊框顯示。用onnxruntime推論,
不依賴PyTorch跑推論本體(PyTorch只在export/FP16轉換腳本裡用到)。

## 如何取得權重

模型權重(6組onnx + 2個tokenizer + yolov8m.pt,合計約1.6GB)沒有放進git歷史
(`.gitignore` 排除了 `*.onnx` / `*.onnx.data` / `*.pt`),改放在HuggingFace Hub
的public model repo:

https://huggingface.co/ScORpioET/thermal-caption-demo

Clone這個repo之後,先下載權重:

```bash
pip install -r Phase3/Day39/requirements-download.txt
python Phase3/Day39/download_weights.py
```

不需要HF token(repo是public的)。下載完成後,`Phase3/Day39/` 底下會補齊以下檔案,
跟Dockerfile的COPY路徑對齊:

```
clip_vision.onnx / clip_vision.onnx.data
clip_vision.fp16.onnx / clip_vision.fp16.onnx.data
gpt.onnx / gpt.onnx.data
gpt.fp16.onnx / gpt.fp16.onnx.data
gpt_rgb.onnx / gpt_rgb.onnx.data
gpt_rgb.fp16.onnx / gpt_rgb.fp16.onnx.data
tokenizer.pkl
tokenizer_rgb.pkl
yolov8m.pt
```

## Build & Run

CPU版:

```bash
cd Phase3/Day39
docker build -t thermal-caption-demo:v1.0 .
docker run --rm -p 7860:7860 thermal-caption-demo:v1.0
```

GPU版(需要`nvidia-container-toolkit`):

```bash
cd Phase3/Day39
docker build -f Dockerfile.gpu -t thermal-caption-demo-gpu:v1.0 .
docker run --rm --gpus all -p 7860:7860 thermal-caption-demo-gpu:v1.0
```

開啟 http://localhost:7860 使用。

## Dockerfile 的兩種權重載入方式

目前的 Dockerfile / Dockerfile.gpu 是「**方式A:build time COPY**」——build image之前
權重要先透過 `download_weights.py` 下載到本機的 `Phase3/Day39/` 目錄,`docker build`
再用 `COPY` 指令把權重烤進image。這兩種方式的取捨:

**方式A:build time COPY(目前採用)**
- 優點:image本身可攜帶——`docker save`/`docker push`之後,任何人`docker run`
  就能動,不需要額外掛volume或連外網,適合要把image整包丟去其他機器/離線環境跑的情境。
- 缺點:image體積會把1.6GB權重整個包進去(CPU image實測約2GB+,GPU image因為
  CUDA runtime base本身更大,加上權重後更肥),每次改app.py等程式碼重新build,
  即使权重没变,也要看你有没有把COPY权重的层放在COPY程式碼之前(目前是,所以
  程式碼變動不會讓權重層重新下載/複製,但image本身還是一樣大)。

**方式B:runtime volume mount / entrypoint下載**
- 做法:Dockerfile不COPY權重,改成`docker run`時用`-v`掛載host上已下載好的權重
  目錄,或是在container的entrypoint裡跑一次`download_weights.py`再啟動app。
- 優點:image本身小很多(只有程式碼+套件),build更快,同一個image可以配不同
  版本的權重(不用重新build image)。
- 缺點:image不能直接拿去別的地方單獨跑——要嘛使用者自己另外掛volume(等於
  還是要在host先跑過download_weights.py),要嘛entrypoint下載需要container
  執行時有網路,離線環境會失敗;`docker run`指令也變複雜(多一個`-v`參數)。

**這輪先維持方式A**(現有Dockerfile/Dockerfile.gpu不變動邏輯,只更新image
tag到`v1.0`),要不要換成方式B看你想不想要「image可攜帶」還是「image輕量」——
之後要換隨時可以再處理。

## 分支說明

- **Domain**:thermal / rgb,各自獨立的GPT decoder + tokenizer,共用同一份
  domain-agnostic的CLIP視覺encoder(`clip_vision.onnx`)。
- **Device**:有偵測到可用GPU才會出現切換鈕,預設選GPU;沒有GPU的機器只會看到
  CPU,不會出現一個實際上壞掉的GPU選項。
- **YOLO疊框**:沿用既有`KEEP_CLASSES`/`CONF_THRESH=0.25`設定,COCO預訓練
  `yolov8m.pt`,不是針對這個demo另外訓練的模型。
