"""
TensorRT EP實驗共用工具。純好奇心實驗,只讀取Day36/Day39既有的ONNX/checkpoint
產物,不修改它們,也不影響正式Docker demo(Phase3/Day39/app/)的任何呼叫路徑。

環境備註(必須先知道,不然TensorrtExecutionProvider會直接載入失敗):
機器上同時存在兩份不相容的TensorRT:
  - apt裝的系統版TensorRT 11.1.0(libnvinfer.so.11,在/usr/lib/x86_64-linux-gnu/)
  - pip依賴帶進來的TensorRT 10.x(libnvinfer.so.10,在.venv裡的tensorrt_libs套件)
這裡用的onnxruntime-gpu 1.22.0是對TensorRT 10.x ABI編譯的,系統版11.1完全用不了
(直接報 "libnvinfer.so.10: cannot open shared object file")。解法是把
LD_LIBRARY_PATH指到pip那份10.x的位置,讓它排在系統路徑前面。這個實驗的所有腳本
都要在這個環境變數設定好的狀況下執行:

    LD_LIBRARY_PATH=<repo>/.venv/lib/python3.10/site-packages/tensorrt_libs python3 xxx.py

(或用本檔案的 ensure_tensorrt_ld_library_path() 在程式一開始檢查/提醒,不會自動
幫你設,因為LD_LIBRARY_PATH要在python process啟動"前"就生效,程式內部設environ
對已經載入的動態連結器無效。)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
DAY36 = Path.home() / "ai-transition-2026" / "Phase3" / "Day36"
DAY39 = Path.home() / "ai-transition-2026" / "Phase3" / "Day39"
IMAGE_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"

TRT_LIBS_DIR = Path.home() / "ai-transition-2026" / ".venv" / "lib" / "python3.10" / "site-packages" / "tensorrt_libs"

# 模型檔案路徑(全部讀既有的,不新產生任何正式權重)
CLIP_FP32_ONNX = DAY36 / "clip_vision.onnx"
CLIP_FP16_ONNX = DAY39 / "clip_vision.fp16.onnx"          # 現行部署用的CUDA EP FP16版本(keep_io_types轉換法)
CLIP_INT8_QDQ_ONNX = DAY36 / "clip_vision.int8.onnx"      # Day36 baseline QDQ INT8,CUDA EP上cosine sim量到0.547456

GPT_FP32_ONNX = DAY36 / "gpt.onnx"
GPT_FP16_ONNX = DAY39 / "gpt.fp16.onnx"                   # 現行部署用的CUDA EP FP16版本
GPT_INT8_QDQ_ONNX = DAY36 / "gpt.int8.exp19_baseline.onnx"  # Day38 exp19 baseline QDQ INT8


def ensure_tensorrt_ld_library_path():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(TRT_LIBS_DIR) not in ld:
        print(f"[FATAL] LD_LIBRARY_PATH沒有包含 {TRT_LIBS_DIR} ,"
              f"TensorrtExecutionProvider會直接載入失敗(系統版TensorRT 11.1跟這份"
              f"onnxruntime-gpu的ABI不相容)。請用:\n"
              f"  LD_LIBRARY_PATH={TRT_LIBS_DIR} python3 {sys.argv[0]}\n"
              f"重新執行。", file=sys.stderr)
        sys.exit(1)
