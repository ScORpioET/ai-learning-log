"""
gpt.onnx 精度比對:整條pipeline跑到底,比對CUDA EP FP32(基準)/ TensorRT EP FP16 /
TensorRT EP INT8 三者生成出來的caption文字有沒有差、差多少。

img_feat固定用CUDA EP FP32的clip_vision.onnx抽,三個版本都吃同一份img_feat輸入——
這樣才是單純在比「GPT decoder換EP/精度」造成的差異,不會被clip_vision那邊的
變因混進來(clip_vision的精度已經在accuracy_clip_vision.py另外測過)。

decoding用greedy(跟Day41 KV cache實驗一致的理由:排除隨機性,乾淨比較邏輯本身)。

TensorRT EP吃gpt.onnx這種動態shape(seq_len隨生成步數變長)的模型,如果不給
明確的shape profile,TRT會在每次遇到新shape時重新build engine(seq_len每次
+1就要重build一次,完全不可行)。這裡用trt_profile_min/opt/max_shapes明確
告訴TensorRT整個shape範圍(1~64),讓它一次build好一個涵蓋整個範圍的engine,
之後每個不同長度的推論都用同一個engine,不會重複build。
"""
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor

from common import (
    ensure_tensorrt_ld_library_path, IMAGE_ROOT, DAY39,
    CLIP_FP32_ONNX, GPT_FP32_ONNX, GPT_INT8_QDQ_ONNX,
)

ensure_tensorrt_ld_library_path()

import sys  # noqa: E402
from common import DAY36  # noqa: E402
sys.path.insert(0, str(DAY39 / "app"))
from minbpe import minbpe  # noqa: E402

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319
MAX_NEW_TOKENS = 40

TEST_IMAGES = [
    "data/video-JhYLiFCieHQHaY8o7-frame-000000-xT7BXRKKyuWEsnywX.jpg",
    "data/video-JhYLiFCieHQHaY8o7-frame-000600-54cr88GJsdAyksdNj.jpg",
    "data/video-JhYLiFCieHQHaY8o7-frame-000900-pdb96S7B7fgmguxPE.jpg",
]

TRT_SHAPE_PROFILE = {
    "trt_profile_min_shapes": "input_ids:1x1,img_feat:1x512",
    "trt_profile_max_shapes": f"input_ids:1x{MAX_NEW_TOKENS + 1},img_feat:1x512",
    "trt_profile_opt_shapes": "input_ids:1x20,img_feat:1x512",
}


def greedy_generate(gpt_sess, img_feat):
    idx = np.array([[IMAGE_TOKEN_ID]], dtype=np.int64)
    for _ in range(MAX_NEW_TOKENS):
        logits = gpt_sess.run(None, {"input_ids": idx, "img_feat": img_feat})[0]
        next_id = int(np.argmax(logits[0, -1]))
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == EOS_TOKEN_ID:
            break
    return idx[0].tolist()


def decode(tokenizer, token_ids):
    clean = [i for i in token_ids if i not in (IMAGE_TOKEN_ID, EOS_TOKEN_ID)]
    return tokenizer.decode([clean])


def main():
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    # GPT_FP32_ONNX (Day36/gpt.onnx) 跟 GPT_INT8_QDQ_ONNX (exp19 baseline) 都是從
    # 舊checkpoint(exp2_reweight2x)匯出的,要配Day36自己的tokenizer.pkl,不是
    # Day39那份配capfix checkpoint用的tokenizer.pkl(配錯會生出亂碼文字,不是
    # generation本身壞掉——這裡只是還原成可讀文字,不影響token id序列本身的比較)。
    tokenizer = minbpe.load(str(DAY36 / "tokenizer.pkl"))

    print("[info] extracting img_feat with CUDA EP FP32 clip_vision.onnx (fixed reference for all 3 GPT variants)")
    clip_sess = ort.InferenceSession(str(CLIP_FP32_ONNX), providers=["CUDAExecutionProvider"])
    img_feats = {}
    for fn in TEST_IMAGES:
        img = Image.open(IMAGE_ROOT / fn).convert("RGB")
        pv = processor(images=[img], return_tensors="pt")["pixel_values"].numpy()
        img_feats[fn] = clip_sess.run(None, {"pixel_values": pv})[0]

    print("[info] loading GPT sessions: CUDA-FP32 baseline, TensorRT-FP16, TensorRT-INT8")
    cuda_fp32_sess = ort.InferenceSession(str(GPT_FP32_ONNX), providers=["CUDAExecutionProvider"])

    t0 = time.perf_counter()
    trt_fp16_sess = ort.InferenceSession(str(GPT_FP32_ONNX), providers=[
        ("TensorrtExecutionProvider", {**TRT_SHAPE_PROFILE, "trt_fp16_enable": True}),
    ])
    print(f"[info] TRT FP16 GPT session built in {time.perf_counter()-t0:.1f}s")

    trt_int8_sess = None
    trt_int8_status = None
    t0 = time.perf_counter()
    try:
        trt_int8_sess = ort.InferenceSession(str(GPT_INT8_QDQ_ONNX), providers=[
            ("TensorrtExecutionProvider", {**TRT_SHAPE_PROFILE, "trt_int8_enable": True}),
        ])
        trt_int8_status = "ok (trt_int8_enable alone)"
    except Exception as e:
        print(f"[warn] trt_int8_enable alone failed for gpt.onnx: {str(e)[:200]}")
        try:
            trt_int8_sess = ort.InferenceSession(str(GPT_INT8_QDQ_ONNX), providers=[
                ("TensorrtExecutionProvider", {**TRT_SHAPE_PROFILE, "trt_int8_enable": True, "trt_fp16_enable": True}),
            ])
            trt_int8_status = "ok (needed trt_int8_enable + trt_fp16_enable together)"
        except Exception as e2:
            trt_int8_status = f"FAILED even with both flags: {str(e2)[:300]}"
    print(f"[info] TRT INT8 GPT session build status: {trt_int8_status} (took {time.perf_counter()-t0:.1f}s)")

    print("\n" + "=" * 90)
    for fn in TEST_IMAGES:
        img_feat = img_feats[fn]
        cuda_tokens = greedy_generate(cuda_fp32_sess, img_feat)
        trt_fp16_tokens = greedy_generate(trt_fp16_sess, img_feat)
        cuda_caption = decode(tokenizer, cuda_tokens)
        trt_fp16_caption = decode(tokenizer, trt_fp16_tokens)

        print(f"\nimage: {fn}")
        print(f"  CUDA-FP32 (基準)      : {cuda_caption!r}")
        print(f"  TensorRT-FP16          : {trt_fp16_caption!r}  "
              f"[{'一致' if trt_fp16_tokens == cuda_tokens else '不一致'}]")

        if trt_int8_sess is not None:
            trt_int8_tokens = greedy_generate(trt_int8_sess, img_feat)
            trt_int8_caption = decode(tokenizer, trt_int8_tokens)
            print(f"  TensorRT-INT8          : {trt_int8_caption!r}  "
                  f"[{'一致' if trt_int8_tokens == cuda_tokens else '不一致'}]")
        else:
            print(f"  TensorRT-INT8          : 無法建立engine,跳過。status: {trt_int8_status}")

    print("\n" + "=" * 90)
    print(f"[info] TRT INT8 build status: {trt_int8_status}")


if __name__ == "__main__":
    main()
