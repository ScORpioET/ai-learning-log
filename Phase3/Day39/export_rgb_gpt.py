"""
v1.1 階段四:RGB分支。

盤點結論:train_vlm.py 訓練時只吃「預先算好的CLIP特徵」(features_train_path:
clip_features_rgb_train.pt),從頭到尾沒有載入/微調CLIP視覺encoder本身
(grep過train_vlm.py,沒有任何 clip_model 相關程式碼)。也就是說 clip_vision.onnx
是domain-agnostic的通用CLIP權重,跟checkpoint(thermal或RGB)無關,RGB分支
不需要重新export一份clip_vision.onnx,只有 gpt.onnx 這半邊(GPT decoder,
真正吃caption/domain訓練的部分)需要從 best_model_rgb_full_reweight2x.pt
重新export。

沿用跟export_capfix.py一樣的邏輯,只是:
- ckpt換成 best_model_rgb_full_reweight2x.pt
- 只export gpt部分(clip_vision.onnx直接沿用Day39現有那份)
- 輸出檔名 gpt_rgb.onnx(跟thermal的gpt.onnx分開,不覆蓋)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "Day36"))
import torch  # noqa: E402
from torch import nn  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
from GPT import GPT, GPTConfig  # noqa: E402

device = 'cpu'
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CKPT_PATH = Path.home() / "ai-transition-2026" / "Phase3" / "Day32" / "checkpoints" / "best_model_rgb_full_reweight2x.pt"
HERE = Path(__file__).parent
SAMPLE_IMAGE = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_rgb_val" / "data" / "video-CgGWcFRZQwyk48xKp-frame-000150-YbQHWyoYefb5kTX7g.jpg"


class ClipVisionWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()

    def forward(self, pixel_values):
        vision_out = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        feat = self.clip_model.visual_projection(pooled)
        return feat


class GPTDecoderWrapper(nn.Module):
    def __init__(self, ckpt_path, block_size, vocab_size, n_layer, n_head, n_embd, clip_vector_size):
        super().__init__()
        self.gpt_model = GPT(GPTConfig(
            block_size=block_size, vocab_size=vocab_size, clip_vector_size=clip_vector_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        )).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.gpt_model.load_state_dict(checkpoint['model'])

    def forward(self, idx, image_feature):
        logits, loss = self.gpt_model(idx.to(device), None, image_feature.to(device))
        return logits


def main():
    clip_wrapper = ClipVisionWrapper(CLIP_MODEL_NAME)
    clip_wrapper.eval()

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    img = Image.open(SAMPLE_IMAGE).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)
    example_input = inputs['pixel_values']

    gpt_wrapper = GPTDecoderWrapper(
        ckpt_path=CKPT_PATH,
        block_size=1024, vocab_size=320, n_layer=12, n_head=12, n_embd=768, clip_vector_size=512,
    )
    gpt_wrapper.eval()
    print(f"[info] checkpoint: {CKPT_PATH}")

    idx_dummy = torch.randint(0, 320, (1, 1), dtype=torch.long)
    dummy_input = (idx_dummy, clip_wrapper(example_input))
    torch.onnx.export(
        gpt_wrapper, dummy_input, str(HERE / "gpt_rgb.onnx"),
        input_names=['input_ids', 'img_feat'], output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}, 'logits': {0: 'batch', 1: 'seq_len'}},
    )
    print(f"[done] {HERE / 'gpt_rgb.onnx'}")

    # ---- sanity check ----
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = gpt_wrapper(torch.tensor([[318]]), clip_wrapper(example_input))
    session = ort.InferenceSession(str(HERE / "gpt_rgb.onnx"))
    onnx_out = session.run(None, {
        "input_ids": torch.tensor([[318]]).cpu().numpy(),
        "img_feat": clip_wrapper(example_input).detach().cpu().numpy(),
    })[0]
    diff = np.abs(torch_out.detach().cpu().numpy() - onnx_out).max()
    cos = np.dot(torch_out.detach().cpu().numpy().flatten(), onnx_out.flatten()) / (
        np.linalg.norm(torch_out.detach().cpu().numpy()) * np.linalg.norm(onnx_out))
    print(f"[sanity check] gpt_rgb.onnx: max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")


if __name__ == "__main__":
    main()
