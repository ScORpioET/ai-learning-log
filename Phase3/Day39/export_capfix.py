"""
沿用 Day37 (Phase3/Day36/export.py) 的 export 邏輯,改成從
best_model_full_capfix_reweight2x.pt(Day39定案版本)重新匯出 clip_vision.onnx / gpt.onnx,
並沿用同一份腳本內建的 PyTorch vs ONNX cosine similarity sanity check。

跟原本 export.py 的差異只有:
1. ckpt_path 改指向 best_model_full_capfix_reweight2x.pt(不再用 hydra config,
   因為模型超參數本身沒變,直接沿用 Day36/config/model/gpt2_124m.yaml 的值寫死)
2. 輸出檔案放在 Phase3/Day39/,不覆蓋 Day36 那份(舊的、對應舊checkpoint的
   clip_vision.onnx/gpt.onnx 保留下來當作歷史紀錄,不要重蹈之前"直接覆寫忘記備份"的錯)
3. tokenizer 改用 tokenizer_full_capfix.pkl(已複製到本資料夾 tokenizer.pkl)
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
CKPT_PATH = Path.home() / "ai-transition-2026" / "Phase3" / "Day32" / "checkpoints" / "best_model_full_capfix_reweight2x.pt"
HERE = Path(__file__).parent
SAMPLE_IMAGE = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val" / "data" / "video-57kWWRyeqqHs3Byei-frame-000816-b6tuLjNco8MfoBs3d.jpg"


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

    gpt_wrapper = GPTDecoderWrapper(
        ckpt_path=CKPT_PATH,
        block_size=1024, vocab_size=320, n_layer=12, n_head=12, n_embd=768, clip_vector_size=512,
    )
    gpt_wrapper.eval()
    print(f"[info] checkpoint: {CKPT_PATH}")

    example_input = inputs['pixel_values']

    torch.onnx.export(
        clip_wrapper, example_input, str(HERE / "clip_vision.onnx"),
        input_names=['pixel_values'], output_names=['img_feat'],
    )
    print(f"[done] {HERE / 'clip_vision.onnx'}")

    idx_dummy = torch.randint(0, 320, (1, 1), dtype=torch.long)
    dummy_input = (idx_dummy, clip_wrapper(example_input))
    torch.onnx.export(
        gpt_wrapper, dummy_input, str(HERE / "gpt.onnx"),
        input_names=['input_ids', 'img_feat'], output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}, 'logits': {0: 'batch', 1: 'seq_len'}},
    )
    print(f"[done] {HERE / 'gpt.onnx'}")

    # ---- sanity check:PyTorch vs ONNX cosine similarity(沿用Day37同一套做法) ----
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = clip_wrapper(example_input)
    session = ort.InferenceSession(str(HERE / "clip_vision.onnx"))
    onnx_out = session.run(None, {"pixel_values": example_input.detach().cpu().numpy()})[0]
    diff = np.abs(torch_out.detach().cpu().numpy() - onnx_out).max()
    cos = np.dot(torch_out.detach().cpu().numpy().flatten(), onnx_out.flatten()) / (
        np.linalg.norm(torch_out.detach().cpu().numpy()) * np.linalg.norm(onnx_out))
    print(f"\n[sanity check] clip_vision.onnx: max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")

    with torch.no_grad():
        torch_out = gpt_wrapper(torch.tensor([[318]]), clip_wrapper(example_input))
    session = ort.InferenceSession(str(HERE / "gpt.onnx"))
    onnx_out = session.run(None, {
        "input_ids": torch.tensor([[318]]).cpu().numpy(),
        "img_feat": clip_wrapper(example_input).detach().cpu().numpy(),
    })[0]
    diff = np.abs(torch_out.detach().cpu().numpy() - onnx_out).max()
    cos = np.dot(torch_out.detach().cpu().numpy().flatten(), onnx_out.flatten()) / (
        np.linalg.norm(torch_out.detach().cpu().numpy()) * np.linalg.norm(onnx_out))
    print(f"[sanity check] gpt.onnx:         max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")


if __name__ == "__main__":
    main()
