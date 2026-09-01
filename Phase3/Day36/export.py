import json
import hydra
import pickle
import random
from pathlib import Path
from PIL import Image
from omegaconf import DictConfig
import inspect
from GPT import GPT, GPTConfig

import regex as re
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from minbpe import minbpe

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

root = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

class ClipVisionWrapper(nn.Module):
    def __init__ (self, model_name):
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
            block_size=block_size,
            vocab_size=vocab_size,
            clip_vector_size=clip_vector_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.gpt_model.load_state_dict(checkpoint['model'])

    def forward(self, idx, image_feature):
        logits, loss = self.gpt_model(idx.to(device), None, image_feature.to(device))

        return logits
 
@hydra.main(version_base=None, config_path="config", config_name="config")
def export(cfg: DictConfig):

    clip_wrapper = ClipVisionWrapper(CLIP_MODEL_NAME)
    clip_wrapper.eval()

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    name = '/home/jack/ai-transition-2026/thermal_dataset/images_thermal_val/data/video-57kWWRyeqqHs3Byei-frame-000816-b6tuLjNco8MfoBs3d.jpg'
    img = Image.open(name).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)

    gpt_wrapper = GPTDecoderWrapper(
        ckpt_path=cfg.train.ckpt_path,
        block_size=cfg.model.block_size,
        vocab_size=cfg.model.vocab_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        clip_vector_size=cfg.model.clip_vector_size,
    )
    gpt_wrapper.eval()


    example_input = inputs['pixel_values']

    torch.onnx.export(
        clip_wrapper,              # model 是誰？
        example_input,              # example inputs，注意這裡通常要包成 tuple
        "clip_vision.onnx",
        input_names=['pixel_values'],
        output_names=['img_feat'],
    )

    idx_dummy = torch.randint(0, cfg.data.base_vocab_size+cfg.data.special_token_size, (1, 1), dtype=torch.long)

    dummpy_input = (idx_dummy, clip_wrapper(example_input))

    input_names = ['input_ids', 'img_feat']
    output_names = ['logits']
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'seq_len'},
        'logits': {0: 'batch', 1: 'seq_len'},
    }
    

    torch.onnx.export(
        gpt_wrapper,
        dummpy_input,
        'gpt.onnx',
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    import onnxruntime as ort
    import numpy as np

    # PyTorch 原始版本的輸出（用同一組 example_input）
    with torch.no_grad():
        torch_out = clip_wrapper(example_input)

    # 用 ONNX Runtime 載入剛存好的檔案、跑同一組輸入
    session = ort.InferenceSession("clip_vision.onnx")
    onnx_out = session.run(
        None,  # 不指定要哪個輸出，全部回傳
        {"pixel_values": example_input.detach().cpu().numpy()}  # ONNX Runtime 吃 numpy,不是 torch tensor
    )[0]

    diff = np.abs(torch_out.detach().cpu().numpy() - onnx_out).max()
    cos = np.dot(torch_out.detach().cpu().numpy().flatten(), onnx_out.flatten()) / (
        np.linalg.norm(torch_out.detach().cpu().numpy()) * np.linalg.norm(onnx_out)
    )
    print(f"max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")

    with torch.no_grad():
        torch_out = gpt_wrapper(torch.tensor([[318]]), clip_wrapper(example_input))

    session = ort.InferenceSession('gpt.onnx')
    onnx_out = session.run(
        None,  # 不指定要哪個輸出，全部回傳
        {"input_ids": torch.tensor([[318]]).cpu().numpy(),
         "img_feat": clip_wrapper(example_input).detach().cpu().numpy()}  # ONNX Runtime 吃 numpy,不是 torch tensor
    )[0]

    diff = np.abs(torch_out.detach().cpu().numpy() - onnx_out).max()
    cos = np.dot(torch_out.detach().cpu().numpy().flatten(), onnx_out.flatten()) / (
        np.linalg.norm(torch_out.detach().cpu().numpy()) * np.linalg.norm(onnx_out)
    )
    print(f"max abs diff = {diff:.6e}, cosine sim = {cos:.6f}")

    # tokenizer = minbpe.load('tokenizer.pkl')

    # def generate(img_feat, max_new_tokens=40):

    #     image_token_id = 318
    #     eos_token_id = 319
    #     idx = torch.tensor([[image_token_id]], dtype=torch.long, device=device)
    #     for i in range(max_new_tokens):

    #         logits = g(idx,  img_feat.to(device))

    #         prob = F.softmax(logits[:,-1], dim=-1)
    #         idx_next = torch.multinomial(prob, num_samples=1)

    #         idx = torch.cat((idx, idx_next), dim=-1)

    #         if idx_next.item() == eos_token_id:
    #             break

    #     generated_ids = idx[0].tolist()
    #     clean_ids = [i for i in generated_ids if i not in (image_token_id, eos_token_id)]
    #     captions = tokenizer.decode([clean_ids])
    #     return captions

    # print(generate(c(inputs['pixel_values'])))

if __name__ == '__main__':
    export()