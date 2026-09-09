"""
KV cache實驗共用的loader跟兩版generate()。

跟Phase3/Day36的既有腳本共用同一顆checkpoint(best_model_full_capfix_reweight2x.pt)
跟同樣的CLIP前處理方式,只是import的GPT改成本目錄下修過bug的GPT_kv_cache.py
(原因見該檔案docstring),不動Day36/GPT.py本身。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from GPT_kv_cache import GPT, GPTConfig  # noqa: E402

# checkpoint裡的GPTConfig是用__main__.GPTConfig pickle的(訓練腳本以腳本模式跑),
# 跟Day36其他載入腳本一樣的workaround。
import __main__  # noqa: E402
__main__.GPTConfig = GPTConfig

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
CKPT_PATH = DAY32 / "checkpoints" / "best_model_full_capfix_reweight2x.pt"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_thermal_val"

IMAGE_TOKEN_ID = 318
EOS_TOKEN_ID = 319

TEST_IMAGES = [
    "data/video-JhYLiFCieHQHaY8o7-frame-000000-xT7BXRKKyuWEsnywX.jpg",
    "data/video-JhYLiFCieHQHaY8o7-frame-000600-54cr88GJsdAyksdNj.jpg",
    "data/video-JhYLiFCieHQHaY8o7-frame-000900-pdb96S7B7fgmguxPE.jpg",
]


def load_gpt_model(device="cpu"):
    model = GPT(GPTConfig(
        block_size=1024, vocab_size=320, clip_vector_size=512, n_layer=12, n_head=12, n_embd=768,
    )).to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def load_clip(device="cpu"):
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return clip_model, processor


@torch.no_grad()
def get_image_feature(clip_model, processor, image_path, device="cpu"):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)
    vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    pooled = vision_out.pooler_output
    return clip_model.visual_projection(pooled)


@torch.no_grad()
def generate_no_cache(gpt_model, img_feat, max_new_tokens=40, mode="greedy",
                       device="cpu", force_full_length=False):
    """現行版本:每一步都把目前為止的完整序列重新餵一次(past_key_value恆為None)。"""
    idx = torch.tensor([[IMAGE_TOKEN_ID]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = gpt_model(idx, None, img_feat.to(device))
        next_logits = logits[:, -1]
        idx_next = _pick_next(next_logits, mode)
        idx = torch.cat((idx, idx_next), dim=-1)
        if not force_full_length and idx_next.item() == EOS_TOKEN_ID:
            break
    return idx[0].tolist()


@torch.no_grad()
def generate_with_cache(gpt_model, img_feat, max_new_tokens=40, mode="greedy",
                         device="cpu", force_full_length=False):
    """實驗版本:prefill跑一次完整(目前只有image token)序列拿到cache,
    之後每一步只餵新產生的1個token,past_key_value/past_length帶著累積的k/v跟位置。"""
    idx_full = torch.tensor([[IMAGE_TOKEN_ID]], dtype=torch.long, device=device)
    logits, _, past_kv = gpt_model(idx_full, None, img_feat.to(device), use_cache=True)
    next_logits = logits[:, -1]
    generated = [IMAGE_TOKEN_ID]
    past_length = idx_full.size(1)

    for _ in range(max_new_tokens):
        idx_next = _pick_next(next_logits, mode)
        token_id = idx_next.item()
        generated.append(token_id)
        if not force_full_length and token_id == EOS_TOKEN_ID:
            break
        logits, _, past_kv = gpt_model(
            idx_next, None, img_feat.to(device),
            past_key_value=past_kv, use_cache=True, past_length=past_length,
        )
        next_logits = logits[:, -1]
        past_length += 1

    return generated


def _pick_next(next_logits, mode):
    if mode == "greedy":
        return torch.argmax(next_logits, dim=-1, keepdim=True)
    prob = F.softmax(next_logits, dim=-1)
    return torch.multinomial(prob, num_samples=1)
