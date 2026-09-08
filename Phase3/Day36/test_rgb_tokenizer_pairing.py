"""
實測驗證:best_model_rgb_full_reweight2x.pt 該配哪份 tokenizer(跟
test_capfix_tokenizer_pairing.py同一套方法)。

候選:
1. tokenizer_rgb_full.pkl (Day32, 重建) -- 用同一輪訓練的captions_rgb_train_full.jsonl
   重建出來的候選(理論上應該對)
2. tokenizer_full_capfix.pkl (Day32) -- thermal capfix版,理論上不對(不同domain+不同輪)
3. tokenizer.pkl (Day32, exp2舊版) -- 理論上也不對
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
from GPT import GPT, GPTConfig  # noqa: E402
from minbpe import minbpe  # noqa: E402

DAY32 = Path.home() / "ai-transition-2026" / "Phase3" / "Day32"
CKPT_PATH = DAY32 / "checkpoints" / "best_model_rgb_full_reweight2x.pt"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_ROOT = Path.home() / "ai-transition-2026" / "thermal_dataset" / "images_rgb_val"

TEST_IMAGES = [
    "data/video-CgGWcFRZQwyk48xKp-frame-000150-YbQHWyoYefb5kTX7g.jpg",
    "data/video-CgGWcFRZQwyk48xKp-frame-000300-7NtCcLPKSNACFGNKx.jpg",
    "data/video-CgGWcFRZQwyk48xKp-frame-000450-3LahX4oD5ipb9zmv7.jpg",
]
GT_CAPTIONS = [
    "Three cars, the nearest ahead.",
    "Several cars, the nearest on the left.",
    "Three cars, one nearby ahead.",
]

CANDIDATES = {
    "tokenizer_rgb_full.pkl (重建, Day32)": DAY32 / "tokenizer_rgb_full.pkl",
    "tokenizer_full_capfix.pkl (thermal capfix版, Day32)": DAY32 / "tokenizer_full_capfix.pkl",
    "tokenizer.pkl (exp2舊版, Day32)": DAY32 / "tokenizer.pkl",
}

device = "cpu"


@torch.no_grad()
def generate(gpt_model, img_feat, tokenizer, image_token_id, eos_token_id, max_new_tokens=40):
    idx = torch.tensor([[image_token_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = gpt_model(idx, None, img_feat.to(device))
        prob = F.softmax(logits[:, -1], dim=-1)
        idx_next = torch.multinomial(prob, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=-1)
        if idx_next.item() == eos_token_id:
            break
    generated_ids = idx[0].tolist()
    clean_ids = [i for i in generated_ids if i not in (image_token_id, eos_token_id)]
    return tokenizer.decode([clean_ids])


def main():
    torch.manual_seed(1337)

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    gpt_model = GPT(GPTConfig(
        block_size=1024, vocab_size=320, clip_vector_size=512, n_layer=12, n_head=12, n_embd=768,
    )).to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    gpt_model.load_state_dict(checkpoint['model'])
    gpt_model.eval()
    print(f"[info] checkpoint loaded: {CKPT_PATH}  (epoch: {checkpoint.get('epoch', 'n/a')}, "
          f"val_loss: {checkpoint.get('val_loss', 'n/a')})")

    image_token_id, eos_token_id = 318, 319

    img_feats = []
    for fn in TEST_IMAGES:
        img = Image.open(IMAGE_ROOT / fn).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            feat = clip_model.visual_projection(pooled)
        img_feats.append(feat)

    for cand_name, cand_path in CANDIDATES.items():
        print(f"\n{'='*70}\n候選: {cand_name}  ({cand_path})\n{'='*70}")
        if not cand_path.exists():
            print("  [找不到檔案,跳過]")
            continue
        tokenizer = minbpe.load(str(cand_path))
        for fn, gt, feat in zip(TEST_IMAGES, GT_CAPTIONS, img_feats):
            torch.manual_seed(1337)
            caption = generate(gpt_model, feat, tokenizer, image_token_id, eos_token_id)
            print(f"  file: {fn}")
            print(f"    GT       : {gt}")
            print(f"    generated: {caption!r}")


if __name__ == "__main__":
    main()
