# 1. 只有問題(不含答案)的版本,用來測量 prompt 佔幾個 token
import torch
import random
from qwen_vl_utils import process_vision_info

def tokenization(dataset, processor):

    samples = []
    for idx in range(len(dataset)):
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dataset[idx]["image"]},
                    {"type": "text", "text": dataset[idx]["query"]},
                ],
            },
        ]
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # 2. 完整版本:問題 + 答案都放進去
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dataset[idx]["image"]},
                    {"type": "text", "text": dataset[idx]["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": dataset[idx]["label"][0]},
                ],
            },
        ]
        full_text = processor.apply_chat_template(full_messages, tokenize=False)


        # 3. 分別轉成 tensor,量出 prompt 部分實際佔幾個 token
        image_inputs, _ = process_vision_info(full_messages)
        prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
        full_inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt")

        print([key for key in prompt_inputs.keys()])

        prompt_len = prompt_inputs.input_ids.shape[1]

        # 4. labels 複製一份 input_ids,把 prompt 那段蓋成 -100
        labels = full_inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        samples.append({
            "input_ids": full_inputs.input_ids[0],
            "labels": labels[0],
            "pixel_values": full_inputs.pixel_values,
            "image_grid_thw": full_inputs.image_grid_thw,
            "mm_token_type_ids": full_inputs.mm_token_type_ids[0],   # 補這個
        })

    return samples

def collate_fn(batch, pad_token_id):
    max_len = 0

    for d in batch:
        assert len(d['input_ids']) == len(d['labels']), f'請確保ids和labels長度一致'
        max_len = len(d['input_ids']) if len(d['input_ids']) > max_len else max_len

    padded_batch = []
    for d in batch:
        d = dict(d) 
        d['attention_mask'] = torch.ones_like(d['input_ids'])
        padding_length = max_len - len(d['input_ids'])
        d['input_ids'] = torch.cat((d['input_ids'], torch.tensor([pad_token_id for _ in range(padding_length)], dtype=d['input_ids'].dtype)))
        d['labels'] = torch.cat((d['labels'], torch.tensor([-100 for _ in range(padding_length)], dtype=d['labels'].dtype)))
        d['attention_mask'] = torch.cat((d['attention_mask'], torch.tensor([0 for _ in range(padding_length)], dtype=d['attention_mask'].dtype)))
        d['mm_token_type_ids'] = torch.cat((
            d['mm_token_type_ids'],
            torch.tensor([0 for _ in range(padding_length)], dtype=d['mm_token_type_ids'].dtype)
        ))
        padded_batch.append(d)

    batch = {
        'input_ids': torch.stack([d['input_ids'] for d in padded_batch]),
        'labels': torch.stack([d['labels'] for d in padded_batch]),
        'attention_mask': torch.stack([d['attention_mask'] for d in padded_batch]),
        'pixel_values': torch.cat([d['pixel_values'] for d in padded_batch], dim=0),
        'image_grid_thw': torch.cat([d['image_grid_thw'] for d in padded_batch]),
        'mm_token_type_ids': torch.stack([d['mm_token_type_ids'] for d in padded_batch])
    }

    return batch

def evaluate(peft_model, val_inputs, pad_token_id, device, batch_size=4):
    peft_model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_inputs), batch_size):
            val_batch = val_inputs[i:i+batch_size]
            inputs = collate_fn(val_batch, pad_token_id)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = peft_model(**inputs)
            losses.append(outputs.loss.item())
    peft_model.train()
    return sum(losses) / len(losses)


def get_batch(dataset, batch_size):

    idx = [random.randrange(len(dataset)) for _ in range(batch_size)]
    batch = [dataset[i] for i in idx]

    return batch