# test_data_utils.py
import torch
import copy
from data_utils import collate_fn, get_batch


def make_fake_sample(seq_len, pad_value=0):
    return {
        "input_ids": torch.arange(seq_len),
        "labels": torch.arange(seq_len),
        "pixel_values": torch.randn(6, 4),
        "image_grid_thw": torch.tensor([[1, 2, 3]]),
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
    }


def test_collate_fn_does_not_mutate_original_samples():
    samples = [make_fake_sample(5), make_fake_sample(3)]
    original_lens = [len(d['input_ids']) for d in samples]

    collate_fn(samples, pad_token_id=999)

    new_lens = [len(d['input_ids']) for d in samples]
    assert original_lens == new_lens


def test_collate_fn_pads_to_longest_in_batch():
    samples = [make_fake_sample(5), make_fake_sample(3)]
    batch = collate_fn(samples, pad_token_id=999)
    assert batch['input_ids'].shape[1] == 5
    assert batch['labels'].shape[1] == 5
    assert batch['attention_mask'].shape[1] == 5


def test_collate_fn_attention_mask_marks_padding_correctly():
    samples = [make_fake_sample(5), make_fake_sample(3)]
    batch = collate_fn(samples, pad_token_id=999)
    second_sample_mask = batch['attention_mask'][1]
    assert second_sample_mask.tolist() == [1, 1, 1, 0, 0]


def test_collate_fn_labels_padding_is_minus_100():
    samples = [make_fake_sample(5), make_fake_sample(3)]
    batch = collate_fn(samples, pad_token_id=999)
    second_sample_labels = batch['labels'][1]
    assert second_sample_labels.tolist()[-2:] == [-100, -100]


def test_get_batch_returns_correct_size():
    dataset = [make_fake_sample(5) for _ in range(10)]
    batch = get_batch(dataset, batch_size=4)
    assert len(batch) == 4


def test_get_batch_indices_never_out_of_range():
    dataset = [make_fake_sample(5) for _ in range(10)]
    for _ in range(1000):
        batch = get_batch(dataset, batch_size=4)
        assert len(batch) == 4 