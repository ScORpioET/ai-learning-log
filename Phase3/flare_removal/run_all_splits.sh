#!/bin/bash
set -e
cd /home/jack/ai-transition-2026/Phase3/flare_removal/Flare7K
export PYTHONPATH="/home/jack/ai-transition-2026/Phase3/caption_fusion/.pylibs:$PYTHONPATH"

TD=/home/jack/ai-transition-2026/thermal_dataset
CKPT=experiments/flare7kpp/net_g_last.pth

echo "=== [1/3] train ==="
python3 test_large.py --input "$TD/images_rgb_train/data" --output ../out_train --model_path "$CKPT" --flare7kpp

echo "=== [2/3] val ==="
python3 test_large.py --input "$TD/images_rgb_val/data" --output ../out_val --model_path "$CKPT" --flare7kpp

echo "=== [3/3] test ==="
python3 test_large.py --input "$TD/video_rgb_test/data" --output ../out_test --model_path "$CKPT" --flare7kpp

echo "[all splits done]"
