#!/bin/bash
# Download dataset từ Google Drive về server
# Chạy 1 lần trước khi train: bash prepare.sh

set -e

echo "=== Cài gdown ==="
pip install -q gdown

echo "=== Tải dataset từ Google Drive ==="
mkdir -p /workspace/dataset

gdown --folder "1QJXC5EpH-ww5d2cyc8VzYtrRkBOX6anQ" \
      --output /workspace/dataset/ \
      --remaining-ok

echo "=== Cài dependencies ==="
pip install -q monai[all] nibabel scipy scikit-learn tqdm pyyaml matplotlib

echo ""
echo "=== Kiểm tra dataset ==="
echo "Images: $(ls /workspace/dataset/Task_7/imagesTr/*.nii.gz 2>/dev/null | wc -l) files"
echo "Labels: $(ls /workspace/dataset/Task_7/labelsTr/*.nii.gz 2>/dev/null | wc -l) files"
echo ""
echo "=== Xong! Chạy train: ==="
echo "    python monai_benchmark/train_models.py"
