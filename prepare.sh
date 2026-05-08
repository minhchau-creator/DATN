#!/bin/bash
# Download dataset từ Google Drive
# Dùng được cả trên vast.ai server lẫn máy local

set -e

# Tự detect đang chạy ở đâu
if [ -w "/workspace" ]; then
    DATASET_ROOT="/workspace/dataset"
else
    DATASET_ROOT="$(pwd)/dataset"
fi

echo "=== Dataset sẽ được lưu vào: $DATASET_ROOT ==="
mkdir -p "$DATASET_ROOT"

echo "=== Cài gdown ==="
pip install -q gdown

echo "=== Tải dataset từ Google Drive ==="
gdown --folder "1QJXC5EpH-ww5d2cyc8VzYtrRkBOX6anQ" \
      --output "$DATASET_ROOT/" \
      --remaining-ok

echo "=== Cài dependencies ==="
pip install -q monai[all] nibabel scipy scikit-learn tqdm pyyaml matplotlib

echo ""
echo "=== Kiểm tra dataset ==="
echo "Images: $(ls $DATASET_ROOT/Task_7/imagesTr/*.nii.gz 2>/dev/null | wc -l) files"
echo "Labels: $(ls $DATASET_ROOT/Task_7/labelsTr/*.nii.gz 2>/dev/null | wc -l) files"

# Cập nhật DATA_DIR trong code theo đúng path vừa download
sed -i "s|DATA_DIR = \".*\"|DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/train_models.py
sed -i "s|DATA_DIR = \".*\"|DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/data/transforms.py
sed -i "s|DEFAULT_DATA_DIR = \".*\"|DEFAULT_DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/evaluate.py

echo ""
echo "=== Xong! Chạy train: ==="
echo "    python monai_benchmark/train_models.py"
