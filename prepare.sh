#!/bin/bash
# Download Task07_Pancreas từ AWS S3 (Medical Segmentation Decathlon)
# Hỗ trợ resume nếu bị gián đoạn

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AWS_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
TAR_FILE="Task07_Pancreas.tar"

# Tự detect đang chạy ở đâu
if [ -w "/workspace" ]; then
    DATASET_ROOT="/workspace/dataset"
elif [ -d "/mnt/d/DATN" ]; then
    DATASET_ROOT="/mnt/d/DATN/dataset"
else
    DATASET_ROOT="$SCRIPT_DIR/dataset"
fi

echo "=== Dataset sẽ được lưu vào: $DATASET_ROOT ==="
mkdir -p "$DATASET_ROOT"

# ── Download ──────────────────────────────────────────────────────────────────
TAR_PATH="$DATASET_ROOT/$TAR_FILE"

if [ -d "$DATASET_ROOT/Task_7/imagesTr" ]; then
    echo "=== Dataset đã tồn tại, bỏ qua download ==="
else
    echo "=== Tải từ AWS S3... ==="
    if command -v wget &>/dev/null; then
        # -c: tiếp tục nếu bị ngắt giữa chừng
        wget -c "$AWS_URL" -O "$TAR_PATH"
    else
        # curl fallback — dùng -C - để resume
        curl -L -C - "$AWS_URL" -o "$TAR_PATH"
    fi

    # ── Giải nén ─────────────────────────────────────────────────────────────
    echo "=== Giải nén $TAR_FILE... ==="
    tar -xf "$TAR_PATH" -C "$DATASET_ROOT/"

    # Dataset giải nén ra thư mục Task07_Pancreas → đổi tên thành Task_7
    if [ -d "$DATASET_ROOT/Task07_Pancreas" ]; then
        mv "$DATASET_ROOT/Task07_Pancreas" "$DATASET_ROOT/Task_7"
        echo "=== Đổi tên Task07_Pancreas → Task_7 ==="
    fi

    # Xoá file tar sau khi giải nén thành công
    rm -f "$TAR_PATH"
    echo "=== Giải nén xong ==="
fi

# ── Kiểm tra ──────────────────────────────────────────────────────────────────
n_images=$(ls "$DATASET_ROOT/Task_7/imagesTr/"*.nii.gz 2>/dev/null | wc -l)
n_labels=$(ls "$DATASET_ROOT/Task_7/labelsTr/"*.nii.gz 2>/dev/null | wc -l)
n_test=$(ls "$DATASET_ROOT/Task_7/imagesTest/"*.nii.gz 2>/dev/null | wc -l)
echo "=== Images: $n_images | Labels: $n_labels | Test: $n_test ==="

if [ "$n_images" -lt 10 ]; then
    echo "=== CẢNH BÁO: Số file ít hơn dự kiến — kiểm tra lại dataset ==="
fi

# ── Cài dependencies ──────────────────────────────────────────────────────────
echo "=== Cài dependencies ==="
pip install -q monai[all] nibabel scipy scikit-learn tqdm pyyaml matplotlib

# ── Cập nhật DATA_DIR trong code ──────────────────────────────────────────────
sed -i "s|DATA_DIR = \".*\"|DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/train_models.py
sed -i "s|DATA_DIR = \".*\"|DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/data/transforms.py
sed -i "s|DEFAULT_DATA_DIR = \".*\"|DEFAULT_DATA_DIR = \"$DATASET_ROOT/Task_7\"|g" monai_benchmark/evaluate.py

echo ""
echo "=== Xong! DATA_DIR = $DATASET_ROOT/Task_7 ==="
echo "    python monai_benchmark/train_models.py"
