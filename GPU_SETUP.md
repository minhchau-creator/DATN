# GPU Setup Guide for WSL2 (RTX 3050 Laptop)

Your GPU: **NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM)**, CUDA 13.0, Driver 581.83.
WSL2 already supports CUDA natively — the Windows driver exposes the GPU to Linux.
You do **not** need to install a Linux GPU driver.

---

## Step 1 – Install Miniconda for Linux (inside WSL2)

Run these in your WSL2 terminal:

```bash
# Download the Linux Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install silently
bash ~/miniconda.sh -b -p ~/miniconda3

# Activate conda in this session
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Add to .bashrc so it persists
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## Step 2 – Create the `datn` Conda Environment

```bash
conda create -n datn python=3.11 -y
conda activate datn
```

---

## Step 3 – Install PyTorch with CUDA

For CUDA 12.x (compatible with your CUDA 13.0 runtime):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 4 – Install Project Dependencies

```bash
pip install nibabel numpy scipy matplotlib ipywidgets pydicom jupyter jupyterlab pandas tqdm scikit-image
```

---

## Step 5 – Verify GPU Access

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"
```

Expected output:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4 GB
```

---

## Step 6 – Launch Jupyter with the `datn` Kernel

```bash
conda activate datn
jupyter lab --no-browser --port=8888
```

Then open the URL shown in the terminal (e.g. `http://localhost:8888/lab?token=...`).
Select **kernel → datn** when opening `data_exploration.ipynb`.

---

## Notes on 4 GB VRAM

- **Batch size**: keep batch size = 1 or 2 for 3D volumes.
- **Patch-based training**: use 96×96×96 or 128×128×128 patches, not full volumes.
- **Mixed precision**: always use `torch.cuda.amp.autocast()` — halves memory usage.
- **Gradient checkpointing**: enable if the model still OOMs.
- **Monitor VRAM**: run `watch -n1 nvidia-smi` in a separate terminal while training.

---

## Quick Command Reference

```bash
# Activate env
conda activate datn

# Check GPU live
watch -n1 nvidia-smi

# Run training with GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# Run exploration notebook
jupyter lab /home/minhc/DATN/code/data_exploration.ipynb
```
