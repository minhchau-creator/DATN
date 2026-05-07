"""
Outlier analysis for Task 7 Pancreas CT dataset.

Output: reports/outlier_lof.png
Cache:  reports/features_cache.csv  (re-used on next run)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
TASK7_IMG_DIR = Path("/mnt/d/DATN/dataset/Task_7/imagesTr")
TASK7_LBL_DIR = Path("/mnt/d/DATN/dataset/Task_7/labelsTr")
REPORTS_DIR   = Path("/home/minhc/DATN/reports")
CACHE_CSV     = REPORTS_DIR / "features_cache.csv"
OUTPUT_PNG    = REPORTS_DIR / "outlier_lof.png"
REPORTS_DIR.mkdir(exist_ok=True)

TOP_N = 10


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img_path: Path) -> dict:
    img     = nib.load(str(img_path))
    data    = img.get_fdata(dtype=np.float32)
    spacing = np.abs(np.diag(img.affine)[:3])

    flat = data.ravel()
    return {
        "filename":    img_path.stem,
        "n_slices":    int(data.shape[2]),
        "xy_spacing":  float(spacing[0]),
        "z_spacing":   float(spacing[2]),
        "mean_global": float(flat.mean()),
        "std_global":  float(flat.std()),
        "min_global":  float(flat.min()),
        "max_global":  float(flat.max()),
    }


def load_features() -> pd.DataFrame:
    img_files = sorted(TASK7_IMG_DIR.glob("*.nii.gz"))

    if CACHE_CSV.exists():
        df = pd.read_csv(CACHE_CSV)
        # Chỉ cần các cột mới dùng — nếu cache cũ (nhiều cột hơn) vẫn dùng được
        needed = {"filename", "n_slices", "mean_global"}
        if needed.issubset(set(df.columns)) and len(df) == len(img_files):
            print(f"Loaded {len(df)} cached features from {CACHE_CSV}")
            return df
        print("Cache mismatch — re-extracting...")

    print(f"Extracting features from {len(img_files)} volumes...")
    records = []
    for i, img_path in enumerate(img_files):
        records.append(extract_features(img_path))
        print(f"  [{i+1}/{len(img_files)}] {img_path.name}", end="\r")

    print()
    df = pd.DataFrame(records)
    df.to_csv(CACHE_CSV, index=False)
    print(f"Cached → {CACHE_CSV}")
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_top_bottom(ax, names, values, mean_val, title, xlabel,
                    color_high="crimson", color_low="steelblue", color_rest="lightgray"):
    """
    Vẽ sorted bar chart:
      - TOP_N cao nhất  → màu đỏ
      - TOP_N thấp nhất → màu xanh
      - Còn lại         → xám
    Đường kẻ đứt = mean toàn dataset
    """
    order  = np.argsort(values)          # tăng dần
    s_vals = values[order]
    s_names = [names[i] for i in order]
    n = len(s_vals)

    colors = [color_rest] * n
    for i in range(TOP_N):              # thấp nhất
        colors[i] = color_low
    for i in range(n - TOP_N, n):      # cao nhất
        colors[i] = color_high

    y_pos = np.arange(n)
    ax.barh(y_pos, s_vals, color=colors, edgecolor="none", height=0.8)
    ax.axvline(mean_val, color="black", lw=1.5, ls="--", label=f"mean = {mean_val:.1f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, fontsize=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)

    # Chú thích 2 đầu
    for i in range(TOP_N):
        ax.text(s_vals[i] + abs(s_vals[i]) * 0.01, y_pos[i],
                f"{s_vals[i]:.1f}", va="center", fontsize=5.5, color=color_low)
    for i in range(n - TOP_N, n):
        ax.text(s_vals[i] + abs(s_vals[i]) * 0.01, y_pos[i],
                f"{s_vals[i]:.1f}", va="center", fontsize=5.5, color=color_high)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df    = load_features()
    names = df["filename"].tolist()

    n_slices   = df["n_slices"].values.astype(float)
    mean_hu    = df["mean_global"].values

    mean_slices = float(n_slices.mean())
    mean_hu_val = float(mean_hu.mean())

    # ── Figure: 2 panels ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(10, len(df) * 0.07)))
    fig.suptitle("Task 7 Pancreas CT — Volume Analysis", fontsize=14, fontweight="bold")

    plot_top_bottom(
        ax1, names, n_slices, mean_slices,
        title=f"Number of Slices  (mean={mean_slices:.1f})",
        xlabel="n_slices",
    )

    plot_top_bottom(
        ax2, names, mean_hu, mean_hu_val,
        title=f"Mean HU  (mean={mean_hu_val:.1f})",
        xlabel="mean HU (Hounsfield Units)",
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {OUTPUT_PNG}")

    # ── Console report ────────────────────────────────────────────────────────
    SEP = "=" * 55

    # n_slices
    print(f"\n{SEP}")
    print(f"  N_SLICES  —  mean = {mean_slices:.1f}")
    print(SEP)
    order = np.argsort(n_slices)
    print(f"\n  Top {TOP_N} ÍT SLICE NHẤT:")
    for rank, idx in enumerate(order[:TOP_N], 1):
        print(f"  {rank:2d}. {names[idx]:<28s}  {int(n_slices[idx])} slices")
    print(f"\n  Top {TOP_N} NHIỀU SLICE NHẤT:")
    for rank, idx in enumerate(order[-TOP_N:][::-1], 1):
        print(f"  {rank:2d}. {names[idx]:<28s}  {int(n_slices[idx])} slices")

    # mean HU
    print(f"\n{SEP}")
    print(f"  MEAN HU  —  mean = {mean_hu_val:.1f} HU")
    print(SEP)
    order_hu = np.argsort(mean_hu)
    print(f"\n  Top {TOP_N} HU THẤP NHẤT:")
    for rank, idx in enumerate(order_hu[:TOP_N], 1):
        print(f"  {rank:2d}. {names[idx]:<28s}  {mean_hu[idx]:.1f} HU")
    print(f"\n  Top {TOP_N} HU CAO NHẤT:")
    for rank, idx in enumerate(order_hu[-TOP_N:][::-1], 1):
        print(f"  {rank:2d}. {names[idx]:<28s}  {mean_hu[idx]:.1f} HU")

    print(f"\n{SEP}")
    print(f"  Done. Chart: {OUTPUT_PNG}")
    print(SEP)


if __name__ == "__main__":
    main()
