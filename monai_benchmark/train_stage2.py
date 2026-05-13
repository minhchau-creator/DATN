#!/usr/bin/env python3
"""
Stage-2 Training — fine-grained pancreas/tumor segmentation on cropped ROI.

Workflow:
  - Dùng GT label để crop quanh vùng pancreas+tumor (transforms_stage2.py)
  - Train nnUNet với class weights [0.05, 0.35, 0.60] — ưu tiên tumor
  - Lưu checkpoint vào checkpoints/nnunet_stage2/fold_{i}/

Usage:
    python monai_benchmark/train_stage2.py
    python monai_benchmark/train_stage2.py --folds 0 --epochs 50
"""

import sys, json, logging, signal, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monai_benchmark.data.transforms import build_data_list
from monai_benchmark.data.transforms_stage2 import get_train_transforms_stage2, get_val_transforms_stage2
from monai_benchmark.models.builder import get_model
from monai_benchmark.models.losses import DiceCELoss, DeepSupervisionLoss
from monai_benchmark.models.eval_metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_stop_requested = False

def _handle_sigterm(*_):
    global _stop_requested
    _stop_requested = True
    logger.warning("SIGTERM — lưu checkpoint và dừng sau epoch này...")

signal.signal(signal.SIGTERM, _handle_sigterm)

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL      = "nnunet"
DEVICE     = "cuda:0"
DATA_DIR   = str(REPO_ROOT / "dataset" / "Task_7")
SPLITS_DIR = Path(__file__).resolve().parent / "splits"


def make_loader(file_names, is_train):
    data_list = build_data_list(file_names, data_dir=DATA_DIR)
    tfm = get_train_transforms_stage2() if is_train else get_val_transforms_stage2()
    ds  = CacheDataset(data=data_list, transform=tfm, cache_rate=0.1, num_workers=4)
    return DataLoader(ds, batch_size=1,
                      shuffle=is_train, num_workers=4,
                      pin_memory=True, collate_fn=list_data_collate)


def _save_resume(path, fold_idx, epoch, model, optimizer, scheduler, scaler, best_dice, history):
    torch.save({
        "fold": fold_idx, "epoch": epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_dice":       best_dice,
        "history":         history,
    }, path)


def _save_plot(history, fold_idx, ckpt_dir, total_epochs):
    epochs   = [h["epoch"]         for h in history]
    losses   = [h["loss"]          for h in history]
    pan_dice = [h["pancreas_dice"] for h in history]
    tum_dice = [h["tumor_dice"]    for h in history]
    avg_dice = [h["avg_dice"]      for h in history]
    pan_iou  = [h["pancreas_iou"]  for h in history]
    tum_iou  = [h["tumor_iou"]     for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Stage-2 Fold {fold_idx} — Training Progress", fontsize=13)

    axes[0].plot(epochs, losses, "b-o", markersize=3)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, pan_dice, "g-o", markersize=3, label="Pancreas")
    axes[1].plot(epochs, tum_dice, "r-o", markersize=3, label="PDAC")
    axes[1].plot(epochs, avg_dice, "k--o", markersize=3, label="Average")
    axes[1].set_title("Dice Score"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, pan_iou, "g-o", markersize=3, label="Pancreas")
    axes[2].plot(epochs, tum_iou, "r-o", markersize=3, label="PDAC")
    axes[2].set_title("IoU Score"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = ckpt_dir / f"training_curve_stage2_fold{fold_idx}.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    logger.info(f"  Plot saved → {plot_path}")


def train_fold(fold_idx, total_epochs):
    global _stop_requested
    device = torch.device(DEVICE)

    fold_data   = json.loads((SPLITS_DIR / f"fold_{fold_idx}.json").read_text())
    train_names = fold_data["train"]
    val_names   = fold_data["val"]
    logger.info(f"\n{'='*60}\n  Stage-2  Fold {fold_idx}  |  train={len(train_names)}  val={len(val_names)}\n{'='*60}")

    model = get_model(MODEL, pretrained=False, device=device)
    train_loader = make_loader(train_names, is_train=True)
    val_loader   = make_loader(val_names,   is_train=False)

    # Tumor class weight tăng lên 0.60 so với 0.40 ở stage 1
    criterion = DeepSupervisionLoss(DiceCELoss(class_weights=[0.05, 0.35, 0.60]))
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler    = GradScaler()

    ckpt_dir = REPO_ROOT / f"checkpoints/nnunet_stage2/fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results").mkdir(exist_ok=True)

    resume_path = ckpt_dir / "resume.pth"
    start_epoch, best_dice, history = 1, 0.0, []

    if resume_path.exists():
        r = torch.load(resume_path, map_location=device)
        model.load_state_dict(r["model_state"])
        optimizer.load_state_dict(r["optimizer_state"])
        scheduler.load_state_dict(r["scheduler_state"])
        scaler.load_state_dict(r["scaler_state"])
        start_epoch = r["epoch"] + 1
        best_dice   = r["best_dice"]
        history     = r.get("history", [])
        logger.info(f"Resume stage-2 fold {fold_idx} từ epoch {r['epoch']}  best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, total_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"S2-E{epoch} train", leave=False, dynamic_ncols=True):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            torch.cuda.empty_cache()
            model.eval()
            all_m = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"S2-E{epoch} val", leave=False, dynamic_ncols=True):
                    images = batch["image"].to(device)
                    labels = batch["label"]

                    def _fwd(x):
                        o = model(x)
                        return o[0] if isinstance(o, (list, tuple)) else o

                    logits = sliding_window_inference(images, [96, 96, 96], 2, _fwd, overlap=0.25)
                    pred   = torch.argmax(logits, 1).squeeze(0).cpu().numpy()
                    target = labels.squeeze().numpy()
                    for k, v in calculate_metrics(pred, target).items():
                        all_m[k].append(v)

            avg_m    = {k: float(np.mean(v)) for k, v in all_m.items()}
            avg_dice = avg_m.get("avg_dice", 0.0)
            pan_dice = avg_m.get("pancreas_dice", 0.0)
            tum_dice = avg_m.get("tumor_dice", 0.0)

            marker = "  ✓ BEST" if avg_dice > best_dice else ""
            logger.info(
                f"Epoch {epoch:3d}/{total_epochs} | loss={avg_loss:.4f} | "
                f"avg_dice={avg_dice:.4f}  pan={pan_dice:.3f}  tum={tum_dice:.3f}{marker}"
            )

            history.append({"epoch": epoch, "loss": avg_loss, **avg_m})
            _save_plot(history, fold_idx, ckpt_dir, total_epochs)

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save({"model_state": model.state_dict(), "metrics": avg_m,
                            "fold": fold_idx}, ckpt_dir / "best_model.pth")
                (REPO_ROOT / f"results/nnunet_stage2_fold{fold_idx}_metrics.json").write_text(
                    json.dumps(avg_m, indent=2))
        else:
            logger.info(f"Epoch {epoch:3d}/{total_epochs} | loss={avg_loss:.4f}")

        # ── Resume checkpoint mỗi 10 epoch ───────────────────────────────────
        if epoch % 10 == 0:
            _save_resume(resume_path, fold_idx, epoch,
                         model, optimizer, scheduler, scaler, best_dice, history)

        if _stop_requested:
            logger.warning(f"Dừng theo yêu cầu tại epoch {epoch}")
            _save_resume(resume_path, fold_idx, epoch,
                         model, optimizer, scheduler, scaler, best_dice, history)
            _stop_requested = False
            return best_dice

    logger.info(f"Stage-2 Fold {fold_idx} done — best dice: {best_dice:.4f}")
    return best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    logger.info(f"Stage-2 training  folds={args.folds}  epochs={args.epochs}")
    for fold in args.folds:
        train_fold(fold, args.epochs)
        torch.cuda.empty_cache()
    logger.info("Stage-2 training hoàn thành.")


if __name__ == "__main__":
    main()
