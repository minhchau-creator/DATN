#!/usr/bin/env python3
import sys, json, logging, signal
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
from monai_benchmark.data.transforms import build_data_list, get_train_transforms, get_val_transforms
from monai_benchmark.models.builder import get_model
from monai_benchmark.models.losses import DiceCELoss, DeepSupervisionLoss
from monai_benchmark.models.eval_metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Signal handler — dừng an toàn khi nhận SIGTERM ───────────────────────────
_stop_requested = False

def _handle_sigterm(*_):
    global _stop_requested
    _stop_requested = True
    logger.warning("⚠️  SIGTERM nhận được — sẽ lưu checkpoint và dừng sau epoch này...")

signal.signal(signal.SIGTERM, _handle_sigterm)
# ─────────────────────────────────────────────────────────────────────────────

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MODEL     = "nnunet"   # "nnunet" | "swin_unetr"
FOLDS     = [0, 1, 2, 3, 4]
DEVICE    = "cuda:0"
REPO_ROOT = Path(__file__).resolve().parent.parent   # .../DATN/
DATA_DIR  = str(REPO_ROOT / "dataset" / "Task_7")
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
#EARLY_STOP = 10
# ─────────────────────────────────────────────────────────────────────────────


def make_loader(file_names, is_train):
    data_list = build_data_list(file_names, data_dir=DATA_DIR)
    tfm = get_train_transforms() if is_train else get_val_transforms()
    ds  = CacheDataset(data=data_list, transform=tfm, cache_rate=0.1, num_workers=4)
    return DataLoader(ds, batch_size=1,
                      shuffle=is_train, num_workers=4,
                      pin_memory=True, collate_fn=list_data_collate)


def _save_resume_ckpt(path, fold_idx, epoch, model, optimizer, scheduler, scaler, best_dice):
    torch.save({
        "fold":            fold_idx,
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_dice":       best_dice,
    }, path)


def _print_metrics_table(epoch, total_epochs, avg_loss, avg_m, best_dice):
    """In bảng metrics rõ ràng sau mỗi lần validate."""
    pan_dice = avg_m.get("pancreas_dice", 0.0)
    tum_dice = avg_m.get("tumor_dice",    0.0)
    avg_dice = avg_m.get("avg_dice",      0.0)
    pan_iou  = avg_m.get("pancreas_iou",  0.0)
    tum_iou  = avg_m.get("tumor_iou",     0.0)
    avg_iou  = avg_m.get("avg_iou",       0.0)
    marker   = "  ✓ BEST" if avg_dice >= best_dice else ""

    print(
        f"\n  ┌─────────────────────────────────────────────────┐\n"
        f"  │  Epoch {epoch:3d}/{total_epochs}   Loss: {avg_loss:.4f}{marker:<10}│\n"
        f"  ├────────────┬──────────────┬──────────────────────┤\n"
        f"  │            │    Dice      │       IoU            │\n"
        f"  ├────────────┼──────────────┼──────────────────────┤\n"
        f"  │ Pancreas   │   {pan_dice:.4f}     │     {pan_iou:.4f}           │\n"
        f"  │ PDAC       │   {tum_dice:.4f}     │     {tum_iou:.4f}           │\n"
        f"  │ Average    │   {avg_dice:.4f}     │     {avg_iou:.4f}           │\n"
        f"  └────────────┴──────────────┴──────────────────────┘"
    )


def _save_plot(history, fold_idx, ckpt_dir, total_epochs):
    """Lưu ảnh loss + dice + iou theo epoch."""
    epochs   = [h["epoch"]        for h in history]
    losses   = [h["loss"]         for h in history]
    pan_dice = [h["pancreas_dice"] for h in history]
    tum_dice = [h["tumor_dice"]    for h in history]
    avg_dice = [h["avg_dice"]      for h in history]
    pan_iou  = [h["pancreas_iou"]  for h in history]
    tum_iou  = [h["tumor_iou"]     for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Fold {fold_idx} — Training Progress", fontsize=13)

    axes[0].plot(epochs, losses, "b-o", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, pan_dice, "g-o", markersize=3, label="Pancreas")
    axes[1].plot(epochs, tum_dice, "r-o", markersize=3, label="PDAC")
    axes[1].plot(epochs, avg_dice, "k--o", markersize=3, label="Average")
    axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, pan_iou, "g-o", markersize=3, label="Pancreas")
    axes[2].plot(epochs, tum_iou, "r-o", markersize=3, label="PDAC")
    axes[2].set_title("IoU Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = ckpt_dir / f"training_curve_fold{fold_idx}.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    logger.info(f"  Plot saved → {plot_path}")


def train_fold(fold_idx):
    global _stop_requested
    device = torch.device(DEVICE)

    fold_file = SPLITS_DIR / f"fold_{fold_idx}.json"
    fold_data  = json.loads(fold_file.read_text())
    train_names, val_names = fold_data["train"], fold_data["val"]
    logger.info(f"\n{'='*55}\n  Fold {fold_idx}  |  train={len(train_names)}  val={len(val_names)}\n{'='*55}")

    model = get_model(MODEL, pretrained=(MODEL != "nnunet"), device=device)
    train_loader = make_loader(train_names, is_train=True)
    val_loader   = make_loader(val_names,   is_train=False)

    criterion = DeepSupervisionLoss(DiceCELoss(class_weights=[0.1, 0.5, 0.4]))
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler    = GradScaler()

    ckpt_dir = REPO_ROOT / f"checkpoints/{MODEL}/fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results").mkdir(exist_ok=True)

    # ── Resume từ checkpoint nếu có ──────────────────────────────────────────
    resume_path = ckpt_dir / "resume.pth"
    start_epoch, best_dice = 1, 0.0
    history = []   # lưu metrics để vẽ plot
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        history     = ckpt.get("history", [])
        logger.info(f"▶ Resume fold {fold_idx} từ epoch {ckpt['epoch']} "
                    f"(best_dice={best_dice:.4f})")
    # ─────────────────────────────────────────────────────────────────────────

    total_epochs = 50

    for epoch in range(start_epoch, total_epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{epoch} train", leave=False, dynamic_ncols=True):
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

        # ── validate ──
        if epoch % 5 == 0 or epoch == 1:
            torch.cuda.empty_cache()
            model.eval()
            all_m = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"E{epoch} val", leave=False, dynamic_ncols=True):
                    images = batch["image"].to(device)
                    labels = batch["label"]

                    def _fwd(x):
                        o = model(x)
                        return o[0] if isinstance(o, (list, tuple)) else o

                    logits = sliding_window_inference(images, [96,96,96], 2, _fwd, overlap=0.25)
                    pred   = torch.argmax(logits, 1).squeeze(0).cpu().numpy()
                    target = labels.squeeze().numpy()
                    for k, v in calculate_metrics(pred, target).items():
                        all_m[k].append(v)

            avg_m    = {k: float(np.mean(v)) for k, v in all_m.items()}
            avg_dice = avg_m.get("avg_dice", 0.0)

            # In bảng metrics
            _print_metrics_table(epoch, total_epochs, avg_loss, avg_m, best_dice)

            # Lưu vào history và vẽ plot
            history.append({"epoch": epoch, "loss": avg_loss, **avg_m})
            _save_plot(history, fold_idx, ckpt_dir, total_epochs)

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save({"model_state": model.state_dict(), "metrics": avg_m,
                            "fold": fold_idx}, ckpt_dir / "best_model.pth")
                (REPO_ROOT / f"results/{MODEL}_fold{fold_idx}_metrics.json").write_text(
                    json.dumps(avg_m, indent=2))
        else:
            logger.info(f"Epoch {epoch:3d}/{total_epochs} | loss={avg_loss:.4f}")

        # ── Lưu resume checkpoint mỗi 10 epoch ──────────────────────────────
        if epoch % 10 == 0:
            _save_resume_ckpt(resume_path, fold_idx, epoch,
                              model, optimizer, scheduler, scaler, best_dice)
            # Ghi thêm history vào resume để không mất khi resume
            ckpt_data = torch.load(resume_path, map_location="cpu")
            ckpt_data["history"] = history
            torch.save(ckpt_data, resume_path)

        # ── Dừng an toàn nếu monitor gửi SIGTERM ─────────────────────────────
        if _stop_requested:
            logger.warning(f"🛑 Dừng theo yêu cầu — lưu resume checkpoint tại epoch {epoch}...")
            _save_resume_ckpt(resume_path, fold_idx, epoch,
                              model, optimizer, scheduler, scaler, best_dice)
            ckpt_data = torch.load(resume_path, map_location="cpu")
            ckpt_data["history"] = history
            torch.save(ckpt_data, resume_path)
            logger.info(f"   Checkpoint lưu tại: {resume_path}")
            logger.info(f"   Chạy lại để tiếp tục từ epoch {epoch + 1}")
            _stop_requested = False
            return best_dice

    logger.info(f"Fold {fold_idx} done — best dice: {best_dice:.4f}")
    return best_dice


if __name__ == "__main__":
    for fold in FOLDS:
        train_fold(fold)
        torch.cuda.empty_cache()
    logger.info("All folds complete.")
