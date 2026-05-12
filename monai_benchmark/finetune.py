#!/usr/bin/env python3
"""
Fine-tune script — tiếp tục train từ best_model.pth với LR thấp hơn.

Thay đổi so với train_models.py:
  - Load weights từ best_model.pth (chỉ model state, reset optimizer)
  - LR = 1e-5 (thấp hơn 10x so với train gốc 1e-4)
  - Epoch thêm: 30 (có thể chỉnh EXTRA_EPOCHS)
  - Checkpoint lưu vào checkpoints/{MODEL}_finetuned/ (không ghi đè bản gốc)

Usage:
    python monai_benchmark/finetune.py
    python monai_benchmark/finetune.py --folds 0 2 --epochs 20 --lr 5e-6
"""

import sys, json, logging, signal, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
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

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL      = "nnunet"
DEVICE     = "cuda:0"
DATA_DIR   = str(REPO_ROOT / "dataset" / "Task_7")
SPLITS_DIR = Path(__file__).resolve().parent / "splits"

_stop_requested = False

def _handle_sigterm(*_):
    global _stop_requested
    _stop_requested = True
    logger.warning("SIGTERM — lưu checkpoint và dừng sau epoch này...")

signal.signal(signal.SIGTERM, _handle_sigterm)


def make_loader(file_names, is_train):
    data_list = build_data_list(file_names, data_dir=DATA_DIR)
    tfm = get_train_transforms() if is_train else get_val_transforms()
    ds  = CacheDataset(data=data_list, transform=tfm, cache_rate=0.1, num_workers=4)
    return DataLoader(ds, batch_size=1,
                      shuffle=is_train, num_workers=4,
                      pin_memory=True, collate_fn=list_data_collate)


def finetune_fold(fold_idx, extra_epochs, lr):
    global _stop_requested
    device = torch.device(DEVICE)

    # ── Load data split ───────────────────────────────────────────────────────
    fold_data   = json.loads((SPLITS_DIR / f"fold_{fold_idx}.json").read_text())
    train_names = fold_data["train"]
    val_names   = fold_data["val"]
    logger.info(f"\n{'='*55}\n  Fine-tune Fold {fold_idx}  |  train={len(train_names)}  val={len(val_names)}\n{'='*55}")

    # ── Load model từ best checkpoint gốc ────────────────────────────────────
    src_ckpt = REPO_ROOT / f"checkpoints/{MODEL}/fold_{fold_idx}/best_model.pth"
    if not src_ckpt.exists():
        logger.error(f"Không tìm thấy checkpoint: {src_ckpt}. Bỏ qua fold {fold_idx}.")
        return None

    model = get_model(MODEL, pretrained=False, device=device)
    ckpt  = torch.load(src_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    logger.info(f"Loaded weights từ: {src_ckpt}")

    train_loader = make_loader(train_names, is_train=True)
    val_loader   = make_loader(val_names,   is_train=False)

    criterion = DeepSupervisionLoss(DiceCELoss(class_weights=[0.1, 0.5, 0.4]))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=extra_epochs, eta_min=lr / 10)
    scaler    = GradScaler()

    # ── Checkpoint dir riêng để không ghi đè bản gốc ─────────────────────────
    ckpt_dir = REPO_ROOT / f"checkpoints/{MODEL}_finetuned/fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pth"
    best_dice   = ckpt.get("metrics", {}).get("avg_dice", 0.0)

    # Resume nếu fine-tune đã chạy dở
    start_epoch = 1
    if resume_path.exists():
        r = torch.load(resume_path, map_location=device)
        model.load_state_dict(r["model_state"])
        optimizer.load_state_dict(r["optimizer_state"])
        scheduler.load_state_dict(r["scheduler_state"])
        scaler.load_state_dict(r["scaler_state"])
        start_epoch = r["epoch"] + 1
        best_dice   = r["best_dice"]
        logger.info(f"Resume fine-tune từ epoch {r['epoch']}  best_dice={best_dice:.4f}")

    logger.info(f"Fine-tune {extra_epochs} epoch  LR={lr}  (best_dice ban đầu={best_dice:.4f})")

    for epoch in range(start_epoch, extra_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"FT-E{epoch} train", leave=False, dynamic_ncols=True):
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

        # ── Validate ──────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            torch.cuda.empty_cache()
            model.eval()
            all_m = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"FT-E{epoch} val", leave=False, dynamic_ncols=True):
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
            logger.info(f"Epoch {epoch:3d}/{extra_epochs} | loss={total_loss/len(train_loader):.4f} | "
                        f"dice={avg_dice:.4f}  (pan={avg_m.get('pancreas_dice',0):.3f} "
                        f"tum={avg_m.get('tumor_dice',0):.3f})")

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save({"model_state": model.state_dict(), "metrics": avg_m,
                            "fold": fold_idx}, ckpt_dir / "best_model.pth")
                logger.info(f"  ✓ best saved  (dice={best_dice:.4f})")
        else:
            logger.info(f"Epoch {epoch:3d}/{extra_epochs} | loss={total_loss/len(train_loader):.4f}")

        # ── Resume checkpoint mỗi 5 epoch ────────────────────────────────────
        if epoch % 5 == 0:
            torch.save({
                "fold": fold_idx, "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_dice": best_dice,
            }, resume_path)

        if _stop_requested:
            logger.warning(f"Dừng theo yêu cầu tại epoch {epoch}")
            torch.save({
                "fold": fold_idx, "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_dice": best_dice,
            }, resume_path)
            _stop_requested = False
            return best_dice

    logger.info(f"Fold {fold_idx} fine-tune done — best dice: {best_dice:.4f}")
    return best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30,  help="Số epoch fine-tune thêm")
    parser.add_argument("--lr",     type=float, default=1e-5, help="Learning rate (mặc định 1e-5)")
    args = parser.parse_args()

    logger.info(f"Fine-tune {args.folds}  epochs={args.epochs}  lr={args.lr}")
    for fold in args.folds:
        finetune_fold(fold, args.epochs, args.lr)
        torch.cuda.empty_cache()
    logger.info("Fine-tune hoàn thành.")


if __name__ == "__main__":
    main()
