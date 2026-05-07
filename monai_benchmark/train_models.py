#!/usr/bin/env python3
import sys, json, logging
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

# ── Cấu hình — sửa ở đây ──────────────────────────────────────────────────────
MODEL      = "nnunet"                          # "nnunet" | "swin_unetr"
FOLDS      = [0, 1, 2, 3, 4]
DEVICE     = "cuda:0"
DATA_DIR   = "/mnt/d/DATN/dataset/Task_7"
CKPT_DIR   = Path("checkpoints")
RESULTS_DIR = Path("results")

HU_WINDOW  = [-74, 140]
PATCH_SIZE = [96, 96, 96]
BATCH_SIZE = 2          # số volume mỗi batch (mỗi volume sinh 4 patch → 8 patch/step)
EPOCHS     = 50
LR         = 1e-4
WEIGHT_DECAY = 1e-5
VAL_FREQ   = 5          # validate mỗi N epoch
PATIENCE   = 10
CACHE_RATE = 0.2
NUM_WORKERS = 4
SW_BATCH   = 4          # sliding window batch size khi validate
SW_OVERLAP = 0.25
CLASS_WEIGHTS = [0.1, 0.5, 0.4]   # background / pancreas / tumor
AUGMENTATION = {"flip_prob": 0.5, "rotate_prob": 0.3, "scale_prob": 0.2}
# ─────────────────────────────────────────────────────────────────────────────


def make_loader(file_names, is_train):
    data_list = build_data_list(file_names, data_dir=DATA_DIR)
    tfm = (get_train_transforms(HU_WINDOW, PATCH_SIZE, AUGMENTATION)
           if is_train else get_val_transforms(HU_WINDOW))
    ds = CacheDataset(data=data_list, transform=tfm,
                      cache_rate=CACHE_RATE, num_workers=NUM_WORKERS)
    return DataLoader(ds, batch_size=BATCH_SIZE if is_train else 1,
                      shuffle=is_train, num_workers=NUM_WORKERS,
                      pin_memory=True, collate_fn=list_data_collate)


def train_fold(fold_idx):
    device = torch.device(DEVICE)

    fold_file = Path(f"monai_benchmark/splits/fold_{fold_idx}.json")
    fold_data  = json.loads(fold_file.read_text())
    train_names, val_names = fold_data["train"], fold_data["val"]
    logger.info(f"\n{'='*55}\n  Fold {fold_idx}  |  train={len(train_names)}  val={len(val_names)}\n{'='*55}")

    model = get_model(MODEL, pretrained=(MODEL != "nnunet"), device=device)
    train_loader = make_loader(train_names, is_train=True)
    val_loader   = make_loader(val_names,   is_train=False)

    criterion = DeepSupervisionLoss(DiceCELoss(class_weights=CLASS_WEIGHTS))
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler()

    ckpt_dir = CKPT_DIR / MODEL / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_dice, patience_ctr = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
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

        # ── validate ──
        if epoch % VAL_FREQ == 0 or epoch == 1:
            model.eval()
            all_m = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"E{epoch} val", leave=False, dynamic_ncols=True):
                    images = batch["image"].to(device)
                    labels = batch["label"]

                    def _fwd(x):
                        o = model(x)
                        return o[0] if isinstance(o, (list, tuple)) else o

                    logits = sliding_window_inference(images, PATCH_SIZE, SW_BATCH, _fwd, overlap=SW_OVERLAP)
                    pred   = torch.argmax(logits, 1).squeeze(0).cpu().numpy()
                    target = labels.squeeze().numpy()
                    for k, v in calculate_metrics(pred, target).items():
                        all_m[k].append(v)

            avg_m    = {k: float(np.mean(v)) for k, v in all_m.items()}
            avg_dice = avg_m.get("avg_dice", 0.0)
            logger.info(f"Epoch {epoch:3d}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | "
                        f"dice={avg_dice:.4f} (pan={avg_m.get('pancreas_dice',0):.3f} "
                        f"tum={avg_m.get('tumor_dice',0):.3f})")

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save({"model_state": model.state_dict(), "metrics": avg_m,
                            "fold": fold_idx}, ckpt_dir / "best_model.pth")
                (RESULTS_DIR / f"{MODEL}_fold{fold_idx}_metrics.json").write_text(
                    json.dumps(avg_m, indent=2))
                patience_ctr = 0
                logger.info(f"  ✓ best saved  (dice={best_dice:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break
        else:
            logger.info(f"Epoch {epoch:3d}/{EPOCHS} | loss={total_loss/len(train_loader):.4f}")

        if epoch % 10 == 0:
            torch.save({"model_state": model.state_dict(), "fold": fold_idx},
                       ckpt_dir / "latest_model.pth")

    logger.info(f"Fold {fold_idx} done — best dice: {best_dice:.4f}")
    return best_dice


if __name__ == "__main__":
    for fold in FOLDS:
        train_fold(fold)
        torch.cuda.empty_cache()
    logger.info("All folds complete.")
