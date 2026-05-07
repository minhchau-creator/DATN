#!/usr/bin/env python3
"""
Main Training Script — 8-Model Benchmark
Uses MONAI CacheDataset, dict-based transforms, sliding-window validation, and AMP.

Usage:
    python monai_benchmark/train_models.py \
        --model nnunet \
        --folds 0 1 2 \
        --device cuda:0 \
        --config monai_benchmark/configs/training.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference

# Project-local imports (add project root to path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monai_benchmark.data.transforms import build_data_list, get_train_transforms, get_val_transforms
from monai_benchmark.models.builder import get_model
from monai_benchmark.models.losses import DiceCELoss, DeepSupervisionLoss
from monai_benchmark.models.eval_metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset root — override via env variable or --data-dir flag
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.environ.get("DATN_DATA_DIR", "/mnt/d/DATN/dataset/Task_7")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model_name, fold_idx, config, device="cuda:0", data_dir=DEFAULT_DATA_DIR):
        self.model_name = model_name
        self.fold_idx   = fold_idx
        self.config     = config
        self.device     = torch.device(device)
        self.data_dir   = data_dir

        self.ckpt_dir   = Path(f"checkpoints/{model_name}/fold_{fold_idx}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_path   = self.ckpt_dir / "best_model.pth"
        self.latest_path = self.ckpt_dir / "latest_model.pth"
        self.metrics_path = self.results_dir / f"{model_name}_fold{fold_idx}_metrics.json"

        logger.info("=" * 60)
        logger.info(f"  Model: {model_name}  |  Fold: {fold_idx}  |  Device: {device}")
        logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────

    def _make_loader(self, file_names, is_train):
        data_list = build_data_list(file_names, data_dir=self.data_dir)
        hu   = self.config.get("hu_window", [-74, 140])
        patch = self.config.get("patch_size", [96, 96, 96])
        aug  = self.config.get("augmentation", {})

        if is_train:
            tfm = get_train_transforms(hu_window=hu, patch_size=patch,
                                       augmentation_config=aug)
            bs  = self.config.get("batch_size", 2)
        else:
            tfm = get_val_transforms(hu_window=hu)
            bs  = 1

        ds = CacheDataset(
            data=data_list,
            transform=tfm,
            cache_rate=self.config.get("cache_rate", 0.2),
            num_workers=self.config.get("num_workers", 4),
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=is_train,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            collate_fn=list_data_collate,
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, train_names, val_names):
        logger.info(f"Train samples: {len(train_names)}  |  Val samples: {len(val_names)}")

        self.model = get_model(
            self.model_name,
            pretrained=(self.config.get("strategy") == "fine_tune"),
            device=self.device,
            config=self.config,
        )

        self.train_loader = self._make_loader(train_names, is_train=True)
        self.val_loader   = self._make_loader(val_names,   is_train=False)

        base_loss = DiceCELoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=self.config.get("class_weights", [0.1, 0.5, 0.4]),
        )
        # Wrap with deep-supervision handler (transparent when model output is a tensor)
        self.criterion = DeepSupervisionLoss(base_loss)

        lr = self.config.get("lr", 1e-4)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.get("weight_decay", 1e-5),
        )
        epochs = self.config.get("epochs", 50)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.scaler    = GradScaler()   # AMP

        logger.info("Setup complete: model | loaders | loss | optimizer | AMP")

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(self.train_loader, desc="Train", leave=False, dynamic_ncols=True):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)  # (B, 1, D, H, W)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self):
        self.model.eval()
        roi_size = self.config.get("patch_size", [96, 96, 96])
        all_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False, dynamic_ncols=True):
                images = batch["image"].to(self.device)   # (1, 1, D, H, W)
                labels = batch["label"]                   # (1, 1, D, H, W) cpu

                # Sliding-window inference — returns full-res logits
                def _predict(x):
                    out = self.model(x)
                    return out[0] if isinstance(out, (list, tuple)) else out

                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=self.config.get("sw_batch_size", 4),
                    predictor=_predict,
                    overlap=self.config.get("sw_overlap", 0.25),
                )

                pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (D,H,W)
                target = labels.squeeze().numpy()                               # (D,H,W)

                m = calculate_metrics(pred, target)
                for k, v in m.items():
                    all_metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, metrics, is_best):
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_name": self.model_name,
            "fold": self.fold_idx,
        }
        path = self.best_path if is_best else self.latest_path
        torch.save(ckpt, path)

        if is_best and metrics is not None:
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"New best saved → {self.best_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self, train_names, val_names):
        self.setup(train_names, val_names)

        epochs   = self.config.get("epochs", 50)
        val_freq = self.config.get("val_freq", 5)
        patience = self.config.get("patience", 10)

        best_dice, patience_ctr = 0.0, 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            self.scheduler.step()

            if epoch % val_freq == 0 or epoch == 1:
                val_m = self.validate()
                avg_dice = val_m.get("avg_dice", 0.0)

                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"loss={train_loss:.4f} | "
                    f"val_dice={avg_dice:.4f} "
                    f"(pan={val_m.get('pancreas_dice', 0):.3f}, "
                    f"tum={val_m.get('tumor_dice', 0):.3f})"
                )

                if avg_dice > best_dice:
                    best_dice = avg_dice
                    self.save_checkpoint(val_m, is_best=True)
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info(f"Early stopping at epoch {epoch}.")
                        break
            else:
                logger.info(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f}")

            if epoch % 10 == 0:
                self.save_checkpoint(None, is_best=False)

        logger.info(f"Training done. Best avg Dice: {best_dice:.4f}")
        return best_dice


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a model on the pancreas segmentation benchmark.")
    p.add_argument("--model",    type=str, required=True, help="Model name (e.g. nnunet, swin_unetr)")
    p.add_argument("--folds",    type=int, nargs="+", default=[0], help="Fold indices to train (e.g. 0 1 2)")
    p.add_argument("--device",   type=str, default="cuda:0")
    p.add_argument("--config",   type=str, default="monai_benchmark/configs/training.yaml")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--epochs",   type=int, default=None, help="Override epochs from config")
    return p.parse_args()


def load_config(config_path, model_name):
    with open(config_path) as f:
        full = yaml.safe_load(f)
    common = full.get("common", {})
    model_cfg = full.get("models", {}).get(model_name, {})
    # Model-specific settings override common
    merged = {**common, **model_cfg}
    # Also merge training section
    merged.update(full.get("training", {}))
    return merged


def main():
    args = parse_args()

    config = load_config(args.config, args.model)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    splits_dir = Path("monai_benchmark/splits")

    for fold_idx in args.folds:
        fold_file = splits_dir / f"fold_{fold_idx}.json"
        with open(fold_file) as f:
            fold_data = json.load(f)

        train_names = fold_data["train"]
        val_names   = fold_data["val"]

        trainer = Trainer(
            model_name=args.model,
            fold_idx=fold_idx,
            config=config,
            device=args.device,
            data_dir=args.data_dir,
        )
        trainer.train(train_names, val_names)
        torch.cuda.empty_cache()

    logger.info(f"All folds done for {args.model}.")


if __name__ == "__main__":
    main()
