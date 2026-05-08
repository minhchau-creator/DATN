#!/usr/bin/env python3
"""
Evaluation Script — Sliding Window Inference + Metrics

Loads the best checkpoint for a given model/fold and runs full-volume
inference on the validation set.  Outputs per-case and aggregated
Dice / IoU / Hausdorff to JSON and CSV.

Usage:
    python monai_benchmark/evaluate.py \
        --model nnunet \
        --folds 0 1 2 3 4 \
        --device cuda:0 \
        --config monai_benchmark/configs/training.yaml
"""

import sys
import json
import logging
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monai_benchmark.data.transforms import build_data_list, get_val_transforms
from monai_benchmark.models.builder import get_model
from monai_benchmark.models.eval_metrics import calculate_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/workspace/dataset/Task_7"


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name, ckpt_path, device, config):
    """Load model architecture and restore weights from checkpoint."""
    model = get_model(
        model_name,
        pretrained=False,
        device=device,
        config=config,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    return model


def predict_volume(model, image_tensor, roi_size, sw_batch_size, overlap, device):
    """Run sliding-window inference and return (D, H, W) argmax prediction."""
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 4:          # (1, D, H, W) → (1, 1, D, H, W)
        image_tensor = image_tensor.unsqueeze(0)

    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=_fwd,
            overlap=overlap,
        )

    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()   # (D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(model_name, fold_idx, config, device, data_dir, results_dir):
    ckpt_path = Path(f"checkpoints/{model_name}/fold_{fold_idx}/best_model.pth")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Skipping fold {fold_idx}.")
        return None

    model = load_model(model_name, ckpt_path, device, config)

    fold_file = Path(f"monai_benchmark/splits/fold_{fold_idx}.json")
    with open(fold_file) as f:
        val_names = json.load(f)["val"]

    hu      = config.get("hu_window", [-74, 140])
    roi     = config.get("patch_size", [96, 96, 96])
    sw_bs   = config.get("sw_batch_size", 4)
    overlap = config.get("sw_overlap", 0.25)

    val_tf   = get_val_transforms(hu_window=hu)
    data_list = build_data_list(val_names, data_dir=data_dir)
    dataset  = Dataset(data=data_list, transform=val_tf)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = defaultdict(list)
    case_rows   = []

    for i, batch in enumerate(loader):
        image  = batch["image"]
        label  = batch["label"].squeeze().numpy()   # (D, H, W)
        name   = val_names[i] if i < len(val_names) else f"case_{i}"

        pred = predict_volume(model, image, roi, sw_bs, overlap, device)
        m    = calculate_metrics(pred, label)

        for k, v in m.items():
            all_metrics[k].append(v)

        row = {"case": name, "fold": fold_idx, **{k: f"{v:.4f}" for k, v in m.items()}}
        case_rows.append(row)
        print_metrics(m, scan_name=name)

    fold_avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    logger.info(
        f"\nFold {fold_idx} avg — "
        f"Pancreas Dice: {fold_avg.get('pancreas_dice', 0):.4f} | "
        f"Tumor Dice:    {fold_avg.get('tumor_dice', 0):.4f} | "
        f"Avg Dice:      {fold_avg.get('avg_dice', 0):.4f}"
    )

    # Save fold-level JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_json = results_dir / f"{model_name}_fold{fold_idx}_eval.json"
    with open(fold_json, "w") as f:
        json.dump({"avg": fold_avg, "cases": case_rows}, f, indent=2)

    return fold_avg, case_rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models on validation folds.")
    p.add_argument("--model",    type=str, required=True)
    p.add_argument("--folds",    type=int, nargs="+", default=[0])
    p.add_argument("--device",   type=str, default="cuda:0")
    p.add_argument("--config",   type=str, default="monai_benchmark/configs/training.yaml")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def load_config(config_path, model_name):
    with open(config_path) as f:
        full = yaml.safe_load(f)
    merged = {**full.get("common", {}), **full.get("models", {}).get(model_name, {})}
    merged.update(full.get("training", {}))
    merged.update(full.get("inference", {}))
    return merged


def main():
    args    = parse_args()
    config  = load_config(args.config, args.model)
    device  = torch.device(args.device)
    results = Path(args.results_dir)

    all_cases   = []
    fold_avgs   = []

    for fold_idx in args.folds:
        result = evaluate_fold(args.model, fold_idx, config, device, args.data_dir, results)
        if result is None:
            continue
        fold_avg, case_rows = result
        fold_avgs.append(fold_avg)
        all_cases.extend(case_rows)

    if not fold_avgs:
        logger.error("No folds evaluated.")
        return

    # Cross-fold summary
    metric_keys = list(fold_avgs[0].keys())
    summary = {k: float(np.mean([fa[k] for fa in fold_avgs])) for k in metric_keys}

    logger.info("\n" + "=" * 60)
    logger.info(f"CROSS-FOLD SUMMARY — {args.model}")
    logger.info("=" * 60)
    logger.info(f"  Pancreas Dice:  {summary.get('pancreas_dice', 0):.4f} ± "
                f"{float(np.std([fa.get('pancreas_dice',0) for fa in fold_avgs])):.4f}")
    logger.info(f"  Tumor Dice:     {summary.get('tumor_dice', 0):.4f} ± "
                f"{float(np.std([fa.get('tumor_dice',0) for fa in fold_avgs])):.4f}")
    logger.info(f"  Avg Dice:       {summary.get('avg_dice', 0):.4f} ± "
                f"{float(np.std([fa.get('avg_dice',0) for fa in fold_avgs])):.4f}")
    logger.info(f"  Pancreas IoU:   {summary.get('pancreas_iou', 0):.4f}")
    logger.info(f"  Tumor IoU:      {summary.get('tumor_iou', 0):.4f}")
    logger.info("=" * 60)

    # Save summary JSON
    summary_path = results / f"{args.model}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "folds": args.folds, "summary": summary,
                   "fold_avgs": fold_avgs}, f, indent=2)
    logger.info(f"Summary saved → {summary_path}")

    # Save all-cases CSV
    csv_path = results / f"{args.model}_cases.csv"
    if all_cases:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cases[0].keys())
            writer.writeheader()
            writer.writerows(all_cases)
        logger.info(f"Per-case CSV  → {csv_path}")


if __name__ == "__main__":
    main()
