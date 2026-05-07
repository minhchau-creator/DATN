#!/usr/bin/env python3
"""
Predictive Uncertainty via Monte Carlo Dropout.

Answers: "How confident is the model that this voxel is PDAC tumor?"

Method (Gal & Ghahramani, 2016):
  1. Enable dropout at inference time (model.train() only for Dropout layers)
  2. Run N stochastic forward passes
  3. Compute mean prediction (= soft label) and variance (= uncertainty)

A high-variance voxel that the model still calls "tumor" is a harder case —
useful for radiologists to scrutinise more carefully.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

logger = logging.getLogger(__name__)

TUMOR_CLASS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Enable MC Dropout at test time
# ─────────────────────────────────────────────────────────────────────────────

def _enable_mc_dropout(model: nn.Module):
    """Set only Dropout layers to train mode so stochasticity is active."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout3d, nn.AlphaDropout)):
            module.train()


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_uncertainty(
    model: nn.Module,
    image: torch.Tensor,
    n_passes: int = 20,
    roi_size=(96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes and return uncertainty statistics.

    Args:
        model:        trained segmentation model (must have dropout layers)
        image:        (1, 1, D, H, W) preprocessed CT tensor on model device
        n_passes:     number of MC samples
        roi_size:     sliding window patch size
        sw_batch_size: sliding window batch size
        overlap:      sliding window overlap ratio

    Returns:
        mean_pred:    (D, H, W) int — mean argmax prediction (most likely class)
        tumor_prob:   (D, H, W) float [0,1] — average probability of tumor class
        uncertainty:  (D, H, W) float [0,1] — predictive entropy (higher = less sure)
    """
    device = image.device
    _enable_mc_dropout(model)

    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    prob_sum   = None
    prob_sq_sum = None

    with torch.no_grad():
        for i in range(n_passes):
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=_fwd,
                overlap=overlap,
            )
            probs = torch.softmax(logits, dim=1)   # (1, C, D, H, W)

            if prob_sum is None:
                prob_sum    = probs
                prob_sq_sum = probs ** 2
            else:
                prob_sum    = prob_sum    + probs
                prob_sq_sum = prob_sq_sum + probs ** 2

            logger.debug(f"MC pass {i+1}/{n_passes}")

    # Restore eval mode
    model.eval()

    mean_probs = (prob_sum / n_passes).squeeze(0)   # (C, D, H, W)
    mean_pred  = torch.argmax(mean_probs, dim=0).cpu().numpy()  # (D, H, W)
    tumor_prob = mean_probs[TUMOR_CLASS].cpu().numpy()          # (D, H, W)

    # Predictive entropy: H = -sum_c p_c * log(p_c)
    eps     = 1e-8
    entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=0)
    # Normalise by log(n_classes) so it lives in [0, 1]
    n_classes   = mean_probs.shape[0]
    entropy_norm = (entropy / np.log(n_classes)).cpu().numpy()

    return (
        mean_pred.astype(np.int8),
        tumor_prob.astype(np.float32),
        entropy_norm.astype(np.float32),
    )
