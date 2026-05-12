#!/usr/bin/env python3
"""
Loss Functions for Medical Image Segmentation.
Includes DiceCELoss and a DeepSupervisionLoss wrapper for nnUNet / DynUNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Handles class imbalance better than cross-entropy alone.
    """
    def __init__(self, smooth=1e-6, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, D, H, W) raw model output
            targets: (B, D, H, W) integer class labels  OR  (B, 1, D, H, W)
        """
        if targets.dim() == logits.dim():   # (B,1,D,H,W) → squeeze
            targets = targets.squeeze(1)

        probs = torch.softmax(logits, dim=1)
        n_cls = logits.shape[1]
        targets_oh = F.one_hot(targets.long(), num_classes=n_cls)  # (B,D,H,W,C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        dims = (2, 3, 4)
        inter = torch.sum(probs * targets_oh, dim=dims)
        card  = torch.sum(probs + targets_oh, dim=dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)  # (B, C)

        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, device=dice.device, dtype=dice.dtype)
            dice = (dice * w).sum(dim=1) / w.sum()
        else:
            dice = dice.mean(dim=1)

        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """
    50 % Dice + 50 % Cross-Entropy.  Stable default for pancreas/tumor seg.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None, smooth=1e-6):
        super().__init__()
        self.ce_w  = ce_weight
        self.dice_w = dice_weight
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        if targets.dim() == logits.dim():
            targets = targets.squeeze(1)
        return self.dice_w * self.dice(logits, targets) + self.ce_w * self.ce(logits, targets.long())


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper that applies a base loss at multiple output scales.

    DynUNet with deep_supervision=True returns a tuple:
        outputs[0]: full-resolution prediction  (B, C, D, H, W)
        outputs[1]: half-resolution prediction
        outputs[2]: quarter-resolution prediction

    Losses are weighted as [1, 0.5, 0.25] (normalised so they sum to 1).
    """
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets):
        # DynUNet deep_supervision=True returns (B, n_heads, C, D, H, W)
        if isinstance(outputs, torch.Tensor) and outputs.ndim == 6:
            outputs = [outputs[:, i] for i in range(outputs.shape[1])]
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)

        n = len(outputs)
        raw_w = [1.0 / (2 ** i) for i in range(n)]
        total = sum(raw_w)
        weights = [w / total for w in raw_w]

        loss = 0.0
        for out, w in zip(outputs, weights):
            if out.shape[2:] != targets.shape[2:]:
                # Downsample label to match this scale
                t = F.interpolate(
                    targets.float(),
                    size=out.shape[2:],
                    mode="nearest",
                ).long()
            else:
                t = targets
            loss = loss + w * self.base_loss(out, t)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss — useful for extremely imbalanced classes."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        if targets.dim() == logits.dim():
            targets = targets.squeeze(1)
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(targets.long(), logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        p = (probs * oh).sum(dim=1)
        focal = -self.alpha * ((1 - p) ** self.gamma) * torch.log(p + 1e-8)
        return focal.mean()


if __name__ == "__main__":
    B, C, D, H, W = 2, 3, 32, 32, 32
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, 1, D, H, W))

    base = DiceCELoss()
    ds_loss = DeepSupervisionLoss(base)

    # Single output
    print(f"DiceCE:           {base(logits, targets):.4f}")

    # Deep supervision output (3 scales)
    outputs_ds = [logits, logits[:, :, ::2, ::2, ::2], logits[:, :, ::4, ::4, ::4]]
    print(f"DeepSupervision:  {ds_loss(outputs_ds, targets):.4f}")
    print("Loss functions OK")
