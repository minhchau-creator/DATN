#!/usr/bin/env python3
"""
Loss Functions for Medical Image Segmentation
Includes DiceCELoss (combination of Dice + CrossEntropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    Good for imbalanced classes (background << pancreas << tumor).
    """
    def __init__(self, smooth=1e-6, reduction='mean', class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, D, H, W) model output
            targets: (B, D, H, W) integer class labels
        """
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Compute Dice per class
        dims = (2, 3, 4)  # Spatial dimensions
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Average across classes
        if self.class_weights is not None:
            dice = (dice * torch.tensor(self.class_weights, device=dice.device, dtype=dice.dtype)).mean()
        else:
            dice = dice.mean()
        
        return 1.0 - dice  # Return loss (minimize)


class DiceCELoss(nn.Module):
    """
    Combination of Dice Loss + Cross Entropy Loss.
    Good for both shape (Dice) and intensity (CE) optimization.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None, smooth=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.soft_dice = DiceLoss(smooth=smooth, class_weights=class_weights)
        self.ce = nn.CrossEntropyLoss(weight=None)
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, D, H, W) model output
            targets: (B, D, H, W) integer class labels
        """
        dice_loss = self.soft_dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        # Compute focal loss
        p = (probs * one_hot_targets).sum(dim=1)  # Probability of correct class
        focal = -self.alpha * ((1 - p) ** self.gamma) * torch.log(p + 1e-6)
        
        return focal.mean()


if __name__ == "__main__":
    # Test losses
    B, C, D, H, W = 2, 3, 32, 32, 32
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))
    
    dice_loss = DiceLoss()
    dice_ce_loss = DiceCELoss()
    
    print(f"Dice Loss: {dice_loss(logits, targets).item():.4f}")
    print(f"DiceCE Loss: {dice_ce_loss(logits, targets).item():.4f}")
    print("✅ Loss functions working")
