#!/usr/bin/env python3
"""
GradCAM-3D and Saliency Maps for segmentation models.

For a predicted tumor voxel, GradCAM answers:
  "Which spatial regions in the feature space drove the tumor prediction?"

Saliency answers:
  "Which input voxels, if perturbed, would change the tumor prediction most?"
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TUMOR_CLASS = 2   # class index for PDAC tumor


# ─────────────────────────────────────────────────────────────────────────────
# Layer finder
# ─────────────────────────────────────────────────────────────────────────────

def find_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Return the last Conv3d (or ConvTranspose3d) layer in the model.
    This is a reasonable GradCAM target for any 3D segmentation network.
    """
    target = None
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            target = module
    return target


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM-3D
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Map for 3D segmentation.

    Specifically designed to explain tumor (class-2) predictions:
    - hooks the target convolutional layer
    - computes gradient of the *tumor class logit sum* w.r.t. that layer
    - returns a normalised heatmap in the same spatial resolution as the input

    Usage:
        cam = GradCAM3D(model)
        heatmap = cam.generate(image_tensor)   # (D, H, W) in [0, 1]
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        if self.target_layer is None:
            raise RuntimeError("No Conv3d found in model for GradCAM.")

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image: torch.Tensor,         # (1, 1, D, H, W) normalised CT
        target_class: int = TUMOR_CLASS,
        mask_with_prediction: bool = True,
    ) -> np.ndarray:
        """
        Args:
            image:              (1, 1, D, H, W) float tensor on model device
            target_class:       class index to explain (default 2 = tumor)
            mask_with_prediction: if True, only keep heatmap where model
                                  actually predicted target_class

        Returns:
            heatmap: (D, H, W) numpy array in [0, 1]
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward
        output = self.model(image)
        if isinstance(output, (list, tuple)):
            output = output[0]   # full-resolution head

        # Score = sum of tumor logits over all spatial positions
        score = output[:, target_class, :, :, :].sum()

        # Backward
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Gradient-weighted activation (Global Average Pooling of grads)
        grads  = self._gradients   # (1, C_feat, d, h, w)
        acts   = self._activations # (1, C_feat, d, h, w)
        weights = grads.mean(dim=[2, 3, 4], keepdim=True)   # (1, C_feat, 1, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)      # (1, 1, d, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()                    # (D, H, W)

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        # Optionally zero-out heatmap where model did not predict tumor
        if mask_with_prediction:
            with torch.no_grad():
                pred = self.model(image.detach())
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_cls = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            cam = cam * (pred_cls == target_class).astype(np.float32)

        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Saliency Map (input-gradient)
# ─────────────────────────────────────────────────────────────────────────────

def saliency_map(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int = TUMOR_CLASS,
) -> np.ndarray:
    """
    Vanilla saliency: gradient of tumor score w.r.t. input voxels.
    Shows *which input voxels* most affect the tumor classification.

    Returns:
        saliency: (D, H, W) numpy array in [0, 1]
    """
    model.eval()
    image = image.clone().detach().requires_grad_(True)

    output = model(image)
    if isinstance(output, (list, tuple)):
        output = output[0]

    score = output[:, target_class, :, :, :].sum()
    model.zero_grad()
    score.backward()

    sal = image.grad.data.abs().squeeze().cpu().numpy()  # (D, H, W)
    lo, hi = sal.min(), sal.max()
    if hi - lo > 1e-8:
        sal = (sal - lo) / (hi - lo)
    return sal.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int = TUMOR_CLASS,
    n_steps: int = 30,
    baseline: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al., 2017).
    More faithful than plain saliency: attribution satisfies completeness axiom.

    baseline: reference image (default: zero image = air HU after normalisation)
    Returns:
        attribution: (D, H, W) numpy array, positive = supports tumor prediction
    """
    model.eval()
    device = next(model.parameters()).device

    if baseline is None:
        baseline = torch.zeros_like(image)

    baseline = baseline.to(device)
    image    = image.to(device)

    # Interpolate from baseline to image in n_steps
    alphas     = torch.linspace(0, 1, n_steps, device=device)
    grad_accum = torch.zeros_like(image)

    for alpha in alphas:
        interp = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        output = model(interp)
        if isinstance(output, (list, tuple)):
            output = output[0]
        score = output[:, target_class, :, :, :].sum()
        model.zero_grad()
        score.backward()
        grad_accum += interp.grad.data

    # IG = (image - baseline) * mean gradient
    ig = ((image - baseline) * grad_accum / n_steps).squeeze().cpu().numpy()

    # Return absolute value, normalised
    ig = np.abs(ig)
    lo, hi = ig.min(), ig.max()
    if hi - lo > 1e-8:
        ig = (ig - lo) / (hi - lo)
    return ig.astype(np.float32)
