#!/usr/bin/env python3
"""
Unified Model Factory for 8-Model Benchmark.
Handles initialization and pretrained weight loading for all architectures.
"""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, SegResNet, UNet, DynUNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom architectures
# ---------------------------------------------------------------------------

class UNetPlusPlus(nn.Module):
    """
    Simplified UNet++ (Nested U-Net) with deep supervision.
    Ref: Zhou et al., 2020
    """
    def __init__(self, in_channels=1, out_channels=3,
                 channels=(32, 64, 128, 256), strides=(2, 2, 2), deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for ch in channels:
            self.encoders.append(conv_block(in_ch, ch))
            in_ch = ch

        self.pool = nn.MaxPool3d(2, 2)

        # Decoder: upsample from channels[i] + skip channels[i-1] → channels[i-1]
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose3d(channels[i], channels[i - 1], 2, stride=2)
            )
            self.decoders.append(conv_block(channels[i - 1] * 2, channels[i - 1]))

        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encoders[:-1]:
            out = enc(out)
            skips.append(out)
            out = self.pool(out)
        out = self.encoders[-1](out)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            out = up(out)
            out = dec(torch.cat([out, skip], dim=1))

        return self.out_conv(out)


class AttentionUNetCustom(nn.Module):
    """3D Attention U-Net with attention gates."""
    def __init__(self, in_channels=1, out_channels=3,
                 channels=(32, 64, 128, 256)):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for ch in channels[:-1]:
            self.encoders.append(conv_block(in_ch, ch))
            in_ch = ch

        self.bottleneck = conv_block(channels[-2], channels[-1])
        self.pool = nn.MaxPool3d(2, 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose3d(channels[i], channels[i - 1], 2, stride=2)
            )
            # Attention gate: gate (channels[i-1]) + skip (channels[i-1]) → 1
            self.att_gates.append(nn.Sequential(
                nn.Conv3d(channels[i - 1] * 2, channels[i - 1], 1),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(channels[i - 1], 1, 1),
                nn.Sigmoid(),
            ))
            self.decoders.append(conv_block(channels[i - 1] * 2, channels[i - 1]))

        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        for up, att, dec, skip in zip(
            self.upsamples, self.att_gates, self.decoders, reversed(skips)
        ):
            out = up(out)
            att_map = att(torch.cat([out, skip], dim=1))
            gated_skip = skip * att_map
            out = dec(torch.cat([out, gated_skip], dim=1))

        return self.out_conv(out)


# ---------------------------------------------------------------------------
# nnUNet-style DynUNet configuration helper
# ---------------------------------------------------------------------------

def _nnunet_config(patch_size=(96, 96, 96), in_channels=1, out_channels=3):
    """
    Auto-configure DynUNet (nnUNet equivalent) for a given patch size.
    5-level 3D full-resolution setup.
    """
    n_levels = 5
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_levels - 1)   # (5,)
    kernel_size = [[3, 3, 3]] * n_levels                     # (5,)
    upsample_kernel_size = [[2, 2, 2]] * (n_levels - 1)      # (4,)
    filters = [32, 64, 128, 256, 320]

    return dict(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=0.0,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=True,
        deep_supr_num=2,
        res_block=True,
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(model_name, pretrained=True, in_channels=1, out_channels=3,
              device="cuda", config=None):
    """
    Return a model instance on the specified device.

    Args:
        model_name: one of 'swin_unetr', 'segresnet', 'unet', 'attention_unet',
                    'unetplusplus', 'nnunet', 'resnet_unet'
        pretrained:  try to load pretrained weights when True
        in_channels: 1 for single-channel CT
        out_channels: 3 for (background, pancreas, tumor)
        device: torch device string or object
        config: optional dict with model-specific overrides

    Returns:
        nn.Module on device
    """
    if config is None:
        config = {}

    name = model_name.lower()

    try:
        if name == "dints":
            logger.warning("DiNTS: load via MONAI bundle, not this factory. Returning None.")
            return None

        elif name == "swin_unetr":
            logger.info("Loading Swin-UNETR...")
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
            if pretrained:
                _load_swin_unetr_weights(model)

        elif name == "segresnet":
            logger.info("Loading SegResNet...")
            model = SegResNet(
                spatial_dims=3,
                init_filters=8,
                in_channels=in_channels,
                out_channels=out_channels,
                use_checkpoint=True,
            )
            if pretrained:
                logger.warning("No reliable public SegResNet checkpoint — training from random init.")

        elif name == "unet":
            logger.info("Loading U-Net...")
            model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            if pretrained:
                logger.warning("No reliable public 3-class U-Net checkpoint — training from random init.")

        elif name == "attention_unet":
            logger.info("Building Attention U-Net...")
            model = AttentionUNetCustom(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(32, 64, 128, 256),
            )

        elif name == "unetplusplus":
            logger.info("Building UNet++...")
            model = UNetPlusPlus(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(32, 64, 128, 256),
                strides=(2, 2, 2),
                deep_supervision=True,
            )

        elif name == "nnunet":
            logger.info("Building nnUNet (MONAI DynUNet)...")
            cfg = _nnunet_config(
                patch_size=config.get("patch_size", (96, 96, 96)),
                in_channels=in_channels,
                out_channels=out_channels,
            )
            model = DynUNet(**cfg)
            logger.info("nnUNet (DynUNet) built with 5-level 3D full-res config + deep supervision.")

        elif name == "resnet_unet":
            logger.info("Building ResNet-UNet (DynUNet variant)...")
            cfg = _nnunet_config(
                patch_size=config.get("patch_size", (96, 96, 96)),
                in_channels=in_channels,
                out_channels=out_channels,
            )
            # Slightly wider filters for ResNet-UNet
            cfg["filters"] = [32, 64, 128, 256, 512]
            cfg["deep_supervision"] = False
            model = DynUNet(**cfg)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{model_name} on {device} | params: {n_params:,}")
        return model

    except Exception:
        logger.exception(f"Failed to create model '{model_name}'")
        raise


def _load_swin_unetr_weights(model):
    """Attempt to download and load BTCV pretrained Swin-UNETR weights."""
    try:
        import os
        import torch

        model_dir = os.path.expanduser("~/.cache/monai/models")
        os.makedirs(model_dir, exist_ok=True)
        weight_path = os.path.join(model_dir, "swin_unetr_btcv_segmentation_fold0.pt")

        if not os.path.exists(weight_path):
            url = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/"
                "releases/download/0.4.0/swin_unetr_btcv_segmentation_fold0.pt"
            )
            logger.info(f"Downloading Swin-UNETR BTCV weights from {url}")
            torch.hub.download_url_to_file(url, weight_path)

        state_dict = torch.load(weight_path, map_location="cpu")
        # Some checkpoints wrap under 'state_dict' key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"BTCV weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    except Exception as e:
        logger.warning(f"Could not load Swin-UNETR pretrained weights: {e}. Using random init.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name in ["swin_unetr", "segresnet", "unet", "attention_unet",
                 "unetplusplus", "nnunet", "resnet_unet"]:
        try:
            m = get_model(name, pretrained=False, device="cpu")
            if m is not None:
                x = torch.randn(1, 1, 96, 96, 96)
                with torch.no_grad():
                    out = m(x)
                shape = out[0].shape if isinstance(out, (list, tuple)) else out.shape
                print(f"  {name}: output={shape}, params={count_parameters(m):,}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
