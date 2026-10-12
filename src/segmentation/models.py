"""Segmentation model factory.

Default: segmentation_models_pytorch U-Net with a ResNet-18 encoder
(ImageNet-pretrained when `pretrained: true`; the ~45 MB encoder weights are
downloaded once by smp/timm on first use). It is small enough to fine-tune
on a CPU for short runs and on a T4-class GPU comfortably.

`legacy_unet` reproduces the plain U-Net from 02_Instrument_Segmentation.ipynb
so the existing models/unet_model.pth can be loaded and evaluated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class LegacyUNet(nn.Module):
    """Exact architecture of `UNet` in 02_Instrument_Segmentation.ipynb
    (1 output logit, no BatchNorm). State-dict keys match unet_model.pth."""

    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.enc1, self.enc2 = _DoubleConv(3, 64), _DoubleConv(64, 128)
        self.enc3, self.enc4 = _DoubleConv(128, 256), _DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(512, 1024)
        self.up4, self.dec4 = nn.ConvTranspose2d(1024, 512, 2, stride=2), _DoubleConv(1024, 512)
        self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, stride=2), _DoubleConv(512, 256)
        self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, stride=2), _DoubleConv(256, 128)
        self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2), _DoubleConv(128, 64)
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.output(d1)


def build_model(model_cfg: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    """Build a model that returns raw logits of shape [B, num_classes, H, W]."""
    arch = model_cfg.get("arch", "unet").lower()
    if arch == "legacy_unet":
        return LegacyUNet(out_channels=num_classes)

    import segmentation_models_pytorch as smp

    encoder = model_cfg.get("encoder", "resnet18")
    use_pretrained = model_cfg.get("pretrained", True) if pretrained is None else pretrained
    weights = model_cfg.get("encoder_weights", "imagenet") if use_pretrained else None
    archs = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN,
             "deeplabv3plus": smp.DeepLabV3Plus}
    if arch not in archs:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {sorted(archs)} or legacy_unet")
    return archs[arch](encoder_name=encoder, encoder_weights=weights,
                       in_channels=3, classes=num_classes)
