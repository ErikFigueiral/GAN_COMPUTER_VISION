"""Experimental transfer-learning models for Pix2Pix.

This module is intentionally separate from ``src.models`` because transfer
learning is an optional extension for the report, not the default solution.
The main candidate is a ResNet encoder initialized from ImageNet and connected
to a U-Net-like decoder. It can be useful when the label maps are RGB semantic
renderings with edges and colour regions, but it is less natural than transfer
learning from real RGB photos because the input domain is not photographic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .models import NormName, PatchGANDiscriminator, UpBlock, init_weights

ResNetName = Literal["resnet18", "resnet34"]


class ImageNetInputNormalize(nn.Module):
    """Map project tensors in [-1, 1] to ImageNet-normalized RGB tensors."""

    def __init__(self) -> None:
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        return (x.clamp(0.0, 1.0) - self.mean) / self.std


def _load_resnet(name: ResNetName, pretrained: bool, weights_path: str | None) -> nn.Module:
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError(
            "ResNet transfer learning requires torchvision. The baseline project "
            "does not depend on this optional extra."
        ) from exc

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
    else:
        raise ValueError("name must be either 'resnet18' or 'resnet34'.")

    if weights_path:
        state_dict = torch.load(Path(weights_path), map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        resnet.load_state_dict(state_dict, strict=False)
    return resnet


class ResNetUNetGenerator(nn.Module):
    """Pix2Pix generator with an ImageNet-style ResNet encoder.

    The decoder mirrors the baseline U-Net skip-connection pattern. The encoder
    can be frozen for the first training epochs and unfrozen later.
    """

    def __init__(
        self,
        out_channels: int = 3,
        resnet_name: ResNetName = "resnet18",
        pretrained: bool = True,
        weights_path: str | None = None,
        norm: NormName = "batch",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        resnet = _load_resnet(resnet_name, pretrained=pretrained, weights_path=weights_path)

        self.input_norm = ImageNetInputNormalize()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up4 = UpBlock(512, 256, norm=norm, dropout=dropout)
        self.up3 = UpBlock(512, 128, norm=norm, dropout=dropout)
        self.up2 = UpBlock(256, 64, norm=norm, dropout=0.0)
        self.up1 = UpBlock(128, 64, norm=norm, dropout=0.0)
        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        for module in [self.up4, self.up3, self.up2, self.up1, self.final]:
            module.apply(init_weights)

    @property
    def encoder_modules(self) -> list[nn.Module]:
        return [self.stem, self.pool, self.layer1, self.layer2, self.layer3, self.layer4]

    def set_encoder_trainable(self, trainable: bool, train_norm_layers: bool = False) -> None:
        for module in self.encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable
            if not train_norm_layers:
                for submodule in module.modules():
                    if isinstance(submodule, nn.BatchNorm2d):
                        submodule.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.input_norm(x)

        s0 = self.stem(x)
        x = self.pool(s0)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x = self.layer4(s3)

        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = self.final(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def maybe_unfreeze_encoder(generator: nn.Module, epoch: int, freeze_epochs: int) -> None:
    """Unfreeze a transfer encoder after ``freeze_epochs`` epochs."""

    if freeze_epochs <= 0 or epoch != freeze_epochs + 1:
        return
    if hasattr(generator, "set_encoder_trainable"):
        generator.set_encoder_trainable(True)  # type: ignore[attr-defined]


def build_transfer_pix2pix_models(
    output_channels: int = 3,
    ndf: int = 64,
    resnet_name: ResNetName = "resnet18",
    pretrained: bool = True,
    weights_path: str | None = None,
    norm: NormName = "batch",
    dropout: float = 0.5,
    freeze_encoder: bool = True,
) -> tuple[ResNetUNetGenerator, PatchGANDiscriminator]:
    """Build the optional ResNet-encoder generator and baseline PatchGAN."""

    generator = ResNetUNetGenerator(
        out_channels=output_channels,
        resnet_name=resnet_name,
        pretrained=pretrained,
        weights_path=weights_path,
        norm=norm,
        dropout=dropout,
    )
    if freeze_encoder:
        generator.set_encoder_trainable(False)

    discriminator = PatchGANDiscriminator(in_channels=3, target_channels=output_channels, ndf=ndf, norm=norm)
    discriminator.apply(init_weights)
    return generator, discriminator
