"""Pix2PixHD-lite models for the optional architecture experiment.

This module intentionally lives next to the baseline Pix2Pix models instead of
replacing them. The design borrows the two useful Pix2PixHD ideas that fit this
project budget: a residual global generator and a multi-scale PatchGAN
discriminator.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

NormName = Literal["batch", "instance", "none"]


def get_norm_layer(norm: NormName, num_features: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    if norm == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError("norm must be one of: 'batch', 'instance', 'none'.")


class ResnetBlock(nn.Module):
    """Residual block used by the Pix2PixHD global generator."""

    def __init__(self, channels: int, norm: NormName = "instance") -> None:
        super().__init__()
        bias = norm == "none"
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=bias),
            get_norm_layer(norm, channels),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=bias),
            get_norm_layer(norm, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GlobalResnetGenerator(nn.Module):
    """Lite Pix2PixHD-style global generator.

    Compared with the baseline U-Net, this generator spends capacity in a
    residual bottleneck. It is smaller than the original Pix2PixHD generator by
    default so it remains feasible for the CV2 project dataset and hardware.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 48,
        n_downsample: int = 3,
        n_blocks: int = 6,
        norm: NormName = "instance",
    ) -> None:
        super().__init__()
        if n_downsample < 1:
            raise ValueError("n_downsample must be at least 1.")
        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1.")

        bias = norm == "none"
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=bias),
            get_norm_layer(norm, ngf),
            nn.ReLU(inplace=False),
        ]

        channels = ngf
        for _ in range(n_downsample):
            next_channels = min(channels * 2, ngf * 8)
            layers.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                    get_norm_layer(norm, next_channels),
                    nn.ReLU(inplace=False),
                ]
            )
            channels = next_channels

        for _ in range(n_blocks):
            layers.append(ResnetBlock(channels, norm=norm))

        for _ in range(n_downsample):
            next_channels = max(channels // 2, ngf)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=bias,
                    ),
                    get_norm_layer(norm, next_channels),
                    nn.ReLU(inplace=False),
                ]
            )
            channels = next_channels

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels, out_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        y = self.model(x)
        if y.shape[-2:] != input_size:
            y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        return y


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator that exposes intermediate feature maps."""

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm: NormName = "instance",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        sequence: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            )
        ]

        channels = ndf
        for layer_index in range(1, n_layers):
            next_channels = min(ndf * (2**layer_index), 512)
            sequence.append(self._block(channels, next_channels, stride=2, norm=norm))
            channels = next_channels

        next_channels = min(channels * 2, 512)
        sequence.append(self._block(channels, next_channels, stride=1, norm=norm))
        sequence.append(nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1))
        self.layers = nn.ModuleList(sequence)

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int, norm: NormName) -> nn.Sequential:
        bias = norm == "none"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=bias),
            get_norm_layer(norm, out_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class MultiscalePatchGANDiscriminator(nn.Module):
    """Pix2PixHD-style discriminator evaluated at multiple image scales."""

    def __init__(
        self,
        in_channels: int = 3,
        target_channels: int = 3,
        ndf: int = 48,
        n_layers: int = 3,
        num_scales: int = 2,
        norm: NormName = "instance",
    ) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales must be at least 1.")
        total_channels = in_channels + target_channels
        self.discriminators = nn.ModuleList(
            [
                NLayerDiscriminator(
                    in_channels=total_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    norm=norm,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, label_map: torch.Tensor, image: torch.Tensor) -> list[list[torch.Tensor]]:
        x = torch.cat([label_map, image], dim=1)
        outputs: list[list[torch.Tensor]] = []
        for index, discriminator in enumerate(self.discriminators):
            if index > 0:
                x = self.downsample(x)
            outputs.append(discriminator(x))
        return outputs


def init_weights(module: nn.Module, init_gain: float = 0.02) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        nn.init.normal_(module.weight.data, mean=0.0, std=init_gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=init_gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def build_pix2pixhd_lite_models(
    input_channels: int = 3,
    output_channels: int = 3,
    ngf: int = 48,
    ndf: int = 48,
    n_downsample: int = 3,
    n_blocks: int = 6,
    n_layers_d: int = 3,
    num_scales: int = 2,
    norm: NormName = "instance",
) -> tuple[GlobalResnetGenerator, MultiscalePatchGANDiscriminator]:
    """Build and initialize the optional Pix2PixHD-lite architecture."""

    generator = GlobalResnetGenerator(
        in_channels=input_channels,
        out_channels=output_channels,
        ngf=ngf,
        n_downsample=n_downsample,
        n_blocks=n_blocks,
        norm=norm,
    )
    discriminator = MultiscalePatchGANDiscriminator(
        in_channels=input_channels,
        target_channels=output_channels,
        ndf=ndf,
        n_layers=n_layers_d,
        num_scales=num_scales,
        norm=norm,
    )
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    return generator, discriminator
