"""Pix2Pix generator and discriminator models implemented in PyTorch."""

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


class DownBlock(nn.Module):
    """Downsampling block used in the U-Net encoder."""

    def __init__(self, in_channels: int, out_channels: int, norm: NormName = "batch", use_norm: bool = True) -> None:
        super().__init__()
        bias = not use_norm or norm == "none"
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        ]
        if use_norm:
            layers.append(get_norm_layer(norm, out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block used in the U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: NormName = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = norm == "none"
        layers: list[nn.Module] = [
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias),
            get_norm_layer(norm, out_channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.block(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return x


class UNetGenerator(nn.Module):
    """U-Net generator for Pix2Pix.

    The default depth is robust for 256x256 and 256x384 training sizes. For
    very small inputs, reduce ``num_downs`` accordingly.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        num_downs: int = 7,
        norm: NormName = "batch",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_downs < 2:
            raise ValueError("num_downs must be at least 2.")

        encoder_channels = [min(ngf * (2**i), ngf * 8) for i in range(num_downs)]
        encoders: list[nn.Module] = []
        previous_channels = in_channels
        for idx, channels in enumerate(encoder_channels):
            encoders.append(
                DownBlock(
                    previous_channels,
                    channels,
                    norm=norm,
                    use_norm=idx != 0,
                )
            )
            previous_channels = channels
        self.encoders = nn.ModuleList(encoders)

        decoders: list[nn.Module] = []
        current_channels = encoder_channels[-1]
        reversed_skip_channels = list(reversed(encoder_channels[:-1]))
        for idx, skip_channels in enumerate(reversed_skip_channels):
            out_channels_decoder = skip_channels
            block_dropout = dropout if idx < 3 else 0.0
            decoders.append(
                UpBlock(
                    current_channels,
                    out_channels_decoder,
                    norm=norm,
                    dropout=block_dropout,
                )
            )
            current_channels = out_channels_decoder + skip_channels
        self.decoders = nn.ModuleList(decoders)

        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # The last encoder output is the bottleneck. The remaining encoder
        # outputs are used as skip connections in reverse order.
        skip_features = list(reversed(skips[:-1]))
        for decoder, skip in zip(self.decoders, skip_features):
            x = decoder(x, skip)

        x = self.final(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


class PatchGANDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    It receives the label map and either the real or generated image, then
    predicts a grid of real/fake logits over local patches.
    """

    def __init__(
        self,
        in_channels: int = 3,
        target_channels: int = 3,
        ndf: int = 64,
        norm: NormName = "batch",
    ) -> None:
        super().__init__()
        total_channels = in_channels + target_channels

        layers: list[nn.Module] = [
            nn.Conv2d(total_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            self._block(ndf, ndf * 2, stride=2, norm=norm),
            self._block(ndf * 2, ndf * 4, stride=2, norm=norm),
            self._block(ndf * 4, ndf * 8, stride=1, norm=norm),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int, norm: NormName) -> nn.Sequential:
        out_channels = min(out_channels, 512)
        bias = norm == "none"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=bias),
            get_norm_layer(norm, out_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, label_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([label_map, image], dim=1)
        return self.model(x)


def init_weights(module: nn.Module, init_gain: float = 0.02) -> None:
    """Initialize convolutional and normalization layers following Pix2Pix practice."""

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


def build_pix2pix_models(
    input_channels: int = 3,
    output_channels: int = 3,
    ngf: int = 64,
    ndf: int = 64,
    num_downs: int = 7,
    norm: NormName = "batch",
    dropout: float = 0.5,
) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """Build and initialize the Pix2Pix generator and discriminator."""

    generator = UNetGenerator(
        in_channels=input_channels,
        out_channels=output_channels,
        ngf=ngf,
        num_downs=num_downs,
        norm=norm,
        dropout=dropout,
    )
    discriminator = PatchGANDiscriminator(
        in_channels=input_channels,
        target_channels=output_channels,
        ndf=ndf,
        norm=norm,
    )
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    return generator, discriminator


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad
