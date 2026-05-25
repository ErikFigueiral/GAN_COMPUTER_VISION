"""Self-attention Pix2Pix models.

This optional variant keeps the baseline CNN U-Net and PatchGAN, adding one
light self-attention block at the generator bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import DownBlock, NormName, PatchGANDiscriminator, UpBlock, init_weights


class SelfAttention2d(nn.Module):
    """Small SAGAN-style attention block for 2D feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        positions = height * width

        query = self.query(x).view(batch, -1, positions).transpose(1, 2)
        key = self.key(x).view(batch, -1, positions)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)

        value = self.value(x).view(batch, channels, positions)
        attended = torch.bmm(value, attention.transpose(1, 2)).view(batch, channels, height, width)
        return x + self.gamma * attended


class AttentionUNetGenerator(nn.Module):
    """Baseline U-Net generator with self-attention at the bottleneck."""

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
        for index, channels in enumerate(encoder_channels):
            encoders.append(DownBlock(previous_channels, channels, norm=norm, use_norm=index != 0))
            previous_channels = channels
        self.encoders = nn.ModuleList(encoders)
        self.attention = SelfAttention2d(encoder_channels[-1])

        decoders: list[nn.Module] = []
        current_channels = encoder_channels[-1]
        for index, skip_channels in enumerate(reversed(encoder_channels[:-1])):
            block_dropout = dropout if index < 3 else 0.0
            decoders.append(UpBlock(current_channels, skip_channels, norm=norm, dropout=block_dropout))
            current_channels = skip_channels + skip_channels
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

        x = self.attention(skips[-1])
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(x, skip)

        x = self.final(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def build_attention_pix2pix_models(
    input_channels: int = 3,
    output_channels: int = 3,
    ngf: int = 64,
    ndf: int = 64,
    num_downs: int = 7,
    norm: NormName = "batch",
    dropout: float = 0.5,
) -> tuple[AttentionUNetGenerator, PatchGANDiscriminator]:
    """Build and initialize the self-attention Pix2Pix variant."""

    generator = AttentionUNetGenerator(
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
