"""Loss functions used by the Pix2Pix training loop."""

from __future__ import annotations

import torch
from torch import nn


class Pix2PixLosses:
    """Convenience wrapper around adversarial and reconstruction losses."""

    def __init__(self, lambda_l1: float = 100.0, gan_loss: str = "vanilla", reconstruction_loss: str = "l1") -> None:
        self.lambda_l1 = lambda_l1
        self.gan_loss = gan_loss
        self.reconstruction_loss = reconstruction_loss
        if gan_loss == "vanilla":
            self.adversarial = nn.BCEWithLogitsLoss()
        elif gan_loss == "lsgan":
            self.adversarial = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported GAN loss {gan_loss!r}. Expected 'vanilla' or 'lsgan'.")
        # Optional ablation: original Pix2Pix uses L1; L2 is exposed by CLI for comparison.
        if reconstruction_loss == "l1":
            self.reconstruction = nn.L1Loss()
        elif reconstruction_loss == "l2":
            self.reconstruction = nn.MSELoss()
        else:
            raise ValueError("reconstruction_loss must be 'l1' or 'l2'.")

    def discriminator_loss(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        real_targets = torch.ones_like(pred_real)
        fake_targets = torch.zeros_like(pred_fake)
        loss_real = self.adversarial(pred_real, real_targets)
        loss_fake = self.adversarial(pred_fake, fake_targets)
        loss_total = 0.5 * (loss_real + loss_fake)
        return loss_total, {
            "loss_D_real": float(loss_real.detach().cpu()),
            "loss_D_fake": float(loss_fake.detach().cpu()),
            "loss_D": float(loss_total.detach().cpu()),
        }

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
        fake_image: torch.Tensor,
        real_image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        real_targets = torch.ones_like(pred_fake)
        loss_gan = self.adversarial(pred_fake, real_targets)
        loss_reconstruction = self.reconstruction(fake_image, real_image)
        loss_total = loss_gan + self.lambda_l1 * loss_reconstruction
        return loss_total, {
            "loss_G_GAN": float(loss_gan.detach().cpu()),
            "loss_G_reconstruction": float(loss_reconstruction.detach().cpu()),
            "loss_G_L1": float(loss_reconstruction.detach().cpu()),
            "loss_G": float(loss_total.detach().cpu()),
        }
