"""Experiment definitions for the Pix2Pix project.

The project keeps one clean Python entry point for all variants.  Each
experiment describes what changes from the baseline instead of duplicating the
training loop in notebooks or shell scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BasePix2PixExperiment:
    """Configuration shared by all image-to-image experiments."""

    name: str = "baseline"
    trainer_module: str = "src.train"
    data_root: str = "data/raw"
    split_dir: str = "data/splits"
    output_dir: str = "outputs/runs/baseline"
    label_side: str = "right"
    image_size: tuple[int, int] = (256, 256)
    scale_size: tuple[int, int] | None = None
    augmentation: bool = False
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    gan_loss: str = "vanilla"
    ngf: int = 64
    ndf: int = 64
    num_downs: int = 7
    norm: str = "batch"
    dropout: float = 0.5
    seed: int = 42
    split_seed: int = 42
    amp: bool = True
    sample_every: int = 5
    save_every: int = 10
    sample_count: int = 4
    resume: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def train_args(self) -> list[str]:
        args = [
            "--data-root",
            self.data_root,
            "--split-dir",
            self.split_dir,
            "--output-dir",
            self.output_dir,
            "--label-side",
            self.label_side,
            "--image-size",
            str(self.image_size[0]),
            str(self.image_size[1]),
            "--epochs",
            str(self.epochs),
            "--batch-size",
            str(self.batch_size),
            "--num-workers",
            str(self.num_workers),
            "--lr",
            str(self.lr),
            "--beta1",
            str(self.beta1),
            "--beta2",
            str(self.beta2),
            "--lambda-l1",
            str(self.lambda_l1),
            "--ngf",
            str(self.ngf),
            "--ndf",
            str(self.ndf),
            "--norm",
            self.norm,
            "--seed",
            str(self.seed),
            "--split-seed",
            str(self.split_seed),
            "--sample-every",
            str(self.sample_every),
            "--save-every",
            str(self.save_every),
            "--sample-count",
            str(self.sample_count),
        ]
        if self.scale_size is not None:
            args += ["--scale-size", str(self.scale_size[0]), str(self.scale_size[1])]
        if self.augmentation:
            args.append("--augmentation")
        if self.amp:
            args.append("--amp")
        if self.resume is not None:
            args += ["--resume", self.resume]
        args += self.model_specific_args()
        args += list(self.extra_args)
        return args

    def model_specific_args(self) -> list[str]:
        return [
            "--gan-loss",
            self.gan_loss,
            "--num-downs",
            str(self.num_downs),
            "--dropout",
            str(self.dropout),
        ]

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.trainer_module, *self.train_args()]

    def run(self, dry_run: bool = False) -> int:
        cmd = self.command()
        print(" ".join(cmd))
        if dry_run:
            return 0
        return subprocess.call(cmd)


@dataclass(frozen=True)
class BaselinePix2PixExperiment(BasePix2PixExperiment):
    """Reference Pix2Pix U-Net and PatchGAN experiment."""

    name: str = "baseline"
    output_dir: str = "outputs/runs/baseline"


@dataclass(frozen=True)
class LSGANExperiment(BaselinePix2PixExperiment):
    """Same Pix2Pix architecture, smoother LSGAN adversarial objective."""

    name: str = "lsgan"
    output_dir: str = "outputs/runs/lsgan"
    gan_loss: str = "lsgan"
    augmentation: bool = False
    scale_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class AttentionPix2PixExperiment(LSGANExperiment):
    """CNN Pix2Pix variant with a lightweight self-attention bottleneck."""

    name: str = "attention"
    trainer_module: str = "src.train_attention_pix2pix"
    output_dir: str = "outputs/runs/attention_pix2pix"

    def model_specific_args(self) -> list[str]:
        return [
            "--gan-loss",
            self.gan_loss,
            "--num-downs",
            str(self.num_downs),
            "--dropout",
            str(self.dropout),
        ]


@dataclass(frozen=True)
class Pix2PixHDLiteExperiment(BasePix2PixExperiment):
    """Residual generator plus multi-scale PatchGAN discriminators."""

    name: str = "pix2pixhd_lite"
    trainer_module: str = "src.train_pix2pixhd_lite"
    output_dir: str = "outputs/runs/pix2pixhd_lite"
    batch_size: int = 2
    ngf: int = 48
    ndf: int = 48
    norm: str = "instance"
    augmentation: bool = True
    scale_size: tuple[int, int] | None = (286, 286)
    n_downsample: int = 3
    n_blocks: int = 6
    n_layers_d: int = 3
    num_scales: int = 2
    lambda_fm: float = 10.0

    def model_specific_args(self) -> list[str]:
        return [
            "--n-downsample",
            str(self.n_downsample),
            "--n-blocks",
            str(self.n_blocks),
            "--n-layers-d",
            str(self.n_layers_d),
            "--num-scales",
            str(self.num_scales),
            "--lambda-fm",
            str(self.lambda_fm),
        ]


@dataclass(frozen=True)
class TransferLearningExperiment(BasePix2PixExperiment):
    """Optional ResNet-encoder transfer-learning ablation."""

    name: str = "transfer"
    trainer_module: str = "src.train_transfer"
    output_dir: str = "outputs/runs/transfer"
    augmentation: bool = True
    scale_size: tuple[int, int] | None = (286, 286)


EXPERIMENTS: dict[str, type[BasePix2PixExperiment]] = {
    "baseline": BaselinePix2PixExperiment,
    "lsgan": LSGANExperiment,
    "attention": AttentionPix2PixExperiment,
    "pix2pixhd_lite": Pix2PixHDLiteExperiment,
    "transfer": TransferLearningExperiment,
}


def build_experiment(name: str, overrides: argparse.Namespace) -> BasePix2PixExperiment:
    cls = EXPERIMENTS[name]
    values = {
        key: value
        for key, value in vars(overrides).items()
        if value is not None and key in cls.__dataclass_fields__
    }
    return cls(**values)


def available_experiments() -> str:
    return ", ".join(sorted(EXPERIMENTS))
