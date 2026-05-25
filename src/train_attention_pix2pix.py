"""Train the self-attention Pix2Pix variant.

Run from the project root, for example:

    python -m src.train_attention_pix2pix --data-root data/raw --output-dir outputs/runs/self_attention
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from torch import optim
from tqdm import tqdm

from .dataset import SplitConfig, build_dataloaders, build_datasets
from .losses import Pix2PixLosses
from .metrics import compute_batch_metrics, summarize_metric_rows
from .models import set_requires_grad
from .models_attention import build_attention_pix2pix_models
from .train import average_dicts, load_resume_checkpoint, save_checkpoint, save_epoch_samples
from .utils import append_csv_row, ensure_dir, get_device, make_reproducible_torch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pix2Pix with bottleneck self-attention.")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Directory containing paired images.")
    parser.add_argument("--split-dir", type=str, default="data/splits", help="Directory containing train/val/test txt files.")
    parser.add_argument("--output-dir", type=str, default="outputs/runs/self_attention", help="Directory for logs, figures and checkpoints.")
    parser.add_argument("--label-side", type=str, choices=["left", "right"], default="right", help="Side of the semantic label map in each paired image.")

    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--scale-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--augmentation", action="store_true", help="Enable synchronized random crop and horizontal flip.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--reconstruction-loss", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--gan-loss", type=str, choices=["vanilla", "lsgan"], default="lsgan")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--num-downs", type=int, default=7)
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "none"], default="batch")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force-splits", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for smoke tests.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training.")
    return parser.parse_args()


def train_one_epoch(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Pix2PixLosses,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    loss_rows: list[dict[str, float]] = []

    progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [self-attn]", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        label_maps = batch["label_map"].to(device, non_blocking=True)
        real_images = batch["real_image"].to(device, non_blocking=True)

        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_images_for_d = generator(label_maps)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_real = discriminator(label_maps, real_images)
            pred_fake = discriminator(label_maps, fake_images_for_d)
            loss_d, loss_d_items = loss_fn.discriminator_loss(pred_real, pred_fake)

        scaler.scale(loss_d).backward()
        scaler.step(optimizer_d)

        set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            fake_images = generator(label_maps)
            pred_fake_for_g = discriminator(label_maps, fake_images)
            loss_g, loss_g_items = loss_fn.generator_loss(pred_fake_for_g, fake_images, real_images)

        scaler.scale(loss_g).backward()
        scaler.step(optimizer_g)
        scaler.update()
        set_requires_grad(discriminator, True)

        row = {**loss_d_items, **loss_g_items}
        loss_rows.append(row)
        progress.set_postfix({"G": f"{row['loss_G']:.3f}", "D": f"{row['loss_D']:.3f}", "L1": f"{row['loss_G_L1']:.3f}"})

    return average_dicts(loss_rows)


@torch.no_grad()
def validate(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    generator.eval()
    metric_rows: list[dict[str, float]] = []

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        label_maps = batch["label_map"].to(device, non_blocking=True)
        real_images = batch["real_image"].to(device, non_blocking=True)
        fake_images = generator(label_maps)
        metric_rows.extend(compute_batch_metrics(fake_images.cpu(), real_images.cpu()))

    summary = summarize_metric_rows(metric_rows)
    return {f"val_{key}": value for key, value in summary.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    make_reproducible_torch()

    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "checkpoints")
    history_path = output_dir / "history.csv"

    config: dict[str, Any] = vars(args).copy()
    config["architecture"] = "self_attention_pix2pix"
    config["image_size"] = list(args.image_size)
    if args.scale_size is not None:
        config["scale_size"] = list(args.scale_size)
    save_json(config, output_dir / "run_config.json")

    device = get_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    datasets = build_datasets(
        data_root=args.data_root,
        split_dir=args.split_dir,
        image_size=tuple(args.image_size),
        split_config=SplitConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.split_seed),
        force_splits=args.force_splits,
        label_side=args.label_side,
        use_augmentation=args.augmentation,
        scale_size=tuple(args.scale_size) if args.scale_size is not None else None,
    )
    dataloaders = build_dataloaders(datasets, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Split sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    generator, discriminator = build_attention_pix2pix_models(
        input_channels=3,
        output_channels=3,
        ngf=args.ngf,
        ndf=args.ndf,
        num_downs=args.num_downs,
        norm=args.norm,
        dropout=args.dropout,
    )
    generator.to(device)
    discriminator.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    loss_fn = Pix2PixLosses(
        lambda_l1=args.lambda_l1,
        gan_loss=args.gan_loss,
        reconstruction_loss=args.reconstruction_loss,
    )
    print(f"Adversarial loss: {args.gan_loss}")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val_mae = float("inf")
    if args.resume:
        start_epoch, best_val_mae = load_resume_checkpoint(args.resume, generator, discriminator, optimizer_g, optimizer_d, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    fieldnames = [
        "epoch",
        "seconds",
        "loss_D",
        "loss_D_real",
        "loss_D_fake",
        "loss_G",
        "loss_G_GAN",
        "loss_G_L1",
        "val_mae",
        "val_psnr",
        "val_ssim",
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloaders["train"],
            loss_fn=loss_fn,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        val_metrics = validate(generator, dataloaders["val"], device=device, max_batches=args.max_val_batches)
        seconds = time.time() - start_time
        row = {"epoch": epoch, "seconds": round(seconds, 2), **train_metrics, **val_metrics}
        append_csv_row(history_path, row, fieldnames=fieldnames)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"G={row.get('loss_G', float('nan')):.4f} | "
            f"D={row.get('loss_D', float('nan')):.4f} | "
            f"val_MAE={row.get('val_mae', float('nan')):.4f} | "
            f"val_PSNR={row.get('val_psnr', float('nan')):.2f} | "
            f"val_SSIM={row.get('val_ssim', float('nan')):.4f} | "
            f"{seconds:.1f}s"
        )

        save_checkpoint(output_dir, "latest.pt", epoch, generator, discriminator, optimizer_g, optimizer_d, config, best_val_mae)
        if row["val_mae"] < best_val_mae:
            best_val_mae = row["val_mae"]
            save_checkpoint(output_dir, "best_generator.pt", epoch, generator, discriminator, optimizer_g, optimizer_d, config, best_val_mae)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir, f"epoch_{epoch:03d}.pt", epoch, generator, discriminator, optimizer_g, optimizer_d, config, best_val_mae)
        if args.sample_every > 0 and (epoch == 1 or epoch % args.sample_every == 0):
            save_epoch_samples(generator, dataloaders["val"], device, output_dir / "figures" / f"samples_epoch_{epoch:03d}.png", args.sample_count, f"Validation samples - epoch {epoch}")

    print(f"Training finished. Best validation MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    main()
