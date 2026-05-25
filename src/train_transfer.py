"""Train the optional ResNet-encoder Pix2Pix transfer-learning ablation.

This entry point is deliberately separate from ``src.train`` so the main
baseline remains stable for the assignment.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import optim

from .dataset import SplitConfig, build_dataloaders, build_datasets
from .losses import Pix2PixLosses
from .models import set_requires_grad
from .models_transfer import build_transfer_pix2pix_models, maybe_unfreeze_encoder
from .train import load_resume_checkpoint, save_checkpoint, save_epoch_samples, train_one_epoch, validate
from .utils import append_csv_row, ensure_dir, get_device, make_reproducible_torch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optional transfer-learning Pix2Pix ablation.")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs/runs/transfer_resnet18")
    parser.add_argument("--label-side", type=str, choices=["left", "right"], default="right")

    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--scale-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--augmentation", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--reconstruction-loss", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "none"], default="batch")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--resnet-name", type=str, choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained-resnet", action="store_true", help="Use torchvision ImageNet weights if available.")
    parser.add_argument("--resnet-weights", type=str, default=None, help="Optional local ResNet weights path.")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force-splits", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    make_reproducible_torch()

    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "checkpoints")
    history_path = output_dir / "history.csv"

    config = vars(args).copy()
    config["architecture"] = "transfer_resnet_unet"
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

    generator, discriminator = build_transfer_pix2pix_models(
        output_channels=3,
        ndf=args.ndf,
        resnet_name=args.resnet_name,
        pretrained=args.pretrained_resnet,
        weights_path=args.resnet_weights,
        norm=args.norm,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder_epochs > 0,
    )
    generator.to(device)
    discriminator.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    loss_fn = Pix2PixLosses(lambda_l1=args.lambda_l1, reconstruction_loss=args.reconstruction_loss)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val_mae = float("inf")
    if args.resume:
        start_epoch, best_val_mae = load_resume_checkpoint(
            args.resume,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            device,
        )
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
        maybe_unfreeze_encoder(generator, epoch=epoch, freeze_epochs=args.freeze_encoder_epochs)
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
        )
        set_requires_grad(discriminator, True)
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

        save_checkpoint(
            output_dir=output_dir,
            name="latest.pt",
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            config=config,
            best_val_mae=best_val_mae,
        )
        if row["val_mae"] < best_val_mae:
            best_val_mae = row["val_mae"]
            save_checkpoint(
                output_dir=output_dir,
                name="best_generator.pt",
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config=config,
                best_val_mae=best_val_mae,
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                name=f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config=config,
                best_val_mae=best_val_mae,
            )
        if args.sample_every > 0 and (epoch == 1 or epoch % args.sample_every == 0):
            save_epoch_samples(
                generator=generator,
                dataloader=dataloaders["val"],
                device=device,
                output_path=output_dir / "figures" / f"samples_epoch_{epoch:03d}.png",
                sample_count=args.sample_count,
                title=f"Transfer validation samples - epoch {epoch}",
            )

    print(f"Training finished. Best validation MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    main()
