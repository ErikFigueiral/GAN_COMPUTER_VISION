"""Evaluate a trained Pix2Pix generator on validation or test data.

Run from the project root, for example:

    python -m src.evaluate --data-root data/raw --checkpoint outputs/runs/baseline/checkpoints/best_generator.pt --split test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .dataset import SplitConfig, build_dataloaders, build_datasets
from .metrics import (
    DistributionMetricConfig,
    clean_metric_dict,
    compute_batch_metrics,
    compute_distribution_metrics,
    compute_fid,
    compute_lpips_mean,
    summarize_metric_rows,
)
from .models import UNetGenerator
from .models_attention import AttentionUNetGenerator
from .models_pix2pixhd import build_pix2pixhd_lite_models
from .utils import ensure_dir, get_device, save_comparison_grid, save_csv_rows, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Pix2Pix generator.")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--label-side", type=str, choices=["left", "right"], default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--save-generated", action="store_true", help="Save every generated image separately.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-distribution-images", type=int, default=512)
    parser.add_argument("--distribution-feature-size", type=int, default=64)
    parser.add_argument("--precision-recall-k", type=int, default=3)
    parser.add_argument("--skip-distribution-metrics", action="store_true")
    parser.add_argument(
        "--include-perceptual-metrics",
        action="store_true",
        help="Also try LPIPS and FID. They may require optional pretrained weights.",
    )
    return parser.parse_args()


def get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def load_generator(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    architecture = config.get("architecture")
    if architecture == "pix2pixhd_lite":
        generator, _ = build_pix2pixhd_lite_models(
            input_channels=3,
            output_channels=3,
            ngf=int(get_config_value(config, "ngf", 48)),
            ndf=int(get_config_value(config, "ndf", 48)),
            n_downsample=int(get_config_value(config, "n_downsample", 3)),
            n_blocks=int(get_config_value(config, "n_blocks", 6)),
            n_layers_d=int(get_config_value(config, "n_layers_d", 3)),
            num_scales=int(get_config_value(config, "num_scales", 2)),
            norm=get_config_value(config, "norm", "instance"),
        )
    else:
        generator_cls = AttentionUNetGenerator if architecture == "self_attention_pix2pix" else UNetGenerator
        generator = generator_cls(
            in_channels=3,
            out_channels=3,
            ngf=int(get_config_value(config, "ngf", 64)),
            num_downs=int(get_config_value(config, "num_downs", 7)),
            norm=get_config_value(config, "norm", "batch"),
            dropout=float(get_config_value(config, "dropout", 0.5)),
        )
    generator = generator.to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, config, int(checkpoint.get("epoch", -1))


@dataclass
class EvaluationContext:
    """Shared state passed through the metric evaluation chain."""

    model: torch.nn.Module
    checkpoint_path: Path | None
    dataloader: torch.utils.data.DataLoader
    device: torch.device
    output_dir: Path
    split_name: str
    sample_count: int
    save_generated: bool
    distribution_config: DistributionMetricConfig | None
    include_perceptual_metrics: bool
    metric_rows: list[dict[str, float | str]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    fake_batches: list[torch.Tensor] = field(default_factory=list)
    real_batches: list[torch.Tensor] = field(default_factory=list)
    first_batch_saved: bool = False


class MetricHandler:
    """Base handler for Chain of Responsibility metric evaluation."""

    def __init__(self, next_handler: "MetricHandler | None" = None) -> None:
        self.next_handler = next_handler

    def set_next(self, next_handler: "MetricHandler") -> "MetricHandler":
        self.next_handler = next_handler
        return next_handler

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        if self.next_handler is not None:
            return self.next_handler.handle(context)
        return context


class PairwiseMetricHandler(MetricHandler):
    """Generate samples, save qualitative outputs, and compute paired metrics."""

    @torch.no_grad()
    def handle(self, context: EvaluationContext) -> EvaluationContext:
        generated_dir = ensure_dir(context.output_dir / "generated") if context.save_generated else None

        for batch in tqdm(context.dataloader, desc=f"Evaluating {context.split_name}"):
            label_maps = batch["label_map"].to(context.device, non_blocking=True)
            real_images = batch["real_image"].to(context.device, non_blocking=True)
            fake_images = context.model(label_maps)

            fake_cpu = fake_images.cpu()
            real_cpu = real_images.cpu()
            batch_metrics = compute_batch_metrics(fake_cpu, real_cpu)
            for path, metrics in zip(batch["path"], batch_metrics):
                context.metric_rows.append({"path": path, **metrics})

            if context.distribution_config is not None or context.include_perceptual_metrics:
                context.fake_batches.append(fake_cpu)
                context.real_batches.append(real_cpu)

            if not context.first_batch_saved:
                save_comparison_grid(
                    label_maps=label_maps.cpu(),
                    real_images=real_cpu,
                    fake_images=fake_cpu,
                    output_path=context.output_dir / f"{context.split_name}_comparison_grid.png",
                    max_items=context.sample_count,
                    title=f"{context.split_name.capitalize()} qualitative comparison",
                )
                context.first_batch_saved = True

            if generated_dir is not None:
                from .utils import save_tensor_image

                for relative_path, fake_image in zip(batch["path"], fake_cpu):
                    safe_stem = str(relative_path).replace("\\", "/").replace("/", "__")
                    output_path = generated_dir / f"{Path(safe_stem).stem}_generated.png"
                    save_tensor_image(fake_image, output_path)

        numeric_rows = [{k: float(v) for k, v in row.items() if k != "path"} for row in context.metric_rows]
        context.summary.update(summarize_metric_rows(numeric_rows))
        return super().handle(context)


class DistributionMetricHandler(MetricHandler):
    """Append dataset-level distribution metrics when enough samples are available."""

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        if context.distribution_config is not None and context.fake_batches and context.real_batches:
            fake_all = torch.cat(context.fake_batches, dim=0)
            real_all = torch.cat(context.real_batches, dim=0)
            context.summary.update(compute_distribution_metrics(fake_all, real_all, config=context.distribution_config))
        return super().handle(context)


class PerceptualMetricHandler(MetricHandler):
    """Append optional pretrained perceptual metrics when explicitly requested."""

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        if context.include_perceptual_metrics and context.fake_batches and context.real_batches:
            fake_all = torch.cat(context.fake_batches, dim=0)
            real_all = torch.cat(context.real_batches, dim=0)
            context.summary["lpips"] = compute_lpips_mean(fake_all, real_all, device=context.device)
            context.summary["fid"] = compute_fid(fake_all, real_all, device=context.device)
        return super().handle(context)


def build_metric_chain() -> MetricHandler:
    """Build the common metric chain used by all experiment evaluations."""

    pairwise = PairwiseMetricHandler()
    distribution = pairwise.set_next(DistributionMetricHandler())
    distribution.set_next(PerceptualMetricHandler())
    return pairwise


def evaluate_split(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    sample_count: int,
    save_generated: bool,
    distribution_config: DistributionMetricConfig | None,
    include_perceptual_metrics: bool,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    context = EvaluationContext(
        model=generator,
        checkpoint_path=checkpoint_path,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        split_name=split_name,
        sample_count=sample_count,
        save_generated=save_generated,
        distribution_config=distribution_config,
        include_perceptual_metrics=include_perceptual_metrics,
    )
    result = build_metric_chain().handle(context)
    return result.summary, result.metric_rows


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    generator, config, epoch = load_generator(args.checkpoint, device)

    image_size = tuple(get_config_value(config, "image_size", [256, 256]))
    label_side = args.label_side or get_config_value(config, "label_side", "right")

    datasets = build_datasets(
        data_root=args.data_root,
        split_dir=args.split_dir,
        image_size=image_size,
        split_config=SplitConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.split_seed),
        force_splits=False,
        label_side=label_side,
        use_augmentation=False,
    )
    dataloaders = build_dataloaders(datasets, batch_size=args.batch_size, num_workers=args.num_workers)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent.parent / "evaluation" / args.split
    output_dir = ensure_dir(output_dir)

    distribution_config = None
    if not args.skip_distribution_metrics:
        distribution_config = DistributionMetricConfig(
            feature_size=args.distribution_feature_size,
            max_items=args.max_distribution_images,
            seed=args.split_seed,
            precision_recall_k=args.precision_recall_k,
        )

    summary, rows = evaluate_split(
        generator=generator,
        dataloader=dataloaders[args.split],
        device=device,
        output_dir=output_dir,
        split_name=args.split,
        sample_count=args.sample_count,
        save_generated=args.save_generated,
        distribution_config=distribution_config,
        include_perceptual_metrics=args.include_perceptual_metrics,
        checkpoint_path=Path(args.checkpoint),
    )

    summary_with_meta = clean_metric_dict(
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": epoch,
            "num_images": len(rows),
            **summary,
        }
    )
    rows = [clean_metric_dict(row) for row in rows]  # type: ignore[list-item]
    save_csv_rows(output_dir / f"{args.split}_metrics_per_image.csv", rows)  # type: ignore[arg-type]
    save_csv_rows(output_dir / f"{args.split}_metrics_summary.csv", [summary_with_meta])
    save_json(summary_with_meta, output_dir / f"{args.split}_metrics_summary.json")

    print("Evaluation summary:")
    for key, value in summary_with_meta.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
