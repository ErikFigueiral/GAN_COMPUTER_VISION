"""Chain of responsibility for evaluation metrics.

Each handler receives the same evaluation context, adds the metrics it owns and
passes the context to the next handler. This keeps metric computation modular:
new metrics can be added without changing model code or experiment classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from .metrics import (
    DistributionMetricConfig,
    compute_batch_metrics,
    compute_distribution_metrics,
    compute_fid,
    compute_lpips_mean,
    summarize_metric_rows,
)


@dataclass
class EvaluationContext:
    """Shared state passed through the metric chain."""

    fake_images: torch.Tensor
    real_images: torch.Tensor
    device: torch.device | str = "cpu"
    distribution_config: DistributionMetricConfig = field(default_factory=DistributionMetricConfig)
    rows: list[dict[str, float]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


class MetricHandler(Protocol):
    """Interface implemented by metric handlers."""

    next_handler: "MetricHandler | None"

    def set_next(self, handler: "MetricHandler") -> "MetricHandler":
        ...

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        ...


class BaseMetricHandler:
    """Base chain node with default pass-through behavior."""

    def __init__(self) -> None:
        self.next_handler: MetricHandler | None = None

    def set_next(self, handler: MetricHandler) -> MetricHandler:
        self.next_handler = handler
        return handler

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        if self.next_handler is None:
            return context
        return self.next_handler.handle(context)


class PixelStructureMetricsHandler(BaseMetricHandler):
    """Compute MAE, PSNR, SSIM and MS-SSIM."""

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        context.rows = compute_batch_metrics(context.fake_images.cpu(), context.real_images.cpu())
        context.summary.update(summarize_metric_rows(context.rows))
        return super().handle(context)


class DistributionMetricsHandler(BaseMetricHandler):
    """Compute C2ST and generative precision/recall."""

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        context.summary.update(
            compute_distribution_metrics(
                context.fake_images.cpu(),
                context.real_images.cpu(),
                config=context.distribution_config,
            )
        )
        return super().handle(context)


class PerceptualMetricsHandler(BaseMetricHandler):
    """Compute optional LPIPS and FID when their dependencies are available."""

    def __init__(self, include_fid: bool = True, include_lpips: bool = True) -> None:
        super().__init__()
        self.include_fid = include_fid
        self.include_lpips = include_lpips

    def handle(self, context: EvaluationContext) -> EvaluationContext:
        if self.include_lpips:
            context.summary["lpips"] = compute_lpips_mean(
                context.fake_images,
                context.real_images,
                device=context.device,
            )
        if self.include_fid:
            context.summary["fid"] = compute_fid(
                context.fake_images,
                context.real_images,
                device=context.device,
            )
        return super().handle(context)


def build_default_metric_chain(include_perceptual: bool = False) -> MetricHandler:
    """Build the project default metric chain."""

    first = PixelStructureMetricsHandler()
    second = first.set_next(DistributionMetricsHandler())
    if include_perceptual:
        second.set_next(PerceptualMetricsHandler())
    return first


def run_metric_chain(
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    device: torch.device | str = "cpu",
    include_perceptual: bool = False,
    distribution_config: DistributionMetricConfig | None = None,
) -> EvaluationContext:
    """Convenience helper for notebooks and experiment summaries."""

    context = EvaluationContext(
        fake_images=fake_images,
        real_images=real_images,
        device=device,
        distribution_config=distribution_config or DistributionMetricConfig(),
    )
    return build_default_metric_chain(include_perceptual=include_perceptual).handle(context)

