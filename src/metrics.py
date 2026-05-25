"""Image quality and distribution metrics for paired Pix2Pix evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:  # Keep train/evaluate usable when scikit-image is not installed.
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:  # pragma: no cover - depends on local environment
    peak_signal_noise_ratio = None
    structural_similarity = None

from .utils import denormalize_tensor, tensor_to_numpy_image

try:  # Optional: listed in requirements, but keep evaluation usable without it.
    import piq
except Exception:  # pragma: no cover - depends on local environment
    piq = None


@dataclass(frozen=True)
class DistributionMetricConfig:
    """Small, explicit knobs for dataset-level metrics."""

    feature_size: int = 64
    max_items: int = 512
    seed: int = 42
    precision_recall_k: int = 3


def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def _to_01_batch(images: torch.Tensor) -> torch.Tensor:
    """Return BCHW float images in [0, 1]. Accepts CHW or BCHW."""

    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError("Expected image tensor with shape CHW or BCHW.")
    if images.min() < 0:
        return denormalize_tensor(images)
    return images.detach().float().cpu().clamp(0.0, 1.0)


def compute_pair_metrics(fake_image: torch.Tensor, real_image: torch.Tensor) -> dict[str, float]:
    """Compute MAE, PSNR, SSIM and MS-SSIM for one generated/target pair.

    Input tensors must have shape CHW and are expected in [-1, 1]. MS-SSIM is
    computed through PIQ when available; otherwise it is reported as NaN.
    """

    fake = tensor_to_numpy_image(fake_image)
    real = tensor_to_numpy_image(real_image)
    mae = float(np.mean(np.abs(fake - real)))
    psnr = compute_psnr(real, fake)
    ssim = compute_ssim(real, fake)
    ms_ssim = compute_ms_ssim(fake_image.unsqueeze(0), real_image.unsqueeze(0))
    return {"mae": mae, "psnr": psnr, "ssim": ssim, "ms_ssim": ms_ssim}


def compute_batch_metrics(fake_images: torch.Tensor, real_images: torch.Tensor) -> list[dict[str, float]]:
    return [compute_pair_metrics(fake, real) for fake, real in zip(fake_images, real_images)]


def compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
    """Compute PSNR in [0, 1], using scikit-image when available."""

    if peak_signal_noise_ratio is not None:
        return float(peak_signal_noise_ratio(real, fake, data_range=1.0))
    mse = float(np.mean((real - fake) ** 2))
    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    """Compute SSIM, or NaN when the optional implementation is unavailable."""

    if structural_similarity is None:
        return float("nan")
    return float(structural_similarity(real, fake, channel_axis=-1, data_range=1.0))


def compute_ms_ssim(fake_images: torch.Tensor, real_images: torch.Tensor) -> float:
    """Compute mean MS-SSIM for a batch, or NaN when PIQ cannot run it."""

    if piq is None:
        return float("nan")

    fake = _to_01_batch(fake_images)
    real = _to_01_batch(real_images)
    if min(fake.shape[-2:]) < 32:
        return float("nan")

    try:
        value = piq.multi_scale_ssim(fake, real, data_range=1.0, reduction="mean")
    except Exception:
        return float("nan")
    return _finite_or_nan(float(value.item()))


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"mae": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "ms_ssim": float("nan")}
    keys = [key for key in rows[0].keys() if isinstance(rows[0][key], (int, float))]
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
    return summary


def image_feature_matrix(images: torch.Tensor, feature_size: int = 64) -> np.ndarray:
    """Downsample images and flatten them into deterministic pixel features.

    These features are intentionally simple. They make C2ST and precision/recall
    available without adding a hidden pretrained model dependency.
    """

    x = _to_01_batch(images)
    x = F.interpolate(x, size=(feature_size, feature_size), mode="area")
    return x.flatten(start_dim=1).numpy().astype(np.float32)


def _euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances without materializing NxMxD arrays."""

    with torch.no_grad():
        a_tensor = torch.from_numpy(a.astype(np.float32, copy=False))
        b_tensor = torch.from_numpy(b.astype(np.float32, copy=False))
        return torch.cdist(a_tensor, b_tensor).numpy()


def _balanced_sample(real: np.ndarray, fake: np.ndarray, max_items: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(real), len(fake), max_items)
    if n == 0:
        return real[:0], fake[:0]
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), size=n, replace=False) if len(real) > n else np.arange(len(real))
    fake_idx = rng.choice(len(fake), size=n, replace=False) if len(fake) > n else np.arange(len(fake))
    return real[real_idx], fake[fake_idx]


def c2st_1nn_accuracy(real_features: np.ndarray, fake_features: np.ndarray, max_items: int = 512, seed: int = 42) -> float:
    """Classifier two-sample test using leave-one-out 1-nearest-neighbour.

    Accuracy near 0.5 means generated and real samples are hard to separate in
    this feature space. Higher values mean the two distributions differ.
    """

    real, fake = _balanced_sample(real_features, fake_features, max_items=max_items, seed=seed)
    if len(real) < 2 or len(fake) < 2:
        return float("nan")

    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(len(real), dtype=np.int64), np.zeros(len(fake), dtype=np.int64)])
    distances = _euclidean_distances(x, x)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(y[nearest] == y))


def c2st_logistic_accuracy(real_features: np.ndarray, fake_features: np.ndarray, max_items: int = 512, seed: int = 42) -> float:
    """Classifier two-sample test with a small logistic classifier."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except Exception:  # pragma: no cover - optional dependency path
        return float("nan")

    real, fake = _balanced_sample(real_features, fake_features, max_items=max_items, seed=seed)
    if len(real) < 4 or len(fake) < 4:
        return float("nan")

    x = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(len(real), dtype=np.int64), np.zeros(len(fake), dtype=np.int64)])
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))
    except Exception:
        return float("nan")


def generative_precision_recall(real_features: np.ndarray, fake_features: np.ndarray, k: int = 3) -> dict[str, float]:
    """Approximate generative precision/recall with k-NN radii.

    Precision: generated samples that fall inside the real manifold.
    Recall: real samples covered by the generated manifold.
    """

    if len(real_features) <= k or len(fake_features) <= k:
        return {"precision": float("nan"), "recall": float("nan")}

    real = real_features.astype(np.float32)
    fake = fake_features.astype(np.float32)
    real_real = _euclidean_distances(real, real)
    fake_fake = _euclidean_distances(fake, fake)
    real_fake = _euclidean_distances(real, fake)

    np.fill_diagonal(real_real, np.inf)
    np.fill_diagonal(fake_fake, np.inf)
    real_radii = np.partition(real_real, kth=k - 1, axis=1)[:, k - 1]
    fake_radii = np.partition(fake_fake, kth=k - 1, axis=1)[:, k - 1]

    precision = np.mean(np.any(real_fake.T <= real_radii[None, :], axis=1))
    recall = np.mean(np.any(real_fake <= fake_radii[None, :], axis=1))
    return {"precision": float(precision), "recall": float(recall)}


def compute_distribution_metrics(
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    config: DistributionMetricConfig = DistributionMetricConfig(),
) -> dict[str, float]:
    """Compute lightweight dataset-level metrics from generated and real images."""

    real_features = image_feature_matrix(real_images, feature_size=config.feature_size)
    fake_features = image_feature_matrix(fake_images, feature_size=config.feature_size)
    real_features, fake_features = _balanced_sample(
        real_features,
        fake_features,
        max_items=config.max_items,
        seed=config.seed,
    )

    pr = generative_precision_recall(real_features, fake_features, k=config.precision_recall_k)
    return {
        "c2st_1nn_acc": c2st_1nn_accuracy(real_features, fake_features, max_items=config.max_items, seed=config.seed),
        "c2st_logistic_acc": c2st_logistic_accuracy(real_features, fake_features, max_items=config.max_items, seed=config.seed),
        "gen_precision": pr["precision"],
        "gen_recall": pr["recall"],
    }


def compute_lpips_mean(fake_images: torch.Tensor, real_images: torch.Tensor, device: torch.device | str = "cpu") -> float:
    """Compute mean LPIPS if the optional package and weights are available."""

    try:
        import lpips
    except Exception:  # pragma: no cover - optional dependency path
        return float("nan")

    try:
        metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        fake = fake_images.detach().float().to(device)
        real = real_images.detach().float().to(device)
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for fake_item, real_item in zip(fake, real):
                values.append(metric(fake_item.unsqueeze(0), real_item.unsqueeze(0)).flatten().cpu())
        return float(torch.cat(values).mean().item()) if values else float("nan")
    except Exception:
        return float("nan")


def compute_fid(fake_images: torch.Tensor, real_images: torch.Tensor, device: torch.device | str = "cpu") -> float:
    """Compute FID through torchmetrics when torch-fidelity is installed."""

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:  # pragma: no cover - optional dependency path
        return float("nan")

    try:
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        metric.update(_to_01_batch(real_images).to(device), real=True)
        metric.update(_to_01_batch(fake_images).to(device), real=False)
        return float(metric.compute().item())
    except Exception:
        return float("nan")


def clean_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Replace NumPy scalars and non-finite floats with JSON/CSV friendly values."""

    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = "nan"
        else:
            cleaned[key] = value
    return cleaned