"""General utilities for training, evaluation and visualization."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def denormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor from [-1, 1] to [0, 1]."""

    return ((x.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """Convert a CHW or BCHW tensor in [-1, 1] or [0, 1] to a HWC float image in [0, 1]."""

    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError("Batched input must contain exactly one image.")
        x = x[0]
    if x.min() < 0:
        x = denormalize_tensor(x)
    else:
        x = x.detach().float().cpu().clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    """Save a CHW or 1xCHW tensor image to disk."""

    from PIL import Image

    image = (tensor_to_numpy_image(x) * 255.0).round().astype(np.uint8)
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(image).save(path)


def save_comparison_grid(
    label_maps: torch.Tensor,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_path: str | Path,
    max_items: int = 4,
    title: str | None = None,
) -> None:
    """Save a visual grid with label map, generated image and target image."""

    n_items = min(max_items, label_maps.shape[0], real_images.shape[0], fake_images.shape[0])
    if n_items <= 0:
        return

    fig_height = 3.2 * n_items
    fig, axes = plt.subplots(n_items, 3, figsize=(10, fig_height), squeeze=False)
    column_titles = ["Input label map", "Generated image", "Target image"]

    for row in range(n_items):
        images = [label_maps[row], fake_images[row], real_images[row]]
        for col, image_tensor in enumerate(images):
            axes[row, col].imshow(tensor_to_numpy_image(image_tensor))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(column_titles[col])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def append_csv_row(path: str | Path, row: dict[str, Any], fieldnames: Iterable[str] | None = None) -> None:
    """Append a row to a CSV file, creating it with a header if necessary."""

    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(row.keys())
    fieldnames = list(fieldnames)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_reproducible_torch() -> None:
    """Set deterministic-ish flags without disabling CUDA performance completely."""

    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTHONHASHSEED", "0")
