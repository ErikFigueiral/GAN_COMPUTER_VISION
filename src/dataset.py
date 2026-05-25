"""Dataset utilities for paired image-to-image translation data.

The expected default format is the one used by the project statement:
each file contains a real RGB image on the left half and the corresponding
semantic label map on the right half.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Iterable, Literal, Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PairedImageDatasetConfig:
    """Generic configuration for a paired image-to-image dataset.

    The code is dataset-agnostic. TU-Graz, Facades, Maps or any other paired
    dataset are selected by changing this configuration, not by changing model
    code.
    """

    name: str = "paired-image-dataset"
    data_root: str | Path = "data/raw"
    split_dir: str | Path = "data/splits"
    label_side: Literal["left", "right"] = "right"
    image_size: tuple[int, int] = (256, 256)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_seed: int = 42

    @property
    def split_config(self) -> "SplitConfig":
        return SplitConfig(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.split_seed,
        )


@dataclass(frozen=True)
class PairedImageSample:
    """One paired training example before tensor preprocessing."""

    path: Path
    real_image: Image.Image
    label_map: Image.Image


@dataclass(frozen=True)
class SplitConfig:
    """Configuration used to create deterministic train/validation/test splits."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    def validate(self) -> None:
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be lower than 1.")


class PairedResizeCropFlip:
    """Apply paired preprocessing to a label map and its target real image.

    The two images receive exactly the same geometric transforms. The target
    real image is resized with bicubic interpolation, while the label map is
    resized with nearest-neighbour interpolation to preserve discrete colours.
    Both outputs are converted to tensors in [-1, 1].
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        augment: bool = False,
        scale_size: tuple[int, int] | None = None,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size  # (height, width)
        self.augment = augment
        self.horizontal_flip_prob = horizontal_flip_prob
        if scale_size is None and augment:
            scale_size = (image_size[0] + 30, image_size[1] + 30)
        self.scale_size = scale_size or image_size

    def __call__(self, label_map: Image.Image, real_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        target_h, target_w = self.image_size
        scale_h, scale_w = self.scale_size

        label_map = label_map.resize((scale_w, scale_h), resample=Image.Resampling.NEAREST)
        real_image = real_image.resize((scale_w, scale_h), resample=Image.Resampling.BICUBIC)

        if self.augment:
            max_top = max(0, scale_h - target_h)
            max_left = max(0, scale_w - target_w)
            top = random.randint(0, max_top) if max_top > 0 else 0
            left = random.randint(0, max_left) if max_left > 0 else 0
            label_map = TF.crop(label_map, top, left, target_h, target_w)
            real_image = TF.crop(real_image, top, left, target_h, target_w)

            if random.random() < self.horizontal_flip_prob:
                label_map = TF.hflip(label_map)
                real_image = TF.hflip(real_image)
        else:
            if (scale_h, scale_w) != (target_h, target_w):
                label_map = label_map.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
                real_image = real_image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)

        label_tensor = pil_to_tensor(label_map) * 2.0 - 1.0
        real_tensor = pil_to_tensor(real_image) * 2.0 - 1.0
        return label_tensor, real_tensor



def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a CHW float tensor in [0, 1]."""

    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


class PairedImageReader(Iterator[PairedImageSample]):
    """Small reusable reader for browsing paired datasets.

    This class is intentionally independent from PyTorch. It is useful in the
    tutorial notebook and demo code when we want to inspect examples one by one:

    ```python
    reader = PairedImageReader.from_config(config)
    sample = next(reader)
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        relative_paths: Sequence[str | Path] | None = None,
        label_side: Literal["left", "right"] = "right",
    ) -> None:
        self.data_root = Path(data_root)
        self.relative_paths = [Path(p) for p in (relative_paths or list_image_files(self.data_root))]
        self.label_side = label_side
        self._index = 0

        if not self.relative_paths:
            raise ValueError(f"No paired images were found under: {self.data_root}")

    @classmethod
    def from_config(cls, config: PairedImageDatasetConfig, split: SplitName | None = None) -> "PairedImageReader":
        if split is None:
            return cls(data_root=config.data_root, label_side=config.label_side)

        splits = create_or_load_splits(
            data_root=config.data_root,
            split_dir=config.split_dir,
            config=config.split_config,
            force=False,
        )
        return cls(
            data_root=config.data_root,
            relative_paths=splits[split],
            label_side=config.label_side,
        )

    def __iter__(self) -> "PairedImageReader":
        return self

    def __next__(self) -> PairedImageSample:
        if self._index >= len(self.relative_paths):
            self._index = 0
            raise StopIteration
        sample = self.get(self._index)
        self._index += 1
        return sample

    def __len__(self) -> int:
        return len(self.relative_paths)

    def reset(self) -> None:
        self._index = 0

    def get(self, index: int) -> PairedImageSample:
        relative_path = self.relative_paths[index]
        path = self.data_root / relative_path

        with Image.open(path) as img:
            paired = img.convert("RGB")
            real_image, label_map = split_paired_image(paired, label_side=self.label_side)

        return PairedImageSample(
            path=relative_path,
            real_image=real_image.copy(),
            label_map=label_map.copy(),
        )


class PairedTranslationDataset(Dataset):
    """Dataset for paired image-to-image translation.

    Parameters
    ----------
    data_root:
        Root directory containing the paired images.
    relative_paths:
        List of image paths relative to ``data_root``.
    image_size:
        Output size as ``(height, width)``.
    augment:
        Whether to apply synchronized random crop and horizontal flip.
    label_side:
        Side where the label map is stored in the concatenated image.
    """

    def __init__(
        self,
        data_root: str | Path,
        relative_paths: Sequence[str | Path],
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
        scale_size: tuple[int, int] | None = None,
        label_side: Literal["left", "right"] = "right",
    ) -> None:
        self.data_root = Path(data_root)
        self.relative_paths = [Path(p) for p in relative_paths]
        self.label_side = label_side
        self.transform = PairedResizeCropFlip(
            image_size=image_size,
            augment=augment,
            scale_size=scale_size,
        )

        if not self.relative_paths:
            raise ValueError(f"No images were found for dataset split under: {self.data_root}")

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        relative_path = self.relative_paths[index]
        path = self.data_root / relative_path

        with Image.open(path) as img:
            paired = img.convert("RGB")

        real_image, label_map = split_paired_image(paired, label_side=self.label_side)
        label_tensor, real_tensor = self.transform(label_map=label_map, real_image=real_image)

        return {
            "label_map": label_tensor,
            "real_image": real_tensor,
            "path": str(relative_path).replace("\\", "/"),
        }


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(data_root: str | Path) -> list[Path]:
    """Return image files relative to ``data_root`` in deterministic order."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    files = [p.relative_to(root) for p in root.rglob("*") if p.is_file() and is_image_file(p)]
    files = [p for p in files if not any(part.startswith(".") for part in p.parts)]
    return sorted(files, key=lambda p: str(p).lower())


def split_paired_image(
    paired_image: Image.Image,
    label_side: Literal["left", "right"] = "right",
) -> tuple[Image.Image, Image.Image]:
    """Split a horizontally concatenated pair into ``(real_image, label_map)``."""

    width, height = paired_image.size
    if width < 2:
        raise ValueError("The paired image is too narrow to be split into two halves.")

    half_width = width // 2
    left = paired_image.crop((0, 0, half_width, height))
    right = paired_image.crop((half_width, 0, width, height))

    if label_side == "right":
        return left, right
    if label_side == "left":
        return right, left
    raise ValueError("label_side must be either 'left' or 'right'.")


def _write_split_file(paths: Iterable[Path], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p).replace("\\", "/") for p in paths]
    file_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_split_file(file_path: Path) -> list[Path]:
    return [Path(line.strip()) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_or_load_splits(
    data_root: str | Path,
    split_dir: str | Path,
    config: SplitConfig = SplitConfig(),
    force: bool = False,
) -> dict[SplitName, list[Path]]:
    """Create or load deterministic train/validation/test split files."""

    config.validate()
    split_dir = Path(split_dir)
    split_files = {
        "train": split_dir / "train.txt",
        "val": split_dir / "val.txt",
        "test": split_dir / "test.txt",
    }

    if not force and all(path.exists() for path in split_files.values()):
        return {split: _read_split_file(path) for split, path in split_files.items()}

    files = list_image_files(data_root)
    if len(files) < 3:
        raise ValueError("At least three paired images are required to create train/validation/test splits.")

    rng = random.Random(config.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * config.train_ratio))
    n_val = int(round(n_total * config.val_ratio))
    n_train = min(max(n_train, 1), n_total - 2)
    n_val = min(max(n_val, 1), n_total - n_train - 1)

    splits: dict[SplitName, list[Path]] = {
        "train": sorted(shuffled[:n_train], key=lambda p: str(p).lower()),
        "val": sorted(shuffled[n_train : n_train + n_val], key=lambda p: str(p).lower()),
        "test": sorted(shuffled[n_train + n_val :], key=lambda p: str(p).lower()),
    }

    for split, paths in splits.items():
        _write_split_file(paths, split_files[split])

    return splits


def build_datasets(
    data_root: str | Path,
    split_dir: str | Path,
    image_size: tuple[int, int] = (256, 256),
    split_config: SplitConfig = SplitConfig(),
    force_splits: bool = False,
    label_side: Literal["left", "right"] = "right",
    use_augmentation: bool = False,
    scale_size: tuple[int, int] | None = None,
) -> dict[SplitName, PairedTranslationDataset]:
    """Create datasets for all three splits."""

    splits = create_or_load_splits(
        data_root=data_root,
        split_dir=split_dir,
        config=split_config,
        force=force_splits,
    )

    return {
        "train": PairedTranslationDataset(
            data_root=data_root,
            relative_paths=splits["train"],
            image_size=image_size,
            augment=use_augmentation,
            scale_size=scale_size,
            label_side=label_side,
        ),
        "val": PairedTranslationDataset(
            data_root=data_root,
            relative_paths=splits["val"],
            image_size=image_size,
            augment=False,
            scale_size=None,
            label_side=label_side,
        ),
        "test": PairedTranslationDataset(
            data_root=data_root,
            relative_paths=splits["test"],
            image_size=image_size,
            augment=False,
            scale_size=None,
            label_side=label_side,
        ),
    }


def build_datasets_from_config(
    config: PairedImageDatasetConfig,
    force_splits: bool = False,
    use_augmentation: bool = False,
    scale_size: tuple[int, int] | None = None,
) -> dict[SplitName, PairedTranslationDataset]:
    """Create train/validation/test datasets from one generic dataset config."""

    return build_datasets(
        data_root=config.data_root,
        split_dir=config.split_dir,
        image_size=config.image_size,
        split_config=config.split_config,
        force_splits=force_splits,
        label_side=config.label_side,
        use_augmentation=use_augmentation,
        scale_size=scale_size,
    )


def build_dataloaders(
    datasets: dict[SplitName, PairedTranslationDataset],
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict[SplitName, DataLoader]:
    """Build dataloaders for train, validation and test splits."""

    pin_memory = torch.cuda.is_available()
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }
