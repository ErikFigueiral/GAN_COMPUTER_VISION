"""Inference-only demo for trained Pix2Pix generators.

Examples
--------
From the project root:

    python demo/infer.py --checkpoint outputs/runs/baseline/checkpoints/best_generator.pt --input demo/sample_inputs/example.png --output demo/output.png

If the input is one of the original paired images, use:

    python demo/infer.py --checkpoint outputs/runs/baseline/checkpoints/best_generator.pt --input data/raw/example.png --input-mode paired-right --output demo/output.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import numpy as np

# Allow running this script from either the project root or the demo directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import split_paired_image  # noqa: E402
from src.models import UNetGenerator  # noqa: E402
from src.models_attention import AttentionUNetGenerator  # noqa: E402
from src.models_pix2pixhd import GlobalResnetGenerator  # noqa: E402
from src.models_transfer import ResNetUNetGenerator  # noqa: E402
from src.utils import get_device, save_tensor_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference on a semantic label map.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint .pt file.")
    parser.add_argument("--input", type=str, required=True, help="Input label map or paired image.")
    parser.add_argument("--output", type=str, required=True, help="Output generated RGB image path.")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["label", "paired-right", "paired-left"],
        default="label",
        help="Use 'label' for a standalone label map. Use 'paired-right' if the label map is on the right half of a paired image.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--keep-input-size", action="store_true", help="Resize the generated output back to the original input label-map size.")
    return parser.parse_args()


def get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def load_generator(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    architecture = get_config_value(config, "architecture", "pix2pix")
    if architecture == "self_attention_pix2pix":
        generator = AttentionUNetGenerator(
            in_channels=3,
            out_channels=3,
            ngf=int(get_config_value(config, "ngf", 64)),
            num_downs=int(get_config_value(config, "num_downs", 7)),
            norm=get_config_value(config, "norm", "batch"),
            dropout=float(get_config_value(config, "dropout", 0.5)),
        ).to(device)
    elif architecture == "pix2pixhd_lite":
        generator = GlobalResnetGenerator(
            in_channels=3,
            out_channels=3,
            ngf=int(get_config_value(config, "ngf", 48)),
            n_downsample=int(get_config_value(config, "n_downsample", 3)),
            n_blocks=int(get_config_value(config, "n_blocks", 6)),
            norm=get_config_value(config, "norm", "instance"),
        ).to(device)
    elif architecture == "transfer_resnet_unet" or "resnet_name" in config:
        generator = ResNetUNetGenerator(
            out_channels=3,
            resnet_name=get_config_value(config, "resnet_name", "resnet18"),
            pretrained=False,
            weights_path=None,
            norm=get_config_value(config, "norm", "batch"),
            dropout=float(get_config_value(config, "dropout", 0.5)),
        ).to(device)
    else:
        generator = UNetGenerator(
            in_channels=3,
            out_channels=3,
            ngf=int(get_config_value(config, "ngf", 64)),
            num_downs=int(get_config_value(config, "num_downs", 7)),
            norm=get_config_value(config, "norm", "batch"),
            dropout=float(get_config_value(config, "dropout", 0.5)),
        ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, config


def load_input_label(path: str | Path, input_mode: str) -> Image.Image:
    with Image.open(path) as img:
        image = img.convert("RGB")

    if input_mode == "label":
        return image
    if input_mode == "paired-right":
        _, label_map = split_paired_image(image, label_side="right")
        return label_map
    if input_mode == "paired-left":
        _, label_map = split_paired_image(image, label_side="left")
        return label_map
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def preprocess_label(label_map: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    resized = label_map.resize((width, height), resample=Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float() * 2.0 - 1.0
    return tensor.unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    generator, config = load_generator(args.checkpoint, device)
    image_size = tuple(get_config_value(config, "image_size", [256, 256]))

    label_map = load_input_label(args.input, args.input_mode)
    original_size = label_map.size
    input_tensor = preprocess_label(label_map, image_size=image_size).to(device)

    with torch.no_grad():
        fake_image = generator(input_tensor).cpu()[0]

    output_path = Path(args.output)
    save_tensor_image(fake_image, output_path)

    if args.keep_input_size:
        with Image.open(output_path) as generated:
            generated = generated.resize(original_size, resample=Image.Resampling.BICUBIC)
            generated.save(output_path)

    print(f"Saved generated image to: {output_path}")


if __name__ == "__main__":
    main()
