"""Gradio visual demo for Pix2Pix image-to-image inference.

Run from the project root:

    python demo/visual_demo.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    raise SystemExit("Gradio is required for demo/visual_demo.py. Install it with: pip install -r requirements.txt") from exc

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.infer import load_generator, load_input_label, preprocess_label  # noqa: E402
from src.utils import get_device, save_tensor_image  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "demo" / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
DEFAULT_PORT = 7860

MODEL_ALIASES = {
    "baseline": "Baseline Pix2Pix",
    "attention": "Attention Pix2Pix",
    "pix2pixhd": "Pix2PixHD-lite",
    "improved": "Improved / LSGAN",
    "lsgan": "Improved / LSGAN",
    "transfer": "Transfer model",
}

MODEL_CACHE: dict[tuple[Path, str], tuple[torch.nn.Module, dict[str, Any], torch.device]] = {}


def get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def load_manifest() -> list[dict[str, str]]:
    manifest_path = SAMPLES_DIR / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows = []
    for input_path in sorted((SAMPLES_DIR / "inputs").glob("*_label.png")):
        sample_id = input_path.stem.replace("_label", "")
        reference_path = SAMPLES_DIR / "references" / f"{sample_id}_real.png"
        rows.append(
            {
                "sample_id": sample_id,
                "input_label_path": project_relative(input_path),
                "reference_real_path": project_relative(reference_path) if reference_path.exists() else "",
            }
        )
    return rows


def model_label(path: Path) -> str:
    run_name = path.parents[1].name if path.parent.name == "checkpoints" else path.stem
    name = run_name.lower()
    for token, label in MODEL_ALIASES.items():
        if token in name:
            return f"{label} ({run_name})"
    return run_name


def discover_run_checkpoints() -> list[Path]:
    patterns = (
        "final_*_e50*/checkpoints/best_generator.pt",
        "final_*_e40*/checkpoints/best_generator.pt",
        "transfer*/checkpoints/best_generator.pt",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(RUNS_DIR.glob(pattern)))
    return candidates


def discover_checkpoints(extra_paths: list[str], include_runs: bool = True) -> dict[str, Path]:
    candidates = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend(sorted(MODELS_DIR.glob(pattern)))
    if include_runs:
        candidates.extend(discover_run_checkpoints())
    candidates.extend(Path(path) for path in extra_paths)

    choices: dict[str, Path] = {}
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        label = model_label(path)
        if label in choices:
            label = f"{label} [{project_relative(path)}]"
        choices[label] = path
    return choices


def sample_choices(samples: list[dict[str, str]]) -> list[str]:
    return ["Upload / drag image"] + [sample["sample_id"] for sample in samples]


def find_sample_by_id(samples: list[dict[str, str]], sample_id: str) -> dict[str, str] | None:
    return next((sample for sample in samples if sample.get("sample_id") == sample_id), None)


def find_sample_by_filename(samples: list[dict[str, str]], image_path: Path) -> dict[str, str] | None:
    filename = image_path.name
    stem = image_path.stem
    for sample in samples:
        input_name = Path(sample.get("input_label_path", "")).name
        if filename == input_name or stem == sample.get("sample_id") or stem == input_name.replace("_label.png", ""):
            return sample
    return None


def load_reference(reference_upload: str | None, sample: dict[str, str] | None) -> Image.Image | None:
    if reference_upload:
        with Image.open(reference_upload) as image:
            return image.convert("RGB")
    if sample and sample.get("reference_real_path"):
        reference_path = PROJECT_ROOT / sample["reference_real_path"]
        if reference_path.exists():
            with Image.open(reference_path) as image:
                return image.convert("RGB")
    return None


def load_cached_model(checkpoint_path: Path, device_name: str) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device = get_device(device_name)
    key = (checkpoint_path, str(device))
    if key not in MODEL_CACHE:
        generator, config = load_generator(checkpoint_path, device)
        MODEL_CACHE[key] = (generator, config, device)
    return MODEL_CACHE[key]


def run_generator(
    checkpoint_path: Path,
    device_name: str,
    image_path: Path,
    input_mode: str,
) -> tuple[Image.Image, Image.Image]:
    generator, config, device = load_cached_model(checkpoint_path, device_name)
    label_map = load_input_label(image_path, input_mode)
    original_size = label_map.size
    image_size = tuple(get_config_value(config, "image_size", [256, 256]))
    input_tensor = preprocess_label(label_map, image_size=image_size).to(device)

    with torch.no_grad():
        generated_tensor = generator(input_tensor).cpu()[0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    save_tensor_image(generated_tensor, temp_path)
    with Image.open(temp_path) as generated:
        generated_image = generated.convert("RGB").resize(original_size, resample=Image.Resampling.BICUBIC)
    temp_path.unlink(missing_ok=True)
    return label_map.convert("RGB"), generated_image


def build_predict_fn(checkpoints: dict[str, Path], samples: list[dict[str, str]]):
    def predict(
        checkpoint_label: str,
        sample_id: str,
        input_upload: str | None,
        reference_upload: str | None,
        input_mode: str,
        device_name: str,
    ) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]:
        if checkpoint_label not in checkpoints:
            raise gr.Error("Select a valid checkpoint.")

        selected_sample = None
        if sample_id and sample_id != "Upload / drag image":
            selected_sample = find_sample_by_id(samples, sample_id)
            if selected_sample is None:
                raise gr.Error("Selected sample was not found.")
            if input_mode != "label":
                raise gr.Error("Bundled samples are already label maps. Use input_mode='label' or upload a paired image.")
            image_path = PROJECT_ROOT / selected_sample["input_label_path"]
        elif input_upload:
            image_path = Path(input_upload)
            selected_sample = find_sample_by_filename(samples, image_path)
        else:
            raise gr.Error("Choose a sample or drag a label map / paired image.")

        label_map, generated = run_generator(checkpoints[checkpoint_label], device_name, image_path, input_mode)
        reference = load_reference(reference_upload, selected_sample)
        source = selected_sample["sample_id"] if selected_sample else image_path.name
        status = f"Generated {source} with {checkpoint_label}."
        if reference is None:
            status += " No reference image was provided or detected."
        return label_map, generated, reference, status

    return predict


def build_app(checkpoints: dict[str, Path], samples: list[dict[str, str]]) -> gr.Blocks:
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in models/. Pass one with --checkpoint.")

    default_checkpoint = next(iter(checkpoints.keys()))
    choices = sample_choices(samples)

    with gr.Blocks(title="Pix2Pix Visual Demo") as app:
        gr.Markdown(
            "# TU-Graz Landing Pix2Pix visual demo\n"
            "Erik Figueiral Alonso and Roi Cores Cabaleiro"
        )
        with gr.Row():
            checkpoint = gr.Dropdown(
                choices=list(checkpoints.keys()),
                value=default_checkpoint,
                label="Model checkpoint",
            )
            sample = gr.Dropdown(choices=choices, value=choices[0], label="Sample input")
            input_mode = gr.Radio(
                choices=["label", "paired-right", "paired-left"],
                value="label",
                label="Input mode",
            )
            device = gr.Radio(choices=["auto", "cpu", "cuda"], value="auto", label="Device")

        with gr.Row():
            input_upload = gr.Image(type="filepath", label="Drag label map or paired image")
            reference_upload = gr.Image(type="filepath", label="Optional reference image")

        run_button = gr.Button("Generate", variant="primary")

        with gr.Row():
            label_preview = gr.Image(type="pil", label="Label map used by the generator")
            generated = gr.Image(type="pil", label="Generated output")
            reference = gr.Image(type="pil", label="Reference")

        status = gr.Textbox(label="Status", interactive=False)
        run_button.click(
            fn=build_predict_fn(checkpoints, samples),
            inputs=[checkpoint, sample, input_upload, reference_upload, input_mode, device],
            outputs=[label_preview, generated, reference, status],
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio Pix2Pix visual demo.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Extra checkpoint path. Can be passed more than once.")
    parser.add_argument("--no-auto-runs", action="store_true", help="Only show models/ and explicit --checkpoint paths.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args.checkpoint, include_runs=not args.no_auto_runs)
    samples = load_manifest()
    app = build_app(checkpoints, samples)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
