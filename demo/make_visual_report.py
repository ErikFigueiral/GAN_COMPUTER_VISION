"""Generate a lightweight Pix2Pix visual demo report.

The report is static and delivery-friendly: it writes generated images,
comparison PNGs and an index.html page using only project dependencies.

Example
-------
python demo/make_visual_report.py --checkpoint models/improved_best_generator.pt
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.infer import load_generator, load_input_label, preprocess_label  # noqa: E402
from src.utils import get_device, save_tensor_image  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "demo" / "samples"
DEFAULT_OUTPUT_DIR = SAMPLES_DIR / "outputs" / "visual_report"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "improved_best_generator.pt"


def get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config and config[key] is not None else default


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def load_samples(manifest_path: Path) -> list[dict[str, str]]:
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [row for row in rows if row.get("input_label_path")]

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


def resize_for_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#eef1f5")
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_comparison_png(
    label_path: Path,
    generated_path: Path,
    reference_path: Path | None,
    output_path: Path,
    title: str,
) -> None:
    panel_size = (420, 280)
    title_height = 44
    label_height = 30
    margin = 18
    gap = 14
    columns = [("Label map", label_path), ("Generated", generated_path)]
    if reference_path and reference_path.exists():
        columns.append(("Reference", reference_path))

    width = margin * 2 + len(columns) * panel_size[0] + (len(columns) - 1) * gap
    height = margin * 2 + title_height + label_height + panel_size[1]
    canvas = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, margin), title, fill="#17202a", font=font)
    y_label = margin + title_height
    y_image = y_label + label_height

    for index, (label, path) in enumerate(columns):
        x = margin + index * (panel_size[0] + gap)
        draw.text((x, y_label), label, fill="#17202a", font=font)
        with Image.open(path) as img:
            panel = resize_for_panel(img, panel_size)
        canvas.paste(panel, (x, y_image))
        draw.rectangle((x, y_image, x + panel_size[0] - 1, y_image + panel_size[1] - 1), outline="#d9dee7")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_inference(
    generator: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    input_path: Path,
    output_path: Path,
) -> None:
    image_size = tuple(get_config_value(config, "image_size", [256, 256]))
    label_map = load_input_label(input_path, "label")
    original_size = label_map.size
    input_tensor = preprocess_label(label_map, image_size=image_size).to(device)

    with torch.no_grad():
        fake_image = generator(input_tensor).cpu()[0]

    save_tensor_image(fake_image, output_path)
    with Image.open(output_path) as generated:
        generated.resize(original_size, resample=Image.Resampling.BICUBIC).save(output_path)


def write_html_report(
    rows: list[dict[str, str]],
    output_dir: Path,
    checkpoint: Path,
    device: torch.device,
) -> Path:
    cards = []
    for row in rows:
        reference_html = ""
        if row.get("reference_rel"):
            reference_html = f'<a href="{html.escape(row["reference_rel"])}">reference</a>'
        cards.append(
            f"""
            <article>
              <h2>{html.escape(row['sample_id'])}</h2>
              <img src="{html.escape(row['comparison_rel'])}" alt="Comparison for {html.escape(row['sample_id'])}">
              <p>
                <a href="{html.escape(row['input_rel'])}">label map</a>
                <a href="{html.escape(row['generated_rel'])}">generated</a>
                {reference_html}
              </p>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pix2Pix visual report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #17202a; background: #f6f7f9; }}
    header, main, footer {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }}
    header {{ padding: 32px 0 16px; }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    p {{ color: #5d6978; }}
    main {{ display: grid; gap: 16px; padding-bottom: 24px; }}
    article {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d9dee7; border-radius: 6px; background: #eef1f5; }}
    a {{ color: #0f766e; font-weight: 700; margin-right: 12px; }}
    footer {{ padding: 0 0 30px; color: #5d6978; font-size: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>TU-Graz Landing Pix2Pix visual report</h1>
    <p>Erik Figueiral Alonso - Roi Cores Cabaleiro</p>
    <p>Checkpoint: {html.escape(project_relative(checkpoint))} - Device: {html.escape(str(device))}</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
  <footer>Generated with demo/make_visual_report.py. No extra dependencies required.</footer>
</body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(document, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static PNG/HTML comparisons for the Pix2Pix demo samples.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Generator checkpoint .pt file.")
    parser.add_argument("--manifest", type=Path, default=SAMPLES_DIR / "manifest.csv", help="Demo samples manifest CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated report files.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu or cuda.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples. 0 means all samples.")
    parser.add_argument("--force", action="store_true", help="Regenerate images even when cached outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    samples = load_samples(args.manifest)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not samples:
        raise FileNotFoundError("No demo samples found. Expected demo/samples/manifest.csv or demo/samples/inputs/*_label.png")

    device = get_device(args.device)
    generator, config = load_generator(checkpoint, device)
    checkpoint_name = checkpoint.stem.replace("_best_generator", "")
    generated_dir = output_dir / "generated" / checkpoint_name
    comparison_dir = output_dir / "comparisons" / checkpoint_name
    report_rows = []

    for sample in samples:
        sample_id = sample.get("sample_id") or Path(sample["input_label_path"]).stem
        input_path = (PROJECT_ROOT / sample["input_label_path"]).resolve()
        reference_text = sample.get("reference_real_path", "")
        reference_path = (PROJECT_ROOT / reference_text).resolve() if reference_text else None
        generated_path = generated_dir / f"{sample_id}_generated.png"
        comparison_path = comparison_dir / f"{sample_id}_comparison.png"

        if args.force or not generated_path.exists():
            run_inference(generator, config, device, input_path, generated_path)
        if args.force or not comparison_path.exists():
            make_comparison_png(input_path, generated_path, reference_path, comparison_path, f"{sample_id} - {checkpoint.name}")

        report_rows.append(
            {
                "sample_id": sample_id,
                "input_rel": project_relative(input_path),
                "generated_rel": project_relative(generated_path),
                "reference_rel": project_relative(reference_path) if reference_path and reference_path.exists() else "",
                "comparison_rel": project_relative(comparison_path),
            }
        )
        print(f"Generated comparison for {sample_id}: {project_relative(comparison_path)}")

    html_path = write_html_report(report_rows, output_dir, checkpoint, device)
    print(f"Visual report saved to: {project_relative(html_path)}")


if __name__ == "__main__":
    main()
