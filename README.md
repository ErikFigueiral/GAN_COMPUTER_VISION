# CV2 GAN Assignment Submission

Authors: Erik Figueiral Alonso and Roi Cores Cabaleiro

This package contains a conditional GAN project for semantic label-to-image synthesis on the TU-Graz Landing dataset. The input is a semantic label map and the output is an aerial RGB image generated with Pix2Pix-style models.

## Main Deliverables

- `src/`: clean Python implementation for dataset handling, models, losses, training, evaluation, and metric chain.
- `demo/visual_demo.py`: Gradio drag-and-drop demo with automatic checkpoint discovery.
- `notebooks/00_tutorial_entrega_pix2pix.ipynb`: tutorial notebook.
- `notebooks/00_tutorial_entrega_pix2pix.executed.ipynb`: executed tutorial notebook.
- `report/main.tex`: IEEE-style report with inline references.
- `report/figures/`: qualitative grids and result figures used by the report.
- `requirements.txt`: Python dependencies.
- `outputs/runs/*/checkpoints/best_generator.pt`: selected best checkpoints for trained models.

## Final Package Status

The Windows submission folder is:

```text
C:\Users\erikf\OneDrive\Escritorio\CV2_ASSIGMENT_GAN
```

It contains the Python code, executed notebook, LaTeX report, report figures, requirements, demo code, and the best selected checkpoint for each main model family.

Selected best checkpoints:

- Baseline Pix2Pix: `outputs/runs/final_baseline_bce_l1_e40/checkpoints/best_generator.pt`
- LSGAN: `outputs/runs/final_lsgan_l1_100_e40/checkpoints/best_generator.pt`
- Self-attention Pix2Pix: `outputs/runs/final_attention_lsgan_e40/checkpoints/best_generator.pt`
- Pix2PixHD-lite: `outputs/runs/final_pix2pixhd_lite_ms2_fm10_e40/checkpoints/best_generator.pt`
- Transfer learning ResNet18: `outputs/runs/final_transfer_resnet18_e50/checkpoints/best_generator.pt`

Best completed model by validation MAE:

```text
final_pix2pixhd_lite_ms2_fm10_e40
MAE 0.1335, PSNR 15.33, SSIM 0.303
```

## Setup

```bash
cd /home/erik/cv2-image-to-image
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the virtual environment already exists:

```bash
cd /home/erik/cv2-image-to-image
source .venv/bin/activate
```

## Run the Demo

The demo auto-discovers final checkpoints from `models/` and `outputs/runs/final_*`.

```bash
python demo/visual_demo.py --host 0.0.0.0 --port 7860
```

Open:

```text
http://localhost:7860
```

Use `paired-right` for the original TU-Graz paired images, where the right half is the semantic label map. Use `label` only when uploading a standalone label map. The optional reference image is only displayed for comparison and is not used by the generator.

## Run the Tutorial Notebook

```bash
jupyter notebook notebooks/00_tutorial_entrega_pix2pix.ipynb
```

The executed version included in the package is:

```text
notebooks/00_tutorial_entrega_pix2pix.executed.ipynb
```

## Train Models

Examples:

```bash
python run_experiment.py baseline
python run_experiment.py lsgan
python run_experiment.py attention
python run_experiment.py pix2pixhd_lite
python -m src.train_transfer --help
```

Every run saves its best generator in its own folder:

```text
outputs/runs/<run_name>/checkpoints/best_generator.pt
```

This avoids overwriting checkpoints between variants.

## Report Compilation

The report is self-contained and uses inline `thebibliography`, so no external `.bib` file is required.

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

If using Overleaf, upload `main.tex` and the complete `figures/` folder. Recompile from scratch if Overleaf keeps old BibTeX warnings.

## Current Best Completed Result

The best completed run by validation MAE is:

```text
final_pix2pixhd_lite_ms2_fm10_e40
```

Its best checkpoint is:

```text
outputs/runs/final_pix2pixhd_lite_ms2_fm10_e40/checkpoints/best_generator.pt
```
