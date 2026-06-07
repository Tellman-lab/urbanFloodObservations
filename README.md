# UFO SegFormer-B2

Code for training and evaluating the SegFormer-B2 urban inundation segmentation
model described for Urban Flood Observations (UFOv2).

The repository is intentionally code-only. UFOv2 data, model checkpoints,
prediction rasters, and other large artifacts should stay outside Git and be
referenced through data releases such as Zenodo.

## Model

The model is SegFormer-B2 initialized from Hugging Face pretrained weights and
adapted from three input channels to four PlanetScope bands:

1. Blue
2. Green
3. Red
4. Near infrared

The model predicts a binary semantic segmentation mask:

- `0`: non-inundated
- `1`: inundated visible surface water

The first patch-embedding layer is modified for four input channels. The RGB
pretrained filters are mapped to the UFO band order, and the NIR channel is
initialized from the red filter weights.

## Expected Data

Training expects paired 256x256 GeoTIFF tiles grouped by flood event:

```text
data/
  PS_256/
    BEI/*.tif
    BNA/*.tif
    ...
  labels_256/
    BEI/*.tif
    BNA/*.tif
    ...
```

Each file in `PS_256/<EVENT>/` must have the same filename as its label in
`labels_256/<EVENT>/`. UFOv2 source chips are 1024x1024 pixels; use the tiling
helper if starting from full chips.

See [docs/DATA.md](docs/DATA.md) for the full data contract.

## Install

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA setup if the default wheel is
not appropriate for your machine.

## Prepare Tiles

If your UFOv2 data are still 1024x1024 chips:

```bash
python scripts/tile_geotiffs_256.py \
  --ps-root data/raw/PS \
  --labels-root data/raw/labels \
  --out-ps-root data/PS_256 \
  --out-labels-root data/labels_256
```

## Train And Evaluate

Run leave-one-event-out cross-validation with the paper settings:

```bash
python train_segformer_b2_loeo.py \
  --ps-root data/PS_256 \
  --labels-root data/labels_256 \
  --output-dir outputs/ufo_segformer_b2_loeo \
  --model-variant segformer_b2 \
  --epochs 30 \
  --batch-size 8 \
  --lr 6e-5 \
  --weight-decay 1e-2 \
  --valid-pct 0.10 \
  --gpu 0 \
  --num-workers 8 \
  --use-tta \
  --save-predictions
```

The first run downloads the SegFormer checkpoint from Hugging Face.

## Outputs

The pipeline writes:

```text
outputs/ufo_segformer_b2_loeo/
  config.json
  fold_summary.csv
  patch_metrics_all.csv
  overall_summary.json
  pipeline.log
  folds/<EVENT>/
    best_model.pt
    normalization_stats.json
    train_history.csv
    val_metrics_best.json
    test_fold_metrics.json
    test_patch_metrics.csv
    predictions/*.tif
```

See [docs/OUTPUTS.md](docs/OUTPUTS.md) for details.

## Mosaic And Full-Chip Evaluation

When `--save-predictions` is used, the trainer saves held-out 256x256
prediction tiles. Mosaic them back to full chips:

```bash
python scripts/mosaic_predictions.py \
  --loeo-output outputs/ufo_segformer_b2_loeo
```

Evaluate the mosaics against full-size UFO labels:

```bash
python scripts/evaluate_mosaics.py \
  --pred-root outputs/ufo_segformer_b2_loeo/mosaics \
  --labels-root data/raw/labels \
  --output-csv outputs/ufo_segformer_b2_loeo/mosaic_metrics.csv
```

## Data Availability

The UFOv2 dataset is archived on Zenodo under CC BY 4.0:
`10.5281/zenodo.19698577`.

## Publication Checklist

Before making the GitHub repository public:

- Confirm the repository name and GitHub organization.
- Add the final code license selected by the project team.
- Confirm the Zenodo DOI and paper citation text.
- Optionally add trained model weights to a GitHub Release or Zenodo, not Git.
