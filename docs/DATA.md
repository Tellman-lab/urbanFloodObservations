# Data Contract

The training script expects paired PlanetScope image tiles and binary label
tiles grouped by event. Event folder names define the leave-one-event-out folds.

## Input Layout

```text
data/
  PS_256/
    <EVENT>/
      <CHIP_OR_TILE_NAME>.tif
  labels_256/
    <EVENT>/
      <CHIP_OR_TILE_NAME>.tif
```

Rules:

- `PS_256` and `labels_256` must contain the same event folder names.
- Within each event folder, image and label filenames must match exactly.
- PlanetScope tiles must be 4-band GeoTIFFs in Blue, Green, Red, NIR order.
- Label tiles must be single-band GeoTIFFs.
- Label values are binary: `0` for non-inundated and `1` for inundated.
- If labels contain any value greater than `0`, the default behavior treats it
  as class `1`. Use `--strict-label-binary` to require exactly `{0, 1}`.

## Tiling Full UFO Chips

UFOv2 source chips are 1024x1024 pixels. To create 256x256 training patches:

```bash
python scripts/tile_geotiffs_256.py \
  --ps-root data/raw/PS \
  --labels-root data/raw/labels \
  --out-ps-root data/PS_256 \
  --out-labels-root data/labels_256
```

The tiling script writes matching filenames for images and labels and preserves
the source georeferencing for each tile.

## Normalization

For each held-out event, normalization statistics are computed only from the
training events. The pipeline samples training pixels, computes per-band upper
clip values using the configured percentile, scales each band to `[0, 1]`, and
then standardizes by fold-specific mean and standard deviation.
