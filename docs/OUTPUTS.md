# Output Contract

The training script writes all generated artifacts under `--output-dir`.

```text
outputs/ufo_segformer_b2_loeo/
  config.json
  fold_summary.csv
  patch_metrics_all.csv
  overall_summary.json
  pipeline.log
  mosaics/<EVENT>/*.tif
  mosaic_metrics.csv
  folds/<EVENT>/
    best_model.pt
    normalization_stats.json
    train_history.csv
    val_metrics_best.json
    test_fold_metrics.json
    test_patch_metrics.csv
    predictions/
      <EVENT>/*.tif
```

## Main Files

- `config.json`: command-line arguments captured at run time.
- `fold_summary.csv`: one row per completed held-out event.
- `patch_metrics_all.csv`: per-tile metrics across all completed folds.
- `overall_summary.json`: micro-averaged counts and macro fold summaries.
- `pipeline.log`: full run log.
- `mosaics/<EVENT>/*.tif`: full-chip predictions created by
  `scripts/mosaic_predictions.py`.
- `mosaic_metrics.csv`: full-chip metrics created by
  `scripts/evaluate_mosaics.py`.

## Per-Fold Files

- `best_model.pt`: checkpoint selected by validation IoU.
- `normalization_stats.json`: fold-specific clip, mean, and standard deviation.
- `train_history.csv`: epoch-level training and validation history.
- `val_metrics_best.json`: validation metrics for the selected checkpoint.
- `test_fold_metrics.json`: aggregate metrics on the held-out event.
- `test_patch_metrics.csv`: per-tile metrics on the held-out event.
- `predictions/*.tif`: predicted class masks when `--save-predictions` is used.

All large outputs are ignored by Git. Publish checkpoints or prediction rasters
through a release or archival data repository rather than committing them.
