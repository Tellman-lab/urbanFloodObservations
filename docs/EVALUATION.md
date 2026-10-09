# Verified archived B2 predictions

The B2-named prediction mosaics and evaluation CSV supplied for the UFO revision
were verified on 9 October 2026 against all 215 local UFO image/label pairs.

## Artifacts

- [Prediction ZIP](https://drive.google.com/file/d/1FpjzPr2eXEQhPJlDl5M16nOlFQ9FVKeW/view):
  `PS_256_localNorm_segformer_b2_Preds_ProcessedMosaics.zip`.
- [Original evaluation CSV](https://drive.google.com/file/d/1AerMLVjzH_etOor4oyKyz8ekDjIm_d-2/view):
  `PS_256_localNorm_segformer_b2_Evaluation_mosaics.csv`.
- [CSV copy in this repository](../results/segformer_b2_archived_mosaic_metrics.csv).
- [Verification summary](../results/segformer_b2_artifact_verification.json).
- [Checkpoint and Hub package](PRETRAINED.md).

The ZIP contains 215 TIFFs across 14 event folders. The CSV contains 215 image
rows, an `OVERALL_MEAN` row and a totals row. There are no missing filenames,
extra image rows or duplicate event/file keys.

All predictions are single-band 1024 x 1024 binary GeoTIFFs with values 0 and 1.
For every image, event membership, filename, dimensions, CRS and affine
transform match the corresponding local image and reference label. TP, FP, TN
and FN recomputed from the masks match the CSV exactly for all 215 images.
Precision, Recall, IoU, F1 and Accuracy agree within `1e-6` rounding tolerance.
The overall mean and totals rows are also consistent with the image rows.

## Water IoU

| Aggregation | Definition | Verified score |
| --- | --- | ---: |
| Mean per image | Arithmetic mean of the 215 image water IoUs; CSV `OVERALL_MEAN` | 77.340578% |
| Pooled pixels | Sum TP divided by sum TP + FP + FN | 85.081899% |
| Mean per event | Arithmetic mean of the 14 within-event mean water IoUs | 73.937440% |

These are different aggregations of the same archived masks. The 77.34% value
is the mean water IoU across images; it is not a mean over both semantic classes.

Total confusion counts:

| TP | FP | TN | FN |
| ---: | ---: | ---: | ---: |
| 74,159,184 | 7,193,473 | 138,281,723 | 5,809,460 |

Event counts: BEI 2, BNA 5, CMO 14, CTO 7, DKA 12, GIL 20, HTX 39,
KTM 23, MID 4, NSW 27, PNE 13, QUE 12, SLC 6, SPS 31.

## Scope of verification

The audit establishes that the ZIP, CSV and reference labels describe the same
215 images and that the reported counts and scores reproduce from the supplied
masks. It does not verify the training/test split, the held-out checkpoint used
for each event, or linkage to the separately supplied `segformer_b2_best.pt`.
Do not interpret these scores as a newly measured evaluation of that individual
checkpoint. The repository's current four-band training code and the archived
three-channel checkpoint have different input contracts.
