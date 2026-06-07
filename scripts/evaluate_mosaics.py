#!/usr/bin/env python3
"""Evaluate mosaicked binary predictions against full-size UFO labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mosaicked UFO predictions")
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("outputs/ufo_segformer_b2_loeo/mosaics"),
        help="Mosaicked prediction root with <EVENT>/*.tif",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("data/raw/labels"),
        help="Full-size label root with <EVENT>/*.tif",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/ufo_segformer_b2_loeo/mosaic_metrics.csv"),
        help="CSV path for per-chip and summary metrics",
    )
    return parser


def align_to_label(pred: np.ndarray, pred_profile: dict, label_profile: dict, label_shape: tuple[int, int]) -> np.ndarray:
    if (
        pred.shape == label_shape
        and pred_profile.get("crs") == label_profile.get("crs")
        and pred_profile.get("transform") == label_profile.get("transform")
    ):
        return pred

    aligned = np.zeros(label_shape, dtype=pred.dtype)
    reproject(
        source=pred,
        destination=aligned,
        src_transform=pred_profile["transform"],
        src_crs=pred_profile["crs"],
        dst_transform=label_profile["transform"],
        dst_crs=label_profile["crs"],
        resampling=Resampling.nearest,
    )
    return aligned


def counts(pred: np.ndarray, true: np.ndarray) -> Dict[str, int]:
    pred = (pred.astype(np.int32) > 0).astype(np.uint8)
    true = (true.astype(np.int32) > 0).astype(np.uint8)
    return {
        "TP": int(np.logical_and(pred == 1, true == 1).sum()),
        "FP": int(np.logical_and(pred == 1, true == 0).sum()),
        "TN": int(np.logical_and(pred == 0, true == 0).sum()),
        "FN": int(np.logical_and(pred == 0, true == 1).sum()),
    }


def metrics(c: Dict[str, int]) -> Dict[str, float]:
    tp = float(c["TP"])
    fp = float(c["FP"])
    tn = float(c["TN"])
    fn = float(c["FN"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "IoU": iou,
        "F1": f1,
        "Accuracy": accuracy,
        "Balanced_Accuracy": 0.5 * (recall + specificity),
    }


def main() -> None:
    args = build_parser().parse_args()
    if not args.pred_root.exists():
        raise FileNotFoundError(f"Missing prediction root: {args.pred_root}")
    if not args.labels_root.exists():
        raise FileNotFoundError(f"Missing labels root: {args.labels_root}")

    rows: List[Dict[str, object]] = []
    total = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for event_dir in sorted(p for p in args.pred_root.iterdir() if p.is_dir()):
        label_dir = args.labels_root / event_dir.name
        if not label_dir.exists():
            print(f"Skipping {event_dir.name}: missing labels folder {label_dir}")
            continue

        for pred_path in sorted(event_dir.glob("*.tif")):
            label_path = label_dir / pred_path.name
            if not label_path.exists():
                print(f"Skipping {pred_path.name}: missing label {label_path}")
                continue

            with rio.open(pred_path) as pred_ds, rio.open(label_path) as label_ds:
                pred = pred_ds.read(1)
                true = label_ds.read(1)
                pred = align_to_label(pred, pred_ds.profile, label_ds.profile, true.shape)

            c = counts(pred, true)
            for key in total:
                total[key] += c[key]
            rows.append({"event": event_dir.name, "file": pred_path.name, **c, **metrics(c)})

    if not rows:
        raise RuntimeError("No matching prediction/label pairs were evaluated")

    summary = {"event": "ALL", "file": "MICRO_AVERAGE", **total, **metrics(total)}
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame([summary])], ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
