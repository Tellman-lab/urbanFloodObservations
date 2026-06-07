#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-one-event-out SegFormer-B2 training pipeline for Urban Flood Observations
(UFOv2).

Dataset layout expected:
    data/
      PS_256/
        <EVENT_NAME>/*.tif
      labels_256/
        <EVENT_NAME>/*.tif

Key behavior:
- Outer loop: leave one event/folder out as the held-out test fold.
- Inner split: small random validation split inside the remaining training events.
- Input: 4-band Blue/Green/Red/NIR uint16 patches.
- Labels: 1-band binary masks with values 0/1 (or values >0 treated as 1).
- Patch-level evaluation only.
- Fold-specific normalization statistics computed from training events only:
    * clip each band to [0, p99.5]
    * scale to [0, 1]
    * standardize with training-fold mean/std
- Geometric augmentation only (flip/rot90).
- Model: SegFormer-B2 with 4-channel input.

Outputs:
<output_dir>/
  config.json
  folds/<EVENT>/
    best_model.pt
    normalization_stats.json
    train_history.csv
    val_metrics_best.json
    test_fold_metrics.json
    test_patch_metrics.csv
    predictions/*.tif   (if enabled)
  fold_summary.csv
  patch_metrics_all.csv
  overall_summary.json
  pipeline.log

Example:
python train_segformer_b2_loeo.py \
    --ps-root data/PS_256 \
    --labels-root data/labels_256 \
    --output-dir outputs/ufo_segformer_b2_loeo \
    --gpu 0 \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 8 \
    --use-tta
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class PairRecord:
    event: str
    filename: str
    ps_path: Path
    label_path: Path


# =============================================================================
# ARGPARSE / CONFIG
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOEO SegFormer-B2 training for UFOv2")

    parser.add_argument("--ps-root", type=str, default="data/PS_256", help="Path to 256x256 PlanetScope event folders")
    parser.add_argument("--labels-root", type=str, default="data/labels_256", help="Path to 256x256 label event folders")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ufo_segformer_b2_loeo",
        help="Directory for all outputs",
    )

    parser.add_argument("--model-variant", type=str, default="segformer_b2",
                        choices=["segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--valid-pct", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on validation IoU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--sample-stats-files", type=int, default=2500,
                        help="Max number of training files sampled per fold for normalization stats")
    parser.add_argument("--sample-stats-pixels", type=int, default=2048,
                        help="Pixels sampled per band per file for normalization stats")
    parser.add_argument("--upper-percentile", type=float, default=99.5,
                        help="Upper clip percentile per band, computed on training files only")

    parser.add_argument("--save-predictions", action="store_true", help="Save held-out patch predictions as GeoTIFFs")
    parser.add_argument("--save-probability", action="store_true",
                        help="Also save held-out class-1 probability rasters as float32 GeoTIFFs")
    parser.add_argument("--use-tta", action="store_true", help="Use scale and flip TTA for held-out evaluation")
    parser.add_argument(
        "--tta-scales",
        type=str,
        default="0.75,1.0,1.25",
        help="Comma-separated TTA scales used when --use-tta is enabled",
    )
    parser.add_argument(
        "--only-events",
        type=str,
        default="",
        help="Comma-separated subset of held-out events to run, e.g. 'UFO_HTX,SW'",
    )
    parser.add_argument("--strict-label-binary", action="store_true",
                        help="Raise if a label contains values outside {0,1} instead of binarizing >0")
    parser.add_argument("--disable-amp", action="store_true", help="Disable automatic mixed precision")

    return parser


# =============================================================================
# LOGGING / UTILITIES
# =============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_fp = output_dir / "pipeline.log"

    logger = logging.getLogger("ufo_segformer_b2_loeo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_fp)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(gpu: int, logger: logging.Logger) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if gpu < 0 or gpu >= n:
            logger.warning(f"Requested GPU {gpu} is unavailable; using GPU 0 instead.")
            gpu = 0
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using CUDA:{gpu} | {torch.cuda.get_device_name(gpu)}")
        return torch.device(f"cuda:{gpu}")

    logger.warning("CUDA not available; using CPU.")
    return torch.device("cpu")


def json_dump(obj: dict, fp: Path) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_tta_scales(value: str) -> List[float]:
    scales = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not scales:
        raise ValueError("--tta-scales must contain at least one numeric scale")
    if any(s <= 0 for s in scales):
        raise ValueError(f"--tta-scales must be positive, got: {scales}")
    return scales


# =============================================================================
# DATASET DISCOVERY / VALIDATION
# =============================================================================
def list_event_dirs(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    event_dirs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    if not event_dirs:
        raise ValueError(f"No event directories found under: {root}")
    return event_dirs


def build_event_pairs(ps_root: Path, labels_root: Path, logger: logging.Logger) -> Dict[str, List[PairRecord]]:
    ps_events = list_event_dirs(ps_root)
    label_events = list_event_dirs(labels_root)

    ps_names = set(ps_events.keys())
    label_names = set(label_events.keys())

    if ps_names != label_names:
        only_ps = sorted(ps_names - label_names)
        only_labels = sorted(label_names - ps_names)
        raise ValueError(
            "Event folder mismatch between PS and labels. "
            f"Only in PS: {only_ps} | Only in labels: {only_labels}"
        )

    event_pairs: Dict[str, List[PairRecord]] = {}
    for event in sorted(ps_names):
        ps_files = sorted(ps_events[event].glob("*.tif"))
        label_files = sorted(label_events[event].glob("*.tif"))

        ps_map = {p.name: p for p in ps_files}
        label_map = {p.name: p for p in label_files}

        if set(ps_map.keys()) != set(label_map.keys()):
            only_ps = sorted(set(ps_map.keys()) - set(label_map.keys()))
            only_labels = sorted(set(label_map.keys()) - set(ps_map.keys()))
            raise ValueError(
                f"Filename mismatch in event '{event}'. "
                f"Only in PS: {only_ps[:10]} | Only in labels: {only_labels[:10]}"
            )

        pairs = [
            PairRecord(event=event, filename=name, ps_path=ps_map[name], label_path=label_map[name])
            for name in sorted(ps_map.keys())
        ]

        if not pairs:
            raise ValueError(f"No paired TIFFs found in event: {event}")

        event_pairs[event] = pairs
        logger.info(f"Discovered event {event}: {len(pairs)} paired patches")

    total_pairs = sum(len(v) for v in event_pairs.values())
    logger.info(f"Total events: {len(event_pairs)} | Total paired patches: {total_pairs}")
    return event_pairs


def split_train_val(
    pairs: Sequence[PairRecord],
    valid_pct: float,
    seed: int,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    if not 0.0 < valid_pct < 1.0:
        raise ValueError(f"valid_pct must be between 0 and 1, got {valid_pct}")
    if len(pairs) < 2:
        raise ValueError("Need at least 2 training pairs to create a validation split")

    idx = np.arange(len(pairs))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_valid = max(1, int(round(len(pairs) * valid_pct)))
    n_valid = min(n_valid, len(pairs) - 1)

    valid_idx = set(idx[:n_valid].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in valid_idx]
    valid_pairs = [pairs[i] for i in range(len(pairs)) if i in valid_idx]
    return train_pairs, valid_pairs


# =============================================================================
# NORMALIZATION STATISTICS
# =============================================================================
def read_multiband_float32(ps_path: Path, expected_bands: int = 4) -> np.ndarray:
    with rio.open(ps_path) as ds:
        arr = ds.read().astype(np.float32)  # (C, H, W)
    if arr.ndim != 3 or arr.shape[0] != expected_bands:
        raise ValueError(f"Expected shape (4,H,W) for {ps_path}, got {arr.shape}")
    return arr


def compute_fold_normalization_stats(
    train_pairs: Sequence[PairRecord],
    sample_stats_files: int,
    sample_stats_pixels: int,
    upper_percentile: float,
    seed: int,
    logger: logging.Logger,
) -> dict:
    """
    Approximate fold-specific normalization stats from a random subset of training files.

    Procedure:
    1. Randomly choose up to sample_stats_files training patches.
    2. Sample up to sample_stats_pixels pixels per band per chosen patch.
    3. Compute per-band upper clip value = p(upper_percentile).
    4. Clip to [0, clip_hi], scale to [0,1], and compute sampled mean/std.

    This avoids repeated full-data scans per fold.
    """
    if len(train_pairs) == 0:
        raise ValueError("Cannot compute normalization stats with zero training pairs")

    rng = np.random.default_rng(seed)
    n_select = min(len(train_pairs), sample_stats_files)
    selected_idx = rng.choice(len(train_pairs), size=n_select, replace=False)
    selected_pairs = [train_pairs[i] for i in selected_idx]

    logger.info(
        f"Computing fold normalization stats from {n_select}/{len(train_pairs)} training files "
        f"with {sample_stats_pixels} sampled pixels per band per file"
    )

    band_samples: List[List[np.ndarray]] = [[] for _ in range(4)]
    zero_chip_count = 0

    for i, pair in enumerate(selected_pairs, start=1):
        arr = read_multiband_float32(pair.ps_path, expected_bands=4)
        if not np.isfinite(arr).all():
            raise ValueError(f"Found NaN/Inf in training image: {pair.ps_path}")

        if np.all(arr == 0):
            zero_chip_count += 1

        for b in range(4):
            flat = arr[b].reshape(-1)
            if flat.size <= sample_stats_pixels:
                sample = flat.copy()
            else:
                idx = rng.choice(flat.size, size=sample_stats_pixels, replace=False)
                sample = flat[idx]
            band_samples[b].append(sample.astype(np.float32, copy=False))

        if i % 500 == 0 or i == len(selected_pairs):
            logger.info(f"  stats sampling progress: {i}/{len(selected_pairs)}")

    clip_upper: List[float] = []
    mean: List[float] = []
    std: List[float] = []
    total_sampled_pixels_per_band: List[int] = []

    for b in range(4):
        samples = np.concatenate(band_samples[b]).astype(np.float32, copy=False)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            raise ValueError(f"No valid sampled pixels found for band {b + 1}")

        hi = float(np.percentile(samples, upper_percentile))
        if not np.isfinite(hi) or hi <= 0:
            raise ValueError(f"Invalid clip upper value for band {b + 1}: {hi}")

        clipped_scaled = np.clip(samples, 0.0, hi) / hi
        mu = float(clipped_scaled.mean())
        sd = float(clipped_scaled.std())
        if sd < 1e-6:
            sd = 1.0

        clip_upper.append(hi)
        mean.append(mu)
        std.append(sd)
        total_sampled_pixels_per_band.append(int(samples.size))

    stats = {
        "clip_lower": [0.0, 0.0, 0.0, 0.0],
        "clip_upper": clip_upper,
        "mean": mean,
        "std": std,
        "upper_percentile": float(upper_percentile),
        "sample_stats_files": int(n_select),
        "sample_stats_pixels_per_file_per_band": int(sample_stats_pixels),
        "sampled_pixels_per_band": total_sampled_pixels_per_band,
        "zero_value_full_chip_count_in_sample": int(zero_chip_count),
        "zero_value_full_chip_fraction_in_sample": float(zero_chip_count / n_select),
    }

    logger.info(f"Fold normalization clip upper: {clip_upper}")
    logger.info(f"Fold normalization mean: {mean}")
    logger.info(f"Fold normalization std: {std}")
    logger.info(f"Zero-value full-chip fraction in stats sample: {zero_chip_count}/{n_select}")
    return stats


# =============================================================================
# DATASET / AUGMENTATION
# =============================================================================
def preprocess_image(arr: np.ndarray, stats: dict) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    clip_lower = np.asarray(stats["clip_lower"], dtype=np.float32)[:, None, None]
    clip_upper = np.asarray(stats["clip_upper"], dtype=np.float32)[:, None, None]
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]

    arr = np.clip(arr, clip_lower, clip_upper)
    arr = arr / np.maximum(clip_upper, 1e-6)
    arr = (arr - mean) / np.maximum(std, 1e-6)
    return arr.astype(np.float32, copy=False)


def read_label_uint8(label_path: Path, strict_label_binary: bool = False) -> np.ndarray:
    with rio.open(label_path) as ds:
        arr = ds.read(1)

    if strict_label_binary:
        uniq = np.unique(arr)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError(f"Label has non-binary values {uniq.tolist()} in {label_path}")
        out = arr.astype(np.uint8, copy=False)
    else:
        out = (arr > 0).astype(np.uint8)

    return out


def apply_geometric_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geometric-only augmentation:
    - random hflip
    - random vflip
    - random rot90 k in {0,1,2,3}

    image shape: (C,H,W)
    mask shape:  (H,W)
    """
    if rng.random() < 0.5:
        image = image[:, :, ::-1]
        mask = mask[:, ::-1]

    if rng.random() < 0.5:
        image = image[:, ::-1, :]
        mask = mask[::-1, :]

    k = int(rng.integers(0, 4))
    if k > 0:
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

    return image, mask


class FloodPatchDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairRecord],
        stats: dict,
        training: bool,
        seed: int,
        strict_label_binary: bool,
        return_meta: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.stats = stats
        self.training = training
        self.strict_label_binary = strict_label_binary
        self.return_meta = return_meta
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image = read_multiband_float32(pair.ps_path, expected_bands=4)
        if not np.isfinite(image).all():
            raise ValueError(f"Found NaN/Inf in image {pair.ps_path}")

        label = read_label_uint8(pair.label_path, strict_label_binary=self.strict_label_binary)
        if label.ndim != 2:
            raise ValueError(f"Expected 2D label for {pair.label_path}, got shape {label.shape}")
        if image.shape[1:] != label.shape:
            raise ValueError(
                f"Image/label size mismatch for {pair.filename}: image {image.shape[1:]}, label {label.shape}"
            )

        image = preprocess_image(image, self.stats)

        if self.training:
            image, label = apply_geometric_augmentation(image, label, self.rng)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()

        if self.return_meta:
            return image_t, label_t, pair.event, pair.filename, str(pair.ps_path)
        return image_t, label_t


# =============================================================================
# MODEL
# =============================================================================
class SegFormer4Ch(nn.Module):
    def __init__(self, model_variant: str, num_classes: int = 2, num_channels: int = 4) -> None:
        super().__init__()
        model_map = {
            "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
            "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "segformer_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
            "segformer_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
            "segformer_b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        }
        if model_variant not in model_map:
            raise ValueError(f"Unsupported model variant: {model_variant}")

        model_name = model_map[model_variant]
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.id2label = {0: "other", 1: "water"}
        config.label2id = {"other": 0, "water": 1}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        original_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        with torch.no_grad():
            # Original pretrained channels correspond to RGB.
            # Our order is Blue, Green, Red, NIR.
            # Map B,G,R directly into the first three channels expected by the adapted conv.
            # Initialize NIR from the red filter weights, which is a reasonable starting point.
            new_conv.weight[:, 0, :, :] = original_conv.weight[:, 2, :, :]  # Blue
            new_conv.weight[:, 1, :, :] = original_conv.weight[:, 1, :, :]  # Green
            new_conv.weight[:, 2, :, :] = original_conv.weight[:, 0, :, :]  # Red
            new_conv.weight[:, 3, :, :] = original_conv.weight[:, 0, :, :]  # NIR ~ initialize from Red
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# =============================================================================
# LOSS / METRICS
# =============================================================================
class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma * ce).mean()

        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        target_f = target.float()

        intersection = (probs * target_f).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth))
        dice = dice.mean()

        return focal + self.dice_weight * dice


def confusion_counts(pred: np.ndarray, true: np.ndarray) -> Dict[str, int]:
    pred = pred.astype(np.uint8, copy=False)
    true = true.astype(np.uint8, copy=False)

    tp = int(np.logical_and(pred == 1, true == 1).sum())
    fp = int(np.logical_and(pred == 1, true == 0).sum())
    tn = int(np.logical_and(pred == 0, true == 0).sum())
    fn = int(np.logical_and(pred == 0, true == 1).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = float(counts["TP"])
    fp = float(counts["FP"])
    tn = float(counts["TN"])
    fn = float(counts["FN"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "IoU": float(iou),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "Balanced_Accuracy": float(balanced_accuracy),
    }


# =============================================================================
# PREDICTION / IO
# =============================================================================
def save_mask_geotiff(mask: np.ndarray, reference_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(reference_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype=rio.uint8, compress="lzw")
    with rio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_probability_geotiff(prob: np.ndarray, reference_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(reference_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype=rio.float32, compress="lzw")
    with rio.open(out_path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)


def predict_logits_tta(model: nn.Module, x: torch.Tensor, scales: Sequence[float]) -> torch.Tensor:
    preds: List[torch.Tensor] = []
    out_size = x.shape[-2:]

    for scale in scales:
        if scale != 1.0:
            scaled_size = (max(1, int(out_size[0] * scale)), max(1, int(out_size[1] * scale)))
            x_scaled = F.interpolate(x, size=scaled_size, mode="bilinear", align_corners=False)
        else:
            x_scaled = x

        logits = model(x_scaled)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        preds.append(logits)

        logits_h = model(torch.flip(x_scaled, dims=[3]))
        logits_h = torch.flip(logits_h, dims=[3])
        if logits_h.shape[-2:] != out_size:
            logits_h = F.interpolate(logits_h, size=out_size, mode="bilinear", align_corners=False)
        preds.append(logits_h)

        logits_v = model(torch.flip(x_scaled, dims=[2]))
        logits_v = torch.flip(logits_v, dims=[2])
        if logits_v.shape[-2:] != out_size:
            logits_v = F.interpolate(logits_v, size=out_size, mode="bilinear", align_corners=False)
        preds.append(logits_v)

        logits_hv = model(torch.flip(x_scaled, dims=[2, 3]))
        logits_hv = torch.flip(logits_hv, dims=[2, 3])
        if logits_hv.shape[-2:] != out_size:
            logits_hv = F.interpolate(logits_hv, size=out_size, mode="bilinear", align_corners=False)
        preds.append(logits_hv)

    return torch.stack(preds, dim=0).mean(dim=0)


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
    tta_scales: Sequence[float],
    save_predictions_dir: Optional[Path] = None,
    save_probability: bool = False,
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    total_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    patch_rows: List[Dict[str, object]] = []

    for batch in loader:
        if len(batch) == 5:
            images, labels, events, filenames, ps_paths = batch
        else:
            images, labels = batch
            events = [""] * images.shape[0]
            filenames = [""] * images.shape[0]
            ps_paths = [""] * images.shape[0]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = predict_logits_tta(model, images, tta_scales) if use_tta else model(images)
                loss = loss_fn(logits, labels)
        else:
            logits = predict_logits_tta(model, images, tta_scales) if use_tta else model(images)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.detach().cpu().item())
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1, :, :].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        trues = labels.detach().cpu().numpy().astype(np.uint8)

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            true_i = trues[i]
            counts = confusion_counts(pred_i, true_i)
            for k in total_counts:
                total_counts[k] += counts[k]

            row = {
                "event": events[i],
                "filename": filenames[i],
                **counts,
                **metrics_from_counts(counts),
            }
            patch_rows.append(row)

            if save_predictions_dir is not None:
                out_mask = save_predictions_dir / str(events[i]) / str(filenames[i])
                save_mask_geotiff(pred_i, str(ps_paths[i]), out_mask)
                if save_probability:
                    out_prob = save_predictions_dir / str(events[i]) / (Path(str(filenames[i])).stem + "_prob.tif")
                    save_probability_geotiff(probs[i], str(ps_paths[i]), out_prob)

    avg_loss = running_loss / max(n_batches, 1)
    aggregate = {**total_counts, **metrics_from_counts(total_counts)}
    return avg_loss, aggregate, patch_rows


# =============================================================================
# FOLD RUNNER
# =============================================================================
def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def run_one_fold(
    holdout_event: str,
    event_pairs: Dict[str, List[PairRecord]],
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    fold_dir = Path(args.output_dir) / "folds" / holdout_event
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_fp = fold_dir / "best_model.pt"
    stats_fp = fold_dir / "normalization_stats.json"
    history_fp = fold_dir / "train_history.csv"
    val_best_fp = fold_dir / "val_metrics_best.json"
    test_metrics_fp = fold_dir / "test_fold_metrics.json"
    test_patch_fp = fold_dir / "test_patch_metrics.csv"

    logger.info("=" * 100)
    logger.info(f"Starting fold | held-out event = {holdout_event}")
    logger.info("=" * 100)

    test_pairs = list(event_pairs[holdout_event])
    training_events = [e for e in sorted(event_pairs.keys()) if e != holdout_event]
    all_train_pairs = [p for e in training_events for p in event_pairs[e]]

    if len(all_train_pairs) == 0:
        raise ValueError(f"No training pairs left after holding out {holdout_event}")
    if len(test_pairs) == 0:
        raise ValueError(f"No test pairs found in held-out event {holdout_event}")

    train_pairs, val_pairs = split_train_val(all_train_pairs, valid_pct=args.valid_pct, seed=args.seed)

    logger.info(
        f"Fold split sizes | train={len(train_pairs)} | val={len(val_pairs)} | test={len(test_pairs)} | "
        f"train_events={len(training_events)}"
    )

    stats = compute_fold_normalization_stats(
        train_pairs=train_pairs,
        sample_stats_files=args.sample_stats_files,
        sample_stats_pixels=args.sample_stats_pixels,
        upper_percentile=args.upper_percentile,
        seed=args.seed,
        logger=logger,
    )
    json_dump(stats, stats_fp)

    train_ds = FloodPatchDataset(
        pairs=train_pairs,
        stats=stats,
        training=True,
        seed=args.seed,
        strict_label_binary=args.strict_label_binary,
        return_meta=False,
    )
    val_ds = FloodPatchDataset(
        pairs=val_pairs,
        stats=stats,
        training=False,
        seed=args.seed,
        strict_label_binary=args.strict_label_binary,
        return_meta=True,
    )
    test_ds = FloodPatchDataset(
        pairs=test_pairs,
        stats=stats,
        training=False,
        seed=args.seed,
        strict_label_binary=args.strict_label_binary,
        return_meta=True,
    )

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, device)

    model = SegFormer4Ch(model_variant=args.model_variant, num_classes=2, num_channels=4).to(device)
    loss_fn = CombinedFocalDiceLoss(gamma=2.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_iou = -np.inf
    best_epoch = -1
    patience_counter = 0
    history_rows: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_metrics, _ = evaluate_dataset(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            use_tta=False,
            tta_scales=[1.0],
            save_predictions_dir=None,
            save_probability=False,
        )

        epoch_seconds = time.time() - epoch_start
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "epoch_seconds": epoch_seconds,
            "lr_last": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(history_row)

        logger.info(
            f"Fold={holdout_event} | epoch={epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_IoU={val_metrics['IoU']:.4f} | val_F1={val_metrics['F1']:.4f} | "
            f"val_Recall={val_metrics['Recall']:.4f} | time={epoch_seconds:.1f}s"
        )

        improved = val_metrics["IoU"] > best_val_iou
        if improved:
            best_val_iou = val_metrics["IoU"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "normalization_stats": stats,
                },
                model_fp,
            )
            json_dump({"best_epoch": epoch, **val_metrics}, val_best_fp)
            logger.info(f"  saved new best model at epoch {epoch} | val_IoU={best_val_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping triggered for fold {holdout_event} at epoch {epoch}; "
                    f"best epoch was {best_epoch} with val_IoU={best_val_iou:.4f}"
                )
                break

    pd.DataFrame(history_rows).to_csv(history_fp, index=False)

    if not model_fp.exists():
        raise RuntimeError(f"No best model checkpoint was saved for fold {holdout_event}")

    ckpt = torch.load(model_fp, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    predictions_dir = fold_dir / "predictions" if args.save_predictions else None
    tta_scales = parse_tta_scales(args.tta_scales)
    test_loss, test_metrics, test_patch_rows = evaluate_dataset(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        use_amp=use_amp,
        use_tta=args.use_tta,
        tta_scales=tta_scales,
        save_predictions_dir=predictions_dir,
        save_probability=args.save_probability,
    )

    test_metrics_full = {
        "holdout_event": holdout_event,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        **test_metrics,
    }
    json_dump(test_metrics_full, test_metrics_fp)

    test_df = pd.DataFrame(test_patch_rows)
    test_df.to_csv(test_patch_fp, index=False)

    logger.info(
        f"Completed fold {holdout_event} | test_IoU={test_metrics['IoU']:.4f} | "
        f"test_F1={test_metrics['F1']:.4f} | test_Recall={test_metrics['Recall']:.4f} | "
        f"test_Precision={test_metrics['Precision']:.4f}"
    )

    del model, optimizer, scheduler, scaler, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_metrics_full, test_df


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    try:
        set_seed(args.seed)
        device = choose_device(args.gpu, logger)

        ps_root = Path(args.ps_root)
        labels_root = Path(args.labels_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_dump(vars(args), output_dir / "config.json")

        logger.info("Starting LOEO SegFormer pipeline for UFOv2")
        logger.info(f"PS root: {ps_root}")
        logger.info(f"Labels root: {labels_root}")
        logger.info(f"Output dir: {output_dir}")

        event_pairs = build_event_pairs(ps_root=ps_root, labels_root=labels_root, logger=logger)
        all_events = sorted(event_pairs.keys())

        if args.only_events.strip():
            requested = [x.strip() for x in args.only_events.split(",") if x.strip()]
            missing = sorted(set(requested) - set(all_events))
            if missing:
                raise ValueError(f"Events requested in --only-events not found: {missing}")
            run_events = requested
        else:
            run_events = all_events

        logger.info(f"Held-out folds to run: {run_events}")

        fold_rows: List[Dict[str, object]] = []
        patch_frames: List[pd.DataFrame] = []
        overall_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        for holdout_event in run_events:
            try:
                fold_metrics, patch_df = run_one_fold(
                    holdout_event=holdout_event,
                    event_pairs=event_pairs,
                    args=args,
                    device=device,
                    logger=logger,
                )
                fold_rows.append(fold_metrics)
                patch_frames.append(patch_df)
                for k in overall_counts:
                    overall_counts[k] += int(fold_metrics[k])
            except Exception as e:
                logger.error(f"Fold failed for {holdout_event}: {e}")
                logger.error(traceback.format_exc())
                continue

        if not fold_rows:
            raise RuntimeError("All folds failed; no outputs were produced.")

        fold_summary_df = pd.DataFrame(fold_rows)
        fold_summary_df.to_csv(output_dir / "fold_summary.csv", index=False)

        patch_metrics_df = pd.concat(patch_frames, ignore_index=True)
        patch_metrics_df.to_csv(output_dir / "patch_metrics_all.csv", index=False)

        # Overall summaries
        overall_micro = {**overall_counts, **metrics_from_counts(overall_counts)}
        metric_cols = ["Precision", "Recall", "Specificity", "IoU", "F1", "Accuracy", "Balanced_Accuracy"]
        overall_macro = {
            f"macro_{m}": float(fold_summary_df[m].mean()) for m in metric_cols if m in fold_summary_df.columns
        }
        overall_macro_std = {
            f"macro_{m}_std": float(fold_summary_df[m].std(ddof=1)) if len(fold_summary_df) > 1 else 0.0
            for m in metric_cols if m in fold_summary_df.columns
        }

        overall_summary = {
            "n_folds_completed": int(len(fold_summary_df)),
            "events_completed": fold_summary_df["holdout_event"].tolist(),
            "overall_micro": overall_micro,
            **overall_macro,
            **overall_macro_std,
        }
        json_dump(overall_summary, output_dir / "overall_summary.json")

        logger.info("=" * 100)
        logger.info("Pipeline completed")
        logger.info(f"Completed folds: {len(fold_summary_df)}")
        logger.info(
            f"Overall micro | IoU={overall_micro['IoU']:.4f} | F1={overall_micro['F1']:.4f} | "
            f"Precision={overall_micro['Precision']:.4f} | Recall={overall_micro['Recall']:.4f}"
        )
        logger.info("=" * 100)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
