#!/usr/bin/env python3
"""Mosaic 256x256 prediction tiles back to source-chip prediction rasters."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List

import rasterio as rio
from rasterio.merge import merge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mosaic UFO prediction patches by source chip")
    parser.add_argument(
        "--loeo-output",
        type=Path,
        default=Path("outputs/ufo_segformer_b2_loeo"),
        help="Output directory produced by train_segformer_b2_loeo.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Mosaic output root. Defaults to <loeo-output>/mosaics",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mosaics")
    return parser


def prediction_event_dirs(loeo_output: Path) -> Iterable[Path]:
    folds_dir = loeo_output / "folds"
    if not folds_dir.exists():
        raise FileNotFoundError(f"Missing folds directory: {folds_dir}")

    for fold_dir in sorted(p for p in folds_dir.iterdir() if p.is_dir()):
        pred_root = fold_dir / "predictions"
        if not pred_root.exists():
            continue
        for event_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
            yield event_dir


def source_prefix(tile_path: Path) -> str:
    stem = tile_path.stem
    marker = "_patch_"
    if marker in stem:
        return stem.split(marker, 1)[0]
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def group_tiles(tile_paths: Iterable[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in tile_paths:
        groups[source_prefix(path)].append(path)
    return dict(groups)


def mosaic_group(tile_paths: List[Path], out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_paths = sorted(tile_paths, key=lambda p: p.name)

    with ExitStack() as stack:
        srcs = [stack.enter_context(rio.open(path)) for path in tile_paths]
        mosaic, out_transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=mosaic.shape[0],
            compress=profile.get("compress", "lzw"),
        )
        with rio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root or (args.loeo_output / "mosaics")

    n_written = 0
    for event_dir in prediction_event_dirs(args.loeo_output):
        tiles = sorted(event_dir.glob("*.tif"))
        if not tiles:
            continue

        groups = group_tiles(tiles)
        print(f"{event_dir.name}: {len(groups)} mosaics from {len(tiles)} tiles")

        for prefix, group in sorted(groups.items()):
            out_path = output_root / event_dir.name / f"{prefix}.tif"
            existed = out_path.exists()
            mosaic_group(group, out_path, overwrite=args.overwrite)
            if args.overwrite or not existed:
                n_written += 1

    print(f"Done. Wrote {n_written} mosaics to {output_root}")


if __name__ == "__main__":
    main()
