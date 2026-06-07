#!/usr/bin/env python3
"""Tile paired GeoTIFF event folders into 256x256 patches.

Expected input layout:
    data/raw/PS/<EVENT>/*.tif
    data/raw/labels/<EVENT>/*.tif

Output layout:
    data/PS_256/<EVENT>/*_patch_rXXX_cXXX.tif
    data/labels_256/<EVENT>/*_patch_rXXX_cXXX.tif

The script keeps georeferencing for each patch and skips partial edge windows by
default. UFOv2 source chips are 1024x1024, so the default 256-pixel tile size
produces 16 patches per chip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import rasterio as rio
from rasterio.windows import Window, transform as window_transform


@dataclass(frozen=True)
class TileJob:
    src_path: Path
    out_path: Path
    window: Window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tile UFO GeoTIFF event folders into 256x256 patches")
    parser.add_argument("--ps-root", type=Path, required=True, help="Input PlanetScope event-folder root")
    parser.add_argument("--labels-root", type=Path, required=True, help="Input label event-folder root")
    parser.add_argument("--out-ps-root", type=Path, default=Path("data/PS_256"), help="Output PS tile root")
    parser.add_argument("--out-labels-root", type=Path, default=Path("data/labels_256"), help="Output label tile root")
    parser.add_argument("--tile-size", type=int, default=256, help="Square tile size in pixels")
    parser.add_argument("--include-partial", action="store_true", help="Write partial edge tiles")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output tiles")
    return parser


def event_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing input root: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        raise ValueError(f"No event folders found under: {root}")
    return dirs


def paired_events(ps_root: Path, labels_root: Path) -> List[str]:
    ps_events = {p.name for p in event_dirs(ps_root)}
    label_events = {p.name for p in event_dirs(labels_root)}
    if ps_events != label_events:
        only_ps = sorted(ps_events - label_events)
        only_labels = sorted(label_events - ps_events)
        raise ValueError(f"Event mismatch. Only in PS: {only_ps}; only in labels: {only_labels}")
    return sorted(ps_events)


def windows_for(width: int, height: int, tile_size: int, include_partial: bool) -> Iterable[Tuple[int, int, Window]]:
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            win_width = min(tile_size, width - col_off)
            win_height = min(tile_size, height - row_off)
            if not include_partial and (win_width != tile_size or win_height != tile_size):
                continue
            yield row_off, col_off, Window(col_off=col_off, row_off=row_off, width=win_width, height=win_height)


def tile_one(src_path: Path, out_dir: Path, tile_size: int, include_partial: bool, overwrite: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with rio.open(src_path) as src:
        for row_off, col_off, window in windows_for(src.width, src.height, tile_size, include_partial):
            out_name = f"{src_path.stem}_patch_r{row_off:04d}_c{col_off:04d}{src_path.suffix}"
            out_path = out_dir / out_name
            if out_path.exists() and not overwrite:
                continue

            profile = src.profile.copy()
            profile.update(
                height=int(window.height),
                width=int(window.width),
                transform=window_transform(window, src.transform),
                compress=profile.get("compress", "lzw"),
            )

            with rio.open(out_path, "w", **profile) as dst:
                dst.write(src.read(window=window))
            written += 1

    return written


def validate_event_files(ps_dir: Path, label_dir: Path) -> List[str]:
    ps_files = {p.name for p in ps_dir.glob("*.tif")}
    label_files = {p.name for p in label_dir.glob("*.tif")}
    if ps_files != label_files:
        only_ps = sorted(ps_files - label_files)
        only_labels = sorted(label_files - ps_files)
        raise ValueError(f"Filename mismatch in {ps_dir.name}. Only in PS: {only_ps}; only in labels: {only_labels}")
    return sorted(ps_files)


def main() -> None:
    args = build_parser().parse_args()
    events = paired_events(args.ps_root, args.labels_root)

    total_written = 0
    for event in events:
        ps_dir = args.ps_root / event
        label_dir = args.labels_root / event
        names = validate_event_files(ps_dir, label_dir)

        print(f"{event}: {len(names)} paired source chips")
        for name in names:
            total_written += tile_one(
                ps_dir / name,
                args.out_ps_root / event,
                args.tile_size,
                args.include_partial,
                args.overwrite,
            )
            total_written += tile_one(
                label_dir / name,
                args.out_labels_root / event,
                args.tile_size,
                args.include_partial,
                args.overwrite,
            )

    print(f"Done. Wrote {total_written} tile files.")


if __name__ == "__main__":
    main()
