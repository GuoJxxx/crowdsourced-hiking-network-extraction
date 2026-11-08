from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy as raster_xy
from rasterio.windows import Window


@dataclass
class TileConfig:
    tile_size_m: float = 560.0
    origin_x: Optional[float] = None
    origin_y: Optional[float] = None
    source: Optional[str] = None


def read_tile_config(report_path: Optional[Path], fallback_size_m: float = 560.0) -> TileConfig:
    cfg = TileConfig(tile_size_m=float(fallback_size_m))
    if report_path is None or not Path(report_path).exists():
        return cfg
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    candidates = [data]
    for key in ("config", "grid", "tile", "resolved", "parameters"):
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    def first(keys: Sequence[str]):
        for container in candidates:
            for key in keys:
                if key in container and container[key] is not None:
                    return container[key]
        return None

    size = first(("tile_size_m_actual", "tile_size_m", "gravity_tile_size_m"))
    ox = first(("tile_origin_x_m", "gravity_tile_origin_x_m", "origin_x_m", "origin_x"))
    oy = first(("tile_origin_y_m", "gravity_tile_origin_y_m", "origin_y_m", "origin_y"))
    if size is not None:
        cfg.tile_size_m = float(size)
    if ox is not None:
        cfg.origin_x = float(ox)
    if oy is not None:
        cfg.origin_y = float(oy)
    cfg.source = str(report_path)
    return cfg


def validate_aligned_rasters(paths: Sequence[Path]) -> Dict[str, Any]:
    paths = [Path(p) for p in paths if p is not None]
    if not paths:
        raise ValueError('No input rasters')
    with rasterio.open(paths[0]) as first:
        ref = {
            "width": first.width,
            "height": first.height,
            "shape": first.shape,
            "transform": first.transform,
            "crs": first.crs,
            "cell_size": float(max(abs(first.transform.a), abs(first.transform.e))),
            "profile": first.profile.copy(),
        }
    for path in paths[1:]:
        with rasterio.open(path) as ds:
            if ds.shape != ref["shape"] or ds.transform != ref["transform"] or ds.crs != ref["crs"]:
                raise ValueError(f"""Rasters are not strictly aligned: {path}""")
    return ref


def tile_windows(width: int, height: int, transform, cfg: TileConfig) -> Iterator[Tuple[str, Window, Dict[str, Any]]]:
    cell = float(max(abs(transform.a), abs(transform.e)))
    tile_px = max(1, int(round(float(cfg.tile_size_m) / cell)))

    # S4/S5 rasters usually share the same physical origin. If the report includes an origin, compute the pixel offset
    # of the first 560 m grid boundary within the raster; otherwise start at 0,0.
    x_left = float(transform.c)
    y_top = float(transform.f)
    if cfg.origin_x is None:
        first_col = 0
        base_tile_col = 0
    else:
        base_tile_col = math.floor((x_left - cfg.origin_x) / cfg.tile_size_m)
        first_x = cfg.origin_x + base_tile_col * cfg.tile_size_m
        first_col = int(round((first_x - x_left) / cell))
        while first_col > 0:
            first_col -= tile_px
            base_tile_col -= 1
        while first_col + tile_px <= 0:
            first_col += tile_px
            base_tile_col += 1

    if cfg.origin_y is None:
        first_row = 0
        base_tile_row = 0
    else:
        # tile row increases northward in physical coordinates; raster row increases southward.
        top_tile_row = math.ceil((y_top - cfg.origin_y) / cfg.tile_size_m) - 1
        first_y_top = cfg.origin_y + (top_tile_row + 1) * cfg.tile_size_m
        first_row = int(round((y_top - first_y_top) / cell))
        base_tile_row = top_tile_row
        while first_row > 0:
            first_row -= tile_px
            base_tile_row += 1
        while first_row + tile_px <= 0:
            first_row += tile_px
            base_tile_row -= 1

    row = first_row
    tile_r = base_tile_row
    while row < height:
        col = first_col
        tile_c = base_tile_col
        while col < width:
            r0, c0 = max(0, row), max(0, col)
            r1, c1 = min(height, row + tile_px), min(width, col + tile_px)
            if r1 > r0 and c1 > c0:
                window = Window(c0, r0, c1 - c0, r1 - r0)
                tile_id = f"r{tile_r}_c{tile_c}"
                left, top = transform * (c0, r0)
                right, bottom = transform * (c1, r1)
                info = {
                    "tile_id": tile_id,
                    "tile_row": int(tile_r),
                    "tile_col": int(tile_c),
                    "row_off": int(r0),
                    "col_off": int(c0),
                    "height": int(r1-r0),
                    "width": int(c1-c0),
                    "left": float(min(left, right)),
                    "right": float(max(left, right)),
                    "bottom": float(min(bottom, top)),
                    "top": float(max(bottom, top)),
                }
                yield tile_id, window, info
            col += tile_px
            tile_c += 1
        row += tile_px
        tile_r -= 1


def combine_sparse_flags(flat_arrays: List[np.ndarray], flag_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not flat_arrays:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)
    flat = np.concatenate(flat_arrays).astype(np.int64, copy=False)
    flags = np.concatenate(flag_arrays).astype(np.uint8, copy=False)
    order = np.argsort(flat, kind="mergesort")
    flat, flags = flat[order], flags[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    unique = flat[starts]
    merged = np.bitwise_or.reduceat(flags, starts)
    return unique, merged


def collect_skeleton_pixels(
    primary_path: Path,
    auxiliary_path: Optional[Path],
    rescued_path: Optional[Path],
    tile_cfg: TileConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    paths = [Path(primary_path)]
    if auxiliary_path and Path(auxiliary_path).exists():
        paths.append(Path(auxiliary_path))
    if rescued_path and Path(rescued_path).exists():
        paths.append(Path(rescued_path))
    ref = validate_aligned_rasters(paths)

    primary = rasterio.open(primary_path)
    auxiliary = rasterio.open(auxiliary_path) if auxiliary_path and Path(auxiliary_path).exists() else None
    rescued = rasterio.open(rescued_path) if rescued_path and Path(rescued_path).exists() else None

    flat_parts: List[np.ndarray] = []
    flag_parts: List[np.ndarray] = []
    tile_rows: List[Dict[str, Any]] = []
    width = int(ref["width"])
    try:
        for tile_id, window, info in tile_windows(width, int(ref["height"]), ref["transform"], tile_cfg):
            p = primary.read(1, window=window) > 0
            a = auxiliary.read(1, window=window) > 0 if auxiliary else np.zeros_like(p)
            r = rescued.read(1, window=window) > 0 if rescued else np.zeros_like(p)
            union = p | a | r
            rr, cc = np.nonzero(union)
            global_r = rr.astype(np.int64) + int(window.row_off)
            global_c = cc.astype(np.int64) + int(window.col_off)
            flat = global_r * width + global_c
            flags = (
                p[rr, cc].astype(np.uint8)
                | (a[rr, cc].astype(np.uint8) << 1)
                | (r[rr, cc].astype(np.uint8) << 2)
            )
            if len(flat):
                flat_parts.append(flat)
                flag_parts.append(flags)
            tile_rows.append({
                **info,
                "primary_pixels": int(p.sum()),
                "auxiliary_pixels": int(a.sum()),
                "rescued_pixels": int(r.sum()),
                "active_pixels": int(union.sum()),
            })
    finally:
        primary.close()
        if auxiliary: auxiliary.close()
        if rescued: rescued.close()

    active, flags = combine_sparse_flags(flat_parts, flag_parts)
    return active, flags, ref, tile_rows


class RasterBlockSampler:
    """LRU block sampler for one or more aligned rasters."""

    def __init__(self, paths: Dict[str, Optional[Path]], cache_blocks: int = 64):
        self.datasets: Dict[str, rasterio.DatasetReader] = {}
        for name, path in paths.items():
            if path is not None and Path(path).exists():
                self.datasets[name] = rasterio.open(path)
        if not self.datasets:
            raise ValueError('RasterBlockSampler requires at least one raster')
        first = next(iter(self.datasets.values()))
        self.transform = first.transform
        self.crs = first.crs
        self.width = first.width
        self.height = first.height
        self.block_h, self.block_w = first.block_shapes[0]
        self.cache_blocks = max(4, int(cache_blocks))
        self.cache: OrderedDict[Tuple[str, int, int], np.ndarray] = OrderedDict()
        for name, ds in self.datasets.items():
            if ds.shape != first.shape or ds.transform != first.transform or ds.crs != first.crs:
                raise ValueError(f"""Sampling rasters are not aligned: {name}""")

    def close(self):
        for ds in self.datasets.values():
            ds.close()
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _block(self, name: str, br: int, bc: int) -> np.ndarray:
        key = (name, br, bc)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        ds = self.datasets[name]
        r0, c0 = br * self.block_h, bc * self.block_w
        window = Window(c0, r0, min(self.block_w, ds.width-c0), min(self.block_h, ds.height-r0))
        value = ds.read(1, window=window, masked=False)
        self.cache[key] = value
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return value

    def values_rc(self, name: str, rows: np.ndarray, cols: np.ndarray, default: float = 0.0) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        out = np.full(len(rows), default, dtype=float)
        valid = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        if not valid.any() or name not in self.datasets:
            return out
        idx = np.flatnonzero(valid)
        br = rows[idx] // self.block_h
        bc = cols[idx] // self.block_w
        pairs = np.c_[br, bc]
        for key in np.unique(pairs, axis=0):
            sel_local = np.flatnonzero((br == key[0]) & (bc == key[1]))
            sel = idx[sel_local]
            block = self._block(name, int(key[0]), int(key[1]))
            lr = rows[sel] - int(key[0]) * self.block_h
            lc = cols[sel] - int(key[1]) * self.block_w
            vals = block[lr, lc].astype(float, copy=False)
            vals[~np.isfinite(vals)] = default
            out[sel] = vals
        return out

    def values_xy(self, name: str, xs: np.ndarray, ys: np.ndarray, default: float = 0.0) -> np.ndarray:
        rows, cols = rowcol(self.transform, np.asarray(xs), np.asarray(ys))
        return self.values_rc(name, np.asarray(rows), np.asarray(cols), default=default)


def pixel_centers(transform, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = raster_xy(transform, rows, cols, offset="center")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
