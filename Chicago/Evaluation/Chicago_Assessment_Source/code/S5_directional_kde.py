#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-memory direction-constrained anisotropic KDE on 560 m x 560 m physical tiles.

Purpose
-------
S5: direction-constrained anisotropic KDE. Avoid allocating the full orientation_bins x height x width array, which can exceed memory on large 2 m grids. Process 560 m x 560 m tiles, reading and rasterizing segments within each tile and its halo, then write the central region directly to its window in a single final GeoTIFF. This avoids numerous intermediate tile files and additional mosaic memory.

Keep standard output filenames for downstream S6 skeletonization.

Gravity alignment
-----------------
For S4_uturn_consensus_gravity.py output, supply --gravity-report gravity_uturn_report.json to read gravity_tile_size_m, gravity_tile_origin_x_m and gravity_tile_origin_y_m automatically. This aligns KDE bounds with the 560 m gravity tile origin and supports corresponding tile diagnostics.

Outputs
-------
- independent_track_support.tif
- raw_trajectory_union.tif
- distance_to_raw_trajectory_m.tif
- kde_directional_alongXm_crossYm_float32.tif
- kde_directional_alongXm_crossYm_uint16_0_1024.tif
- kde_directional_alongXm_crossYm_preview.png
- directional_kde_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from shapely.geometry import LineString, box, mapping
from shapely.strtree import STRtree


@dataclass
class Config:
    cell_size_m: float = 2.0
    sigma_along_m: float = 4.0
    sigma_cross_m: float = 1.5
    orientation_bins: int = 16
    truncate: float = 3.0
    padding_m: float = 20.0
    max_link_m: float = 0.0
    scaled_max: int = 1024
    all_touched: bool = False
    clear_output: bool = True
    segment_field: str = "clean_track_id"
    order_field: str = "clean_point_index"
    contribution_fields: Tuple[str, ...] = ("source_file", "track_id")
    road_weight_field: str = "road_extraction_weight"
    min_road_weight: float = 0.35
    require_mainroad_candidate: bool = False

    # Low-memory parameters. Process physical 560 m x 560 m tiles by default.
    tile_size_m: float = 560.0
    # Optional lower-left tile origin aligned with the gravity-concentration tiling.
    # If specified, extend KDE raster bounds to the origin + 560 m grid so tile boundaries match.
    tile_origin_x: float | None = None
    tile_origin_y: float | None = None
    gravity_report: str = ""
    # Optional tile size in pixels; takes priority when >0, mainly for debugging or compatibility with old parameters.
    tile_size_px_override: int = 0
    # With the default S3 relaxed_corridor_m of 5 m, 10 m here is sufficient.
    # Increase this value too if subsequent S6 processing uses a larger relaxed-corridor-m.
    distance_max_m: float = 10.0
    block_size: int = 256


def cid_series(df: pd.DataFrame, fields: Sequence[str]) -> pd.Series:
    return df[fields[0]].fillna("").astype(str) + "||" + df[fields[1]].fillna("").astype(str)


def _bool_series(df: pd.DataFrame, field: str, default: bool = False) -> np.ndarray:
    if field not in df.columns:
        return np.full(len(df), bool(default), dtype=bool)
    value = df[field]
    if pd.api.types.is_bool_dtype(value):
        return value.fillna(default).to_numpy(bool)
    return value.fillna("").astype(str).str.lower().isin({"1", "true", "yes", "y"}).to_numpy(bool)


def road_eligible_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Semantic road eligibility: confirmed activity through-corridors are positive evidence."""
    if cfg.road_weight_field in df.columns:
        weight = pd.to_numeric(df[cfg.road_weight_field], errors="coerce").fillna(1.0).to_numpy(float)
    else:
        weight = np.ones(len(df), dtype=float)
    through = _bool_series(df, "activity_through_corridor", False)
    keep = (
        np.isfinite(pd.to_numeric(df["x_m"], errors="coerce").to_numpy(float))
        & np.isfinite(pd.to_numeric(df["y_m"], errors="coerce").to_numpy(float))
        & df[cfg.segment_field].notna().to_numpy(bool)
        & ((weight >= float(cfg.min_road_weight)) | through)
    )
    if cfg.require_mainroad_candidate and "mainroad_candidate" in df.columns:
        main = _bool_series(df, "mainroad_candidate", False)
        keep &= (main | through)
    return keep


def load_points(path: Path, cfg: Config) -> Tuple[pd.DataFrame, CRS]:
    df = pd.read_csv(path, low_memory=False)
    input_rows_total = int(len(df))
    required = {
        cfg.segment_field, cfg.order_field, "x_m", "y_m", "metric_crs",
        *cfg.contribution_fields,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"""Missing fields: {sorted(missing)}""")
    for col in ["x_m", "y_m", cfg.order_field]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if cfg.road_weight_field not in df.columns:
        df[cfg.road_weight_field] = 1.0
    keep = road_eligible_mask(df, cfg)
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError('No points remain for KDE after road-weight filtering')
    crs_values = df["metric_crs"].dropna().astype(str).unique().tolist()
    if len(crs_values) != 1:
        raise ValueError(f"""metric_crs is not unique: {crs_values}""")
    crs = CRS.from_user_input(crs_values[0])
    if not crs.is_projected:
        raise ValueError('metric_crs must be a projected CRS')
    df["_cid"] = cid_series(df, cfg.contribution_fields)
    df.attrs["input_rows_total"] = input_rows_total
    df.attrs["input_rows_used"] = int(len(df))
    return df, crs


def split_parts(df: pd.DataFrame, cfg: Config) -> List[Tuple[str, np.ndarray]]:
    'Sort by clean_track_id; split at enabled long-edge limits or gaps in original point indices.'
    output: List[Tuple[str, np.ndarray]] = []
    for _, group in df.sort_values([cfg.segment_field, cfg.order_field]).groupby(
        cfg.segment_field, sort=False
    ):
        xy = group[["x_m", "y_m"]].to_numpy(float)
        order = pd.to_numeric(group[cfg.order_field], errors="coerce").to_numpy(float)
        cid = str(group["_cid"].iloc[0])
        if len(xy) < 2:
            continue
        distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        order_gap = np.diff(order) > 1.0
        distance_gap = (distances > cfg.max_link_m) if cfg.max_link_m > 0.0 else np.zeros_like(distances, dtype=bool)
        bad_edges = distance_gap | order_gap | ~np.isfinite(distances)
        start = 0
        for i, bad in enumerate(bad_edges):
            if bad:
                if i + 1 - start >= 2:
                    output.append((cid, xy[start:i + 1]))
                start = i + 1
        if len(xy) - start >= 2:
            output.append((cid, xy[start:]))
    if not output:
        raise ValueError('No continuous main-road track segments available for KDE; check the weight threshold or max_link_m')
    return output


def _actual_tile_size_m_for_grid(cfg: Config) -> float:
    'Return the actual physical KDE tile side length consistent with pixel tiling.'
    if int(cfg.tile_size_px_override) > 0:
        pixels = int(cfg.tile_size_px_override)
    else:
        pixels = int(round(float(cfg.tile_size_m) / float(cfg.cell_size_m)))
    if pixels < 1:
        raise ValueError(
            f"""Invalid tile pixel count: tile_size_m={cfg.tile_size_m}, cell_size_m={cfg.cell_size_m}"""
        )
    return float(pixels) * float(cfg.cell_size_m)


def _floor_to_origin(value: float, origin: float, step: float) -> float:
    return float(origin) + math.floor((float(value) - float(origin)) / float(step)) * float(step)


def _ceil_to_origin(value: float, origin: float, step: float) -> float:
    return float(origin) + math.ceil((float(value) - float(origin)) / float(step)) * float(step)


def build_grid(parts: Sequence[Tuple[str, np.ndarray]], cfg: Config):
    xs = np.concatenate([p[:, 0] for _, p in parts])
    ys = np.concatenate([p[:, 1] for _, p in parts])
    pad = max(
        cfg.padding_m,
        cfg.truncate * max(cfg.sigma_along_m, cfg.sigma_cross_m) + cfg.cell_size_m,
        cfg.distance_max_m + cfg.cell_size_m,
    )
    raw_left = float(xs.min()) - pad
    raw_right = float(xs.max()) + pad
    raw_bottom = float(ys.min()) - pad
    raw_top = float(ys.max()) + pad

    # Align to cell size by default to preserve existing behavior; if a gravity-tile origin is specified, extend the output bounds
    # to the same 560 m tile grid so gravity-concentration and KDE tile diagnostics correspond.
    if cfg.tile_origin_x is not None and cfg.tile_origin_y is not None:
        tile_m = _actual_tile_size_m_for_grid(cfg)
        left = _floor_to_origin(raw_left, float(cfg.tile_origin_x), tile_m)
        right = _ceil_to_origin(raw_right, float(cfg.tile_origin_x), tile_m)
        bottom = _floor_to_origin(raw_bottom, float(cfg.tile_origin_y), tile_m)
        top = _ceil_to_origin(raw_top, float(cfg.tile_origin_y), tile_m)
    else:
        left = math.floor(raw_left / cfg.cell_size_m) * cfg.cell_size_m
        right = math.ceil(raw_right / cfg.cell_size_m) * cfg.cell_size_m
        bottom = math.floor(raw_bottom / cfg.cell_size_m) * cfg.cell_size_m
        top = math.ceil(raw_top / cfg.cell_size_m) * cfg.cell_size_m

    width = int(round((right - left) / cfg.cell_size_m))
    height = int(round((top - bottom) / cfg.cell_size_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"""Invalid grid dimensions: width={width}, height={height}""")
    transform = from_origin(left, top, cfg.cell_size_m, cfg.cell_size_m)
    return transform, width, height, (left, bottom, right, top)


def kernel(theta: float, cfg: Config) -> np.ndarray:
    radius = int(
        math.ceil(
            cfg.truncate * max(cfg.sigma_along_m, cfg.sigma_cross_m) / cfg.cell_size_m
        )
    )
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    dx = xx * cfg.cell_size_m
    dy = -yy * cfg.cell_size_m
    ct, st = math.cos(theta), math.sin(theta)
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    k = np.exp(-0.5 * ((u / cfg.sigma_along_m) ** 2 + (v / cfg.sigma_cross_m) ** 2))
    k /= max(float(k.sum()), 1e-12)
    return k.astype(np.float32)


def kernel_radius_px(cfg: Config) -> int:
    return int(
        math.ceil(
            cfg.truncate * max(cfg.sigma_along_m, cfg.sigma_cross_m) / cfg.cell_size_m
        )
    )


def make_profile(width: int, height: int, transform, crs: CRS, dtype: str, nodata):
    block = int(max(16, min(512, cfg_global_block_size)))
    # GeoTIFF tile block must be multiple of 16.
    block = int(math.ceil(block / 16) * 16)
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": block,
        "blockysize": block,
        "BIGTIFF": "IF_SAFER",
        "predictor": 3 if np.dtype(dtype).kind == "f" else 2,
    }


# Used only by make_profile to avoid passing many parameters.
cfg_global_block_size = 256


def write_empty_rasters(out: Path, width: int, height: int, transform, crs: CRS, cfg: Config):
    global cfg_global_block_size
    cfg_global_block_size = int(cfg.block_size)
    paths = {
        "kde": out / "_tmp_kde_float32.tif",
        "support": out / "independent_track_support.tif",
        "raw_union": out / "raw_trajectory_union.tif",
        "distance": out / "distance_to_raw_trajectory_m.tif",
    }
    profiles = {
        "kde": make_profile(width, height, transform, crs, "float32", 0.0),
        "support": make_profile(width, height, transform, crs, "uint16", 0),
        "raw_union": make_profile(width, height, transform, crs, "uint8", 0),
        "distance": make_profile(width, height, transform, crs, "float32", 0.0),
    }
    for key, path in paths.items():
        with rasterio.open(path, "w", **profiles[key]):
            pass
    return paths


def tile_size_px_from_config(cfg: Config) -> Tuple[int, float]:
    'Convert physical tile side length to pixels and return the actual physical length.'
    if int(cfg.tile_size_px_override) > 0:
        pixels = int(cfg.tile_size_px_override)
    else:
        pixels = int(round(float(cfg.tile_size_m) / float(cfg.cell_size_m)))
    if pixels < 1:
        raise ValueError(
            f"""Invalid tile pixel count: tile_size_m={cfg.tile_size_m}, cell_size_m={cfg.cell_size_m}"""
        )
    actual_m = pixels * float(cfg.cell_size_m)
    if abs(actual_m - float(cfg.tile_size_m)) > max(1e-6, 0.001 * float(cfg.cell_size_m)):
        print(
            f"""[WARNING] tile_size_m={cfg.tile_size_m:g} is not exactly divisible by cell_size_m={cfg.cell_size_m:g}; actual tile side length is {actual_m:g} m ({pixels} pixels)""",
            flush=True,
        )
    return pixels, actual_m


def iter_tiles(width: int, height: int, tile_size_px: int):
    for row in range(0, height, tile_size_px):
        h = min(tile_size_px, height - row)
        for col in range(0, width, tile_size_px):
            w = min(tile_size_px, width - col)
            yield row, col, h, w


def window_bounds(transform, row: int, col: int, height: int, width: int):
    left, top = transform * (col, row)
    right, bottom = transform * (col + width, row + height)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def local_transform(global_transform, row0: int, col0: int):
    x0, y0 = global_transform * (col0, row0)
    return from_origin(x0, y0, abs(global_transform.a), abs(global_transform.e))


def build_segments(parts: Sequence[Tuple[str, np.ndarray]], cfg: Config):
    'Split track polylines into indexable short segments.'
    geoms: List[LineString] = []
    meta: List[Tuple[str, int]] = []
    for cid, xy in parts:
        for a, b in zip(xy[:-1], xy[1:]):
            vector = b - a
            length = float(np.linalg.norm(vector))
            if (
                not np.isfinite(length)
                or length <= 1e-9
                or (cfg.max_link_m > 0.0 and length > cfg.max_link_m)
            ):
                continue
            theta = math.atan2(float(vector[1]), float(vector[0])) % math.pi
            bin_id = int(theta / math.pi * cfg.orientation_bins) % cfg.orientation_bins
            geom = LineString([tuple(a), tuple(b)])
            if not geom.is_empty:
                geoms.append(geom)
                meta.append((cid, bin_id))
    if not geoms:
        raise ValueError('No track edges available for rasterization')
    tree = STRtree(geoms)
    id_to_index = {id(g): i for i, g in enumerate(geoms)}
    wkb_to_index = {g.wkb: i for i, g in enumerate(geoms)}
    return geoms, meta, tree, id_to_index, wkb_to_index


def query_indices(tree, query_geom, id_to_index, wkb_to_index) -> List[int]:
    result = tree.query(query_geom)
    if len(result) == 0:
        return []
    first = result[0]
    # Shapely 2.x returns integer indices.
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in result]
    # Shapely 1.8 returns geometry objects.
    indices: List[int] = []
    for geom in result:
        idx = id_to_index.get(id(geom))
        if idx is None:
            idx = wkb_to_index.get(geom.wkb)
        if idx is not None:
            indices.append(int(idx))
    return indices


def central_slice(row: int, col: int, h: int, w: int, row0: int, col0: int):
    r0 = row - row0
    c0 = col - col0
    return slice(r0, r0 + h), slice(c0, c0 + w)


def rasterize_by_cid(
    cid_to_indices: Dict[str, List[int]],
    geoms: Sequence[LineString],
    shape: Tuple[int, int],
    transform,
    all_touched: bool,
) -> np.ndarray:
    support = np.zeros(shape, dtype=np.uint16)
    for indices in cid_to_indices.values():
        if not indices:
            continue
        shapes = [(mapping(geoms[i]), 1) for i in indices]
        mask = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
        support += (mask > 0).astype(np.uint16)
    return support


def process_tile(
    row: int,
    col: int,
    h: int,
    w: int,
    width: int,
    height: int,
    transform,
    geoms: Sequence[LineString],
    meta: Sequence[Tuple[str, int]],
    tree,
    id_to_index,
    wkb_to_index,
    kernels: Sequence[np.ndarray],
    cfg: Config,
):
    kr = kernel_radius_px(cfg)
    distance_halo_px = int(math.ceil(cfg.distance_max_m / cfg.cell_size_m)) + 4
    halo = max(kr, distance_halo_px)

    row0 = max(0, row - halo)
    col0 = max(0, col - halo)
    row1 = min(height, row + h + halo)
    col1 = min(width, col + w + halo)
    lh = row1 - row0
    lw = col1 - col0
    ltr = local_transform(transform, row0, col0)
    bounds = window_bounds(transform, row0, col0, lh, lw)
    indices = query_indices(tree, box(*bounds), id_to_index, wkb_to_index)

    if not indices:
        support_c = np.zeros((h, w), dtype=np.uint16)
        raw_c = np.zeros((h, w), dtype=np.uint8)
        dist_c = np.full((h, w), cfg.distance_max_m + cfg.cell_size_m, dtype=np.float32)
        kde_c = np.zeros((h, w), dtype=np.float32)
        return kde_c, support_c, raw_c, dist_c, 0.0, 0

    cid_to_indices: Dict[str, List[int]] = defaultdict(list)
    bin_cid_to_indices: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for idx in indices:
        cid, bin_id = meta[idx]
        cid_to_indices[cid].append(idx)
        bin_cid_to_indices[(bin_id, cid)].append(idx)

    support_local = rasterize_by_cid(
        cid_to_indices, geoms, (lh, lw), ltr, cfg.all_touched
    )
    raw_union_local = support_local > 0

    if raw_union_local.any():
        distance_local = distance_transform_edt(~raw_union_local).astype(np.float32) * float(
            cfg.cell_size_m
        )
        distance_local = np.minimum(distance_local, cfg.distance_max_m + cfg.cell_size_m)
    else:
        distance_local = np.full(
            (lh, lw), cfg.distance_max_m + cfg.cell_size_m, dtype=np.float32
        )

    kde_local = np.zeros((lh, lw), dtype=np.float32)
    for bin_id in range(cfg.orientation_bins):
        bin_support = np.zeros((lh, lw), dtype=np.uint16)
        for (b, _cid), seg_indices in bin_cid_to_indices.items():
            if b != bin_id:
                continue
            shapes = [(mapping(geoms[i]), 1) for i in seg_indices]
            mask = rasterize(
                shapes,
                out_shape=(lh, lw),
                transform=ltr,
                fill=0,
                dtype="uint8",
                all_touched=cfg.all_touched,
            )
            bin_support += (mask > 0).astype(np.uint16)
        if bin_support.any():
            kde_local += fftconvolve(
                bin_support.astype(np.float32), kernels[bin_id], mode="same"
            ).astype(np.float32)

    rs, cs = central_slice(row, col, h, w, row0, col0)
    kde_c = kde_local[rs, cs].astype(np.float32, copy=False)
    support_c = support_local[rs, cs].astype(np.uint16, copy=False)
    raw_c = raw_union_local[rs, cs].astype(np.uint8, copy=False)
    dist_c = distance_local[rs, cs].astype(np.float32, copy=False)
    return kde_c, support_c, raw_c, dist_c, float(kde_c.max(initial=0.0)), len(indices)


def create_scaled_and_preview(
    kde_path: Path,
    scaled_path: Path,
    preview_path: Path,
    scaled_max: int,
    kde_max: float,
) -> Tuple[int, float]:
    positive_cells = 0
    sample_values: List[np.ndarray] = []
    with rasterio.open(kde_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint16", nodata=0, predictor=2)
        with rasterio.open(scaled_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                kde = src.read(1, window=window)
                positive = kde > 0
                positive_cells += int(positive.sum())
                if positive.any():
                    sample = kde[positive]
                    # Estimate the preview stretch limit by sampling rather than collecting every positive value.
                    if sample.size > 5000:
                        step = max(1, sample.size // 5000)
                        sample = sample[::step]
                    sample_values.append(sample.astype(np.float32, copy=False))
                scaled = np.rint(
                    np.clip(kde / max(float(kde_max), 1e-12) * scaled_max, 0, scaled_max)
                ).astype(np.uint16)
                scaled[positive & (scaled == 0)] = 1
                dst.write(scaled, 1, window=window)

        # Generate the preview through low-resolution resampling without loading the full raster into memory.
        max_side = 4096
        scale = max(src.width / max_side, src.height / max_side, 1.0)
        out_w = max(1, int(round(src.width / scale)))
        out_h = max(1, int(round(src.height / scale)))
        small = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)

    if sample_values:
        sampled = np.concatenate(sample_values)
        upper = float(np.percentile(sampled, 99.5)) if sampled.size else max(kde_max, 1.0)
    else:
        upper = 1.0
    upper = max(upper, 1e-12)
    image = np.round((1 - np.clip(small / upper, 0, 1)) * 255).astype(np.uint8)
    Image.fromarray(image).save(preview_path)
    return positive_cells, upper


def run(inp: Path, out: Path, cfg: Config):
    tile_size_px, actual_tile_size_m = tile_size_px_from_config(cfg)
    if tile_size_px < 16:
        raise ValueError('Too few pixels per tile; increase --tile-size-m or decrease --cell-size-m')
    if cfg.orientation_bins < 1:
        raise ValueError('orientation-bins must be greater than 0')
    if cfg.max_link_m < 0.0:
        raise ValueError('max-link-m must be >= 0; 0 disables splitting vehicle tracks by adjacent-point distance')
    if cfg.clear_output and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    df, crs = load_points(inp, cfg)
    ps = split_parts(df, cfg)
    transform, width, height, bounds = build_grid(ps, cfg)
    geoms, meta, tree, id_to_index, wkb_to_index = build_segments(ps, cfg)
    kernels = [kernel((i + 0.5) * math.pi / cfg.orientation_bins, cfg) for i in range(cfg.orientation_bins)]

    tag = f"along{cfg.sigma_along_m:g}m_cross{cfg.sigma_cross_m:g}m".replace(".", "p")
    kde_float_path = out / f"kde_directional_{tag}_float32.tif"
    kde_tmp_paths = write_empty_rasters(out, width, height, transform, crs, cfg)
    tmp_kde_path = kde_tmp_paths["kde"]

    tile_count = math.ceil(width / tile_size_px) * math.ceil(height / tile_size_px)
    max_kde = 0.0
    processed = 0
    with rasterio.open(tmp_kde_path, "r+") as kde_ds, rasterio.open(
        kde_tmp_paths["support"], "r+"
    ) as support_ds, rasterio.open(kde_tmp_paths["raw_union"], "r+") as raw_ds, rasterio.open(
        kde_tmp_paths["distance"], "r+"
    ) as distance_ds:
        for row, col, h, w in iter_tiles(width, height, tile_size_px):
            kde_c, support_c, raw_c, dist_c, tile_max, selected_count = process_tile(
                row,
                col,
                h,
                w,
                width,
                height,
                transform,
                geoms,
                meta,
                tree,
                id_to_index,
                wkb_to_index,
                kernels,
                cfg,
            )
            window = Window(col, row, w, h)
            kde_ds.write(kde_c, 1, window=window)
            support_ds.write(support_c, 1, window=window)
            raw_ds.write(raw_c, 1, window=window)
            distance_ds.write(dist_c, 1, window=window)
            max_kde = max(max_kde, tile_max)
            processed += 1
            if processed % 50 == 0 or processed == tile_count:
                print(
                    f"[KDE-560M-TILE] {processed}/{tile_count} "
                    f"row={row} col={col} selected_edges={selected_count} max={tile_max:.6g}",
                    flush=True,
                )

    tmp_kde_path.replace(kde_float_path)
    scaled_path = out / f"kde_directional_{tag}_uint16_0_{cfg.scaled_max}.tif"
    preview_path = out / f"kde_directional_{tag}_preview.png"
    positive_cells, preview_upper = create_scaled_and_preview(
        kde_float_path, scaled_path, preview_path, cfg.scaled_max, max_kde
    )

    report = {
        "input": str(inp),
        "input_rows_total": int(df.attrs.get("input_rows_total", len(df))),
        "rows": int(len(df)),
        "road_weight_retention_ratio": float(len(df) / max(int(df.attrs.get("input_rows_total", len(df))), 1)),
        "segments": int(df[cfg.segment_field].nunique()),
        "independent_tracks": int(df["_cid"].nunique()),
        "continuous_parts": int(len(ps)),
        "rasterized_edges": int(len(geoms)),
        "grid": {"width": width, "height": height, "bounds": bounds},
        "tile_size_px": int(tile_size_px),
        "tile_size_m_requested": float(cfg.tile_size_m),
        "tile_size_m_actual": float(actual_tile_size_m),
        "tile_origin_x": None if cfg.tile_origin_x is None else float(cfg.tile_origin_x),
        "tile_origin_y": None if cfg.tile_origin_y is None else float(cfg.tile_origin_y),
        "tile_alignment_mode": "gravity_report_or_origin" if cfg.tile_origin_x is not None and cfg.tile_origin_y is not None else "kde_extent_cell_aligned",
        "tiles": int(tile_count),
        "kde_max": float(max_kde),
        "positive_cells": int(positive_cells),
        "preview_upper_p995_sampled": float(preview_upper),
        "config": asdict(cfg),
        "float32": str(kde_float_path),
        "uint16": str(scaled_path),
        "support": str(kde_tmp_paths["support"]),
        "raw_union": str(kde_tmp_paths["raw_union"]),
        "distance": str(kde_tmp_paths["distance"]),
        "note": "560 m physical tiled KDE; tiles are written directly into merged GeoTIFF windows; no orientation_bins × height × width array is allocated",
    }
    (out / "directional_kde_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def load_gravity_tile_config(report_path: Path) -> Dict[str, float]:
    'Read 560 m tile settings from the gravity-adjustment report to align the KDE grid.'
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tile_cfg = report.get("gravity_tile_config", {})
    if not tile_cfg:
        raise ValueError(f"""Report is missing gravity_tile_config: {report_path}""")
    required = ["gravity_tile_size_m", "gravity_tile_origin_x_m", "gravity_tile_origin_y_m"]
    missing = [key for key in required if key not in tile_cfg]
    if missing:
        raise ValueError(f"""gravity_tile_config is missing fields: {missing}""")
    return {
        "tile_size_m": float(tile_cfg["gravity_tile_size_m"]),
        "tile_origin_x": float(tile_cfg["gravity_tile_origin_x_m"]),
        "tile_origin_y": float(tile_cfg["gravity_tile_origin_y_m"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Low-memory direction-constrained anisotropic KDE on 560 m x 560 m tiles, with standard outputs for S6 skeletonization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size-m", type=float, default=2.0)
    parser.add_argument("--sigma-along-m", type=float, default=4.0)
    parser.add_argument("--sigma-cross-m", type=float, default=1.5)
    parser.add_argument("--orientation-bins", type=int, default=16)
    parser.add_argument(
        "--max-link-m",
        type=float,
        default=0.0,
        help='Maximum distance between adjacent KDE track points; use 0 in vehicle mode to disable distance splitting',
    )
    parser.add_argument("--min-road-weight", type=float, default=0.35,
                        help='Use only main-road candidate points whose road_extraction_weight is at least this value')
    parser.add_argument("--require-mainroad-candidate", action="store_true",
                        help='Also require mainroad_candidate=True when the input contains this field')
    parser.add_argument("--tile-size-m", type=float, default=560.0, help='Physical KDE tile side length in meters; default 560 m')
    parser.add_argument("--tile-origin-x", type=float, default=None, help='Optional lower-left KDE tile origin X for alignment with gravity tiles')
    parser.add_argument("--tile-origin-y", type=float, default=None, help='Optional lower-left KDE tile origin Y for alignment with gravity tiles')
    parser.add_argument("--gravity-report", type=Path, default=None, help='Optional gravity_uturn_report.json; read tile size/origin automatically to align the KDE grid')
    parser.add_argument("--tile-size", type=int, default=0, help='Optional tile size in pixels; positive values override --tile-size-m for compatibility with older commands')
    parser.add_argument("--distance-max-m", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--no-clear-output", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tile_size_m = float(args.tile_size_m)
    tile_origin_x = args.tile_origin_x
    tile_origin_y = args.tile_origin_y
    gravity_report = ""
    if args.gravity_report is not None:
        gravity_report_path = args.gravity_report.expanduser().resolve()
        tile_cfg = load_gravity_tile_config(gravity_report_path)
        tile_size_m = tile_cfg["tile_size_m"]
        tile_origin_x = tile_cfg["tile_origin_x"]
        tile_origin_y = tile_cfg["tile_origin_y"]
        gravity_report = str(gravity_report_path)
        print(
            f"""[ALIGN] Aligning KDE tiles using gravity report: tile_size={tile_size_m:g} m, origin=({tile_origin_x:.3f}, {tile_origin_y:.3f})""",
            flush=True,
        )
    elif (tile_origin_x is None) ^ (tile_origin_y is None):
        raise ValueError('--tile-origin-x and --tile-origin-y must be specified together')

    cfg = Config(
        cell_size_m=float(args.cell_size_m),
        sigma_along_m=float(args.sigma_along_m),
        sigma_cross_m=float(args.sigma_cross_m),
        orientation_bins=int(args.orientation_bins),
        max_link_m=float(args.max_link_m),
        min_road_weight=float(args.min_road_weight),
        require_mainroad_candidate=bool(args.require_mainroad_candidate),
        tile_size_m=tile_size_m,
        tile_origin_x=None if tile_origin_x is None else float(tile_origin_x),
        tile_origin_y=None if tile_origin_y is None else float(tile_origin_y),
        gravity_report=gravity_report,
        tile_size_px_override=int(args.tile_size),
        distance_max_m=float(args.distance_max_m),
        block_size=int(args.block_size),
        all_touched=bool(args.all_touched),
        clear_output=not bool(args.no_clear_output),
    )
    run(args.input.expanduser().resolve(), args.output.expanduser().resolve(), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
