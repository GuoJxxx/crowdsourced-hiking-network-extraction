#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract and retain centerline candidates at the density thresholds in manuscript Equation 8."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    convolve,
    distance_transform_edt,
    label,
    maximum_filter,
)
from shapely.geometry import LineString, box, mapping
from skimage.morphology import thin

CODE = Path(__file__).resolve().parent
MODULES = CODE / "pipeline_modules"


def loadmod(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, MODULES / file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {MODULES / file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rm = loadmod("rm_lock", "skeleton_route_medoid.py")
seq = loadmod("seq_lock", "skeleton_switchback_sequence.py")
conn = loadmod("conn_lock", "skeleton_connectivity.py")
refine = loadmod("s6_kde_refine", "s6_kde_topology_refine.py")
base = rm.base

# Power-of-two thresholds. The old version started at 512; this version defaults to 1024,
# where 1024 corresponds to a robust reference rather than the single global maximum.
DEFAULT_THRESHOLD_LEVELS = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
LEGACY_THRESHOLD_LEVELS = [512, 256, 128, 64, 32, 16, 8, 4]


# =============================================================================
# Thresholds and scaling
# =============================================================================
def parse_thresholds(text: Optional[str], legacy: bool = False) -> List[int]:
    if not text:
        return LEGACY_THRESHOLD_LEVELS if legacy else DEFAULT_THRESHOLD_LEVELS
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    values = sorted(set(values), reverse=True)
    if not values:
        raise ValueError("Threshold list is empty")
    for value in values:
        if value <= 0 or value > 1024:
            raise ValueError(f"Threshold must be within 1-1024: {value}")
    return values


# Eq. 8: map directional density to the common threshold scale using
# the shared density reference before progressive skeleton extraction.
def scaled_from_float_reference(kde: np.ndarray, reference: float) -> np.ndarray:
    ref = max(float(reference), 1e-12)
    out = np.rint(np.clip(kde / ref * 1024.0, 0.0, 1024.0)).astype(np.uint16)
    out[(kde > 0) & (out == 0)] = 1
    return out


def scaled_from_global_max(kde: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(kde)) if kde.size else 0.0
    return scaled_from_float_reference(kde, mx)


def reservoir_update(sample: List[float], values: np.ndarray, max_samples: int, rng: np.random.Generator, seen: int) -> int:
    """Sample as a stream to avoid storing all KDE values within the full corridor in memory at once."""
    values = values[np.isfinite(values) & (values > 0)]
    for value in values.astype(float, copy=False):
        seen += 1
        if len(sample) < max_samples:
            sample.append(float(value))
        else:
            j = int(rng.integers(0, seen))
            if j < max_samples:
                sample[j] = float(value)
    return seen


def compute_threshold_reference(
    kde_src: rasterio.DatasetReader,
    support_src: rasterio.DatasetReader,
    distance_src: rasterio.DatasetReader,
    relaxed_corridor_m: float,
    percentile: float,
    max_samples: int,
    mode: str,
) -> Dict[str, object]:
    """Compute the threshold reference for floating-point KDE.

    The default mode=active_percentile uses a percentile of positive KDE pixels in trajectory corridors or with support.
    This prevents abundant background zeros and extreme local hotspots in large study areas from dominating thresholds.
    """
    if mode == "legacy_global_max":
        max_value = 0.0
        positive_count = 0
        for _, window in kde_src.block_windows(1):
            kde = kde_src.read(1, window=window, masked=False)
            valid = np.isfinite(kde) & (kde > 0)
            if valid.any():
                positive_count += int(valid.sum())
                max_value = max(max_value, float(np.nanmax(kde[valid])))
        return {
            "threshold_mode": mode,
            "threshold_reference": float(max_value),
            "positive_count_used": int(positive_count),
            "percentile": None,
            "note": "Legacy global-maximum normalization; low-density trails may be suppressed over large areas",
        }

    rng = np.random.default_rng(20260214)
    sample: List[float] = []
    seen = 0
    active_count = 0
    for _, window in kde_src.block_windows(1):
        kde = kde_src.read(1, window=window, masked=False)
        support = support_src.read(1, window=window, masked=False)
        distance = distance_src.read(1, window=window, masked=False)
        active = np.isfinite(kde) & (kde > 0) & ((distance <= relaxed_corridor_m) | (support > 0))
        active_count += int(active.sum())
        if active.any():
            seen = reservoir_update(sample, kde[active], max_samples, rng, seen)
    if not sample:
        raise RuntimeError("Cannot compute KDE threshold reference: no positive KDE pixels in track corridors")
    arr = np.asarray(sample, dtype=float)
    reference = float(np.quantile(arr, float(percentile) / 100.0))
    if not math.isfinite(reference) or reference <= 0:
        reference = float(np.nanmax(arr))
    return {
        "threshold_mode": mode,
        "threshold_reference": reference,
        "positive_count_used": int(active_count),
        "sample_count": int(arr.size),
        "percentile": float(percentile),
        "sample_min": float(np.nanmin(arr)),
        "sample_median": float(np.nanmedian(arr)),
        "sample_max": float(np.nanmax(arr)),
        "note": "Threshold reference uses a robust percentile of positive KDE within track corridors",
    }


# =============================================================================
# Bridge suppression and multithreshold foreground
# =============================================================================
def evidence_connects(comp: np.ndarray, evidence: np.ndarray, prev_labels: np.ndarray, touched: Sequence[int]) -> bool:
    ev = comp & evidence
    labs, n = label(ev, np.ones((3, 3), np.uint8))
    if n == 0:
        return False
    dil = {x: binary_dilation(prev_labels == x, iterations=1) for x in touched}
    for k in range(1, n + 1):
        e = labs == k
        if sum(bool(np.any(e & dil[x])) for x in touched) >= 2:
            return True
    return False


def voronoi_split(comp: np.ndarray, prev_labels: np.ndarray, touched: Sequence[int], margin: float = 0.75) -> np.ndarray:
    ds = np.stack([distance_transform_edt(prev_labels != x) for x in touched])
    ss = np.sort(ds, axis=0)
    watershed = (ss[1] - ss[0]) <= margin
    return comp & ~watershed


def bridge_aware_foreground(
    kde_scaled: np.ndarray,
    support: np.ndarray,
    distance: np.ndarray,
    thresholds: Sequence[int],
    strict_corridor_m: float = 3.0,
    relaxed_corridor_m: float = 5.0,
    bridge_merge_support_min: int = 3,
    bridge_evidence_dilation: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # The corridor still admits low-threshold road candidates near measured trajectories to preserve main-road edges.
    repeated_near = binary_dilation(support >= 2, iterations=3)
    corridor = (distance <= strict_corridor_m) | ((distance <= relaxed_corridor_m) & repeated_near)

    # Use a stricter merge criterion: permit a low-threshold area to connect two existing components only when
    # continuous measured evidence inside comp has sufficient independent-trajectory support.
    # The previous support>=2 and dilation=1 criteria could mistake dilated evidence between nearby U-bend arms for a valid merge.
    repeated_evidence = support >= int(max(1, bridge_merge_support_min))
    if int(bridge_evidence_dilation) > 0:
        repeated_evidence = binary_dilation(repeated_evidence, iterations=int(bridge_evidence_dilation))
    accepted = np.zeros_like(corridor, bool)
    rows: List[Dict[str, object]] = []

    for t in thresholds:
        candidate = (kde_scaled >= int(t)) & corridor
        if not accepted.any():
            accepted = candidate.copy()
            rows.append({
                "threshold": int(t),
                "candidate": int(candidate.sum()),
                "accepted": int(accepted.sum()),
                "split_merges": 0,
                "allowed_merges": 0,
            })
            continue
        prev_labels, _ = label(accepted, np.ones((3, 3), np.uint8))
        new = candidate & ~accepted
        new_labels, new_count = label(new, np.ones((3, 3), np.uint8))
        add = np.zeros_like(accepted)
        split_count = 0
        allowed_count = 0
        for k in range(1, new_count + 1):
            comp = new_labels == k
            ring = binary_dilation(comp, iterations=1) & accepted
            touched = np.unique(prev_labels[ring])
            touched = touched[touched > 0].tolist()
            if len(touched) <= 1:
                add |= comp
                continue
            if evidence_connects(comp, repeated_evidence, prev_labels, touched):
                core = comp & binary_dilation(repeated_evidence, iterations=1)
                add |= core
                residual = comp & ~binary_dilation(core, iterations=1)
                if residual.any():
                    add |= voronoi_split(residual, prev_labels, touched)
                allowed_count += 1
            else:
                add |= voronoi_split(comp, prev_labels, touched)
                split_count += 1
        accepted |= add
        rows.append({
            "threshold": int(t),
            "candidate": int(candidate.sum()),
            "accepted": int(accepted.sum()),
            "split_merges": int(split_count),
            "allowed_merges": int(allowed_count),
        })
    return accepted, thin(accepted).astype(bool), corridor, pd.DataFrame(rows)


# =============================================================================
# Tile windows and local data
# =============================================================================
def ceil_int(value: float) -> int:
    return int(math.ceil(float(value)))


def iter_core_windows(width: int, height: int, tile_px: int) -> Iterable[Tuple[int, Window]]:
    tile_id = 0
    for row_off in range(0, height, tile_px):
        win_h = min(tile_px, height - row_off)
        for col_off in range(0, width, tile_px):
            win_w = min(tile_px, width - col_off)
            tile_id += 1
            yield tile_id, Window(col_off, row_off, win_w, win_h)


def expand_window(window: Window, halo_px: int, width: int, height: int) -> Window:
    col0 = max(0, int(window.col_off) - halo_px)
    row0 = max(0, int(window.row_off) - halo_px)
    col1 = min(width, int(window.col_off + window.width) + halo_px)
    row1 = min(height, int(window.row_off + window.height) + halo_px)
    return Window(col0, row0, col1 - col0, row1 - row0)


def core_slices(core: Window, extended: Window) -> Tuple[slice, slice]:
    r0 = int(core.row_off - extended.row_off)
    c0 = int(core.col_off - extended.col_off)
    return slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width))


def window_bbox(transform: Affine, window: Window):
    left, bottom, right, top = window_bounds(window, transform)
    return box(left, bottom, right, top)


def ensure_xy_columns(points: pd.DataFrame) -> pd.DataFrame:
    pts = points.copy()
    if {"x_m", "y_m"}.issubset(pts.columns):
        return pts
    if {"gravity_x_m", "gravity_y_m"}.issubset(pts.columns):
        pts["x_m"] = pd.to_numeric(pts["gravity_x_m"], errors="coerce")
        pts["y_m"] = pd.to_numeric(pts["gravity_y_m"], errors="coerce")
        return pts
    raise ValueError("Track-point table lacks x_m/y_m and gravity_x_m/gravity_y_m")


def subset_points(points: pd.DataFrame, transform: Affine, window: Window, pad_m: float = 0.0) -> pd.DataFrame:
    b = window_bbox(transform, window).buffer(float(pad_m))
    x = pd.to_numeric(points["x_m"], errors="coerce")
    y = pd.to_numeric(points["y_m"], errors="coerce")
    mask = x.between(b.bounds[0], b.bounds[2]) & y.between(b.bounds[1], b.bounds[3])
    return points.loc[mask].copy()


def read_consensus(path: Optional[Path]) -> gpd.GeoDataFrame:
    if path is None or not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=None)
    try:
        return gpd.read_file(path, layer="uturn_consensus_lines")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=None)


def read_activity_zones(path: Optional[Path], layer: str = "activity_suppression_zones") -> gpd.GeoDataFrame:
    if path is None or not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=None)
    try:
        return gpd.read_file(path, layer=layer)
    except Exception as exc:
        raise ValueError(f"Cannot read activity-suppression zones {path} layer={layer}: {exc}") from exc


def subset_activity_zones(
    zones: gpd.GeoDataFrame,
    transform: Affine,
    window: Window,
    crs,
    pad_m: float = 0.0,
) -> gpd.GeoDataFrame:
    if zones is None or zones.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    local_box = window_bbox(transform, window).buffer(float(pad_m))
    work = zones
    if work.crs is not None and crs is not None and str(work.crs) != str(crs):
        work = work.to_crs(crs)
    try:
        idx = work.sindex.query(local_box, predicate="intersects")
        return work.iloc[idx].copy()
    except Exception:
        return work.loc[work.geometry.intersects(local_box)].copy()


def subset_consensus(consensus: gpd.GeoDataFrame, transform: Affine, window: Window, crs, pad_m: float = 0.0) -> gpd.GeoDataFrame:
    if consensus is None or consensus.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    local_box = window_bbox(transform, window).buffer(float(pad_m))
    if consensus.crs is not None and crs is not None and str(consensus.crs) != str(crs):
        consensus = consensus.to_crs(crs)
    try:
        idx = consensus.sindex.query(local_box, predicate="intersects")
        out = consensus.iloc[idx].copy()
    except Exception:
        out = consensus.loc[consensus.geometry.intersects(local_box)].copy()
    return out


def raster_shapes_for_geometries(geometries: Iterable[object], buffer_m: float = 0.0):
    """Return finite, non-empty rasterio shapes without turning zero-buffer lines empty.

    Shapely ``LineString.buffer(0)`` yields an empty polygon.  For a zero buffer,
    rasterize the original geometry directly; only apply ``buffer`` when a positive
    corridor width is explicitly requested.
    """
    shapes = []
    distance = max(0.0, float(buffer_m))
    for geom in geometries:
        if geom is None or getattr(geom, "is_empty", True):
            continue
        work = geom.buffer(distance) if distance > 0.0 else geom
        if work is None or getattr(work, "is_empty", True):
            continue
        try:
            bounds = np.asarray(work.bounds, dtype=float)
        except Exception:
            continue
        if bounds.size != 4 or not np.isfinite(bounds).all():
            continue
        shapes.append((mapping(work), 1))
    return shapes


def zones_mask(zones: Sequence[Dict[str, object]], shape: Tuple[int, int], tr: Affine) -> np.ndarray:
    geoms = [(mapping(z["geometry"]), 1) for z in zones if z.get("geometry") is not None]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(geoms, out_shape=shape, transform=tr, fill=0, dtype="uint8", all_touched=True).astype(bool)


# =============================================================================
# Locked-route endpoint connections
# =============================================================================
def route_endpoint_connectors(
    records: Sequence[Dict[str, object]],
    points: pd.DataFrame,
    base_outside: np.ndarray,
    locked_routes: np.ndarray,
    shape: Tuple[int, int],
    tr: Affine,
    max_along_m: float = 45.0,
    max_snap_m: float = 6.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extend along the same measured trajectory selected for each locked route to connect to the main skeleton outside the zone."""
    connectors = np.zeros(shape, bool)
    rows: List[Dict[str, object]] = []
    target_dil = binary_dilation(base_outside, iterations=1)
    if points.empty or "clean_track_id" not in points.columns:
        return connectors, pd.DataFrame(rows)

    for rec in records:
        tid = rec.get("selected_track_id")
        geom = rec.get("geometry")
        if tid is None or geom is None:
            continue
        group = points[points.clean_track_id.astype(str).eq(str(tid))].sort_values("clean_point_index")
        xy = group[["x_m", "y_m"]].to_numpy(float)
        if len(xy) < 2:
            continue
        cd = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        for end_no, coord in enumerate([np.asarray(geom.coords[0], float), np.asarray(geom.coords[-1], float)]):
            i = int(np.argmin(np.linalg.norm(xy - coord, axis=1)))
            if float(np.linalg.norm(xy[i] - coord)) > max_snap_m:
                continue
            candidates = []
            for step in (-1, 1):
                j = i
                while 0 <= j + step < len(xy) and abs(cd[j + step] - cd[i]) <= max_along_m:
                    j += step
                    rr, cc = rasterio.transform.rowcol(tr, xy[j, 0], xy[j, 1])
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1] and target_dil[rr, cc]:
                        if abs(cd[j] - cd[i]) >= 2.0:
                            candidates.append((abs(cd[j] - cd[i]), j))
                        break
            if not candidates:
                continue
            _, j = min(candidates)
            lo, hi = sorted((i, j))
            seg = xy[lo : hi + 1]
            if len(seg) < 2:
                continue
            line = LineString(seg)
            mask = rasterize(
                [(mapping(line), 1)],
                out_shape=shape,
                transform=tr,
                fill=0,
                dtype="uint8",
                all_touched=False,
            ).astype(bool)
            connectors |= mask
            rows.append({
                "route_type": "cluster_or_sequence",
                "route_id": rec.get("cluster_id", f"{rec.get('pair_a', '')}-{rec.get('pair_b', '')}"),
                "selected_track_id": tid,
                "endpoint": int(end_no),
                "connector_length_m": float(line.length),
            })
    return thin(connectors).astype(bool), pd.DataFrame(rows)


def remove_small_binary_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove tiny rescue fragments so scattered high-support pixels do not enter the main skeleton directly."""
    if int(min_pixels) <= 1 or not mask.any():
        return mask.astype(bool)
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_pixels)
    keep[0] = False
    return keep[labels]


def disk_footprint(radius_px: int) -> np.ndarray:
    """Create a circular neighborhood kernel."""
    radius_px = max(0, int(radius_px))
    if radius_px <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def keep_rescue_components(
    rescue: np.ndarray,
    anchor: np.ndarray,
    min_pixels: int,
) -> np.ndarray:
    """Retain rescue components that meet the length requirement and connect to the main skeleton/protected corridor."""
    rescue = rescue.astype(bool)
    if not rescue.any():
        return rescue

    labels_arr, count = label(
        rescue,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels_arr.ravel())
    anchor_labels = np.unique(labels_arr[anchor & (labels_arr > 0)])
    keep = np.zeros(count + 1, dtype=bool)

    for component_id in range(1, count + 1):
        if sizes[component_id] < max(1, int(min_pixels)):
            continue
        if component_id in anchor_labels:
            keep[component_id] = True

    return keep[labels_arr]


def build_dense_support_rescue(
    support: np.ndarray,
    distance: np.ndarray,
    scaled: np.ndarray,
    activity_mask: np.ndarray,
    protected_mask: np.ndarray,
    existing_skeleton: np.ndarray,
    cell_size_m: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build a main-road rescue skeleton using exact pixel support plus aggregated neighborhood support.

    independent_track_support.tif records independent-trajectory support at pixel level.
    On wide roads or with lateral trajectory dispersion, individual-pixel support may be low while many
    distinct trajectories provide neighborhood evidence. Restore these roads using total neighborhood support and occupied-pixel counts,
    while requiring connection to an existing skeleton or protected road corridor to avoid reintroducing parking-lot clutter.
    """
    cell = max(float(cell_size_m), 1e-9)
    radius_px = max(
        1,
        int(math.ceil(float(args.dense_rescue_radius_m) / cell)),
    )
    footprint = disk_footprint(radius_px)
    kernel = footprint.astype(np.float32)

    support_float = support.astype(np.float32, copy=False)
    local_support_sum = convolve(
        support_float,
        kernel,
        mode="constant",
        cval=0.0,
    )
    local_occupied_pixels = convolve(
        (support > 0).astype(np.float32),
        kernel,
        mode="constant",
        cval=0.0,
    )
    local_support_max = maximum_filter(
        support,
        footprint=footprint,
        mode="constant",
        cval=0,
    )

    exact = support >= int(args.support_rescue_min_tracks)
    dense = (
        (local_support_sum >= float(args.dense_rescue_support_sum_min))
        & (
            local_occupied_pixels
            >= float(args.dense_rescue_occupied_pixels_min)
        )
        & (
            local_support_max
            >= int(args.dense_rescue_near_support_min)
        )
    )

    evidence = (
        (exact | dense)
        & (
            distance
            <= float(args.support_rescue_max_distance_m)
        )
        & (
            scaled
            >= int(args.support_rescue_kde_min_level)
        )
    )

    closing_px = max(
        0,
        int(math.ceil(float(args.dense_rescue_closing_m) / cell)),
    )
    if closing_px > 0:
        evidence = binary_closing(
            evidence,
            structure=disk_footprint(closing_px),
        )

    if int(args.support_rescue_dilation) > 0:
        evidence = binary_dilation(
            evidence,
            iterations=int(args.support_rescue_dilation),
        )

    candidate_skeleton = thin(evidence).astype(bool)

    connection_px = max(
        1,
        int(math.ceil(float(args.dense_rescue_connection_m) / cell)),
    )
    anchor = (
        binary_dilation(
            existing_skeleton.astype(bool),
            iterations=connection_px,
        )
        | protected_mask
        | exact
    )

    rescued = keep_rescue_components(
        candidate_skeleton,
        anchor,
        min_pixels=int(args.support_rescue_min_component_pixels),
    )
    rescued = thin(rescued).astype(bool)

    info = {
        "dense_radius_px": int(radius_px),
        "exact_candidate_pixels": int(exact.sum()),
        "dense_candidate_pixels": int(dense.sum()),
        "combined_evidence_pixels": int(evidence.sum()),
        "candidate_skeleton_pixels": int(candidate_skeleton.sum()),
        "rescued_pixels": int(rescued.sum()),
        "local_support_sum_max": float(
            np.nanmax(local_support_sum)
        ) if local_support_sum.size else 0.0,
        "local_occupied_pixels_max": float(
            np.nanmax(local_occupied_pixels)
        ) if local_occupied_pixels.size else 0.0,
    }
    return rescued, dense, info


# =============================================================================
# Outputs
# =============================================================================
def output_profile_like(src: rasterio.DatasetReader, dtype: str = "uint8") -> Dict[str, object]:
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=0,
        compress="deflate",
        predictor=2 if dtype != "uint8" else 1,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    # GeoTIFF block size should be multiples of 16. Keep safe values for very small rasters.
    profile["blockxsize"] = min(512, max(16, int(math.ceil(min(src.width, 512) / 16.0) * 16)))
    profile["blockysize"] = min(512, max(16, int(math.ceil(min(src.height, 512) / 16.0) * 16)))
    return profile


def create_outputs(output_dir: Path, src: rasterio.DatasetReader) -> Dict[str, rasterio.io.DatasetWriter]:
    names = [
        "foreground_bridge_aware",
        "skeleton_global_bridge_aware",
        "skeleton_global_kde_ridge_refined",
        "skeleton_kde_ridge_moved",
        "skeleton_primary_route_locked",
        "skeleton_auxiliary_singletrack_safe",
        "skeleton_auxiliary_singletrack_raw_after_primary",
        "skeleton_all_evidence_route_locked",
        "skeleton_locked_routes",
        "skeleton_route_endpoint_connectors",
        "skeleton_route_endpoint_connectors_raw",
        "skeleton_lock_zones",
        "skeleton_singletrack_raw",
        "skeleton_singletrack_removed_in_lockzone",
        "skeleton_old_removed_in_lockzone",
        "skeleton_activity_removed",
        "skeleton_support_rescued",
        "skeleton_support_rescued_raw",
        "skeleton_dense_support_candidate",
        "skeleton_protected_corridor_mask",
        "skeleton_effective_lock_zones",
        "skeleton_lockzone_preserved",
    ]
    profile = output_profile_like(src, "uint8")
    return {name: rasterio.open(output_dir / f"{name}.tif", "w", **profile) for name in names}


def close_outputs(outputs: Dict[str, rasterio.io.DatasetWriter]) -> None:
    for dst in outputs.values():
        dst.close()


def write_core(outputs: Dict[str, rasterio.io.DatasetWriter], arrays: Dict[str, np.ndarray], core: Window, extended: Window) -> Dict[str, int]:
    rs, cs = core_slices(core, extended)
    counts: Dict[str, int] = {}
    for name, arr in arrays.items():
        core_arr = arr[rs, cs].astype(np.uint8, copy=False)
        outputs[name].write(core_arr, 1, window=core)
        counts[name] = int(core_arr.sum())
    return counts


# =============================================================================
# Single-tile processing
# =============================================================================
def process_tile(
    tile_id: int,
    core: Window,
    extended: Window,
    kde_src: rasterio.DatasetReader,
    support_src: rasterio.DatasetReader,
    distance_src: rasterio.DatasetReader,
    raw_union_src: rasterio.DatasetReader,
    points: pd.DataFrame,
    consensus: gpd.GeoDataFrame,
    activity_zones: gpd.GeoDataFrame,
    protected_zones: gpd.GeoDataFrame,
    cfg,
    args,
    thresholds: Sequence[int],
    threshold_reference: Optional[float],
) -> Tuple[Dict[str, np.ndarray], List[pd.DataFrame], Dict[str, object]]:
    kde = kde_src.read(1, window=extended, masked=False)
    support = support_src.read(1, window=extended, masked=False)
    distance = distance_src.read(1, window=extended, masked=False)
    raw_union = raw_union_src.read(1, window=extended, masked=False) > 0
    local_transform = window_transform(extended, kde_src.transform)
    shape = kde.shape
    cell = abs(kde_src.transform.a)

    local_points = subset_points(points, kde_src.transform, extended, pad_m=args.local_vector_pad_m)
    local_consensus = subset_consensus(consensus, kde_src.transform, extended, kde_src.crs, pad_m=args.local_vector_pad_m)
    local_activity = subset_activity_zones(
        activity_zones, kde_src.transform, extended, kde_src.crs,
        pad_m=args.activity_zone_pad_m,
    )
    local_protected = subset_activity_zones(
        protected_zones, kde_src.transform, extended, kde_src.crs,
        pad_m=args.activity_zone_pad_m,
    )

    active = (
        raw_union.any()
        or np.any(support > 0)
        or np.any((distance <= args.relaxed_corridor_m) & np.isfinite(kde) & (kde > 0))
        or not local_points.empty
        or not local_consensus.empty
    )
    if not active:
        zeros = np.zeros(shape, dtype=bool)
        arrays = {
            "foreground_bridge_aware": zeros,
            "skeleton_global_bridge_aware": zeros,
            "skeleton_global_kde_ridge_refined": zeros,
            "skeleton_kde_ridge_moved": zeros,
            "skeleton_primary_route_locked": zeros,
            "skeleton_auxiliary_singletrack_safe": zeros,
            "skeleton_auxiliary_singletrack_raw_after_primary": zeros,
            "skeleton_all_evidence_route_locked": zeros,
            "skeleton_locked_routes": zeros,
            "skeleton_route_endpoint_connectors": zeros,
            "skeleton_route_endpoint_connectors_raw": zeros,
            "skeleton_lock_zones": zeros,
            "skeleton_singletrack_raw": zeros,
            "skeleton_singletrack_removed_in_lockzone": zeros,
            "skeleton_old_removed_in_lockzone": zeros,
            "skeleton_activity_removed": zeros,
            "skeleton_support_rescued": zeros,
            "skeleton_support_rescued_raw": zeros,
            "skeleton_dense_support_candidate": zeros,
            "skeleton_protected_corridor_mask": zeros,
            "skeleton_effective_lock_zones": zeros,
            "skeleton_lockzone_preserved": zeros,
        }
        return arrays, [], {"tile_id": tile_id, "skipped": True}

    if kde.dtype.kind == "f":
        # For both active_percentile and legacy_global_max,
        # use the single global threshold_reference obtained by the main pipeline's preliminary scan.
        # Do not renormalize by each tile's local maximum, which would give 512/256/... different meanings across tiles.
        assert threshold_reference is not None
        scaled = scaled_from_float_reference(kde, threshold_reference)
    else:
        scaled = kde.astype(np.uint16, copy=False)

    foreground, global_sk, corridor, stage = bridge_aware_foreground(
        scaled,
        support,
        distance,
        thresholds=thresholds,
        strict_corridor_m=args.strict_corridor_m,
        relaxed_corridor_m=args.relaxed_corridor_m,
        bridge_merge_support_min=args.bridge_merge_support_min,
        bridge_evidence_dilation=args.bridge_evidence_dilation,
    )
    stage["tile_id"] = tile_id
    stage["tile_col_off"] = int(core.col_off)
    stage["tile_row_off"] = int(core.row_off)

    # S3 has already identified activity zones such as parking lots and plazas as suppression polygons;
    # remove only ordinary KDE/thin skeletons inside those polygons. S3 has excluded stable road corridors from suppression zones,
    # and medoid-locked paths through U-bends/continuous switchbacks may be restored in later steps.
    if local_activity is not None and not local_activity.empty:
        activity_geoms = raster_shapes_for_geometries(
            local_activity.geometry,
            buffer_m=float(args.activity_suppression_buffer_m),
        )
        activity_mask = rasterize(
            activity_geoms,
            out_shape=shape,
            transform=local_transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool) if activity_geoms else np.zeros(shape, bool)
    else:
        activity_mask = np.zeros(shape, bool)

    if local_protected is not None and not local_protected.empty:
        protected_geoms = raster_shapes_for_geometries(
            local_protected.geometry,
            buffer_m=float(args.protected_corridor_buffer_m),
        )
        protected_mask = rasterize(
            protected_geoms,
            out_shape=shape,
            transform=local_transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool) if protected_geoms else np.zeros(shape, bool)
    else:
        protected_mask = np.zeros(shape, bool)

    # v0.5: activity zones denote semantically disordered areas rather than road-exclusion zones.
    # S3 already reduces disordered-trajectory contributions through road_weight; S6 no longer hard-deletes
    # KDE/multitrajectory skeletons within activity zones, which would systematically sever main roads through villages, parking lots, and plazas.
    # Retain activity_removed for diagnostics only: it marks skeleton pixels the previous logic would have deleted.
    activity_mask &= ~protected_mask
    activity_removed = global_sk & activity_mask

    # v0.5: use global_bridge_aware only as a topological scaffold.
    # Snap long degree-2 main-road chains to KDE ridges using dynamic programming along local normals;
    # fix endpoints/intersections and leave pure cycles unmoved in S6 for S7's arbitrary-polygon cycle resolver.
    global_sk_before_ridge = global_sk.copy()
    if args.disable_kde_ridge_refine:
        ridge_moved = np.zeros_like(global_sk, dtype=bool)
        ridge_rows = []
    else:
        global_sk, ridge_moved, ridge_rows = refine.refine_primary_to_kde_ridge(
            global_sk,
            scaled,
            support,
            distance,
            cell_size_m=cell,
            search_radius_m=args.kde_ridge_search_radius_m,
            min_chain_m=args.kde_ridge_min_chain_m,
            min_local_ridge_ratio=args.kde_ridge_min_local_ratio,
            support_weight=args.kde_ridge_support_weight,
            distance_weight=args.kde_ridge_distance_weight,
            offset_penalty=args.kde_ridge_offset_penalty,
            transition_penalty=args.kde_ridge_transition_penalty,
            min_mean_kde_gain=args.kde_ridge_min_mean_kde_gain,
            max_distance_m=args.kde_ridge_max_distance_m,
            exclusion_mask=np.zeros_like(activity_mask, dtype=bool),
        )
    global_sk_ridge_refined = global_sk.copy()

    # Rescue through aggregated multitrajectory evidence:
    # use both exact individual-pixel support and total neighborhood support with occupied-pixel counts.
    # This recovers main routes on wide roads/curves where lateral dispersion coexists with dense neighborhood trajectories.
    rescue_raw, dense_candidate, rescue_info = build_dense_support_rescue(
        support=support,
        distance=distance,
        scaled=scaled,
        activity_mask=activity_mask,
        protected_mask=protected_mask,
        existing_skeleton=global_sk,
        cell_size_m=cell,
        args=args,
    )
    rescue, rescue_topology_info = refine.filter_rescue_by_ridge_and_attachments(
        rescue_raw,
        global_sk,
        scaled,
        support,
        cell_size_m=cell,
        min_ridge_ratio=args.support_rescue_min_local_ridge_ratio,
        ridge_radius_m=args.support_rescue_ridge_radius_m,
        max_primary_attachments=args.support_rescue_max_primary_attachments,
        bridge_min_support=args.support_rescue_bridge_min_support,
        bridge_min_support_fraction=args.support_rescue_bridge_min_support_fraction,
        bridge_min_ridge_ratio=args.support_rescue_bridge_min_ridge_ratio,
    )
    global_sk = thin(global_sk | rescue).astype(bool)
    foreground |= rescue

    frames: List[pd.DataFrame] = [stage]
    if ridge_rows:
        ridge_df = pd.DataFrame(ridge_rows)
        ridge_df["table"] = "kde_ridge_refinement"
        ridge_df["tile_id"] = tile_id
        frames.append(ridge_df)

    if local_points.empty:
        single_seed = np.zeros(shape, bool)
        single_table = pd.DataFrame()
    else:
        single_seed, single_table = conn.rasterize_singletrack_edges(
            local_points,
            global_sk.shape,
            local_transform,
            max_link_m=cfg.max_track_link_m,
            transition_edges=0,
        )
        if single_table is not None and not single_table.empty:
            single_table["tile_id"] = tile_id
            frames.append(single_table)

    if local_consensus is not None and not local_consensus.empty and not local_points.empty:
        after_c, ins_c, rem_c, rec_c, z_c = rm.replace_clusters(
            global_sk,
            local_points,
            local_consensus,
            scaled,
            support,
            local_transform,
            cfg,
        )
    else:
        after_c = global_sk
        ins_c = np.zeros(shape, bool)
        rec_c = []
        z_c = []

    if not local_points.empty:
        after_s, ins_s, rem_s, rec_s, z_s = seq.replace_pairs(
            after_c,
            local_points,
            scaled,
            support,
            local_transform,
            cfg,
            args.min_pair_tracks,
            args.pair_buffer_m,
        )
    else:
        after_s = after_c
        ins_s = np.zeros(shape, bool)
        rec_s = []
        z_s = []

    lock_zone = zones_mask(
        z_c + z_s,
        global_sk.shape,
        local_transform,
    )
    locked_routes = thin(ins_c | ins_s).astype(bool)
    route_corr = binary_dilation(
        locked_routes,
        iterations=max(
            1,
            int(math.ceil(args.route_corridor_m / cell)),
        ),
    )

    # The old version cleared ordinary skeletons throughout U-bend/continuous-switchback diagnostic polygons.
    # Overlapping event polygons or complex trajectory clusters could therefore erase genuine connecting roads.
    # The new version cleans only the intersection of the original lock_zone and the locked-medoid path buffer.
    lock_clear_px = max(
        1,
        int(math.ceil(float(args.lock_clear_buffer_m) / cell)),
    )
    if locked_routes.any():
        lock_route_buffer = binary_dilation(
            locked_routes,
            iterations=lock_clear_px,
        )
        effective_lock_zone = lock_zone & lock_route_buffer
    else:
        effective_lock_zone = np.zeros_like(lock_zone, dtype=bool)

    lockzone_preserved = (
        global_sk
        & lock_zone
        & ~effective_lock_zone
    )
    base_outside = thin(
        global_sk & ~effective_lock_zone
    ).astype(bool)
    endpoint_connectors_raw, endpoint_table = route_endpoint_connectors(
        rec_c + rec_s,
        local_points,
        base_outside,
        locked_routes,
        global_sk.shape,
        local_transform,
        max_along_m=args.endpoint_connector_max_along_m,
        max_snap_m=args.endpoint_connector_snap_m,
    )
    connector_base = base_outside | thin(locked_routes).astype(bool)
    if args.disable_endpoint_connector_cycle_filter:
        endpoint_connectors = endpoint_connectors_raw
        connector_info = {
            "connector_components": 0,
            "connector_components_accepted": 0,
            "connector_same_component_cycle_risk_rejected": 0,
            "connector_ambiguous_rejected": 0,
        }
    else:
        endpoint_connectors, connector_info = refine.filter_additions_connecting_distinct_components(
            endpoint_connectors_raw, connector_base
        )
    if endpoint_table is not None and not endpoint_table.empty:
        endpoint_table["tile_id"] = tile_id
        frames.append(endpoint_table)

    primary = (
        base_outside
        | thin(locked_routes).astype(bool)
        | endpoint_connectors
        | (rescue & ~effective_lock_zone)
    )
    primary = thin(primary).astype(bool)

    single_safe = (
        (single_seed & ~effective_lock_zone)
        | (single_seed & route_corr)
    )
    auxiliary_raw = thin(single_safe & ~primary).astype(bool)
    if args.disable_auxiliary_leaf_only:
        auxiliary = auxiliary_raw
        auxiliary_info = {
            "auxiliary_components": 0,
            "auxiliary_isolated_kept": 0,
            "auxiliary_leaf_components_kept": 0,
            "auxiliary_multi_primary_components_removed": 0,
            "auxiliary_primary_hugging_components_removed": 0,
        }
    else:
        auxiliary, auxiliary_info = refine.filter_auxiliary_leaf_only(
            auxiliary_raw,
            primary,
            max_primary_attachments=args.auxiliary_max_primary_attachments,
            max_primary_contact_fraction=args.auxiliary_max_primary_contact_fraction,
        )
    all_evidence = primary | auxiliary

    arrays = {
        "foreground_bridge_aware": foreground,
        "skeleton_global_bridge_aware": global_sk_before_ridge,
        "skeleton_global_kde_ridge_refined": global_sk_ridge_refined,
        "skeleton_kde_ridge_moved": ridge_moved,
        "skeleton_primary_route_locked": primary,
        "skeleton_auxiliary_singletrack_safe": auxiliary,
        "skeleton_auxiliary_singletrack_raw_after_primary": auxiliary_raw,
        "skeleton_all_evidence_route_locked": all_evidence,
        "skeleton_locked_routes": locked_routes,
        "skeleton_route_endpoint_connectors": endpoint_connectors,
        "skeleton_route_endpoint_connectors_raw": endpoint_connectors_raw,
        "skeleton_lock_zones": lock_zone,
        "skeleton_singletrack_raw": single_seed,
        "skeleton_singletrack_removed_in_lockzone": (
            single_seed
            & effective_lock_zone
            & ~route_corr
        ),
        "skeleton_old_removed_in_lockzone": (
            global_sk
            & effective_lock_zone
            & ~locked_routes
        ),
        "skeleton_activity_removed": activity_removed,
        "skeleton_support_rescued": rescue,
        "skeleton_support_rescued_raw": rescue_raw,
        "skeleton_dense_support_candidate": dense_candidate,
        "skeleton_protected_corridor_mask": protected_mask,
        "skeleton_effective_lock_zones": effective_lock_zone,
        "skeleton_lockzone_preserved": lockzone_preserved,
    }

    rows = []
    for rec in rec_c:
        rows.append({"table": "uturn_routes", "tile_id": tile_id, **{k: v for k, v in rec.items() if k != "geometry"}})
    for rec in rec_s:
        rows.append({"table": "sequence_routes", "tile_id": tile_id, **{k: v for k, v in rec.items() if k != "geometry"}})
    if rows:
        frames.append(pd.DataFrame(rows))

    info = {
        "tile_id": tile_id,
        "skipped": False,
        "core_col_off": int(core.col_off),
        "core_row_off": int(core.row_off),
        "core_width": int(core.width),
        "core_height": int(core.height),
        "extended_col_off": int(extended.col_off),
        "extended_row_off": int(extended.row_off),
        "extended_width": int(extended.width),
        "extended_height": int(extended.height),
        "local_points": int(len(local_points)),
        "local_consensus_lines": int(len(local_consensus)),
        "local_activity_zones": int(len(local_activity)),
        "local_protected_corridors": int(len(local_protected)),
        "activity_removed_pixels_extended": int(activity_removed.sum()),
        "global_skeleton_before_ridge_pixels_extended": int(global_sk_before_ridge.sum()),
        "global_skeleton_after_ridge_pixels_extended": int(global_sk_ridge_refined.sum()),
        "kde_ridge_moved_pixels_extended": int(ridge_moved.sum()),
        "kde_ridge_refined_chains_extended": int(len(ridge_rows)),
        "support_rescued_raw_pixels_extended": int(rescue_raw.sum()),
        "support_rescued_pixels_extended": int(rescue.sum()),
        "dense_support_candidate_pixels_extended": int(dense_candidate.sum()),
        "raw_lock_zone_pixels_extended": int(lock_zone.sum()),
        "effective_lock_zone_pixels_extended": int(effective_lock_zone.sum()),
        "lockzone_preserved_pixels_extended": int(lockzone_preserved.sum()),
        **{f"rescue_{key}": value for key, value in rescue_info.items()},
        **{f"rescue_topology_{key}": value for key, value in rescue_topology_info.items()},
        **{f"connector_{key}": value for key, value in connector_info.items()},
        **{f"auxiliary_{key}": value for key, value in auxiliary_info.items()},
        "foreground_pixels_extended": int(foreground.sum()),
        "global_skeleton_pixels_extended": int(global_sk.sum()),
        "primary_pixels_extended": int(primary.sum()),
        "auxiliary_pixels_extended": int(auxiliary.sum()),
        "all_evidence_pixels_extended": int(all_evidence.sum()),
    }
    return arrays, frames, info


# =============================================================================
# Main pipeline
# =============================================================================
def check_alignment(*sources: rasterio.DatasetReader) -> None:
    first = sources[0]
    for src in sources[1:]:
        if src.width != first.width or src.height != first.height:
            raise ValueError("Input raster dimensions do not match")
        if src.transform != first.transform:
            raise ValueError("Input raster transforms do not match")
        if first.crs != src.crs:
            raise ValueError("Input raster CRS values do not match")


def run(args) -> None:
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    diag_dir = args.output / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    points = ensure_xy_columns(pd.read_csv(args.gravity_points, low_memory=False))
    if "gravity_kde_include" in points.columns:
        include = points["gravity_kde_include"]
        if not pd.api.types.is_bool_dtype(include):
            include = include.fillna("").astype(str).str.lower().isin({"1", "true", "yes", "y"})
        points = points.loc[include.to_numpy(bool)].copy()
    elif "road_extraction_weight" in points.columns:
        weight = pd.to_numeric(points["road_extraction_weight"], errors="coerce").fillna(1.0)
        points = points.loc[weight.to_numpy(float) >= args.min_road_weight].copy()
    consensus = read_consensus(args.uturn_gpkg)
    activity_zones = read_activity_zones(args.activity_zones, args.activity_layer)
    try:
        protected_zones = read_activity_zones(
            args.activity_zones, args.protected_activity_layer
        )
    except Exception:
        protected_zones = gpd.GeoDataFrame(geometry=[], crs=None)
        print(
            f"[WARNING] Protected corridors not found for layer={args.protected_activity_layer}; "
            "rerun S3 (the bundled v0.5 version).",
            flush=True,
        )
    try:
        activity_through_zones = read_activity_zones(
            args.activity_zones, args.activity_through_layer
        )
    except Exception:
        activity_through_zones = gpd.GeoDataFrame(geometry=[], crs=None)
    if not activity_through_zones.empty:
        if protected_zones.empty:
            protected_zones = activity_through_zones.copy()
        else:
            if protected_zones.crs is None and activity_through_zones.crs is not None:
                protected_zones = protected_zones.set_crs(activity_through_zones.crs)
            elif activity_through_zones.crs is not None and protected_zones.crs != activity_through_zones.crs:
                activity_through_zones = activity_through_zones.to_crs(protected_zones.crs)
            protected_zones = gpd.GeoDataFrame(
                pd.concat([protected_zones, activity_through_zones], ignore_index=True),
                geometry="geometry", crs=protected_zones.crs or activity_through_zones.crs,
            )

    thresholds = parse_thresholds(args.thresholds, legacy=args.legacy_thresholds)
    cfg = rm.Config()
    cfg.final_branch_prune_m = 0
    cfg.final_branch_support_lt = -1
    cfg.min_component_length_m = 0

    with rasterio.open(args.kde) as kde_src, rasterio.open(args.support) as support_src, rasterio.open(args.distance) as distance_src, rasterio.open(args.raw_union) as raw_union_src:
        check_alignment(kde_src, support_src, distance_src, raw_union_src)
        cell = abs(kde_src.transform.a)
        tile_px = max(16, int(round(args.tile_size_m / cell)))
        if args.halo_m is None:
            halo_m = max(
                args.relaxed_corridor_m + 20.0,
                args.dense_rescue_radius_m
                + args.dense_rescue_connection_m
                + args.dense_rescue_closing_m
                + 20.0,
                args.lock_clear_buffer_m + args.route_corridor_m + 20.0,
                args.route_corridor_m + args.endpoint_connector_max_along_m + args.endpoint_connector_snap_m + 20.0,
                args.pair_buffer_m + 30.0,
            )
        else:
            halo_m = float(args.halo_m)
        halo_px = max(1, int(math.ceil(halo_m / cell)))

        threshold_info: Dict[str, object]
        threshold_reference: Optional[float] = None
        if kde_src.dtypes[0].startswith("float") or kde_src.dtypes[0] in {"float32", "float64"}:
            if args.threshold_reference is not None and args.threshold_reference > 0:
                threshold_reference = float(args.threshold_reference)
                threshold_info = {
                    "threshold_mode": "manual_reference",
                    "threshold_reference": threshold_reference,
                    "note": "User-specified reference; scaled=clip(kde/reference*1024)",
                }
            else:
                threshold_info = compute_threshold_reference(
                    kde_src,
                    support_src,
                    distance_src,
                    relaxed_corridor_m=args.relaxed_corridor_m,
                    percentile=args.threshold_percentile,
                    max_samples=args.threshold_sample_max,
                    mode=args.threshold_mode,
                )
                threshold_reference = float(threshold_info["threshold_reference"])
        else:
            threshold_info = {
                "threshold_mode": "input_uint16_scaled",
                "threshold_reference": None,
                "note": "Input KDE is not float; use existing 0-1024 or approximate integer intensities; global normalization effects remain",
            }

        outputs = create_outputs(args.output, kde_src)
        tile_rows: List[Dict[str, object]] = []
        all_stage: List[pd.DataFrame] = []
        all_single: List[pd.DataFrame] = []
        all_endpoint: List[pd.DataFrame] = []
        all_routes: List[pd.DataFrame] = []
        all_ridge: List[pd.DataFrame] = []
        output_pixel_sums = {name: 0 for name in outputs}

        try:
            total_tiles = int(math.ceil(kde_src.width / tile_px) * math.ceil(kde_src.height / tile_px))
            for idx, (tile_id, core) in enumerate(iter_core_windows(kde_src.width, kde_src.height, tile_px), start=1):
                extended = expand_window(core, halo_px, kde_src.width, kde_src.height)
                arrays, frames, info = process_tile(
                    tile_id,
                    core,
                    extended,
                    kde_src,
                    support_src,
                    distance_src,
                    raw_union_src,
                    points,
                    consensus,
                    activity_zones,
                    protected_zones,
                    cfg,
                    args,
                    thresholds,
                    threshold_reference,
                )
                counts = write_core(outputs, arrays, core, extended)
                for key, value in counts.items():
                    output_pixel_sums[key] += int(value)
                info.update({f"core_{k}": v for k, v in counts.items()})
                tile_rows.append(info)

                for frame in frames:
                    if frame is None or frame.empty:
                        continue
                    if "threshold" in frame.columns:
                        all_stage.append(frame)
                    elif "connector_length_m" in frame.columns:
                        all_endpoint.append(frame)
                    elif "table" in frame.columns:
                        if frame["table"].astype(str).eq("kde_ridge_refinement").any():
                            all_ridge.append(frame.drop(columns=["table"], errors="ignore"))
                        else:
                            all_routes.append(frame)
                    else:
                        all_single.append(frame)

                if idx % max(1, args.progress_every) == 0 or idx == total_tiles:
                    print(
                        f"[TILE] {idx}/{total_tiles} id={tile_id} "
                        f"core=({int(core.col_off)},{int(core.row_off)},{int(core.width)}x{int(core.height)}) "
                        f"primary={counts.get('skeleton_primary_route_locked', 0)} "
                        f"all={counts.get('skeleton_all_evidence_route_locked', 0)}"
                    )
        finally:
            close_outputs(outputs)

        pd.DataFrame(tile_rows).to_csv(diag_dir / "skeleton_tile_summary.csv", index=False, encoding="utf-8-sig")
        if all_stage:
            pd.concat(all_stage, ignore_index=True, sort=False).to_csv(args.output / "multithreshold_bridge_log.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["tile_id", "threshold", "candidate", "accepted", "split_merges", "allowed_merges"]).to_csv(args.output / "multithreshold_bridge_log.csv", index=False, encoding="utf-8-sig")
        if all_ridge:
            pd.concat(all_ridge, ignore_index=True, sort=False).to_csv(args.output / "kde_ridge_refinement.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(args.output / "kde_ridge_refinement.csv", index=False, encoding="utf-8-sig")
        if all_single:
            pd.concat(all_single, ignore_index=True, sort=False).to_csv(args.output / "singletrack_edges.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(args.output / "singletrack_edges.csv", index=False, encoding="utf-8-sig")
        if all_endpoint:
            pd.concat(all_endpoint, ignore_index=True, sort=False).to_csv(args.output / "route_endpoint_connectors.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(args.output / "route_endpoint_connectors.csv", index=False, encoding="utf-8-sig")
        if all_routes:
            routes = pd.concat(all_routes, ignore_index=True, sort=False)
            routes.loc[routes["table"].eq("uturn_routes")].drop(columns=["table"], errors="ignore").to_csv(args.output / "uturn_routes.csv", index=False, encoding="utf-8-sig")
            routes.loc[routes["table"].eq("sequence_routes")].drop(columns=["table"], errors="ignore").to_csv(args.output / "sequence_routes.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(args.output / "uturn_routes.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame().to_csv(args.output / "sequence_routes.csv", index=False, encoding="utf-8-sig")

        report = {
            "mode": "tiled_skeleton",
            "tile_size_m": float(args.tile_size_m),
            "tile_size_px": int(tile_px),
            "halo_m": float(halo_m),
            "halo_px": int(halo_px),
            "cell_size_m": float(cell),
            "raster_width": int(kde_src.width),
            "raster_height": int(kde_src.height),
            "threshold_levels": thresholds,
            "threshold_info": threshold_info,
            "output_pixel_sums": output_pixel_sums,
            "point_rows": int(len(points)),
            "consensus_lines": int(len(consensus)),
            "activity_zone_features": int(len(activity_zones)),
            "protected_corridor_features": int(len(protected_zones)),
            "activity_through_corridor_features": int(len(activity_through_zones)),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "notes": [
                "Process tile+halo windows and write only tile cores to use less memory than reading entire rasters.",
                "Default thresholds are 1024,512,...,4, with 1024 representing the robust threshold_reference rather than the global KDE maximum.",
                "Multi-track rescue combines exact pixel support and aggregated neighborhood support for wide roads and laterally dispersed tracks.",
                "Clear the old hairpin skeleton only within the medoid-path buffer, preserving the rest of the diagnostic polygon.",
                "In v0.5, global_bridge_aware provides topology scaffolding; long degree-2 primary chains snap to local KDE ridges without moving junctions.",
                "By default, support_rescue must follow its local KDE ridge, and ordinary components receive at most one primary attachment.",
                "The default endpoint connector limit is reduced from 45m to 15m; reject connectors that close a cycle within an already connected component.",
                "Single-track auxiliary components use the default leaf-only rule: allow isolated or single-end attachments, but no alternative loop between two primary locations.",
                "This script does not perform global GPKG vectorization; vectorize the output skeleton rasters separately.",
            ],
        }
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v0.5: KDE ridge-constrained primary skeleton + controlled rescue/connector/auxiliary + hairpin locking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kde", type=Path, required=True, help="float32 KDE or scaled uint16 KDE")
    parser.add_argument("--support", type=Path, required=True, help="independent_track_support.tif")
    parser.add_argument("--distance", type=Path, required=True, help="distance_to_raw_trajectory_m.tif")
    parser.add_argument("--raw-union", type=Path, required=True, help="raw_trajectory_union.tif")
    parser.add_argument("--gravity-points", type=Path, required=True, help="all_cleaned_points.csv after gravity concentration")
    parser.add_argument("--uturn-gpkg", type=Path, default=None, help="uturn_consensus.gpkg from the gravity stage; skip hairpin consensus locking if missing")
    parser.add_argument("--activity-zones", type=Path, default=None,
                        help="S3 activity_zones.gpkg; suppress only activity_suppression_zones, preserving stable road corridors")
    parser.add_argument("--activity-layer", default="activity_suppression_zones")
    parser.add_argument("--protected-activity-layer", default="protected_road_corridors",
                        help="Protected stable/curved road corridor layer from S3")
    parser.add_argument("--activity-through-layer", default="activity_zone_backbones",
                        help="True primary through-corridors in activity zones from S3; positive road evidence exempt from activity-zone masks")
    parser.add_argument("--protected-corridor-buffer-m", type=float, default=0.0,
                        help="Extra protected-corridor buffer; default 0, optionally 1-2 m")
    parser.add_argument("--activity-zone-pad-m", type=float, default=0.0,
                        help="Extra query buffer when selecting local activity-zone features")
    parser.add_argument("--activity-suppression-buffer-m", type=float, default=0.0,
                        help="Extra activity-suppression geometry buffer; default 0 to preserve entrance roads")
    parser.add_argument("--min-road-weight", type=float, default=0.35,
                        help="Minimum road weight when gravity_points lacks gravity_kde_include")
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--tile-size-m", type=float, default=560.0, help="Core tile side length in m")
    parser.add_argument("--halo-m", type=float, default=None, help="Tile halo buffer; estimated automatically from connection parameters by default")
    parser.add_argument("--local-vector-pad-m", type=float, default=20.0, help="Extra search distance for local points/consensus lines")
    parser.add_argument("--progress-every", type=int, default=50)

    parser.add_argument("--strict-corridor-m", type=float, default=3.0)
    parser.add_argument("--relaxed-corridor-m", type=float, default=5.0)
    parser.add_argument("--disable-kde-ridge-refine", action="store_true",
                        help="Disable v0.5 local KDE ridge snapping of the global primary skeleton")
    parser.add_argument("--kde-ridge-search-radius-m", type=float, default=4.0,
                        help="Normal-direction search radius for local KDE ridges along the global primary skeleton")
    parser.add_argument("--kde-ridge-min-chain-m", type=float, default=16.0,
                        help="Optimize KDE ridges only for degree-2 road chains at least this long")
    parser.add_argument("--kde-ridge-min-local-ratio", type=float, default=0.20,
                        help="Minimum candidate-pixel KDE relative to the local maximum")
    parser.add_argument("--kde-ridge-support-weight", type=float, default=0.25)
    parser.add_argument("--kde-ridge-distance-weight", type=float, default=0.25)
    parser.add_argument("--kde-ridge-offset-penalty", type=float, default=0.08)
    parser.add_argument("--kde-ridge-transition-penalty", type=float, default=0.16,
                        help="Penalty for lateral-offset changes between samples; larger values give smoother results")
    parser.add_argument("--kde-ridge-min-mean-kde-gain", type=float, default=0.03,
                        help="Minimum fractional improvement in chain-average KDE required to accept repositioning")
    parser.add_argument("--kde-ridge-max-distance-m", type=float, default=6.0,
                        help="Maximum ridge-candidate distance in the original-track distance raster")
    parser.add_argument("--support-rescue-min-tracks", type=int, default=2,
                        help="Minimum independent tracks for exact single-pixel support rescue")
    parser.add_argument("--support-rescue-max-distance-m", type=float, default=5.0,
                        help="Maximum rescue-pixel distance from raw tracks")
    parser.add_argument("--support-rescue-kde-min-level", type=int, default=1,
                        help="Minimum scaled KDE level for rescue pixels")
    parser.add_argument("--support-rescue-dilation", type=int, default=1,
                        help="Rescue skeleton seed dilation iterations")
    parser.add_argument("--support-rescue-min-component-pixels", type=int, default=6,
                        help="Minimum connected pixels in a rescue line")
    parser.add_argument("--support-rescue-min-local-ridge-ratio", type=float, default=0.45,
                        help="Minimum rescue-pixel KDE relative to its local peak")
    parser.add_argument("--support-rescue-ridge-radius-m", type=float, default=6.0,
                        help="Radius for local rescue KDE peaks")
    parser.add_argument("--support-rescue-max-primary-attachments", type=int, default=1,
                        help="Maximum primary attachments for an ordinary rescue component")
    parser.add_argument("--support-rescue-bridge-min-support", type=int, default=3,
                        help="Minimum support to retain multi-attachment rescue as a real bridge")
    parser.add_argument("--support-rescue-bridge-min-support-fraction", type=float, default=0.50,
                        help="Minimum fraction of multi-attachment rescue pixels meeting the strong-support threshold")
    parser.add_argument("--support-rescue-bridge-min-ridge-ratio", type=float, default=0.65,
                        help="Minimum median local KDE ridge ratio for retaining multi-attachment rescue")

    parser.add_argument("--dense-rescue-radius-m", type=float, default=6.0,
                        help="Radius for aggregated neighborhood track evidence")
    parser.add_argument("--dense-rescue-support-sum-min", type=float, default=10.0,
                        help="Minimum total neighborhood support")
    parser.add_argument("--dense-rescue-occupied-pixels-min", type=float, default=6.0,
                        help="Minimum neighborhood pixels with support>0")
    parser.add_argument("--dense-rescue-near-support-min", type=int, default=1,
                        help="Minimum neighborhood peak support; 1 detects laterally dispersed tracks")
    parser.add_argument("--dense-rescue-closing-m", type=float, default=4.0,
                        help="Closing distance for aggregated evidence to fill small gaps within track bundles")
    parser.add_argument("--dense-rescue-connection-m", type=float, default=8.0,
                        help="Maximum connection distance from rescue components to the existing primary skeleton")
    parser.add_argument("--lock-clear-buffer-m", type=float, default=10.0,
                        help="Clear the old skeleton only within this distance of the locked medoid path to preserve real roads elsewhere in the lock zone")
    parser.add_argument("--bridge-merge-support-min", type=int, default=3, help="Minimum independent-track support for low-threshold merges of existing components; increase for false hairpin merges")
    parser.add_argument("--bridge-evidence-dilation", type=int, default=0, help="Merge-evidence support raster dilation iterations; 0 is most conservative against merging adjacent hairpin arms")
    parser.add_argument("--route-corridor-m", type=float, default=4.0)
    parser.add_argument("--min-pair-tracks", type=int, default=3)
    parser.add_argument("--pair-buffer-m", type=float, default=6.0)
    parser.add_argument("--endpoint-connector-max-along-m", type=float, default=15.0,
                        help="Maximum search distance along the same observed track from locked-route endpoints to the primary skeleton; reduced from 45m to 15m in v0.5")
    parser.add_argument("--endpoint-connector-snap-m", type=float, default=6.0)
    parser.add_argument("--disable-endpoint-connector-cycle-filter", action="store_true",
                        help="Disable same-component cycle-risk filtering before writing endpoint connectors")
    parser.add_argument("--disable-auxiliary-leaf-only", action="store_true",
                        help="Disable the default isolated/leaf-only rule for single-track auxiliary components")
    parser.add_argument("--auxiliary-max-primary-attachments", type=int, default=1,
                        help="Maximum primary attachments for a single-track auxiliary component")
    parser.add_argument("--auxiliary-max-primary-contact-fraction", type=float, default=0.25,
                        help="Remove an auxiliary component as a duplicate side line when its contact fraction along primary exceeds this value")

    parser.add_argument(
        "--threshold-mode",
        choices=["active_percentile", "legacy_global_max"],
        default="active_percentile",
        help="Reference method for converting float KDE to 0-1024 threshold levels",
    )
    parser.add_argument("--threshold-percentile", type=float, default=99.5, help="Percentile used in active_percentile mode")
    parser.add_argument("--threshold-sample-max", type=int, default=2_000_000, help="Maximum streaming samples for percentile estimation")
    parser.add_argument("--threshold-reference", type=float, default=None, help="Manual float KDE reference; overrides the automatic percentile")
    parser.add_argument("--thresholds", default=None, help="Comma-separated threshold levels, e.g. 1024,512,256,128,64,32,16,8,4")
    parser.add_argument("--legacy-thresholds", action="store_true", help="Use legacy thresholds 512,256,...,4")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not (0 < args.threshold_percentile <= 100):
            raise ValueError("--threshold-percentile must be within (0, 100]")
        run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
