#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean and clip source trajectory observations while preserving segment identity and sampling order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

try:
    import geopandas as gpd
    import pyogrio
    from pyproj import CRS, Transformer
    from shapely.geometry import LineString, Point, shape
    from shapely.ops import unary_union
    from shapely.prepared import prep
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing geospatial dependencies. Run:\n"
        "conda install -c conda-forge pandas numpy geopandas shapely pyproj pyogrio\n"
        f"Original error: {exc}"
    ) from exc


# =============================================================================
# Default paths and fields
# =============================================================================
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[3] / "data/mainling/observed_tracks"
DEFAULT_BOUNDARY_PATH = Path(__file__).resolve().parents[3] / "data/mainling/FocusArea.shp"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs/S2_Clean"
DEFAULT_DEM_PATH: Optional[Path] = None

ORIGINAL_FIELDS = [
    "track_id",
    "source_file",
    "document_name",
    "placemark_name",
    "segment_id",
    "point_index",
    "raw_point_index",
    "longitude",
    "latitude",
    "elevation_m",
    "time_raw",
    "time_utc",
    "time_china",
    "time_source",
]

CLEAN_FIELDS = [
    "clean_track_id",
    "clip_part_id",
    "temporal_part_id",
    "spatial_part_id",
    "hiking_part_id",
    "clean_point_index",
    "hiking_quality",
    "hiking_confidence",
    "segment_median_speed_mps",
    "segment_p90_speed_mps",
    "segment_high_speed_fraction",
    "segment_valid_speed_fraction",
    "segment_effective_speed_mps",
    "segment_duration_s",
    "segment_length_m",
    "x_m",
    "y_m",
    "metric_crs",
    "elevation_clean_m",
    "dem_elevation_m",
    "elevation_dem_residual_m",
    "elevation_dem_centered_residual_m",
    "dem_vertical_bias_m",
    "elevation_local_residual_m",
    "elevation_quality",
    "elevation_repair_method",
    "elevation_anomaly_confidence",
    "elevation_repaired",
    "dt_s",
    "step_m",
    "speed_mps",
    "vertical_speed_mps",
    "vertical_speed_clean_mps",
    "vertical_speed_original_mps",
    "turn_angle_deg",
    "cumulative_distance_m",
    "quality_flags",
]

REJECT_FIELDS = ["input_file", "source_row"] + ORIGINAL_FIELDS + [
    "x_m",
    "y_m",
    "reject_stage",
    "reject_reason",
]


MANIFEST_FIELDS = [
    "input_file",
    "source_track_id",
    "clean_track_id",
    "hiking_quality",
    "hiking_confidence",
    "clip_part_id",
    "temporal_part_id",
    "spatial_part_id",
    "hiking_part_id",
    "point_count",
    "length_m",
    "duration_s",
    "median_speed_mps",
    "p90_speed_mps",
    "high_speed_fraction",
    "valid_speed_fraction",
    "effective_speed_mps",
    "dem_vertical_bias_m",
    "start_time_utc",
    "end_time_utc",
    "start_longitude",
    "start_latitude",
    "end_longitude",
    "end_latitude",
    "elevation_repaired_count",
    "elevation_flagged_count",
]

SPLIT_EVENT_FIELDS = [
    "input_file",
    "track_id",
    "stage",
    "split_reason",
    "previous_source_row",
    "current_source_row",
    "previous_point_index",
    "current_point_index",
    "previous_time_utc",
    "current_time_utc",
    "dt_s",
    "distance_m",
    "speed_mps",
]


ELEVATION_AUDIT_FIELDS = [
    "input_file",
    "track_id",
    "source_file",
    "clip_part_id",
    "temporal_part_id",
    "spatial_part_id",
    "point_index",
    "raw_point_index",
    "time_utc",
    "longitude",
    "latitude",
    "x_m",
    "y_m",
    "elevation_original_m",
    "elevation_clean_m",
    "dem_elevation_m",
    "dem_vertical_bias_m",
    "elevation_dem_centered_residual_m",
    "elevation_local_residual_m",
    "elevation_quality",
    "elevation_repair_method",
    "elevation_anomaly_confidence",
    "quality_flags",
]


@dataclass(frozen=True)
class CleaningConfig:
    """Parameters for cleaning candidate hiking GPS trajectories in Mainling's alpine valleys."""

    # Boundary clipping is strict by default. To preserve topology at administrative boundaries, set
    # --boundary-context-points 2 or a small --boundary-buffer-m on the command line.
    boundary_buffer_m: float = 0.0
    boundary_context_points: int = 0

    # Minimum final segment size. Retain short mountain branches that may connect the network,
    # while requiring at least 5 points and 20 m to avoid lines formed from only two or three noisy points.
    min_points_per_segment: int = 5
    min_segment_length_m: float = 20.0

    # Temporal issues
    duplicate_position_m: float = 2.0
    duplicate_time_keep_max_m: float = 50.0
    # Brief near-coincident clusters with different timestamps can also inflate KDE density. Compress only when the spatial extent is small
    # and path efficiency is low or the median step is tiny; retain the first and last timestamps.
    near_duplicate_run_radius_m: float = 3.0
    near_duplicate_run_min_points: int = 3
    near_duplicate_run_path_efficiency_max: float = 0.50
    near_duplicate_run_median_step_max_m: float = 0.75
    max_time_gap_s: float = 1800.0

    # Hard splits for physical discontinuities
    hard_step_break_m: float = 300.0
    hard_speed_break_mps: float = 50.0
    hard_speed_min_step_m: float = 100.0

    # Hiking-speed flag: exceeding this value flags points without directly deleting them.
    hiking_speed_flag_mps: float = 5.0
    very_high_speed_flag_mps: float = 20.0

    # Isolated A-B-C spikes
    spike_min_leg_m: float = 20.0
    spike_min_detour_m: float = 80.0
    spike_detour_ratio: float = 4.0
    spike_direct_ratio: float = 0.35
    spike_direct_max_m: float = 80.0
    spike_cross_track_min_m: float = 15.0
    spike_local_window_edges: int = 8
    spike_local_speed_floor_mps: float = 5.0
    spike_zero_dt_leg_m: float = 20.0
    spike_zero_dt_step_ratio: float = 2.0
    spike_missing_time_leg_m: float = 100.0
    # With missing times, a candidate edge must also be much longer than the normal local step length.
    spike_missing_time_step_ratio: float = 6.0
    # Prioritize protection when both entry and exit directions of the candidate path continue the adjacent measured trajectory,
    # preventing removal of mountain switchback or U-bend vertices.
    spike_flank_angle_max_deg: float = 70.0
    spike_max_passes: int = 4

    # Brief A-B-C-D-E drift clusters
    excursion_cluster_max_points: int = 6
    excursion_cluster_max_duration_s: float = 180.0
    excursion_cluster_min_detour_m: float = 100.0
    excursion_cluster_detour_ratio: float = 4.0
    excursion_cluster_direct_max_m: float = 100.0
    excursion_cluster_cross_track_min_m: float = 20.0
    excursion_flank_angle_max_deg: float = 70.0
    excursion_cluster_max_passes: int = 3

    # Stationary drift: time + spatial diameter + path efficiency
    stationary_min_duration_s: float = 120.0
    stationary_diameter_m: float = 20.0
    stationary_endpoint_max_m: float = 10.0
    stationary_path_efficiency_max: float = 0.30
    stationary_median_speed_max_mps: float = 0.25
    stationary_max_scan_s: float = 21600.0
    stationary_robust_radius_m: float = 20.0
    stationary_outlier_tolerance_points: int = 2
    stationary_min_inlier_fraction: float = 0.85
    stationary_alternate_start_checks: int = 8

    # Elevation quality control: elevation anomalies do not remove planimetric points or split trajectories.
    elevation_spike_min_m: float = 8.0
    elevation_side_change_min_m: float = 5.0
    elevation_residual_mad_k: float = 6.0
    elevation_residual_min_samples: int = 10
    elevation_grade_hard: float = 2.0
    vertical_speed_hard_mps: float = 2.0
    elevation_near_same_xy_m: float = 2.0
    elevation_near_same_xy_jump_m: float = 10.0
    elevation_step_change_min_m: float = 15.0
    elevation_min_horizontal_distance_m: float = 0.05
    elevation_stationary_deviation_min_m: float = 5.0
    elevation_stationary_mad_k: float = 4.0
    elevation_max_passes: int = 3
    # Interpolate short elevation gaps along track distance first to avoid mixing vertical datums within a segment.
    elevation_short_gap_max_points: int = 5
    elevation_short_gap_max_distance_m: float = 150.0

    # Use the DEM only as assistance: estimate trajectory-level GPS-DEM vertical bias before assessing repairs.
    dem_repair_trend_tolerance_m: float = 20.0
    dem_repair_steep_tolerance_m: float = 8.0
    dem_local_window_points: int = 2
    dem_steep_grade_threshold: float = 0.50
    dem_trust_max_grade: float = 1.00
    dem_trust_max_local_range_m: float = 30.0
    dem_bias_min_samples: int = 15
    dem_bias_mad_k: float = 4.0
    dem_bias_max_abs_m: float = 200.0
    dem_residual_flag_m: float = 100.0
    fill_missing_elevation_from_dem: bool = True

    # Local adaptive distance gaps. Weak signals in Mainling's valleys can cause sparse sampling, so ordinary candidates
    # require joint evidence from step-length ratios and time/speed rather than turn angle alone.
    adaptive_step_window_edges: int = 8
    adaptive_step_floor_m: float = 80.0
    adaptive_step_mad_k: float = 8.0
    adaptive_step_ratio: float = 8.0
    adaptive_step_strong_ratio: float = 12.0
    adaptive_step_turn_support_deg: float = 60.0
    adaptive_step_turn_min_ratio: float = 10.0
    adaptive_step_speed_min_ratio: float = 8.0
    adaptive_step_missing_time_ratio: float = 12.0
    adaptive_step_time_ratio: float = 6.0
    adaptive_step_min_time_gap_s: float = 60.0

    # Hiking-consistency filtering: downweight isolated timestamp jitter before identifying sustained high speeds.
    hiking_window_edges: int = 15
    hiking_core_median_max_mps: float = 4.0
    hiking_core_effective_max_mps: float = 4.5
    hiking_core_high_fraction: float = 0.35
    hiking_edge_hard_mps: float = 8.0
    hiking_edge_hard_min_step_m: float = 20.0
    hiking_extreme_edge_mps: float = 15.0
    hiking_extreme_edge_min_step_m: float = 80.0
    hiking_hard_run_min_edges: int = 2
    hiking_nonconforming_min_duration_s: float = 20.0
    hiking_nonconforming_min_distance_m: float = 100.0
    hiking_nonconforming_min_median_mps: float = 5.0
    hiking_window_min_duration_s: float = 20.0
    hiking_window_min_distance_m: float = 50.0
    hiking_bridge_max_edges: int = 5
    hiking_bridge_max_duration_s: float = 30.0
    hiking_bridge_max_distance_m: float = 80.0
    hiking_expand_speed_mps: float = 3.5
    hiking_expand_max_edges: int = 5
    hiking_timestamp_jitter_max_step_m: float = 80.0
    hiking_timestamp_jitter_combined_speed_max_mps: float = 4.5
    hiking_timestamp_jitter_local_dt_ratio: float = 0.25

    # Final subsegment speed statistics: do not reject a whole segment from a few speeds when too few edges have valid times.
    hiking_segment_median_max_mps: float = 3.5
    hiking_segment_p90_max_mps: float = 5.0
    hiking_segment_high_speed_fraction_max: float = 0.10
    hiking_segment_robust_min_edges: int = 10
    hiking_segment_sparse_p90_max_mps: float = 8.0
    hiking_segment_sparse_high_edges_max: int = 1
    hiking_min_valid_speed_edges: int = 3
    hiking_min_valid_speed_fraction: float = 0.30

    # Outputs
    write_gpkg: bool = True


# =============================================================================
# General utilities
# =============================================================================
def safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def parse_time_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")



NAT_INT64 = np.iinfo(np.int64).min


def datetime_ns_array(series: pd.Series) -> np.ndarray:
    """Return a datetime64[ns] integer array; represent NaT by the minimum int64 value."""
    return np.asarray(series.array.asi8, dtype=np.int64)


def seconds_between(time_ns: np.ndarray, start: int, end: int) -> float:
    a = int(time_ns[start])
    b = int(time_ns[end])
    if a == NAT_INT64 or b == NAT_INT64:
        return math.nan
    return (b - a) / 1_000_000_000.0

def robust_upper(values: Iterable[float], floor: float, k: float) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float(floor)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        q95 = float(np.quantile(x, 0.95))
        return max(float(floor), q95 * 1.5, median * 2.0)
    return max(float(floor), median + float(k) * sigma)


def append_csv(frame: pd.DataFrame, path: Path, fields: Sequence[str]) -> None:
    """Append CSV using UTF-8-SIG consistently for Chinese-text compatibility in Excel/QGIS."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reindex(columns=list(fields))
    exists = path.exists() and path.stat().st_size > 0
    output.to_csv(
        path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def sanitize_id(value: object) -> str:
    """Create stable identifier text without converting None/NaN into the string ``nan``."""
    missing = value is None
    if not missing:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
    text = "track" if missing else str(value).strip()
    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")
    return text or "track"


def safe_output_stem(value: object, max_length: int = 170) -> str:
    """Create a Windows-compatible output filename; retain a hash when truncating to avoid collisions."""
    text = sanitize_id(value).strip(" ._") or "track"
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    keep = max(20, max_length - len(digest) - 2)
    return f"{text[:keep]}__{digest}"


def local_utm_crs(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def crs_uses_metres(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    try:
        units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
    except Exception:
        return False
    return any("metre" in unit or "meter" in unit for unit in units)


def make_valid_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    try:
        from shapely import make_valid

        return make_valid(geometry)
    except (ImportError, AttributeError):
        return geometry if geometry.is_valid else geometry.buffer(0)


def clean_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = output_dir / "cleaned_hiking_tracks"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for name in (
        "all_cleaned_points.csv",
        "rejected_points.csv",
        "split_events.csv",
        "elevation_audit.csv",
        "segment_manifest.csv",
        "cleaning_summary.csv",
        "cleaned_hiking_tracks.gpkg",
        "cleaning_config.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


# =============================================================================
# Boundary and DEM
# =============================================================================
def load_boundary(
    boundary_path: Path,
    crs_override: Optional[str],
    buffer_m: float,
) -> Tuple[object, CRS, Dict[str, object]]:
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")

    warning = ""
    try:
        boundary = gpd.read_file(boundary_path)
    except Exception as first_error:
        try:
            import shapefile

            reader = shapefile.Reader(shp=str(boundary_path))
            geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
            boundary = gpd.GeoDataFrame(geometry=geometries, crs=None)
            warning = f"geopandas read failed; falling back to pyshp: {first_error}"
        except Exception as second_error:
            raise RuntimeError(
                "Boundary read failed. Ensure .shp/.shx/.dbf/.prj files are present and share the same base name.\n"
                f"geopandas: {first_error}\npyshp: {second_error}"
            ) from second_error

    if boundary.empty or boundary.geometry.isna().all():
        raise ValueError(f"Boundary contains no valid geometry: {boundary_path}")

    if boundary.crs is None:
        if not crs_override:
            raise ValueError(
                "Boundary CRS is missing (usually because the .prj file is absent). Verify the actual CRS in QGIS "
                "and specify it with --boundary-crs EPSG:xxxx; the script will not guess."
            )
        boundary = boundary.set_crs(crs_override, allow_override=True)
        warning = (warning + " | " if warning else "") + f"Manually specified CRS={crs_override}"

    source_crs = CRS.from_user_input(boundary.crs)
    if crs_uses_metres(source_crs):
        metric_crs = source_crs
        boundary_metric = boundary.to_crs(metric_crs)
    else:
        boundary_wgs84 = boundary.to_crs("EPSG:4326")
        centroid = make_valid_geometry(unary_union(boundary_wgs84.geometry)).centroid
        metric_crs = local_utm_crs(float(centroid.x), float(centroid.y))
        boundary_metric = boundary.to_crs(metric_crs)
        warning = (warning + " | " if warning else "") + (
            f"Boundary coordinates are not metric; measurements automatically use {metric_crs.to_string()}"
        )

    geometries = [
        make_valid_geometry(g)
        for g in boundary_metric.geometry
        if g is not None and not g.is_empty
    ]
    boundary_geometry = make_valid_geometry(unary_union(geometries))
    if boundary_geometry is None or boundary_geometry.is_empty:
        raise ValueError("Boundary is empty after merging")
    if buffer_m != 0:
        boundary_geometry = make_valid_geometry(boundary_geometry.buffer(float(buffer_m)))
        if boundary_geometry is None or boundary_geometry.is_empty:
            raise ValueError(
                f"Boundary is empty after applying buffer={buffer_m} m; check the buffer value and boundary geometry"
            )

    return boundary_geometry, metric_crs, {
        "boundary_source_crs": source_crs.to_string(),
        "metric_crs": metric_crs.to_string(),
        "bounds": tuple(round(float(v), 3) for v in boundary_geometry.bounds),
        "area_m2": round(float(boundary_geometry.area), 3),
        "warning": warning,
    }


def points_covered_by_boundary(
    x: np.ndarray,
    y: np.ndarray,
    boundary_geometry,
) -> np.ndarray:
    try:
        from shapely import covers, points

        return np.asarray(covers(boundary_geometry, points(x, y)), dtype=bool)
    except (ImportError, AttributeError, TypeError):
        prepared = prep(boundary_geometry)
        return np.fromiter(
            (prepared.covers(Point(float(px), float(py))) for px, py in zip(x, y)),
            dtype=bool,
            count=len(x),
        )


class DemSampler:
    """Optional DEM sampler accepting WGS84 longitude and latitude."""

    def __init__(self, dem_path: Optional[Path]):
        self.path = dem_path
        self.dataset = None
        self.transformer: Optional[Transformer] = None
        self.nodata = math.nan
        if dem_path is None:
            return
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM does not exist: {dem_path}")
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "--dem requires rasterio: conda install -c conda-forge rasterio"
            ) from exc
        self.dataset = rasterio.open(dem_path)
        if self.dataset.crs is None:
            self.dataset.close()
            self.dataset = None
            raise ValueError(f"DEM CRS is missing: {dem_path}")
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.dataset.crs, always_xy=True
        )
        self.nodata = safe_float(self.dataset.nodata)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        values = np.full(len(lon), np.nan, dtype=float)
        if self.dataset is None or self.transformer is None or len(lon) == 0:
            return values
        x, y = self.transformer.transform(lon, lat)
        samples = self.dataset.sample(zip(x, y), masked=True)
        for i, sample in enumerate(samples):
            if len(sample) == 0 or np.ma.is_masked(sample[0]):
                continue
            value = safe_float(sample[0])
            if not math.isfinite(value):
                continue
            if math.isfinite(self.nodata) and math.isclose(value, self.nodata):
                continue
            values[i] = value
        return values

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()


# =============================================================================
# Trajectory metrics
# =============================================================================
def _vertical_speed_from_elevation(
    z: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Compute absolute vertical speeds between adjacent points from the given elevations."""
    result = np.full(len(z), np.nan, dtype=float)
    if len(z) <= 1:
        return result
    dz = np.full(len(z), np.nan, dtype=float)
    dz[1:] = np.abs(np.diff(z))
    valid = np.isfinite(dt) & (dt > 0) & np.isfinite(dz)
    result[valid] = dz[valid] / dt[valid]
    return result


def compute_metrics(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute point-level trajectory metrics.

    Field semantics:
    - step[i], dt[i], and speed[i] refer to the incoming edge of point i, from point i-1 to i;
    - turn[i] refers to vertex i, formed by edges i-1 to i and i to i+1.

    Thus turn_angle_deg intentionally belongs to middle vertex i and must not be shifted to the next point.
    """
    n = len(frame)
    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    z = frame["_z_clean"].to_numpy(dtype=float)

    step = np.full(n, np.nan, dtype=float)
    if n > 1:
        step[1:] = np.hypot(np.diff(x), np.diff(y))

    dt = np.full(n, np.nan, dtype=float)
    time_ns = datetime_ns_array(frame["_time"])
    if n > 1:
        valid_time = (time_ns[:-1] != NAT_INT64) & (time_ns[1:] != NAT_INT64)
        valid_positions = np.flatnonzero(valid_time) + 1
        dt[valid_positions] = (
            time_ns[valid_positions] - time_ns[valid_positions - 1]
        ) / 1_000_000_000.0

    speed = np.full(n, np.nan, dtype=float)
    valid_dt = np.isfinite(dt) & (dt > 0)
    speed[valid_dt] = step[valid_dt] / dt[valid_dt]

    vertical_speed = _vertical_speed_from_elevation(z, dt)

    turn = np.full(n, np.nan, dtype=float)
    for i in range(1, n - 1):
        v1x, v1y = x[i] - x[i - 1], y[i] - y[i - 1]
        v2x, v2y = x[i + 1] - x[i], y[i + 1] - y[i]
        norm1, norm2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            continue
        cosine = (v1x * v2x + v1y * v2y) / (norm1 * norm2)
        turn[i] = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))

    cumulative = np.zeros(n, dtype=float)
    if n > 1:
        cumulative[1:] = np.cumsum(np.nan_to_num(step[1:], nan=0.0))

    return {
        "step": step,
        "dt": dt,
        "speed": speed,
        "vertical_speed": vertical_speed,
        "turn": turn,
        "cumulative": cumulative,
    }


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    cx: float,
    cy: float,
) -> float:
    vx, vy = cx - ax, cy - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / length2
    t = min(1.0, max(0.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def angle_between_vectors(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return the angle between two 2D vectors; return NaN for zero vectors."""
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return math.nan
    cosine = (ax * bx + ay * by) / (norm_a * norm_b)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def path_flanks_are_continuous(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    max_angle_deg: float,
) -> bool:
    """Check directional continuity between both ends of a candidate path and adjacent trajectory sections.

    On Mainling's mountain roads, genuine U-bends and switchbacks can have long detours with nearby endpoints,
    so the detour/direct ratio alone can misclassify them. If both the entry and exit directions
    continue adjacent trajectory sections, treat the candidate as genuine road geometry and prioritize its protection.
    """
    if start <= 0 or end >= len(x) - 1 or end <= start:
        return False
    entry_angle = angle_between_vectors(
        x[start] - x[start - 1],
        y[start] - y[start - 1],
        x[start + 1] - x[start],
        y[start + 1] - y[start],
    )
    exit_angle = angle_between_vectors(
        x[end] - x[end - 1],
        y[end] - y[end - 1],
        x[end + 1] - x[end],
        y[end + 1] - y[end],
    )
    return (
        math.isfinite(entry_angle)
        and math.isfinite(exit_angle)
        and entry_angle <= max_angle_deg
        and exit_angle <= max_angle_deg
    )


def local_step_reference(
    steps: np.ndarray,
    center_edge: int,
    window_edges: int,
) -> float:
    """Return the median normal step length near a candidate edge, excluding that edge and hard gaps."""
    start = max(1, center_edge - window_edges)
    end = min(len(steps), center_edge + window_edges + 1)
    values = [
        float(steps[i])
        for i in range(start, end)
        if i != center_edge
        and math.isfinite(float(steps[i]))
        and 0 < float(steps[i]) < 300.0
    ]
    return float(np.median(values)) if len(values) >= 3 else math.nan


def local_step_references(
    steps: np.ndarray,
    window_edges: int,
) -> np.ndarray:
    output = np.full(len(steps), np.nan, dtype=float)
    for edge in range(1, len(steps)):
        output[edge] = local_step_reference(steps, edge, window_edges)
    return output


def local_speed_ceiling(
    speed: np.ndarray,
    center_edge: int,
    window_edges: int,
    floor: float,
) -> float:
    start = max(1, center_edge - window_edges)
    end = min(len(speed), center_edge + window_edges + 1)
    values = speed[start:end]
    values = values[np.isfinite(values) & (values >= 0) & (values <= 20.0)]
    if values.size < 3:
        return floor
    return min(20.0, robust_upper(values, floor=floor, k=4.0))


def local_speed_ceilings(
    speed: np.ndarray,
    window_edges: int,
    floor: float,
) -> np.ndarray:
    """Compute local speed limits for all edges once to avoid repeated statistics during candidate searches."""
    output = np.full(len(speed), float(floor), dtype=float)
    for edge in range(1, len(speed)):
        output[edge] = local_speed_ceiling(speed, edge, window_edges, floor)
    return output


# =============================================================================
# Audit records
# =============================================================================
def rejected_rows(frame: pd.DataFrame, stage: str, reason: str) -> pd.DataFrame:
    output = frame.copy()
    output["source_row"] = pd.to_numeric(
        output.get("_source_row", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    )
    output["x_m"] = output.get("_x", np.nan)
    output["y_m"] = output.get("_y", np.nan)
    output["reject_stage"] = stage
    output["reject_reason"] = reason
    return output


def finalize_rejected_frames(
    rejected_frames: List[pd.DataFrame],
    final_segments: List[pd.DataFrame],
    input_file: str,
) -> List[pd.DataFrame]:
    """Create a rejected-point audit table unique by original row and disjoint from final output points.

    Speed-edge splitting makes adjacent accepted and rejected segments share boundary points. If a shared point remains in the final output,
    it must not also appear in rejected_points.csv. Merge reasons when multiple stages flag the same point,
    ensuring that ``output points + unique rejected points = input points``.
    """
    if not rejected_frames:
        return []

    prepared: List[pd.DataFrame] = []
    order = 0
    for frame in rejected_frames:
        if frame is None or frame.empty:
            continue
        current = frame.copy()
        current["input_file"] = input_file
        if "source_row" not in current.columns:
            current["source_row"] = pd.to_numeric(
                current.get("_source_row", pd.Series(np.nan, index=current.index)),
                errors="coerce",
            )
        else:
            current["source_row"] = pd.to_numeric(current["source_row"], errors="coerce")
        current["_reject_order"] = np.arange(order, order + len(current), dtype=int)
        order += len(current)
        prepared.append(current)

    if not prepared:
        return []
    combined = pd.concat(prepared, ignore_index=True, sort=False)

    accepted_rows: set[int] = set()
    for segment in final_segments:
        if segment is None or segment.empty or "_source_row" not in segment.columns:
            continue
        accepted_rows.update(
            int(value)
            for value in pd.to_numeric(segment["_source_row"], errors="coerce").dropna()
        )
    if accepted_rows:
        combined = combined.loc[
            ~combined["source_row"].isin(accepted_rows)
        ].copy()
    if combined.empty:
        return []

    # source_row should always exist; use unique negative values when missing to avoid erroneous merging.
    missing = combined["source_row"].isna()
    if missing.any():
        combined.loc[missing, "source_row"] = -(
            np.arange(1, int(missing.sum()) + 1, dtype=int)
        )

    combined.sort_values(["source_row", "_reject_order"], inplace=True)
    merged_rows: List[pd.Series] = []
    for _, group in combined.groupby("source_row", sort=False, dropna=False):
        row = group.iloc[0].copy()
        stages = [
            str(value).strip() for value in group.get("reject_stage", pd.Series(dtype=object))
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        reasons = [
            str(value).strip() for value in group.get("reject_reason", pd.Series(dtype=object))
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        row["reject_stage"] = " | ".join(dict.fromkeys(stages))
        row["reject_reason"] = " | ".join(dict.fromkeys(reasons))
        merged_rows.append(row)

    result = pd.DataFrame(merged_rows).drop(columns=["_reject_order"], errors="ignore")
    result["source_row"] = pd.to_numeric(result["source_row"], errors="coerce").astype("Int64")
    return [result.reset_index(drop=True)]


def split_event(
    frame: pd.DataFrame,
    break_pos: int,
    stage: str,
    reason: str,
    input_file: str,
) -> Dict[str, object]:
    """Record a split immediately before break_pos; the argument always denotes an iloc position."""
    if break_pos <= 0 or break_pos >= len(frame):
        raise IndexError(f"Invalid split position: {break_pos}, track points={len(frame)}")
    previous = frame.iloc[break_pos - 1]
    current = frame.iloc[break_pos]
    distance = math.hypot(
        float(current["_x"] - previous["_x"]),
        float(current["_y"] - previous["_y"]),
    )
    dt = math.nan
    if pd.notna(previous["_time"]) and pd.notna(current["_time"]):
        dt = (current["_time"] - previous["_time"]).total_seconds()
    speed = distance / dt if math.isfinite(dt) and dt > 0 else math.nan
    return {
        "input_file": input_file,
        "track_id": current.get("track_id", ""),
        "stage": stage,
        "split_reason": reason,
        "previous_source_row": previous.get("_source_row", ""),
        "current_source_row": current.get("_source_row", ""),
        "previous_point_index": previous.get("point_index", ""),
        "current_point_index": current.get("point_index", ""),
        "previous_time_utc": previous.get("time_utc", ""),
        "current_time_utc": current.get("time_utc", ""),
        "dt_s": round(dt, 3) if math.isfinite(dt) else "",
        "distance_m": round(distance, 3),
        "speed_mps": round(speed, 4) if math.isfinite(speed) else "",
    }


def split_frame_by_breaks(
    frame: pd.DataFrame,
    break_before: np.ndarray,
) -> List[pd.DataFrame]:
    starts = [0] + np.flatnonzero(break_before).tolist()
    starts = sorted(set(starts))
    ends = starts[1:] + [len(frame)]
    return [
        frame.iloc[start:end].copy().reset_index(drop=True)
        for start, end in zip(starts, ends)
        if end > start
    ]


# =============================================================================
# Cleaning step 1: temporal issues
# =============================================================================
def split_by_time(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
    input_file: str,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[Dict[str, object]], Dict[str, int]]:
    """Split at temporal anomalies and handle same-second records conservatively."""
    work = frame.copy().reset_index(drop=True)
    segments: List[pd.DataFrame] = []
    rejected: List[pd.DataFrame] = []
    events: List[Dict[str, object]] = []
    stats = {
        "duplicate_removed": 0,
        "duplicate_time_kept": 0,
        "duplicate_time_splits": 0,
        "reverse_time_splits": 0,
        "time_gap_splits": 0,
    }

    x = work["_x"].to_numpy(dtype=float)
    y = work["_y"].to_numpy(dtype=float)
    time_ns = datetime_ns_array(work["_time"])
    flags = work["_flags"].astype(object).tolist()
    current_positions: List[int] = []
    previous_kept_pos: Optional[int] = None
    same_time_anchor_pos: Optional[int] = None

    def flush() -> None:
        nonlocal current_positions
        if current_positions:
            part = work.iloc[current_positions].copy().reset_index(drop=True)
            part["_flags"] = [flags[pos] for pos in current_positions]
            segments.append(part)
            current_positions = []

    for pos in range(len(work)):
        if previous_kept_pos is None:
            current_positions.append(pos)
            previous_kept_pos = pos
            same_time_anchor_pos = None
            continue

        distance = math.hypot(
            x[pos] - x[previous_kept_pos], y[pos] - y[previous_kept_pos]
        )
        dt = seconds_between(time_ns, previous_kept_pos, pos)

        if math.isfinite(dt) and dt == 0:
            if same_time_anchor_pos is None:
                same_time_anchor_pos = previous_kept_pos
            anchor_distance = math.hypot(
                x[pos] - x[same_time_anchor_pos],
                y[pos] - y[same_time_anchor_pos],
            )
            if distance <= cfg.duplicate_position_m:
                rejected.append(
                    rejected_rows(work.iloc[[pos]], "time", "duplicate_timestamp_position")
                )
                stats["duplicate_removed"] += 1
                continue
            # Consecutive same-second points must be close both to the previous point and to the first point of that same-second sequence,
            # preventing chained drift of 30 m + 30 m + ... from accumulating to hundreds of meters.
            if (
                distance <= cfg.duplicate_time_keep_max_m
                and anchor_distance <= cfg.duplicate_time_keep_max_m
            ):
                flags[pos] = append_flag(flags[pos], "duplicate_timestamp_kept")
                current_positions.append(pos)
                previous_kept_pos = pos
                stats["duplicate_time_kept"] += 1
                continue

            flush()
            current_positions.append(pos)
            previous_kept_pos = pos
            same_time_anchor_pos = pos
            stats["duplicate_time_splits"] += 1
            events.append(
                split_event(work, pos, "time", "duplicate_time_spatial_conflict", input_file)
            )
            continue

        same_time_anchor_pos = None

        if math.isfinite(dt) and dt < 0:
            flush()
            current_positions.append(pos)
            previous_kept_pos = pos
            stats["reverse_time_splits"] += 1
            events.append(split_event(work, pos, "time", "reverse_time", input_file))
            continue

        if math.isfinite(dt) and dt > cfg.max_time_gap_s:
            flush()
            current_positions.append(pos)
            previous_kept_pos = pos
            stats["time_gap_splits"] += 1
            events.append(split_event(work, pos, "time", "long_time_gap", input_file))
            continue

        current_positions.append(pos)
        previous_kept_pos = pos

    flush()
    return segments, rejected, events, stats


def thin_consecutive_near_duplicate_runs(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], int]:
    """Compress brief consecutive near-coincident clusters, retaining the first and last points of each cluster.

    Unlike stationary-stop detection, this step does not require 120 seconds and handles devices repeatedly
    recording nearly identical positions within tens of seconds. Genuine slow movement usually has higher path efficiency,
    so short step lengths alone do not cause compression.
    """
    work = frame.copy().reset_index(drop=True)
    if len(work) < cfg.near_duplicate_run_min_points:
        return work, [], 0
    x = work["_x"].to_numpy(dtype=float)
    y = work["_y"].to_numpy(dtype=float)
    remove: set[int] = set()
    start = 0
    while start < len(work) - 1:
        end = start + 1
        while end < len(work):
            indices = slice(start, end + 1)
            diameter = math.hypot(
                float(np.max(x[indices]) - np.min(x[indices])),
                float(np.max(y[indices]) - np.min(y[indices])),
            )
            if diameter > 2.0 * cfg.near_duplicate_run_radius_m:
                break
            end += 1
        candidate_end = end - 1
        count = candidate_end - start + 1
        if count >= cfg.near_duplicate_run_min_points:
            dx = np.diff(x[start : candidate_end + 1])
            dy = np.diff(y[start : candidate_end + 1])
            steps = np.hypot(dx, dy)
            path_length = float(np.sum(steps))
            endpoint = math.hypot(
                x[candidate_end] - x[start],
                y[candidate_end] - y[start],
            )
            efficiency = endpoint / max(path_length, 1e-9)
            median_step = float(np.median(steps)) if steps.size else math.nan
            compact = (
                endpoint <= cfg.near_duplicate_run_radius_m
                and (
                    efficiency <= cfg.near_duplicate_run_path_efficiency_max
                    or (
                        math.isfinite(median_step)
                        and median_step <= cfg.near_duplicate_run_median_step_max_m
                    )
                )
            )
            if compact and candidate_end - start >= 2:
                remove.update(range(start + 1, candidate_end))
                start = candidate_end
                continue
        start += 1

    if not remove:
        return work, [], 0
    indices = sorted(remove)
    rejected = [
        rejected_rows(
            work.iloc[indices],
            "duplicate_position",
            "consecutive_near_duplicate_thinned",
        )
    ]
    return work.drop(index=indices).reset_index(drop=True), rejected, len(indices)


# =============================================================================
# Cleaning step 2: isolated A-B-C spikes
# =============================================================================
def _edge_is_spike_suspicious(
    distance_m: float,
    speed_mps: float,
    dt_s: float,
    local_speed_ceiling_mps: float,
    local_step_reference_m: float,
    cfg: CleaningConfig,
) -> bool:
    """Determine whether an edge provides evidence of a local spike.

    Use the local speed limit when valid times exist. For same-second or missing times, also require the edge to be much
    longer than normal local sampling steps to avoid deleting sparsely sampled mountain curves under weak signals in Mainling.
    """
    if math.isfinite(speed_mps):
        return speed_mps > local_speed_ceiling_mps
    if math.isfinite(dt_s):
        if dt_s == 0:
            threshold = cfg.spike_zero_dt_leg_m
            if math.isfinite(local_step_reference_m):
                threshold = max(
                    threshold,
                    cfg.spike_zero_dt_step_ratio * local_step_reference_m,
                )
            return distance_m >= threshold
        if dt_s < 0:
            return True
    threshold = cfg.spike_missing_time_leg_m
    if math.isfinite(local_step_reference_m):
        threshold = max(
            threshold,
            cfg.spike_missing_time_step_ratio * local_step_reference_m,
        )
    return distance_m > cfg.hard_step_break_m or distance_m >= threshold


def _direct_connection_is_reasonable(
    direct_m: float,
    time_ns: np.ndarray,
    start: int,
    end: int,
    local_ceiling_mps: float,
    cfg: CleaningConfig,
) -> bool:
    dt = seconds_between(time_ns, start, end)
    if math.isfinite(dt):
        if dt <= 0:
            return False
        direct_speed = direct_m / dt
        return direct_speed <= min(
            cfg.hard_speed_break_mps,
            max(local_ceiling_mps, cfg.hiking_speed_flag_mps) * 1.5,
        )
    # With missing times, accept only if the endpoint distance is short enough to avoid forced gap bridging.
    return direct_m <= cfg.spike_direct_max_m


def remove_isolated_spikes(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], int]:
    """Remove isolated A-B-C planimetric spikes supported by sufficient evidence."""
    work = frame.copy().reset_index(drop=True)
    rejected: List[pd.DataFrame] = []
    total_removed = 0

    for _ in range(cfg.spike_max_passes):
        if len(work) < 3:
            break
        metrics = compute_metrics(work)
        x = work["_x"].to_numpy(dtype=float)
        y = work["_y"].to_numpy(dtype=float)
        time_ns = datetime_ns_array(work["_time"])
        ceilings = local_speed_ceilings(
            metrics["speed"], cfg.spike_local_window_edges,
            cfg.spike_local_speed_floor_mps
        )
        step_references = local_step_references(
            metrics["step"], cfg.spike_local_window_edges
        )
        candidates: List[int] = []

        for i in range(1, len(work) - 1):
            d1 = float(metrics["step"][i])
            d2 = float(metrics["step"][i + 1])
            s1 = float(metrics["speed"][i])
            s2 = float(metrics["speed"][i + 1])
            dt1 = float(metrics["dt"][i])
            dt2 = float(metrics["dt"][i + 1])
            if not (math.isfinite(d1) and math.isfinite(d2)):
                continue
            if d1 < cfg.spike_min_leg_m or d2 < cfg.spike_min_leg_m:
                continue

            ceiling1 = float(ceilings[i])
            ceiling2 = float(ceilings[i + 1])
            edge1_bad = _edge_is_spike_suspicious(
                d1, s1, dt1, ceiling1, float(step_references[i]), cfg
            )
            edge2_bad = _edge_is_spike_suspicious(
                d2, s2, dt2, ceiling2, float(step_references[i + 1]), cfg
            )
            if not (edge1_bad and edge2_bad):
                continue

            detour = d1 + d2
            direct = math.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
            if detour < cfg.spike_min_detour_m:
                continue
            if detour / max(direct, 1.0) < cfg.spike_detour_ratio:
                continue
            if direct > max(cfg.spike_direct_max_m, cfg.spike_direct_ratio * detour):
                continue

            cross_track = point_segment_distance(
                x[i], y[i], x[i - 1], y[i - 1], x[i + 1], y[i + 1]
            )
            if cross_track < cfg.spike_cross_track_min_m:
                continue

            if path_flanks_are_continuous(
                x, y, i - 1, i + 1, cfg.spike_flank_angle_max_deg
            ):
                work.at[i, "_flags"] = append_flag(
                    work.at[i, "_flags"], "switchback_geometry_protected"
                )
                continue

            if not _direct_connection_is_reasonable(
                direct, time_ns, i - 1, i + 1, max(ceiling1, ceiling2), cfg
            ):
                continue

            # Do not remove adjacent points in the same pass, which could progressively flatten genuine continuous bends.
            if candidates and i == candidates[-1] + 1:
                continue
            candidates.append(i)

        if not candidates:
            break
        rejected.append(
            rejected_rows(work.iloc[candidates], "geometry", "isolated_abc_spike")
        )
        total_removed += len(candidates)
        work = work.drop(index=candidates).reset_index(drop=True)

    return work, rejected, total_removed


def remove_short_excursion_clusters(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], int]:
    """Remove brief multipoint drift clusters that leave and then return to the local corridor.

    Unlike a single A-B-C spike, this function allows 2-N consecutive anomalous points. Remove them only when direct endpoint connection is reasonable,
    the detour and lateral deviation are substantial, and both entry and exit edges are anomalous, protecting genuine hairpins.
    """
    work = frame.copy().reset_index(drop=True)
    rejected: List[pd.DataFrame] = []
    total_removed = 0

    for _ in range(cfg.excursion_cluster_max_passes):
        if len(work) < 4:
            break
        metrics = compute_metrics(work)
        x = work["_x"].to_numpy(dtype=float)
        y = work["_y"].to_numpy(dtype=float)
        time_ns = datetime_ns_array(work["_time"])
        ceilings = local_speed_ceilings(
            metrics["speed"], cfg.spike_local_window_edges,
            cfg.spike_local_speed_floor_mps
        )
        step_references = local_step_references(
            metrics["step"], cfg.spike_local_window_edges
        )
        remove_indices: List[int] = []
        anchor = 0

        while anchor < len(work) - 3:
            first_edge = anchor + 1
            d_first = float(metrics["step"][first_edge])
            s_first = float(metrics["speed"][first_edge])
            dt_first = float(metrics["dt"][first_edge])
            ceiling_first = float(ceilings[first_edge])
            if not _edge_is_spike_suspicious(
                d_first, s_first, dt_first, ceiling_first,
                float(step_references[first_edge]), cfg
            ):
                anchor += 1
                continue

            found: Optional[Tuple[int, int]] = None
            max_end = min(
                len(work) - 1,
                anchor + cfg.excursion_cluster_max_points + 1,
            )
            # Remove at least two points; single-point spikes are already handled by A-B-C.
            for end in range(anchor + 3, max_end + 1):
                duration = seconds_between(time_ns, anchor, end)
                if math.isfinite(duration):
                    if duration <= 0:
                        continue
                    if duration > cfg.excursion_cluster_max_duration_s:
                        break

                last_edge = end
                d_last = float(metrics["step"][last_edge])
                s_last = float(metrics["speed"][last_edge])
                dt_last = float(metrics["dt"][last_edge])
                ceiling_last = float(ceilings[last_edge])
                if not _edge_is_spike_suspicious(
                    d_last, s_last, dt_last, ceiling_last,
                    float(step_references[last_edge]), cfg
                ):
                    continue

                direct = math.hypot(x[end] - x[anchor], y[end] - y[anchor])
                path_length = float(np.nansum(metrics["step"][anchor + 1 : end + 1]))
                if path_length < cfg.excursion_cluster_min_detour_m:
                    continue
                if path_length / max(direct, 1.0) < cfg.excursion_cluster_detour_ratio:
                    continue
                if direct > cfg.excursion_cluster_direct_max_m:
                    continue

                cross_track = max(
                    point_segment_distance(
                        x[k], y[k], x[anchor], y[anchor], x[end], y[end]
                    )
                    for k in range(anchor + 1, end)
                )
                if cross_track < cfg.excursion_cluster_cross_track_min_m:
                    continue

                if path_flanks_are_continuous(
                    x, y, anchor, end, cfg.excursion_flank_angle_max_deg
                ):
                    for protected_i in range(anchor + 1, end):
                        work.at[protected_i, "_flags"] = append_flag(
                            work.at[protected_i, "_flags"],
                            "switchback_cluster_geometry_protected",
                        )
                    continue

                if not _direct_connection_is_reasonable(
                    direct, time_ns, anchor, end,
                    max(ceiling_first, ceiling_last), cfg
                ):
                    continue
                found = (anchor + 1, end)
                break

            if found is None:
                anchor += 1
                continue

            start_remove, end_remove_exclusive = found
            remove_indices.extend(range(start_remove, end_remove_exclusive))
            anchor = end_remove_exclusive

        if not remove_indices:
            break
        remove_indices = sorted(set(remove_indices))
        rejected.append(
            rejected_rows(
                work.iloc[remove_indices],
                "geometry",
                "short_excursion_cluster",
            )
        )
        total_removed += len(remove_indices)
        work = work.drop(index=remove_indices).reset_index(drop=True)

    return work, rejected, total_removed


# =============================================================================
# Cleaning step 3: hard spatial/speed splits
# =============================================================================
def _local_positive_values(values: np.ndarray, center: int, window: int) -> np.ndarray:
    start = max(1, center - window)
    end = min(len(values), center + window + 1)
    positions = [i for i in range(start, end) if i != center]
    if not positions:
        return np.asarray([], dtype=float)
    output = values[np.asarray(positions, dtype=int)]
    return output[np.isfinite(output) & (output > 0)]


def _adaptive_step_threshold(
    steps: np.ndarray,
    center: int,
    cfg: CleaningConfig,
) -> Tuple[float, float]:
    local = _local_positive_values(steps, center, cfg.adaptive_step_window_edges)
    local = local[local < cfg.hard_step_break_m]
    if local.size < 3:
        return cfg.adaptive_step_floor_m, math.nan
    median = float(np.median(local))
    mad = float(np.median(np.abs(local - median)))
    robust = median + cfg.adaptive_step_mad_k * 1.4826 * mad
    threshold = max(
        cfg.adaptive_step_floor_m,
        robust,
        cfg.adaptive_step_ratio * median,
    )
    return min(threshold, cfg.hard_step_break_m), median


def split_by_hard_discontinuities(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
    input_file: str,
) -> Tuple[List[pd.DataFrame], List[Dict[str, object]], Dict[str, int]]:
    """Split definite spatial discontinuities.

    In addition to the absolute 300 m threshold, detect intermediate-length edges much longer than local sampling steps.
    Adaptive candidates require at least one additional signal: a strong step-length ratio, speed above hiking limits,
    endpoint turns, or an anomalous time interval, avoiding splits caused solely by sharp mountain bends.
    """
    if len(frame) <= 1:
        return [frame.copy().reset_index(drop=True)], [], {
            "hard_distance_splits": 0,
            "hard_speed_splits": 0,
            "adaptive_distance_splits": 0,
        }

    metrics = compute_metrics(frame)
    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    speed_ceilings = local_speed_ceilings(
        metrics["speed"],
        cfg.adaptive_step_window_edges,
        cfg.hiking_speed_flag_mps,
    )
    breaks = np.zeros(len(frame), dtype=bool)
    events: List[Dict[str, object]] = []
    stats = {
        "hard_distance_splits": 0,
        "hard_speed_splits": 0,
        "adaptive_distance_splits": 0,
    }

    for i in range(1, len(frame)):
        step = float(metrics["step"][i])
        speed = float(metrics["speed"][i])
        dt = float(metrics["dt"][i])
        if not math.isfinite(step):
            continue

        if step > cfg.hard_step_break_m:
            breaks[i] = True
            stats["hard_distance_splits"] += 1
            events.append(split_event(frame, i, "space", "distance_over_hard_threshold", input_file))
            continue

        if (
            math.isfinite(speed)
            and speed > cfg.hard_speed_break_mps
            and step >= cfg.hard_speed_min_step_m
        ):
            breaks[i] = True
            stats["hard_speed_splits"] += 1
            events.append(split_event(frame, i, "space", "extreme_speed_jump", input_file))
            continue

        threshold, local_median = _adaptive_step_threshold(metrics["step"], i, cfg)
        if step <= threshold or step < cfg.adaptive_step_floor_m:
            continue

        ratio = step / max(local_median, 1.0) if math.isfinite(local_median) else math.nan
        turn_before = float(metrics["turn"][i - 1]) if i - 1 >= 0 else math.nan
        turn_after = float(metrics["turn"][i]) if i < len(frame) else math.nan
        both_endpoint_turns = (
            math.isfinite(turn_before)
            and math.isfinite(turn_after)
            and turn_before >= cfg.adaptive_step_turn_support_deg
            and turn_after >= cfg.adaptive_step_turn_support_deg
        )
        turn_support = (
            both_endpoint_turns
            and math.isfinite(ratio)
            and ratio >= cfg.adaptive_step_turn_min_ratio
        )

        local_speed_limit = float(speed_ceilings[i])
        speed_support = (
            math.isfinite(speed)
            and math.isfinite(ratio)
            and ratio >= cfg.adaptive_step_speed_min_ratio
            and speed > max(
                cfg.hiking_speed_flag_mps,
                1.25 * local_speed_limit,
            )
        )
        strong_ratio = (
            math.isfinite(ratio)
            and ratio >= cfg.adaptive_step_strong_ratio
        )

        local_dt = _local_positive_values(metrics["dt"], i, cfg.adaptive_step_window_edges)
        local_dt_median = float(np.median(local_dt)) if local_dt.size >= 3 else math.nan
        time_support = (
            math.isfinite(dt)
            and dt >= cfg.adaptive_step_min_time_gap_s
            and math.isfinite(ratio)
            and ratio >= cfg.adaptive_step_ratio
            and (
                not math.isfinite(local_dt_median)
                or dt >= cfg.adaptive_step_time_ratio * max(local_dt_median, 1.0)
            )
        )

        # A long edge with directional continuity on both sides is more likely sparse sampling under weak signals
        # than an incorrect shortcut across a bend. Prioritize protection when a large step-length ratio is the only evidence.
        direction_aligned = path_flanks_are_continuous(
            x, y, i - 1, i, max_angle_deg=45.0
        )
        missing_time_support = (
            not math.isfinite(dt)
            and math.isfinite(ratio)
            and ratio >= cfg.adaptive_step_missing_time_ratio
            and not direction_aligned
        )
        supported = (
            speed_support
            or (time_support and not direction_aligned)
            or turn_support
            or missing_time_support
            or (strong_ratio and not direction_aligned)
        )

        if supported:
            reasons: List[str] = []
            if strong_ratio:
                reasons.append("strong_step_ratio")
            if speed_support:
                reasons.append("speed")
            if time_support and not direction_aligned:
                reasons.append("time_gap")
            if turn_support:
                reasons.append("two_endpoint_turns")
            if missing_time_support:
                reasons.append("missing_time_strong_ratio")
            breaks[i] = True
            stats["adaptive_distance_splits"] += 1
            events.append(
                split_event(
                    frame,
                    i,
                    "space",
                    "adaptive_distance_discontinuity:" + "+".join(reasons),
                    input_file,
                )
            )
        elif direction_aligned:
            frame.at[i, "_flags"] = append_flag(
                frame.at[i, "_flags"],
                "adaptive_long_edge_protected_direction_continuity",
            )

    return split_frame_by_breaks(frame, breaks), events, stats


# =============================================================================
# Cleaning step 4: stationary-drift compression
# =============================================================================
def stationary_window_evidence(
    frame: pd.DataFrame,
    start: int,
    end: int,
    cfg: CleaningConfig,
    metrics: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[bool, Dict[str, object]]:
    """Assess whether a candidate window represents a stop using a robust spatial center.

    Allow a few transient multipath-drift points, excluding them from robust diameter and endpoint-position calculations.
    """
    if end - start < 2:
        return False, {}
    if metrics is None:
        metrics = compute_metrics(frame)

    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    times = frame["_time"].array
    if pd.isna(times[start]) or pd.isna(times[end]):
        return False, {}

    duration = (times[end] - times[start]).total_seconds()
    if duration < cfg.stationary_min_duration_s or duration > cfg.stationary_max_scan_s:
        return False, {}

    indices = np.arange(start, end + 1)
    xs = x[indices]
    ys = y[indices]
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))
    radial = np.hypot(xs - center_x, ys - center_y)
    inlier = radial <= cfg.stationary_robust_radius_m
    inlier_count = int(inlier.sum())
    outlier_count = int(len(indices) - inlier_count)
    inlier_fraction = inlier_count / max(len(indices), 1)
    if (
        inlier_count < 3
        or outlier_count > cfg.stationary_outlier_tolerance_points
        or inlier_fraction < cfg.stationary_min_inlier_fraction
    ):
        return False, {}

    inlier_indices = indices[inlier]
    inlier_x = x[inlier_indices]
    inlier_y = y[inlier_indices]
    robust_diameter = math.hypot(
        float(np.max(inlier_x) - np.min(inlier_x)),
        float(np.max(inlier_y) - np.min(inlier_y)),
    )
    if robust_diameter > cfg.stationary_diameter_m:
        return False, {}

    effective_start = int(inlier_indices[0])
    effective_end = int(inlier_indices[-1])
    if pd.isna(times[effective_start]) or pd.isna(times[effective_end]):
        return False, {}
    effective_duration = (times[effective_end] - times[effective_start]).total_seconds()
    if effective_duration < cfg.stationary_min_duration_s:
        return False, {}

    endpoint = math.hypot(
        x[effective_end] - x[effective_start],
        y[effective_end] - y[effective_start],
    )
    path_length = float(
        np.nansum(metrics["step"][effective_start + 1 : effective_end + 1])
    )
    efficiency = endpoint / max(path_length, 1.0)
    speeds = metrics["speed"][effective_start + 1 : effective_end + 1]
    valid_speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
    median_speed = float(np.median(valid_speeds)) if valid_speeds.size else math.nan

    stationary = (
        endpoint <= cfg.stationary_endpoint_max_m
        and efficiency <= cfg.stationary_path_efficiency_max
        and (
            not math.isfinite(median_speed)
            or median_speed <= cfg.stationary_median_speed_max_mps
        )
    )
    return stationary, {
        "start": effective_start,
        "end": effective_end,
        "duration": effective_duration,
        "robust_diameter": robust_diameter,
        "endpoint": endpoint,
        "path_length": path_length,
        "path_efficiency": efficiency,
        "median_speed": median_speed,
        "inlier_indices": inlier_indices,
        "outlier_count": outlier_count,
        "inlier_fraction": inlier_fraction,
    }


def _rolling_median_coordinates(values: np.ndarray, window: int = 5) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def detect_stationary_windows(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> List[Tuple[int, int, Dict[str, object]]]:
    """Detect stationary windows in near-linear time.

    The first pass uses raw coordinates for consistency with the conventional spatial-diameter method; the second uses 5-point median
    coordinates only to recover stops interrupted by 1-2 transient multipath-drift points. Final decisions always
    use robust evidence from the raw coordinates.
    """
    n = len(frame)
    if n < 3:
        return []
    metrics = compute_metrics(frame)
    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    time_ns = datetime_ns_array(frame["_time"])

    def scan(x_scan: np.ndarray, y_scan: np.ndarray) -> List[Tuple[int, int, Dict[str, object]]]:
        min_x: deque[int] = deque()
        max_x: deque[int] = deque()
        min_y: deque[int] = deque()
        max_y: deque[int] = deque()
        left = 0
        found: List[Tuple[int, int, Dict[str, object]]] = []

        def push(index: int) -> None:
            while min_x and x_scan[min_x[-1]] >= x_scan[index]:
                min_x.pop()
            min_x.append(index)
            while max_x and x_scan[max_x[-1]] <= x_scan[index]:
                max_x.pop()
            max_x.append(index)
            while min_y and y_scan[min_y[-1]] >= y_scan[index]:
                min_y.pop()
            min_y.append(index)
            while max_y and y_scan[max_y[-1]] <= y_scan[index]:
                max_y.pop()
            max_y.append(index)

        def discard_before(index: int) -> None:
            for queue in (min_x, max_x, min_y, max_y):
                while queue and queue[0] < index:
                    queue.popleft()

        for right in range(n):
            push(right)
            while left < right:
                duration = seconds_between(time_ns, left, right)
                diameter = math.hypot(
                    x_scan[max_x[0]] - x_scan[min_x[0]],
                    y_scan[max_y[0]] - y_scan[min_y[0]],
                )
                too_long = math.isfinite(duration) and duration > cfg.stationary_max_scan_s
                if diameter <= cfg.stationary_diameter_m and not too_long:
                    break
                left += 1
                discard_before(left)

            if left >= right:
                continue
            duration = seconds_between(time_ns, left, right)
            if not math.isfinite(duration) or duration < cfg.stationary_min_duration_s:
                continue
            ok, evidence = stationary_window_evidence(
                frame, left, right, cfg, metrics=metrics
            )
            if not ok:
                # The earliest feasible window can still include a few moving points preceding the stop. Check a bounded
                # number of later start positions rather than reverting to unbounded O(N^2) enumeration.
                target_ns = int(time_ns[right] - cfg.stationary_min_duration_s * 1_000_000_000)
                if time_ns[right] != NAT_INT64:
                    valid_prefix = time_ns[left:right]
                    valid_positions = np.flatnonzero(
                        (valid_prefix != NAT_INT64) & (valid_prefix <= target_ns)
                    )
                    if valid_positions.size:
                        latest_start = left + int(valid_positions[-1])
                        first_candidates = list(range(
                            left + 1,
                            min(latest_start + 1, left + 1 + cfg.stationary_alternate_start_checks),
                        ))
                        last_begin = max(left + 1, latest_start - cfg.stationary_alternate_start_checks + 1)
                        last_candidates = list(range(last_begin, latest_start + 1))
                        for alternative_start in dict.fromkeys(first_candidates + last_candidates):
                            ok, evidence = stationary_window_evidence(
                                frame, alternative_start, right, cfg, metrics=metrics
                            )
                            if ok:
                                break
            if ok:
                found.append((
                    int(evidence["start"]), int(evidence["end"]), evidence
                ))
        return found

    raw_windows = scan(x, y)
    filtered_windows = scan(
        _rolling_median_coordinates(x),
        _rolling_median_coordinates(y),
    )
    candidates = raw_windows + filtered_windows
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1]))

    # Deduplicate and greedily retain longer windows. Merge overlapping windows first; if the merged window fails the criteria,
    # retain the one with greater coverage.
    merged: List[Tuple[int, int, Dict[str, object]]] = []
    for start, end, evidence in candidates:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, evidence))
            continue
        previous_start, previous_end, previous_evidence = merged[-1]
        merged_start = min(previous_start, start)
        merged_end = max(previous_end, end)
        ok, merged_evidence = stationary_window_evidence(
            frame, merged_start, merged_end, cfg, metrics=metrics
        )
        if ok:
            merged[-1] = (
                int(merged_evidence["start"]),
                int(merged_evidence["end"]),
                merged_evidence,
            )
        elif (end - start) > (previous_end - previous_start):
            merged[-1] = (start, end, evidence)
    return merged


def thin_stationary_drift(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
    stationary_windows: Optional[List[Tuple[int, int, Dict[str, object]]]] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], int]:
    """Compress confirmed stationary windows, retaining endpoint representatives and removing intermediate drift points."""
    work = frame.copy().reset_index(drop=True)
    windows = (
        stationary_windows
        if stationary_windows is not None
        else detect_stationary_windows(work, cfg)
    )
    if not windows:
        return work, [], 0

    remove: set[int] = set()
    for start, end, _ in windows:
        if end - start >= 2:
            remove.update(range(start + 1, end))
    if not remove:
        return work, [], 0

    indices = sorted(remove)
    rejected = [
        rejected_rows(work.iloc[indices], "stationary", "stationary_drift_thinned")
    ]
    return work.drop(index=indices).reset_index(drop=True), rejected, len(indices)


# =============================================================================
# Cleaning step 5: elevation
# =============================================================================
def _weighted_elevation_expected(
    za: float,
    zc: float,
    d1: float,
    d2: float,
    minimum_distance: float,
) -> float:
    """Estimate an intermediate elevation along track distance; clamp very short distances for numerical stability."""
    d1 = max(float(d1), minimum_distance)
    d2 = max(float(d2), minimum_distance)
    return (za * d2 + zc * d1) / (d1 + d2)


def _mark_elevation_repair(
    flags: List[object],
    repaired_mask: np.ndarray,
    methods: List[object],
    qualities: List[object],
    confidences: List[object],
    index: int,
    flag: str,
    method: str,
    confidence: str = "high",
) -> None:
    flags[index] = append_flag(flags[index], flag)
    repaired_mask[index] = True
    methods[index] = method
    qualities[index] = "repaired"
    confidences[index] = confidence


def _stationary_elevation_outliers(
    frame: pd.DataFrame,
    z: np.ndarray,
    cfg: CleaningConfig,
    stationary_windows: Optional[List[Tuple[int, int, Dict[str, object]]]] = None,
) -> List[Tuple[int, float]]:
    """Return elevation outliers and the median within confirmed stationary windows."""
    candidates: Dict[int, float] = {}
    windows = (
        stationary_windows
        if stationary_windows is not None
        else detect_stationary_windows(frame, cfg)
    )
    for start, end, evidence in windows:
        indices = np.arange(start, end + 1)
        values = z[indices]
        valid = np.isfinite(values)
        if int(valid.sum()) < 3:
            continue
        valid_values = values[valid]
        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(
            cfg.elevation_stationary_deviation_min_m,
            cfg.elevation_stationary_mad_k * robust_sigma,
        )
        for idx, value in zip(indices[valid], valid_values):
            if abs(float(value) - median) >= threshold:
                candidates[int(idx)] = median
    return sorted(candidates.items())


def estimate_dem_vertical_bias(
    z: np.ndarray,
    dem: np.ndarray,
    cfg: CleaningConfig,
) -> float:
    """Robustly estimate this trajectory's systematic GPS elevation bias relative to the DEM.

    The bias may reflect different vertical datums or systematic device errors. Align the DEM to
    this trajectory's GPS elevation datum before repairs to avoid raising or lowering the entire trajectory.
    """
    valid = np.isfinite(z) & np.isfinite(dem)
    residuals = z[valid] - dem[valid]
    if residuals.size < cfg.dem_bias_min_samples:
        return math.nan
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    sigma = 1.4826 * mad
    threshold = max(10.0, cfg.dem_bias_mad_k * sigma)
    inliers = residuals[np.abs(residuals - median) <= threshold]
    if inliers.size < cfg.dem_bias_min_samples:
        return math.nan
    bias = float(np.median(inliers))
    if not math.isfinite(bias) or abs(bias) > cfg.dem_bias_max_abs_m:
        return math.nan
    return bias


def short_missing_elevation_runs(
    frame: pd.DataFrame,
    z: np.ndarray,
    cfg: CleaningConfig,
) -> List[Tuple[int, int, np.ndarray]]:
    """Return short elevation gaps eligible for along-track interpolation and their interpolated values."""
    missing = ~np.isfinite(z)
    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    output: List[Tuple[int, int, np.ndarray]] = []
    for start, end in boolean_runs(missing, value=True):
        if start == 0 or end >= len(z) - 1:
            continue
        count = end - start + 1
        if count > cfg.elevation_short_gap_max_points:
            continue
        if not (math.isfinite(z[start - 1]) and math.isfinite(z[end + 1])):
            continue
        indices = np.arange(start - 1, end + 2)
        edge_lengths = np.hypot(np.diff(x[indices]), np.diff(y[indices]))
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        total = float(cumulative[-1])
        if total <= 1e-9 or total > cfg.elevation_short_gap_max_distance_m:
            continue
        fraction = cumulative[1:-1] / total
        values = z[start - 1] + fraction * (z[end + 1] - z[start - 1])
        output.append((start, end, np.asarray(values, dtype=float)))
    return output


def _dem_local_reliability(
    frame: pd.DataFrame,
    dem: np.ndarray,
    index: int,
    cfg: CleaningConfig,
) -> Tuple[bool, float, float]:
    """Assess whether the along-track DEM neighborhood is stable enough for automatic elevation repair.

    Returns reliability, maximum local slope, and the suggested trend tolerance.
    """
    if index < 0 or index >= len(frame) or not math.isfinite(dem[index]):
        return False, math.nan, cfg.dem_repair_steep_tolerance_m
    x = frame["_x"].to_numpy(dtype=float)
    y = frame["_y"].to_numpy(dtype=float)
    start = max(0, index - cfg.dem_local_window_points)
    end = min(len(frame), index + cfg.dem_local_window_points + 1)
    local_values: List[float] = [float(dem[index])]
    grades: List[float] = []
    for j in range(start, end):
        if j == index or not math.isfinite(dem[j]):
            continue
        distance = math.hypot(x[j] - x[index], y[j] - y[index])
        if distance <= cfg.elevation_min_horizontal_distance_m:
            continue
        local_values.append(float(dem[j]))
        grades.append(abs(float(dem[j] - dem[index])) / distance)
    if len(local_values) < 2:
        return False, math.nan, cfg.dem_repair_steep_tolerance_m
    local_range = max(local_values) - min(local_values)
    max_grade = max(grades) if grades else 0.0
    reliable = (
        max_grade <= cfg.dem_trust_max_grade
        and local_range <= cfg.dem_trust_max_local_range_m
    )
    tolerance = (
        cfg.dem_repair_steep_tolerance_m
        if max_grade >= cfg.dem_steep_grade_threshold
        else cfg.dem_repair_trend_tolerance_m
    )
    return reliable, max_grade, tolerance


def repair_elevation(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
    stationary_windows: Optional[List[Tuple[int, int, Dict[str, object]]]] = None,
) -> Tuple[pd.DataFrame, int, int, int, int]:
    """
    Elevation quality control.

    Processing rules:
    1. Elevation anomalies do not delete longitude/latitude points or split the planimetric trajectory;
    2. Repair elevation_clean_m only for well-supported isolated spikes and outliers in stationary windows;
    3. Flag sustained steps or insufficiently supported anomalies for subsequent DEM/multitrajectory consistency checks;
    4. Always retain the original elevation_m values.
    """
    work = frame.copy().reset_index(drop=True)
    z = work["_z_clean"].to_numpy(dtype=float).copy()
    dem = work["_dem_z"].to_numpy(dtype=float)
    dem_bias = estimate_dem_vertical_bias(z, dem, cfg)
    dem_aligned = dem + dem_bias if math.isfinite(dem_bias) else dem.copy()
    work["_dem_bias"] = dem_bias
    work["_dem_z_aligned"] = dem_aligned
    flags = work["_flags"].astype(object).tolist()
    repaired_mask = work.get("_elev_repaired", pd.Series(False, index=work.index)).to_numpy(dtype=bool)
    methods = work.get("_elev_method", pd.Series("", index=work.index)).astype(object).tolist()
    qualities = work.get("_elev_quality", pd.Series("normal", index=work.index)).astype(object).tolist()
    confidences = work.get("_elev_confidence", pd.Series("", index=work.index)).astype(object).tolist()
    local_residual = np.full(len(work), np.nan, dtype=float)

    # First repair short gaps bounded by reliable GPS elevations.
    interpolated = 0
    for start, end, values in short_missing_elevation_runs(work, z, cfg):
        for offset, i in enumerate(range(start, end + 1)):
            z[i] = float(values[offset])
            _mark_elevation_repair(
                flags, repaired_mask, methods, qualities, confidences,
                i, "elevation_short_gap_interpolated",
                "along_track_distance_interpolation", "high"
            )
            interpolated += 1

    dem_filled = 0
    dem_skipped_unreliable = 0
    if cfg.fill_missing_elevation_from_dem:
        missing = ~np.isfinite(z) & np.isfinite(dem_aligned)
        for raw_i in np.flatnonzero(missing):
            i = int(raw_i)
            dem_reliable, _, _ = _dem_local_reliability(work, dem, i, cfg)
            if not dem_reliable:
                flags[i] = append_flag(
                    flags[i], "dem_fill_skipped_unreliable_local_terrain"
                )
                if qualities[i] != "repaired":
                    qualities[i] = "flagged"
                confidences[i] = "medium"
                dem_skipped_unreliable += 1
                continue
            z[i] = dem_aligned[i]
            method = "dem_fill_bias_aligned" if math.isfinite(dem_bias) else "dem_fill_unaligned"
            confidence = "high" if math.isfinite(dem_bias) else "medium"
            flag = (
                "elevation_filled_from_bias_aligned_dem"
                if math.isfinite(dem_bias)
                else "elevation_filled_from_unaligned_dem"
            )
            _mark_elevation_repair(
                flags, repaired_mask, methods, qualities, confidences,
                i, flag, method, confidence
            )
            dem_filled += 1

    repaired = interpolated
    repaired_indices: set[int] = set(
        int(i) for i in np.flatnonzero(repaired_mask)
    )

    # A. Confirmed stationary windows: repair only substantial elevation outliers within each window.
    for i, stationary_median in _stationary_elevation_outliers(
        work, z, cfg, stationary_windows=stationary_windows
    ):
        replacement = stationary_median
        dem_reliable, _, dem_tolerance = _dem_local_reliability(work, dem, i, cfg)
        if (
            dem_reliable
            and math.isfinite(dem_aligned[i])
            and abs(dem_aligned[i] - stationary_median) <= dem_tolerance
        ):
            replacement = dem_aligned[i]
            method = "stationary_dem"
            flag = "elevation_stationary_outlier_repaired_with_dem"
        else:
            if math.isfinite(dem_aligned[i]) and not dem_reliable:
                flags[i] = append_flag(flags[i], "dem_not_used_unreliable_local_terrain")
            method = "stationary_median"
            flag = "elevation_stationary_outlier_repaired_local"
        if not repaired_mask[i] or not math.isclose(z[i], replacement, abs_tol=1e-9):
            z[i] = replacement
            _mark_elevation_repair(
                flags, repaired_mask, methods, qualities, confidences,
                i, flag, method, "high"
            )
            if i not in repaired_indices:
                repaired_indices.add(i)
                repaired += 1

    # B. Isolated A-B-C elevation spikes: local residual + opposite changes on both sides + physical-anomaly evidence.
    for _ in range(cfg.elevation_max_passes):
        if len(work) < 3:
            break
        x = work["_x"].to_numpy(dtype=float)
        y = work["_y"].to_numpy(dtype=float)
        times = work["_time"].array

        residual_values: List[float] = []
        expected_values = np.full(len(work), np.nan, dtype=float)
        for i in range(1, len(work) - 1):
            za, zb, zc = z[i - 1], z[i], z[i + 1]
            if not np.isfinite([za, zb, zc]).all():
                continue
            d1 = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            d2 = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            expected = _weighted_elevation_expected(
                za, zc, d1, d2, cfg.elevation_min_horizontal_distance_m
            )
            expected_values[i] = expected
            residual_values.append(abs(zb - expected))

        finite_residuals = np.asarray(residual_values, dtype=float)
        finite_residuals = finite_residuals[np.isfinite(finite_residuals)]
        if finite_residuals.size < cfg.elevation_residual_min_samples:
            threshold = float(cfg.elevation_spike_min_m)
        else:
            threshold = robust_upper(
                finite_residuals,
                floor=cfg.elevation_spike_min_m,
                k=cfg.elevation_residual_mad_k,
            )

        candidates: List[Tuple[int, float, str]] = []
        for i in range(1, len(work) - 1):
            za, zb, zc = z[i - 1], z[i], z[i + 1]
            expected = expected_values[i]
            if not np.isfinite([za, zb, zc, expected]).all():
                continue
            d1 = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            d2 = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            dz1 = zb - za
            dz2 = zc - zb
            residual = abs(zb - expected)
            local_residual[i] = residual
            if repaired_mask[i]:
                continue

            # An isolated spike must rise and then fall, or fall and then rise.
            reversal = dz1 * dz2 < 0
            side_changes_large = (
                abs(dz1) >= cfg.elevation_side_change_min_m
                and abs(dz2) >= cfg.elevation_side_change_min_m
            )
            if not (residual >= threshold and reversal and side_changes_large):
                continue

            grade1 = abs(dz1) / max(d1, cfg.elevation_min_horizontal_distance_m)
            grade2 = abs(dz2) / max(d2, cfg.elevation_min_horizontal_distance_m)
            grade_bad = max(grade1, grade2) > cfg.elevation_grade_hard

            vertical_bad = False
            if pd.notna(times[i - 1]) and pd.notna(times[i]):
                dt1 = (times[i] - times[i - 1]).total_seconds()
                vertical_bad |= dt1 > 0 and abs(dz1) / dt1 > cfg.vertical_speed_hard_mps
            if pd.notna(times[i]) and pd.notna(times[i + 1]):
                dt2 = (times[i + 1] - times[i]).total_seconds()
                vertical_bad |= dt2 > 0 and abs(dz2) / dt2 > cfg.vertical_speed_hard_mps

            near_same_bad = (
                d1 <= cfg.elevation_near_same_xy_m
                and abs(dz1) >= cfg.elevation_near_same_xy_jump_m
            ) or (
                d2 <= cfg.elevation_near_same_xy_m
                and abs(dz2) >= cfg.elevation_near_same_xy_jump_m
            )
            if not (grade_bad or vertical_bad or near_same_bad):
                continue

            replacement = expected
            method = "local_weighted_trend"
            dem_reliable, _, dem_tolerance = _dem_local_reliability(work, dem, i, cfg)
            if (
                dem_reliable
                and math.isfinite(dem_aligned[i])
                and abs(dem_aligned[i] - expected) <= dem_tolerance
                and abs(zb - dem_aligned[i]) > abs(expected - dem_aligned[i])
            ):
                replacement = dem_aligned[i]
                method = "dem_supported"
            elif math.isfinite(dem_aligned[i]) and not dem_reliable:
                flags[i] = append_flag(flags[i], "dem_not_used_unreliable_local_terrain")
            candidates.append((i, replacement, method))

        if not candidates:
            break

        # Avoid changing adjacent points in the same pass, which could mistake a sustained step for several isolated spikes.
        selected: List[Tuple[int, float, str]] = []
        for item in candidates:
            if selected and item[0] == selected[-1][0] + 1:
                continue
            selected.append(item)

        changed = 0
        for i, replacement, method in selected:
            z[i] = replacement
            flag = (
                "elevation_spike_repaired_with_dem"
                if method == "dem_supported"
                else "elevation_spike_repaired_local"
            )
            _mark_elevation_repair(
                flags, repaired_mask, methods, qualities, confidences,
                i, flag, method, "high"
            )
            if i not in repaired_indices:
                repaired_indices.add(i)
                repaired += 1
            changed += 1
        if changed == 0:
            break

    # C. Recompute local residuals after repairs and flag unresolved elevation steps or near-coincident jumps.
    x = work["_x"].to_numpy(dtype=float)
    y = work["_y"].to_numpy(dtype=float)
    times = work["_time"].array
    flagged_indices: set[int] = set()

    for i in range(1, len(work) - 1):
        za, zb, zc = z[i - 1], z[i], z[i + 1]
        if np.isfinite([za, zb, zc]).all():
            d1 = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            d2 = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            expected = _weighted_elevation_expected(
                za, zc, d1, d2, cfg.elevation_min_horizontal_distance_m
            )
            local_residual[i] = abs(zb - expected)

    for i in range(1, len(work)):
        if not np.isfinite([z[i - 1], z[i]]).all():
            continue
        distance = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        dz = abs(z[i] - z[i - 1])
        grade = dz / max(distance, cfg.elevation_min_horizontal_distance_m)
        vertical_speed = math.nan
        if pd.notna(times[i - 1]) and pd.notna(times[i]):
            dt = (times[i] - times[i - 1]).total_seconds()
            if dt > 0:
                vertical_speed = dz / dt

        near_same_jump = (
            distance <= cfg.elevation_near_same_xy_m
            and dz >= cfg.elevation_near_same_xy_jump_m
        )
        physical_step = (
            dz >= cfg.elevation_step_change_min_m
            and (
                grade > cfg.elevation_grade_hard
                or (math.isfinite(vertical_speed) and vertical_speed > cfg.vertical_speed_hard_mps)
            )
        )
        if not (near_same_jump or physical_step):
            continue
        if repaired_mask[i] or repaired_mask[i - 1]:
            continue

        if near_same_jump:
            flag = "elevation_near_same_position_jump_candidate"
            confidence = "high"
        else:
            flag = "elevation_step_change_candidate"
            confidence = "medium"
        flags[i] = append_flag(flags[i], flag)
        if qualities[i] != "repaired":
            qualities[i] = "flagged"
        confidences[i] = confidence
        flagged_indices.add(i)

    work["_z_clean"] = z
    work["_flags"] = flags
    work["_elev_repaired"] = repaired_mask
    work["_elev_method"] = methods
    work["_elev_quality"] = qualities
    work["_elev_confidence"] = confidences
    work["_elev_residual"] = local_residual
    return work, repaired, dem_filled, len(flagged_indices), dem_skipped_unreliable


def build_elevation_audit_rows(
    frame: pd.DataFrame,
    input_file: str,
    clip_part_id: int,
    temporal_part_id: int,
    spatial_part_id: int,
) -> pd.DataFrame:
    """Create a separate audit table of repaired/pending elevation points, including those later removed by stationary compression."""
    quality = frame.get("_elev_quality", pd.Series("normal", index=frame.index)).astype(str)
    selected = frame.loc[quality.isin(["repaired", "flagged"])].copy()
    if selected.empty:
        return pd.DataFrame(columns=ELEVATION_AUDIT_FIELDS)
    output = pd.DataFrame(index=selected.index)
    output["input_file"] = input_file
    output["track_id"] = selected.get("track_id", "")
    output["source_file"] = selected.get("source_file", "")
    output["clip_part_id"] = clip_part_id
    output["temporal_part_id"] = temporal_part_id
    output["spatial_part_id"] = spatial_part_id
    output["point_index"] = selected.get("point_index", "")
    output["raw_point_index"] = selected.get("raw_point_index", "")
    output["time_utc"] = selected.get("time_utc", "")
    output["longitude"] = selected.get("longitude", "")
    output["latitude"] = selected.get("latitude", "")
    output["x_m"] = np.round(selected["_x"].to_numpy(dtype=float), 3)
    output["y_m"] = np.round(selected["_y"].to_numpy(dtype=float), 3)
    output["elevation_original_m"] = pd.to_numeric(
        selected.get("elevation_m"), errors="coerce"
    ).to_numpy(dtype=float)
    output["elevation_clean_m"] = np.round(selected["_z_clean"].to_numpy(dtype=float), 3)
    output["dem_elevation_m"] = np.round(selected["_dem_z"].to_numpy(dtype=float), 3)
    output["dem_vertical_bias_m"] = np.round(
        selected.get("_dem_bias", pd.Series(np.nan, index=selected.index)).to_numpy(dtype=float), 3
    )
    output["elevation_dem_centered_residual_m"] = np.round(
        selected["_z_clean"].to_numpy(dtype=float)
        - selected.get("_dem_z_aligned", selected["_dem_z"]).to_numpy(dtype=float),
        3,
    )
    output["elevation_local_residual_m"] = np.round(
        selected.get("_elev_residual", pd.Series(np.nan, index=selected.index)).to_numpy(dtype=float),
        3,
    )
    output["elevation_quality"] = selected["_elev_quality"].astype(str)
    output["elevation_repair_method"] = selected["_elev_method"].astype(str)
    output["elevation_anomaly_confidence"] = selected["_elev_confidence"].astype(str)
    output["quality_flags"] = selected["_flags"].astype(str)
    return output.reset_index(drop=True).reindex(columns=ELEVATION_AUDIT_FIELDS)


def append_flag(existing: object, flag: str) -> str:
    try:
        missing = existing is None or bool(pd.isna(existing))
    except (TypeError, ValueError):
        missing = existing is None
    text = "" if missing else str(existing).strip(" |")
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if flag not in parts:
        parts.append(flag)
    return " | ".join(parts)


# =============================================================================
# Cleaning step 6: hiking-consistency filtering
# =============================================================================
def boolean_runs(mask: np.ndarray, value: bool = True) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, current in enumerate(mask):
        if bool(current) == value and start is None:
            start = i
        elif bool(current) != value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def hiking_speed_evidence(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Dict[str, object]:
    """Build reliable speed evidence for hiking classification.

    Devices in Mainling's valleys may produce an isolated early/late timestamp, making a short or medium edge appear to have a very high
    instantaneous speed. If its speed remains consistent with hiking across a wider time window on both sides, and adjacent edges
    show no sustained high-speed evidence, only downweight that edge without deleting spatial points.
    """
    metrics = compute_metrics(frame)
    speed_edges = metrics["speed"][1:].copy()
    step_edges = metrics["step"][1:].copy()
    dt_edges = metrics["dt"][1:].copy()
    reliable = np.isfinite(speed_edges) & np.isfinite(dt_edges) & (dt_edges > 0)
    jitter = np.zeros(len(speed_edges), dtype=bool)
    time_ns = datetime_ns_array(frame["_time"])

    for edge in range(len(speed_edges)):
        speed = float(speed_edges[edge])
        step = float(step_edges[edge])
        dt = float(dt_edges[edge])
        if not (
            reliable[edge]
            and speed >= cfg.hiking_edge_hard_mps
            and step <= cfg.hiking_timestamp_jitter_max_step_m
        ):
            continue

        left_speed = speed_edges[edge - 1] if edge > 0 else math.nan
        right_speed = speed_edges[edge + 1] if edge + 1 < len(speed_edges) else math.nan
        adjacent_fast = any(
            math.isfinite(float(value)) and float(value) >= cfg.hiking_speed_flag_mps
            for value in (left_speed, right_speed)
        )
        if adjacent_fast:
            continue

        span_start_point = max(0, edge - 1)
        span_end_point = min(len(frame) - 1, edge + 2)
        total_dt = seconds_between(time_ns, span_start_point, span_end_point)
        if not math.isfinite(total_dt) or total_dt <= 0:
            continue
        total_distance = float(
            np.nansum(metrics["step"][span_start_point + 1 : span_end_point + 1])
        )
        combined_speed = total_distance / total_dt

        local_dts = [
            float(dt_edges[j])
            for j in range(max(0, edge - 3), min(len(dt_edges), edge + 4))
            if j != edge and math.isfinite(float(dt_edges[j])) and float(dt_edges[j]) > 0
        ]
        local_dt_median = float(np.median(local_dts)) if len(local_dts) >= 2 else math.nan
        compressed_dt = (
            math.isfinite(local_dt_median)
            and dt <= cfg.hiking_timestamp_jitter_local_dt_ratio * local_dt_median
        )
        if (
            combined_speed <= cfg.hiking_timestamp_jitter_combined_speed_max_mps
            and (compressed_dt or speed >= cfg.very_high_speed_flag_mps)
        ):
            jitter[edge] = True
            reliable[edge] = False
            point_index = edge + 1
            frame.at[point_index, "_flags"] = append_flag(
                frame.at[point_index, "_flags"],
                "isolated_timestamp_jitter_speed_ignored",
            )

    filtered_speed = speed_edges.copy()
    filtered_speed[~reliable] = np.nan
    valid = filtered_speed[np.isfinite(filtered_speed) & (filtered_speed >= 0)]
    valid_fraction = float(reliable.mean()) if len(reliable) else 0.0
    reliable_distance = float(np.nansum(step_edges[reliable])) if len(step_edges) else 0.0
    reliable_duration = float(np.nansum(dt_edges[reliable])) if len(dt_edges) else 0.0
    effective_speed = (
        reliable_distance / reliable_duration if reliable_duration > 0 else math.nan
    )
    return {
        "metrics": metrics,
        "speed_edges": filtered_speed,
        "raw_speed_edges": speed_edges,
        "step_edges": step_edges,
        "dt_edges": dt_edges,
        "reliable_mask": reliable,
        "jitter_mask": jitter,
        "valid_speeds": valid,
        "valid_fraction": valid_fraction,
        "effective_speed": effective_speed,
    }


def hiking_segment_statistics(frame: pd.DataFrame, cfg: CleaningConfig) -> Dict[str, object]:
    evidence = hiking_speed_evidence(frame, cfg)
    metrics = evidence["metrics"]
    valid = evidence["valid_speeds"]
    median_speed = float(np.median(valid)) if valid.size else math.nan
    p90_speed = float(np.quantile(valid, 0.90)) if valid.size else math.nan
    high_fraction = (
        float(np.mean(valid > cfg.hiking_speed_flag_mps)) if valid.size else math.nan
    )
    length = float(metrics["cumulative"][-1]) if len(frame) else 0.0
    valid_times = frame["_time"].dropna()
    duration = math.nan
    if len(valid_times) >= 2:
        duration = (valid_times.iloc[-1] - valid_times.iloc[0]).total_seconds()
    return {
        "median_speed": median_speed,
        "p90_speed": p90_speed,
        "high_fraction": high_fraction,
        "valid_speed_edges": int(valid.size),
        "high_speed_edges": int(np.sum(valid > cfg.hiking_speed_flag_mps)),
        "hard_speed_edges": int(np.sum(valid >= cfg.hiking_edge_hard_mps)),
        "valid_speed_fraction": float(evidence["valid_fraction"]),
        "effective_speed": safe_float(evidence["effective_speed"]),
        "timestamp_jitter_edges_ignored": int(np.sum(evidence["jitter_mask"])),
        "length": length,
        "duration": duration,
    }


def _bridge_nonhiking_gaps(
    mask: np.ndarray,
    step_edges: np.ndarray,
    dt_edges: np.ndarray,
    cfg: CleaningConfig,
) -> np.ndarray:
    output = mask.copy()
    for start, end in boolean_runs(output, value=False):
        if start == 0 or end == len(output) - 1:
            continue
        if not (output[start - 1] and output[end + 1]):
            continue
        edge_count = end - start + 1
        distance = float(np.nansum(step_edges[start : end + 1]))
        dts = dt_edges[start : end + 1]
        duration = float(np.nansum(dts[np.isfinite(dts) & (dts > 0)]))
        if (
            edge_count <= cfg.hiking_bridge_max_edges
            and distance <= cfg.hiking_bridge_max_distance_m
            and duration <= cfg.hiking_bridge_max_duration_s
        ):
            output[start : end + 1] = True
    return output


def _retain_sustained_nonhiking_runs(
    mask: np.ndarray,
    speed_edges: np.ndarray,
    step_edges: np.ndarray,
    dt_edges: np.ndarray,
    cfg: CleaningConfig,
) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    for start, end in boolean_runs(mask, value=True):
        speeds = speed_edges[start : end + 1]
        steps = step_edges[start : end + 1]
        valid = speeds[np.isfinite(speeds)]
        median_speed = float(np.median(valid)) if valid.size else math.nan
        distance = float(np.nansum(steps))
        dts = dt_edges[start : end + 1]
        duration = float(np.nansum(dts[np.isfinite(dts) & (dts > 0)]))
        hard_count = int(
            np.sum(
                np.isfinite(speeds)
                & (speeds >= cfg.hiking_edge_hard_mps)
                & (steps >= cfg.hiking_edge_hard_min_step_m)
            )
        )
        contains_extreme_edge = bool(
            np.any(
                np.isfinite(speeds)
                & (speeds >= cfg.hiking_extreme_edge_mps)
                & (steps >= cfg.hiking_extreme_edge_min_step_m)
            )
        )
        sustained = (
            duration >= cfg.hiking_nonconforming_min_duration_s
            and distance >= cfg.hiking_nonconforming_min_distance_m
        )
        fast_enough = (
            math.isfinite(median_speed)
            and median_speed >= cfg.hiking_nonconforming_min_median_mps
        )
        repeated_hard = hard_count >= cfg.hiking_hard_run_min_edges
        if contains_extreme_edge or repeated_hard or (sustained and fast_enough):
            output[start : end + 1] = True
    return output


def _expand_nonhiking_runs(
    mask: np.ndarray,
    speed_edges: np.ndarray,
    cfg: CleaningConfig,
) -> np.ndarray:
    output = mask.copy()
    for start, end in boolean_runs(mask, value=True):
        left = start - 1
        count = 0
        while left >= 0 and count < cfg.hiking_expand_max_edges:
            speed = speed_edges[left]
            if not (math.isfinite(speed) and speed >= cfg.hiking_expand_speed_mps):
                break
            output[left] = True
            left -= 1
            count += 1
        right = end + 1
        count = 0
        while right < len(output) and count < cfg.hiking_expand_max_edges:
            speed = speed_edges[right]
            if not (math.isfinite(speed) and speed >= cfg.hiking_expand_speed_mps):
                break
            output[right] = True
            right += 1
            count += 1
    return output


def _hiking_quality(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Tuple[bool, str, float, Dict[str, object]]:
    stats = hiking_segment_statistics(frame, cfg)
    valid_edges = int(stats["valid_speed_edges"])
    median_speed = safe_float(stats["median_speed"])
    p90_speed = safe_float(stats["p90_speed"])
    high_fraction = safe_float(stats["high_fraction"])

    valid_fraction = safe_float(stats.get("valid_speed_fraction"))
    if valid_edges < cfg.hiking_min_valid_speed_edges:
        return True, "time_unverified", 0.50, stats
    if (
        not math.isfinite(valid_fraction)
        or valid_fraction < cfg.hiking_min_valid_speed_fraction
    ):
        return True, "time_partially_verified", 0.55, stats

    if valid_edges < cfg.hiking_segment_robust_min_edges:
        high_edges = int(stats.get("high_speed_edges", 0))
        sparse_accepted = (
            math.isfinite(median_speed)
            and math.isfinite(p90_speed)
            and median_speed <= cfg.hiking_segment_median_max_mps
            and p90_speed <= cfg.hiking_segment_sparse_p90_max_mps
            and high_edges <= cfg.hiking_segment_sparse_high_edges_max
        )
        if sparse_accepted:
            return True, "accepted_sparse_speed_evidence", 0.62, stats
        return False, "rejected_sparse_speed_inconsistent", 0.0, stats

    accepted = (
        math.isfinite(median_speed)
        and math.isfinite(p90_speed)
        and math.isfinite(high_fraction)
        and median_speed <= cfg.hiking_segment_median_max_mps
        and p90_speed <= cfg.hiking_segment_p90_max_mps
        and high_fraction <= cfg.hiking_segment_high_speed_fraction_max
    )
    if not accepted:
        return False, "rejected_speed_inconsistent", 0.0, stats

    median_margin = max(
        0.0,
        1.0 - median_speed / max(cfg.hiking_segment_median_max_mps, 1e-6),
    )
    p90_margin = max(
        0.0,
        1.0 - p90_speed / max(cfg.hiking_segment_p90_max_mps, 1e-6),
    )
    confidence = min(
        0.99,
        0.65 + 0.18 * median_margin + 0.10 * p90_margin
        + 0.06 * max(0.0, 1.0 - high_fraction),
    )
    return True, "accepted", confidence, stats


def split_and_filter_hiking(
    frame: pd.DataFrame,
    cfg: CleaningConfig,
) -> Tuple[
    List[Tuple[pd.DataFrame, str, float, Dict[str, object]]],
    List[pd.DataFrame],
]:
    """Split and exclude persistently fast subsegments using hiking-speed characteristics.

    Return accepted and rejected. Rejected does not denote a vehicle class; it only means the movement statistics
    are inconsistent with candidate hiking trajectories and unsuitable for hiking-network extraction.
    """
    if len(frame) < 2:
        accepted, quality, confidence, stats = _hiking_quality(frame, cfg)
        if accepted:
            return [(frame.copy().reset_index(drop=True), quality, confidence, stats)], []
        return [], [rejected_rows(frame, "hiking_filter", "speed_inconsistent_with_hiking")]

    evidence = hiking_speed_evidence(frame, cfg)
    metrics = evidence["metrics"]
    speed_edges = np.asarray(evidence["speed_edges"], dtype=float)
    step_edges = np.asarray(evidence["step_edges"], dtype=float)
    dt_edges = np.asarray(evidence["dt_edges"], dtype=float)
    edge_count = len(speed_edges)

    speed_series = pd.Series(speed_edges, dtype=float)
    min_periods = max(3, cfg.hiking_window_edges // 3)
    rolling_median = speed_series.rolling(
        cfg.hiking_window_edges, center=True, min_periods=min_periods
    ).median().to_numpy(dtype=float)
    high_series = pd.Series(
        np.where(
            np.isfinite(speed_edges),
            speed_edges > cfg.hiking_speed_flag_mps,
            np.nan,
        ),
        dtype=float,
    )
    rolling_high_fraction = high_series.rolling(
        cfg.hiking_window_edges, center=True, min_periods=min_periods
    ).mean().to_numpy(dtype=float)

    reliable = np.isfinite(speed_edges) & np.isfinite(dt_edges) & (dt_edges > 0)
    reliable_step = np.where(reliable, step_edges, 0.0)
    reliable_dt = np.where(reliable, dt_edges, 0.0)
    rolling_distance = pd.Series(reliable_step, dtype=float).rolling(
        cfg.hiking_window_edges, center=True, min_periods=min_periods
    ).sum().to_numpy(dtype=float)
    rolling_duration = pd.Series(reliable_dt, dtype=float).rolling(
        cfg.hiking_window_edges, center=True, min_periods=min_periods
    ).sum().to_numpy(dtype=float)
    rolling_effective = np.divide(
        rolling_distance,
        rolling_duration,
        out=np.full(edge_count, np.nan, dtype=float),
        where=rolling_duration > 0,
    )
    window_supported = (
        (rolling_distance >= cfg.hiking_window_min_distance_m)
        | (rolling_duration >= cfg.hiking_window_min_duration_s)
    )

    hard_edge = (
        reliable
        & (speed_edges >= cfg.hiking_edge_hard_mps)
        & (step_edges >= cfg.hiking_edge_hard_min_step_m)
    )
    sustained_core = (
        window_supported
        & np.isfinite(rolling_median)
        & np.isfinite(rolling_high_fraction)
        & np.isfinite(rolling_effective)
        & (rolling_high_fraction >= cfg.hiking_core_high_fraction)
        & (
            (rolling_median >= cfg.hiking_core_median_max_mps)
            | (rolling_effective >= cfg.hiking_core_effective_max_mps)
        )
    )
    nonhiking_mask = hard_edge | sustained_core
    nonhiking_mask = _bridge_nonhiking_gaps(
        nonhiking_mask, step_edges, dt_edges, cfg
    )
    nonhiking_mask = _retain_sustained_nonhiking_runs(
        nonhiking_mask, speed_edges, step_edges, dt_edges, cfg
    )
    nonhiking_mask = _expand_nonhiking_runs(nonhiking_mask, speed_edges, cfg)

    if not nonhiking_mask.any():
        accepted, quality, confidence, stats = _hiking_quality(frame, cfg)
        if accepted:
            return [(frame.copy().reset_index(drop=True), quality, confidence, stats)], []
        return [], [
            rejected_rows(
                frame,
                "hiking_filter",
                "segment_speed_statistics_inconsistent_with_hiking",
            )
        ]

    accepted_pieces: List[Tuple[pd.DataFrame, str, float, Dict[str, object]]] = []
    rejected_pieces: List[pd.DataFrame] = []
    edge_labels = np.where(nonhiking_mask, "reject", "candidate")
    start = 0
    for i in range(1, edge_count + 1):
        at_end = i == edge_count
        changed = not at_end and edge_labels[i] != edge_labels[start]
        if not (at_end or changed):
            continue
        end = i - 1 if changed else edge_count - 1
        piece = frame.iloc[start : end + 2].copy().reset_index(drop=True)
        if edge_labels[start] == "reject":
            rejected_pieces.append(
                rejected_rows(
                    piece,
                    "hiking_filter",
                    "sustained_speed_inconsistent_with_hiking",
                )
            )
        else:
            accepted, quality, confidence, stats = _hiking_quality(piece, cfg)
            if accepted:
                accepted_pieces.append((piece, quality, confidence, stats))
            else:
                rejected_pieces.append(
                    rejected_rows(
                        piece,
                        "hiking_filter",
                        "segment_speed_statistics_inconsistent_with_hiking",
                    )
                )
        start = i

    return accepted_pieces, rejected_pieces


# =============================================================================
# Final fields
# =============================================================================
def finalize_segment(
    frame: pd.DataFrame,
    clean_track_id: str,
    clip_part_id: int,
    temporal_part_id: int,
    spatial_part_id: int,
    hiking_part_id: int,
    hiking_quality: str,
    hiking_confidence: float,
    hiking_stats: Dict[str, object],
    metric_crs: CRS,
    cfg: CleaningConfig,
) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    metrics = compute_metrics(output)

    output["clean_track_id"] = clean_track_id
    output["clip_part_id"] = clip_part_id
    output["temporal_part_id"] = temporal_part_id
    output["spatial_part_id"] = spatial_part_id
    output["hiking_part_id"] = hiking_part_id
    output["clean_point_index"] = np.arange(1, len(output) + 1, dtype=int)
    output["hiking_quality"] = hiking_quality
    output["hiking_confidence"] = round(float(hiking_confidence), 4)
    output["segment_median_speed_mps"] = round(safe_float(hiking_stats.get("median_speed")), 4)
    output["segment_p90_speed_mps"] = round(safe_float(hiking_stats.get("p90_speed")), 4)
    output["segment_high_speed_fraction"] = round(safe_float(hiking_stats.get("high_fraction")), 4)
    output["segment_valid_speed_fraction"] = round(
        safe_float(hiking_stats.get("valid_speed_fraction")), 4
    )
    output["segment_effective_speed_mps"] = round(
        safe_float(hiking_stats.get("effective_speed")), 4
    )
    output["segment_duration_s"] = round(safe_float(hiking_stats.get("duration")), 3)
    output["segment_length_m"] = round(safe_float(hiking_stats.get("length")), 3)

    output["x_m"] = np.round(output["_x"].to_numpy(dtype=float), 3)
    output["y_m"] = np.round(output["_y"].to_numpy(dtype=float), 3)
    output["metric_crs"] = metric_crs.to_string()
    output["elevation_clean_m"] = np.round(output["_z_clean"].to_numpy(dtype=float), 3)
    output["dem_elevation_m"] = np.round(output["_dem_z"].to_numpy(dtype=float), 3)
    output["elevation_dem_residual_m"] = np.round(
        output["_z_clean"].to_numpy(dtype=float) - output["_dem_z"].to_numpy(dtype=float), 3
    )
    aligned_dem = output.get("_dem_z_aligned", output["_dem_z"]).to_numpy(dtype=float)
    output["elevation_dem_centered_residual_m"] = np.round(
        output["_z_clean"].to_numpy(dtype=float) - aligned_dem, 3
    )
    output["dem_vertical_bias_m"] = np.round(
        output.get("_dem_bias", pd.Series(np.nan, index=output.index)).to_numpy(dtype=float), 3
    )
    output["elevation_local_residual_m"] = np.round(
        output.get("_elev_residual", pd.Series(np.nan, index=output.index)).to_numpy(dtype=float), 3
    )
    output["elevation_quality"] = output.get(
        "_elev_quality", pd.Series("normal", index=output.index)
    ).astype(str)
    output["elevation_repair_method"] = output.get(
        "_elev_method", pd.Series("", index=output.index)
    ).astype(str)
    output["elevation_anomaly_confidence"] = output.get(
        "_elev_confidence", pd.Series("", index=output.index)
    ).astype(str)
    output["elevation_repaired"] = output.get(
        "_elev_repaired", pd.Series(False, index=output.index)
    ).astype(bool)
    output["dt_s"] = np.round(metrics["dt"], 3)
    output["step_m"] = np.round(metrics["step"], 3)
    output["speed_mps"] = np.round(metrics["speed"], 4)
    clean_vertical_speed = metrics["vertical_speed"]
    original_z = output.get(
        "_z_original", pd.to_numeric(output["elevation_m"], errors="coerce")
    )
    original_vertical_speed = _vertical_speed_from_elevation(
        np.asarray(original_z, dtype=float), metrics["dt"]
    )
    output["vertical_speed_mps"] = np.round(clean_vertical_speed, 4)
    output["vertical_speed_clean_mps"] = np.round(clean_vertical_speed, 4)
    output["vertical_speed_original_mps"] = np.round(original_vertical_speed, 4)
    output["turn_angle_deg"] = np.round(metrics["turn"], 3)
    output["cumulative_distance_m"] = np.round(metrics["cumulative"], 3)

    flags_output: List[str] = []
    internal_flags = output.get("_flags", pd.Series("", index=output.index)).astype(str).tolist()
    time_missing = output["_time"].isna().to_numpy(dtype=bool)
    original_elevation = pd.to_numeric(output["elevation_m"], errors="coerce").to_numpy(dtype=float)
    boundary_context = output.get(
        "_boundary_context", pd.Series(False, index=output.index)
    ).to_numpy(dtype=bool)
    speeds = metrics["speed"]
    dem_residuals = output["elevation_dem_centered_residual_m"].to_numpy(dtype=float)

    for i in range(len(output)):
        flags: List[str] = []
        internal = internal_flags[i].strip(" |")
        if internal:
            flags.extend(part.strip() for part in internal.split("|") if part.strip())
        if time_missing[i]:
            flags.append("time_missing_or_invalid")
        if hiking_quality == "time_unverified":
            flags.append("hiking_speed_not_verified")
        elif hiking_quality == "time_partially_verified":
            flags.append("hiking_speed_partially_verified")
        elif hiking_quality == "accepted_sparse_speed_evidence":
            flags.append("hiking_speed_sparse_sample_accepted")
        if boundary_context[i]:
            flags.append("boundary_context_point")
        if not math.isfinite(original_elevation[i]):
            flags.append("original_elevation_missing")
        speed = float(speeds[i])
        if math.isfinite(speed) and speed > cfg.hiking_speed_flag_mps:
            flags.append("high_speed_for_hiking")
        if math.isfinite(speed) and speed > cfg.very_high_speed_flag_mps:
            flags.append("very_high_speed_observation")
        dem_residual = float(dem_residuals[i])
        if math.isfinite(dem_residual) and abs(dem_residual) > cfg.dem_residual_flag_m:
            flags.append("large_gps_dem_elevation_residual")
        flags_output.append(" | ".join(dict.fromkeys(flags)))
    output["quality_flags"] = flags_output
    return output


# =============================================================================
# Input preparation and continuous segments within the boundary
# =============================================================================
def prepare_input_frame(
    csv_path: Path,
    metric_transformer: Transformer,
    dem_sampler: DemSampler,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    missing = [field for field in ("longitude", "latitude") if field not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    for field in ORIGINAL_FIELDS:
        if field not in frame.columns:
            frame[field] = ""

    frame["_source_row"] = np.arange(1, len(frame) + 1, dtype=int)
    lon = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )

    invalid = frame.loc[~valid].copy()
    if not invalid.empty:
        invalid["x_m"] = np.nan
        invalid["y_m"] = np.nan
        invalid["reject_stage"] = "input"
        invalid["reject_reason"] = "invalid_coordinate"

    work = frame.loc[valid].copy().reset_index(drop=True)
    if work.empty:
        return work, invalid

    lon = pd.to_numeric(work["longitude"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(work["latitude"], errors="coerce").to_numpy(dtype=float)
    x, y = metric_transformer.transform(lon, lat)
    work["_x"] = np.asarray(x, dtype=float)
    work["_y"] = np.asarray(y, dtype=float)
    work["_time"] = parse_time_series(work["time_utc"])
    original_z = pd.to_numeric(work["elevation_m"], errors="coerce").to_numpy(dtype=float)
    work["_z_original"] = original_z
    work["_z_clean"] = original_z.copy()
    work["_dem_z"] = dem_sampler.sample(lon, lat)
    work["_flags"] = ""
    work["_boundary_context"] = False
    work["_elev_repaired"] = False
    work["_elev_method"] = ""
    work["_elev_quality"] = "normal"
    work["_elev_confidence"] = ""
    work["_elev_residual"] = np.nan
    work["_dem_bias"] = np.nan
    work["_dem_z_aligned"] = work["_dem_z"].to_numpy(dtype=float)
    work = mark_input_continuity_breaks(work)
    return work, invalid


def mark_input_continuity_breaks(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark original input boundaries that must not be bridged.

    Respect the original KML track_id / segment_id. After invalid coordinates are removed, gaps in original
    row numbers must also trigger splits; otherwise points on both sides could form a straight line across a valley.
    Point-order resets usually indicate a new original segment and must not be bridged either.
    """
    work = frame.copy().reset_index(drop=True)
    n = len(work)
    breaks = np.zeros(n, dtype=bool)
    reasons = [""] * n
    if n == 0:
        work["_input_break_before"] = breaks
        work["_input_break_reason"] = reasons
        return work
    breaks[0] = True
    reasons[0] = "input_start"

    source_rows = pd.to_numeric(work["_source_row"], errors="coerce").to_numpy(dtype=float)
    track_values = work.get("track_id", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().to_numpy()
    segment_values = work.get("segment_id", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().to_numpy()
    point_values = pd.to_numeric(
        work.get("point_index", pd.Series(np.nan, index=work.index)), errors="coerce"
    ).to_numpy(dtype=float)
    raw_point_values = pd.to_numeric(
        work.get("raw_point_index", pd.Series(np.nan, index=work.index)), errors="coerce"
    ).to_numpy(dtype=float)

    for i in range(1, n):
        current_reasons: List[str] = []
        if (
            math.isfinite(source_rows[i - 1])
            and math.isfinite(source_rows[i])
            and int(source_rows[i]) != int(source_rows[i - 1]) + 1
        ):
            current_reasons.append("invalid_or_removed_source_row_gap")
        if track_values[i] != track_values[i - 1] and (track_values[i] or track_values[i - 1]):
            current_reasons.append("track_id_changed")
        if segment_values[i] != segment_values[i - 1] and (segment_values[i] or segment_values[i - 1]):
            current_reasons.append("segment_id_changed")
        if (
            math.isfinite(point_values[i - 1])
            and math.isfinite(point_values[i])
            and point_values[i] <= point_values[i - 1]
        ):
            current_reasons.append("point_index_reset")
        if (
            math.isfinite(raw_point_values[i - 1])
            and math.isfinite(raw_point_values[i])
            and raw_point_values[i] <= raw_point_values[i - 1]
        ):
            current_reasons.append("raw_point_index_reset")
        if current_reasons:
            breaks[i] = True
            reasons[i] = "+".join(dict.fromkeys(current_reasons))

    work["_input_break_before"] = breaks
    work["_input_break_reason"] = reasons
    return work


def split_input_continuity_parts(
    frame: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[Tuple[int, str]]]:
    if frame.empty:
        return [], []
    breaks = frame.get(
        "_input_break_before", pd.Series(False, index=frame.index)
    ).to_numpy(dtype=bool)
    if len(breaks):
        breaks[0] = True
    positions = np.flatnonzero(breaks).tolist()
    starts = sorted(set([0] + positions))
    ends = starts[1:] + [len(frame)]
    parts = [
        frame.iloc[start:end].copy().reset_index(drop=True)
        for start, end in zip(starts, ends)
        if end > start
    ]
    events = [
        (
            pos,
            str(frame.iloc[pos].get("_input_break_reason", "source_continuity_break")),
        )
        for pos in positions
        if pos > 0
    ]
    return parts, events


def contiguous_inside_parts(
    frame: pd.DataFrame,
    inside: np.ndarray,
    context_points: int = 0,
) -> List[pd.DataFrame]:
    """Extract continuous in-boundary segments, optionally retaining a few external context points at their endpoints.

    context_points=0 performs strict point-level clipping. With positive values, additional points are flagged as
    boundary_context_point to preserve topological connections to the external road network at the boundary.
    """
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(inside):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(frame)))

    parts: List[pd.DataFrame] = []
    context_points = max(0, int(context_points))
    for run_index, (run_start, run_end) in enumerate(runs):
        ext_start = max(0, run_start - context_points)
        ext_end = min(len(frame), run_end + context_points)
        # Adjacent extended segments do not share the same external boundary point, avoiding duplicate output.
        if run_index > 0:
            previous_run_end = runs[run_index - 1][1]
            midpoint = (previous_run_end + run_start) // 2
            ext_start = max(ext_start, midpoint)
        if run_index + 1 < len(runs):
            next_run_start = runs[run_index + 1][0]
            midpoint = (run_end + next_run_start + 1) // 2
            ext_end = min(ext_end, midpoint)

        part = frame.iloc[ext_start:ext_end].copy().reset_index(drop=True)
        original_positions = np.arange(ext_start, ext_end)
        context_mask = ~inside[original_positions]
        if "_boundary_context" not in part.columns:
            part["_boundary_context"] = False
        part.loc[context_mask, "_boundary_context"] = True
        parts.append(part)
    return parts


# =============================================================================
# Single-file processing
# =============================================================================
def process_file(
    csv_path: Path,
    output_dir: Path,
    boundary_geometry,
    metric_crs: CRS,
    metric_transformer: Transformer,
    dem_sampler: DemSampler,
    cfg: CleaningConfig,
) -> Tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[pd.DataFrame],
    List[Dict[str, object]],
    List[pd.DataFrame],
]:
    frame, invalid = prepare_input_frame(csv_path, metric_transformer, dem_sampler)
    input_points = len(frame) + len(invalid)
    rejected_frames: List[pd.DataFrame] = []
    split_events: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    line_rows: List[Dict[str, object]] = []
    final_segments: List[pd.DataFrame] = []

    if not invalid.empty:
        rejected_frames.append(invalid)

    summary: Dict[str, object] = {
        "input_file": csv_path.name,
        "input_points": input_points,
        "invalid_coordinate_points": len(invalid),
        "time_valid_points": int(frame["_time"].notna().sum()) if not frame.empty else 0,
        "time_valid_rate": round(float(frame["_time"].notna().mean()), 6) if not frame.empty else 0.0,
        "dem_enabled": bool(dem_sampler.dataset is not None),
        "dem_coverage_points": int(np.isfinite(frame["_dem_z"].to_numpy(dtype=float)).sum()) if not frame.empty else 0,
        "dem_coverage_rate": round(float(np.isfinite(frame["_dem_z"].to_numpy(dtype=float)).mean()), 6) if not frame.empty else 0.0,
        "source_continuity_parts": 0,
        "source_continuity_splits": 0,
        "outside_boundary_points": 0,
        "boundary_context_points": 0,
        "inside_boundary_points": 0,
        "clip_parts": 0,
        "duplicate_removed": 0,
        "duplicate_time_kept": 0,
        "duplicate_time_splits": 0,
        "near_duplicate_thinned": 0,
        "reverse_time_splits": 0,
        "time_gap_splits": 0,
        "isolated_spike_removed": 0,
        "excursion_cluster_removed": 0,
        "hard_distance_splits": 0,
        "hard_speed_splits": 0,
        "adaptive_distance_splits": 0,
        "stationary_thinned": 0,
        "elevation_repaired": 0,
        "elevation_flagged": 0,
        "elevation_filled_from_dem": 0,
        "dem_fill_skipped_unreliable": 0,
        "nonhiking_rejected_points": 0,
        "nonhiking_rejected_pieces": 0,
        "short_segment_removed": 0,
        "output_segments": 0,
        "output_points": 0,
        "output_length_m": 0.0,
        "unique_rejected_points": 0,
        "accounted_points": 0,
        "accounting_difference": 0,
        "status": "OK",
        "warnings": "",
    }

    if frame.empty:
        summary["status"] = "EMPTY_AFTER_COORDINATE_CHECK"
        return summary, manifest_rows, line_rows, rejected_frames, split_events, final_segments

    warning_messages: List[str] = []
    if int(summary["time_valid_points"]) == 0:
        warning_messages.append("All time_utc values are invalid; speed consistency cannot be verified; applying geometry cleaning only")
    elif float(summary["time_valid_rate"]) < 0.80:
        warning_messages.append(f"Low valid timestamp rate: {float(summary['time_valid_rate']):.1%}")
    if bool(summary["dem_enabled"]):
        if int(summary["dem_coverage_points"]) == 0:
            warning_messages.append("DEM is enabled, but this track has no valid DEM coverage")
        elif float(summary["dem_coverage_rate"]) < 0.80:
            warning_messages.append(f"Low DEM coverage: {float(summary['dem_coverage_rate']):.1%}")

    source_parts, source_breaks = split_input_continuity_parts(frame)
    summary["source_continuity_parts"] = len(source_parts)
    summary["source_continuity_splits"] = len(source_breaks)
    for break_pos, reason in source_breaks:
        split_events.append(
            split_event(frame, break_pos, "source", reason, csv_path.name)
        )

    if boundary_geometry is None:
        inside = np.ones(len(frame), dtype=bool)
        summary["inside_boundary_points"] = len(frame)
        clip_parts = [part.copy().reset_index(drop=True) for part in source_parts]
    else:
        inside = points_covered_by_boundary(
            frame["_x"].to_numpy(dtype=float),
            frame["_y"].to_numpy(dtype=float),
            boundary_geometry,
        )
        summary["outside_boundary_points"] = int((~inside).sum())
        summary["inside_boundary_points"] = int(inside.sum())
        clip_parts: List[pd.DataFrame] = []
        for source_part in source_parts:
            part_inside = points_covered_by_boundary(
                source_part["_x"].to_numpy(dtype=float),
                source_part["_y"].to_numpy(dtype=float),
                boundary_geometry,
            )
            clip_parts.extend(
                contiguous_inside_parts(
                    source_part,
                    part_inside,
                    context_points=cfg.boundary_context_points,
                )
            )
        context_source_rows: set[int] = set()
        for part in clip_parts:
            context_mask = part.get(
                "_boundary_context", pd.Series(False, index=part.index)
            ).astype(bool)
            if context_mask.any():
                context_source_rows.update(
                    int(value)
                    for value in part.loc[context_mask, "_source_row"].tolist()
                )
        summary["boundary_context_points"] = len(context_source_rows)
        outside = frame.loc[~inside].copy()
        if context_source_rows:
            outside = outside.loc[
                ~outside["_source_row"].astype(int).isin(context_source_rows)
            ]
        if not outside.empty:
            rejected_frames.append(rejected_rows(outside, "clip", "outside_boundary"))
    summary["clip_parts"] = len(clip_parts)
    summary["warnings"] = " | ".join(warning_messages)
    global_hiking_counter = 0

    for clip_part_id, clipped in enumerate(clip_parts, start=1):
        temporal_parts, rejected, events, time_stats = split_by_time(clipped, cfg, csv_path.name)
        rejected_frames.extend(rejected)
        split_events.extend(events)
        for key, value in time_stats.items():
            summary[key] = int(summary[key]) + int(value)

        for temporal_part_id, temporal in enumerate(temporal_parts, start=1):
            temporal, rejected, near_duplicate_removed = thin_consecutive_near_duplicate_runs(
                temporal, cfg
            )
            rejected_frames.extend(rejected)
            summary["near_duplicate_thinned"] = (
                int(summary["near_duplicate_thinned"]) + near_duplicate_removed
            )

            temporal, rejected, removed = remove_isolated_spikes(temporal, cfg)
            rejected_frames.extend(rejected)
            summary["isolated_spike_removed"] = int(summary["isolated_spike_removed"]) + removed

            temporal, rejected, cluster_removed = remove_short_excursion_clusters(temporal, cfg)
            rejected_frames.extend(rejected)
            summary["excursion_cluster_removed"] = int(summary["excursion_cluster_removed"]) + cluster_removed

            spatial_parts, events, spatial_stats = split_by_hard_discontinuities(
                temporal, cfg, csv_path.name
            )
            split_events.extend(events)
            for key, value in spatial_stats.items():
                summary[key] = int(summary[key]) + int(value)

            for spatial_part_id, spatial in enumerate(spatial_parts, start=1):
                stationary_windows = detect_stationary_windows(spatial, cfg)
                spatial, repaired, dem_filled, elev_flagged, dem_skipped = repair_elevation(
                    spatial, cfg, stationary_windows=stationary_windows
                )
                summary["elevation_repaired"] = int(summary["elevation_repaired"]) + repaired
                summary["elevation_flagged"] = int(summary["elevation_flagged"]) + elev_flagged
                summary["elevation_filled_from_dem"] = int(summary["elevation_filled_from_dem"]) + dem_filled
                summary["dem_fill_skipped_unreliable"] = int(summary["dem_fill_skipped_unreliable"]) + dem_skipped

                elevation_audit = build_elevation_audit_rows(
                    spatial,
                    input_file=csv_path.name,
                    clip_part_id=clip_part_id,
                    temporal_part_id=temporal_part_id,
                    spatial_part_id=spatial_part_id,
                )
                append_csv(elevation_audit, output_dir / "elevation_audit.csv", ELEVATION_AUDIT_FIELDS)

                spatial, rejected, thinned = thin_stationary_drift(
                    spatial, cfg, stationary_windows=stationary_windows
                )
                rejected_frames.extend(rejected)
                summary["stationary_thinned"] = int(summary["stationary_thinned"]) + thinned

                hiking_pieces, rejected_nonhiking = split_and_filter_hiking(spatial, cfg)
                rejected_frames.extend(rejected_nonhiking)
                summary["nonhiking_rejected_pieces"] = int(summary["nonhiking_rejected_pieces"]) + len(rejected_nonhiking)
                summary["nonhiking_rejected_points"] = int(summary["nonhiking_rejected_points"]) + sum(len(item) for item in rejected_nonhiking)

                accepted_id = 0
                for piece, quality, confidence, hiking_stats in hiking_pieces:
                    if len(piece) < cfg.min_points_per_segment:
                        if not piece.empty:
                            rejected_frames.append(rejected_rows(piece, "final", "short_segment_after_cleaning"))
                            summary["short_segment_removed"] = int(summary["short_segment_removed"]) + len(piece)
                        continue

                    length_m = float(compute_metrics(piece)["cumulative"][-1])
                    if length_m < cfg.min_segment_length_m:
                        rejected_frames.append(rejected_rows(piece, "final", "short_length_after_cleaning"))
                        summary["short_segment_removed"] = int(summary["short_segment_removed"]) + len(piece)
                        continue

                    accepted_id += 1
                    global_hiking_counter += 1
                    base_track_id = sanitize_id(
                        piece["track_id"].iloc[0] if "track_id" in piece.columns else csv_path.stem
                    )
                    clean_track_id = (
                        f"{base_track_id}__clip{clip_part_id:03d}"
                        f"__time{temporal_part_id:03d}"
                        f"__space{spatial_part_id:03d}"
                        f"__hike{global_hiking_counter:03d}"
                    )
                    final = finalize_segment(
                        piece,
                        clean_track_id=clean_track_id,
                        clip_part_id=clip_part_id,
                        temporal_part_id=temporal_part_id,
                        spatial_part_id=spatial_part_id,
                        hiking_part_id=accepted_id,
                        hiking_quality=quality,
                        hiking_confidence=confidence,
                        hiking_stats=hiking_stats,
                        metric_crs=metric_crs,
                        cfg=cfg,
                    )
                    final_segments.append(final)
                    summary["output_segments"] = int(summary["output_segments"]) + 1
                    summary["output_points"] = int(summary["output_points"]) + len(final)
                    summary["output_length_m"] = float(summary["output_length_m"]) + length_m

                    manifest_rows.append({
                        "input_file": csv_path.name,
                        "source_track_id": final["track_id"].iloc[0],
                        "clean_track_id": clean_track_id,
                        "hiking_quality": quality,
                        "hiking_confidence": round(confidence, 4),
                        "clip_part_id": clip_part_id,
                        "temporal_part_id": temporal_part_id,
                        "spatial_part_id": spatial_part_id,
                        "hiking_part_id": accepted_id,
                        "point_count": len(final),
                        "length_m": round(length_m, 3),
                        "duration_s": final["segment_duration_s"].iloc[0],
                        "median_speed_mps": final["segment_median_speed_mps"].iloc[0],
                        "p90_speed_mps": final["segment_p90_speed_mps"].iloc[0],
                        "high_speed_fraction": final["segment_high_speed_fraction"].iloc[0],
                        "valid_speed_fraction": final["segment_valid_speed_fraction"].iloc[0],
                        "effective_speed_mps": final["segment_effective_speed_mps"].iloc[0],
                        "dem_vertical_bias_m": final["dem_vertical_bias_m"].iloc[0],
                        "start_time_utc": final["time_utc"].iloc[0],
                        "end_time_utc": final["time_utc"].iloc[-1],
                        "start_longitude": final["longitude"].iloc[0],
                        "start_latitude": final["latitude"].iloc[0],
                        "end_longitude": final["longitude"].iloc[-1],
                        "end_latitude": final["latitude"].iloc[-1],
                        "elevation_repaired_count": int(final["elevation_repaired"].sum()),
                        "elevation_flagged_count": int((final["elevation_quality"].astype(str) == "flagged").sum()),
                    })

                    coordinates = list(zip(final["x_m"].to_numpy(dtype=float), final["y_m"].to_numpy(dtype=float)))
                    if len(coordinates) >= 2:
                        line_rows.append({
                            "input_file": csv_path.name,
                            "source_track_id": final["track_id"].iloc[0],
                            "clean_track_id": clean_track_id,
                            "hiking_quality": quality,
                            "hiking_confidence": round(confidence, 4),
                            "point_count": len(final),
                            "length_m": round(length_m, 3),
                            "median_speed_mps": final["segment_median_speed_mps"].iloc[0],
                            "p90_speed_mps": final["segment_p90_speed_mps"].iloc[0],
                            "geometry": LineString(coordinates),
                        })

    rejected_frames = finalize_rejected_frames(
        rejected_frames, final_segments, csv_path.name
    )
    unique_rejected = sum(len(item) for item in rejected_frames)
    summary["unique_rejected_points"] = unique_rejected
    summary["accounted_points"] = int(summary["output_points"]) + unique_rejected
    summary["accounting_difference"] = input_points - int(summary["accounted_points"])

    if rejected_frames:
        audit = rejected_frames[0]
        reasons = audit.get("reject_reason", pd.Series("", index=audit.index)).astype(str)
        summary["nonhiking_rejected_points"] = int(
            reasons.str.contains("hiking", regex=False, na=False).sum()
        )
        summary["short_segment_removed"] = int(
            reasons.str.contains("short_", regex=False, na=False).sum()
        )
    else:
        summary["nonhiking_rejected_points"] = 0
        summary["short_segment_removed"] = 0

    if int(summary["accounting_difference"]) != 0:
        warning_messages.append(
            "Point-count audit imbalance: "
            f"input={input_points}, output={summary['output_points']}, "
            f"rejected={unique_rejected}"
        )
    summary["warnings"] = " | ".join(dict.fromkeys(warning_messages))

    if int(summary["inside_boundary_points"]) == 0:
        summary["status"] = "NO_POINT_INSIDE_BOUNDARY"
    elif int(summary["output_segments"]) == 0:
        summary["status"] = "NO_VALID_HIKING_SEGMENT_AFTER_CLEANING"
    summary["output_length_m"] = round(float(summary["output_length_m"]), 3)
    return summary, manifest_rows, line_rows, rejected_frames, split_events, final_segments


# =============================================================================
# Outputs
# =============================================================================
def write_segment_outputs(
    segments: List[pd.DataFrame],
    output_dir: Path,
) -> None:
    fields = ORIGINAL_FIELDS + CLEAN_FIELDS
    all_path = output_dir / "all_cleaned_points.csv"
    folder = output_dir / "cleaned_hiking_tracks"
    for segment in segments:
        clean_id = safe_output_stem(segment["clean_track_id"].iloc[0])
        file_path = folder / f"{clean_id}.csv"
        # MAX_PATH applies to the full Windows path, not just its basename.
        # Keep the scientific track ID in the CSV; compact only the disk name.
        if os.name == "nt" and len(str(file_path.resolve()).encode("utf-16-le")) // 2 >= 260:
            track_id = str(segment["clean_track_id"].iloc[0])
            digest = hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:24]
            file_path = folder / f"track_{digest}.csv"
        segment.reindex(columns=fields).to_csv(
            file_path, index=False, encoding="utf-8-sig", lineterminator="\n"
        )
        append_csv(segment, all_path, fields)


def write_gpkg_layers(
    line_rows: List[Dict[str, object]],
    output_path: Path,
    metric_crs: CRS,
) -> None:
    if not line_rows:
        return
    lines = gpd.GeoDataFrame(line_rows, geometry="geometry", crs=metric_crs)
    if output_path.exists():
        output_path.unlink()
    pyogrio.write_dataframe(
        lines,
        output_path,
        layer="hiking_tracks",
        driver="GPKG",
        append=False,
    )
    if not output_path.exists():
        raise RuntimeError("GPKG file does not exist after writing")
    actual_layers = {str(row[0]) for row in pyogrio.list_layers(output_path)}
    if "hiking_tracks" not in actual_layers:
        raise RuntimeError("GPKG layer validation failed: hiking_tracks is missing")


# =============================================================================
# Parameters and main program
# =============================================================================
def validate_config(cfg: CleaningConfig) -> None:
    """Check relationships between key parameters to avoid silently producing meaningless results."""
    positive = {
        "min_points_per_segment": cfg.min_points_per_segment,
        "min_segment_length_m": cfg.min_segment_length_m,
        "duplicate_position_m": cfg.duplicate_position_m,
        "duplicate_time_keep_max_m": cfg.duplicate_time_keep_max_m,
        "near_duplicate_run_radius_m": cfg.near_duplicate_run_radius_m,
        "near_duplicate_run_min_points": cfg.near_duplicate_run_min_points,
        "near_duplicate_run_median_step_max_m": cfg.near_duplicate_run_median_step_max_m,
        "max_time_gap_s": cfg.max_time_gap_s,
        "hard_step_break_m": cfg.hard_step_break_m,
        "adaptive_step_floor_m": cfg.adaptive_step_floor_m,
        "hiking_speed_flag_mps": cfg.hiking_speed_flag_mps,
        "hiking_edge_hard_mps": cfg.hiking_edge_hard_mps,
        "elevation_spike_min_m": cfg.elevation_spike_min_m,
        "elevation_residual_min_samples": cfg.elevation_residual_min_samples,
        "elevation_short_gap_max_points": cfg.elevation_short_gap_max_points,
        "elevation_short_gap_max_distance_m": cfg.elevation_short_gap_max_distance_m,
        "dem_bias_min_samples": cfg.dem_bias_min_samples,
        "adaptive_step_ratio": cfg.adaptive_step_ratio,
        "adaptive_step_strong_ratio": cfg.adaptive_step_strong_ratio,
        "hiking_window_edges": cfg.hiking_window_edges,
        "hiking_hard_run_min_edges": cfg.hiking_hard_run_min_edges,
        "hiking_segment_robust_min_edges": cfg.hiking_segment_robust_min_edges,
        "hiking_segment_sparse_p90_max_mps": cfg.hiking_segment_sparse_p90_max_mps,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"The following parameters must be greater than 0: {', '.join(invalid)}")
    if cfg.min_points_per_segment < 2:
        raise ValueError("min_points_per_segment must be at least 2")
    if cfg.duplicate_position_m > cfg.duplicate_time_keep_max_m:
        raise ValueError("duplicate_position_m must not exceed duplicate_time_keep_max_m")
    if cfg.adaptive_step_floor_m >= cfg.hard_step_break_m:
        raise ValueError("adaptive_step_floor_m must be less than hard_step_break_m")
    for name, value in {
        "near_duplicate_run_path_efficiency_max": cfg.near_duplicate_run_path_efficiency_max,
        "stationary_path_efficiency_max": cfg.stationary_path_efficiency_max,
        "stationary_min_inlier_fraction": cfg.stationary_min_inlier_fraction,
        "hiking_core_high_fraction": cfg.hiking_core_high_fraction,
        "hiking_segment_high_speed_fraction_max": cfg.hiking_segment_high_speed_fraction_max,
        "hiking_min_valid_speed_fraction": cfg.hiking_min_valid_speed_fraction,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if cfg.dem_steep_grade_threshold > cfg.dem_trust_max_grade:
        raise ValueError("dem_steep_grade_threshold must not exceed dem_trust_max_grade")
    if cfg.adaptive_step_ratio > cfg.adaptive_step_strong_ratio:
        raise ValueError("adaptive_step_ratio must not exceed adaptive_step_strong_ratio")
    if cfg.hiking_speed_flag_mps >= cfg.hiking_edge_hard_mps:
        raise ValueError("hiking_speed_flag_mps must be less than hiking_edge_hard_mps")
    if cfg.hiking_edge_hard_mps >= cfg.hiking_extreme_edge_mps:
        raise ValueError("hiking_edge_hard_mps must be less than hiking_extreme_edge_mps")
    for name, value in {
        "spike_flank_angle_max_deg": cfg.spike_flank_angle_max_deg,
        "excursion_flank_angle_max_deg": cfg.excursion_flank_angle_max_deg,
        "adaptive_step_turn_support_deg": cfg.adaptive_step_turn_support_deg,
    }.items():
        if not 0.0 < float(value) < 180.0:
            raise ValueError(f"{name} must be within (0, 180)")


def build_parser() -> argparse.ArgumentParser:
    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(
        description="Clean candidate hiking gx:Track CSVs for the alpine gorges of Milin (v7.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="S1 observed_tracks directory")
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY_PATH, help="Milin study-area polygon boundary")
    parser.add_argument("--no-boundary", action="store_true", help="Skip boundary clipping")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--input-pattern", default="*__gxtrack.csv", help="Input track CSV filename pattern; use *.csv for test files")
    parser.add_argument("--boundary-crs", default=None, help="Explicit boundary CRS when the .prj file is missing")
    parser.add_argument("--metric-crs", default=None, help="Manual metric CRS with --no-boundary, e.g. EPSG:32647")
    parser.add_argument("--boundary-buffer-m", type=float, default=defaults.boundary_buffer_m, help="Boundary buffer distance")
    parser.add_argument("--boundary-context-points", type=int, default=defaults.boundary_context_points, help="Number of outside points retained before and after each continuous in-boundary segment")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH, help="Optional DEM")
    parser.add_argument("--min-points", type=int, default=defaults.min_points_per_segment, help="Minimum point count per final segment")
    parser.add_argument("--min-length-m", type=float, default=defaults.min_segment_length_m, help="Minimum length per final segment")
    parser.add_argument("--max-time-gap-s", type=float, default=defaults.max_time_gap_s, help="Hard time-gap split threshold")
    parser.add_argument("--hard-step-m", type=float, default=defaults.hard_step_break_m, help="Absolute adjacent-point distance split threshold")
    parser.add_argument("--adaptive-step-floor-m", type=float, default=defaults.adaptive_step_floor_m, help="Minimum candidate step length for local adaptive distance breaks")
    parser.add_argument("--hiking-speed-mps", type=float, default=defaults.hiking_speed_flag_mps, help="Threshold for flagging high hiking speeds")
    parser.add_argument("--hiking-hard-speed-mps", type=float, default=defaults.hiking_edge_hard_mps, help="Edge speed threshold for clear inconsistency with hiking")
    parser.add_argument("--hiking-median-max-mps", type=float, default=defaults.hiking_segment_median_max_mps, help="Maximum median speed for final hiking segments")
    parser.add_argument("--hiking-p90-max-mps", type=float, default=defaults.hiking_segment_p90_max_mps, help="Maximum P90 speed for final hiking segments")
    parser.add_argument("--hiking-high-fraction-max", type=float, default=defaults.hiking_segment_high_speed_fraction_max, help="Maximum high-speed edge fraction for final hiking segments")
    parser.add_argument("--hiking-min-valid-fraction", type=float, default=defaults.hiking_min_valid_speed_fraction, help="Minimum fraction of edges with valid timing required for speed-based segment rejection")
    parser.add_argument("--spike-zero-dt-leg-m", type=float, default=defaults.spike_zero_dt_leg_m, help="Minimum candidate edge distance for same-second spikes")
    parser.add_argument("--elevation-spike-min-m", type=float, default=defaults.elevation_spike_min_m, help="Minimum residual for isolated elevation spikes")
    parser.add_argument("--vertical-speed-mps", type=float, default=defaults.vertical_speed_hard_mps, help="Supporting vertical speed threshold for elevation anomalies")
    parser.add_argument("--elevation-grade-hard", type=float, default=defaults.elevation_grade_hard, help="Supporting apparent grade threshold for elevation anomalies")
    parser.add_argument("--near-same-xy-jump-m", type=float, default=defaults.elevation_near_same_xy_jump_m, help="Elevation jump threshold at nearly identical positions")
    parser.add_argument("--elevation-short-gap-points", type=int, default=defaults.elevation_short_gap_max_points, help="Maximum consecutive missing elevation points allowed for along-track interpolation")
    parser.add_argument("--dem-trust-max-grade", type=float, default=defaults.dem_trust_max_grade, help="Maximum local grade for automatic DEM-assisted repair")
    parser.add_argument("--dem-bias-min-samples", type=int, default=defaults.dem_bias_min_samples, help="Minimum samples required to estimate GPS-DEM vertical bias")
    parser.add_argument("--no-dem-fill", action="store_true", help="Skip DEM filling of remaining missing elevations")
    parser.add_argument("--no-gpkg", action="store_true", help="Skip GPKG output")
    return parser


def infer_metric_crs_from_csvs(csv_files: Sequence[Path]) -> CRS:
    """Infer local UTM from a finite sample of valid coordinates without aggregating all trajectories in memory."""
    lon_values: List[float] = []
    lat_values: List[float] = []
    max_samples = 100_000
    for path in csv_files:
        frame = pd.read_csv(
            path,
            usecols=lambda name: name in {"longitude", "latitude"},
            encoding="utf-8-sig",
            low_memory=False,
        )
        lon = pd.to_numeric(frame.get("longitude"), errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(frame.get("latitude"), errors="coerce").to_numpy(dtype=float)
        valid = (
            np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= -180.0)
            & (lon <= 180.0)
            & (lat >= -90.0)
            & (lat <= 90.0)
        )
        lon = lon[valid]
        lat = lat[valid]
        remaining = max_samples - len(lon_values)
        if remaining <= 0:
            break
        if lon.size > remaining:
            positions = np.linspace(0, lon.size - 1, remaining, dtype=int)
            lon = lon[positions]
            lat = lat[positions]
        lon_values.extend(lon.tolist())
        lat_values.extend(lat.tolist())
    if not lon_values:
        raise ValueError("No input tracks contain valid longitude and latitude; cannot select a metric CRS")
    return local_utm_crs(float(np.median(lon_values)), float(np.median(lat_values)))


def failed_summary_row(input_file: str, warning: str) -> Dict[str, object]:
    """Create a failure summary with all fields to avoid missing or inconsistent columns in cleaning_summary.csv."""
    return {
        "input_file": input_file,
        "input_points": 0,
        "invalid_coordinate_points": 0,
        "time_valid_points": 0,
        "time_valid_rate": 0.0,
        "dem_enabled": False,
        "dem_coverage_points": 0,
        "dem_coverage_rate": 0.0,
        "source_continuity_parts": 0,
        "source_continuity_splits": 0,
        "outside_boundary_points": 0,
        "boundary_context_points": 0,
        "inside_boundary_points": 0,
        "clip_parts": 0,
        "duplicate_removed": 0,
        "duplicate_time_kept": 0,
        "duplicate_time_splits": 0,
        "near_duplicate_thinned": 0,
        "reverse_time_splits": 0,
        "time_gap_splits": 0,
        "isolated_spike_removed": 0,
        "excursion_cluster_removed": 0,
        "hard_distance_splits": 0,
        "hard_speed_splits": 0,
        "adaptive_distance_splits": 0,
        "stationary_thinned": 0,
        "elevation_repaired": 0,
        "elevation_flagged": 0,
        "elevation_filled_from_dem": 0,
        "dem_fill_skipped_unreliable": 0,
        "nonhiking_rejected_points": 0,
        "nonhiking_rejected_pieces": 0,
        "short_segment_removed": 0,
        "output_segments": 0,
        "output_points": 0,
        "output_length_m": 0.0,
        "unique_rejected_points": 0,
        "accounted_points": 0,
        "accounting_difference": 0,
        "status": "FAILED",
        "warnings": warning,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input.expanduser().resolve()
    boundary_path = args.boundary.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    dem_path = args.dem.expanduser().resolve() if args.dem is not None else None

    cfg = CleaningConfig(
        boundary_buffer_m=float(args.boundary_buffer_m),
        boundary_context_points=max(0, int(args.boundary_context_points)),
        min_points_per_segment=int(args.min_points),
        min_segment_length_m=float(args.min_length_m),
        max_time_gap_s=float(args.max_time_gap_s),
        hard_step_break_m=float(args.hard_step_m),
        adaptive_step_floor_m=float(args.adaptive_step_floor_m),
        hiking_speed_flag_mps=float(args.hiking_speed_mps),
        hiking_edge_hard_mps=float(args.hiking_hard_speed_mps),
        hiking_segment_median_max_mps=float(args.hiking_median_max_mps),
        hiking_segment_p90_max_mps=float(args.hiking_p90_max_mps),
        hiking_segment_high_speed_fraction_max=float(args.hiking_high_fraction_max),
        hiking_min_valid_speed_fraction=float(args.hiking_min_valid_fraction),
        spike_zero_dt_leg_m=float(args.spike_zero_dt_leg_m),
        elevation_spike_min_m=float(args.elevation_spike_min_m),
        vertical_speed_hard_mps=float(args.vertical_speed_mps),
        elevation_grade_hard=float(args.elevation_grade_hard),
        elevation_near_same_xy_jump_m=float(args.near_same_xy_jump_m),
        elevation_short_gap_max_points=int(args.elevation_short_gap_points),
        dem_trust_max_grade=float(args.dem_trust_max_grade),
        dem_bias_min_samples=int(args.dem_bias_min_samples),
        fill_missing_elevation_from_dem=not bool(args.no_dem_fill),
        write_gpkg=not bool(args.no_gpkg),
    )

    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"[ERROR] Invalid parameters: {exc}", file=sys.stderr)
        return 2

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    csv_files = sorted(
        path for path in input_dir.glob(str(args.input_pattern)) if path.is_file()
    )
    if not csv_files:
        print(
            f"[ERROR] No CSV files in observed_tracks match {args.input_pattern!r}: {input_dir}",
            file=sys.stderr,
        )
        print(
            "        For temporary test files, add: --input-pattern '*.csv'",
            file=sys.stderr,
        )
        return 2

    try:
        if args.no_boundary:
            boundary_geometry = None
            if args.metric_crs:
                metric_crs = CRS.from_user_input(args.metric_crs)
                if not crs_uses_metres(metric_crs):
                    raise ValueError(
                        "--metric-crs must be a projected CRS in metres, e.g. EPSG:32647; "
                        "geographic coordinate systems such as EPSG:4326 are not supported"
                    )
                metric_crs_warning = f"Manually specified metric CRS {metric_crs.to_string()}"
            else:
                metric_crs = infer_metric_crs_from_csvs(csv_files)
                metric_crs_warning = f"Metric CRS inferred automatically from input tracks: {metric_crs.to_string()}"
            boundary_info = {
                "mode": "DISABLED",
                "metric_crs": metric_crs.to_string(),
                "warning": "Boundary clipping skipped; all valid coordinate points are included in cleaning | " + metric_crs_warning,
            }
        else:
            boundary_geometry, metric_crs, boundary_info = load_boundary(
                boundary_path, args.boundary_crs, cfg.boundary_buffer_m
            )
        dem_sampler = DemSampler(dem_path)
    except Exception as exc:
        print(f"[ERROR] Initialization failed: {exc}", file=sys.stderr)
        return 2

    clean_output_directory(output_dir)
    with (output_dir / "cleaning_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "S2_v7.0_milin",
                "config": asdict(cfg),
                "boundary": boundary_info,
                "dem": str(dem_path) if dem_path else None,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    metric_transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)

    print("[BOUNDARY]")
    for key, value in boundary_info.items():
        print(f"  {key}: {value}")
    print(f"[INPUT] {input_dir}")
    print(f"[FILES] {len(csv_files)}")
    print(f"[DEM] {dem_path if dem_path else 'disabled'}")

    summary_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    line_rows: List[Dict[str, object]] = []
    failed = 0

    try:
        for number, csv_path in enumerate(csv_files, start=1):
            try:
                summary, manifests, lines, rejected, events, segments = process_file(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    boundary_geometry=boundary_geometry,
                    metric_crs=metric_crs,
                    metric_transformer=metric_transformer,
                    dem_sampler=dem_sampler,
                    cfg=cfg,
                )
                summary_rows.append(summary)
                manifest_rows.extend(manifests)
                line_rows.extend(lines)
                write_segment_outputs(segments, output_dir)
                if rejected:
                    rejected_file = output_dir / "rejected_points.csv"
                    for rejected_frame in rejected:
                        append_csv(rejected_frame, rejected_file, REJECT_FIELDS)
                if events:
                    append_csv(pd.DataFrame(events), output_dir / "split_events.csv", SPLIT_EVENT_FIELDS)
                print(
                    f"[{number}/{len(csv_files)}] {csv_path.name}: "
                    f"inside={summary['inside_boundary_points']}, "
                    f"hiking_segments={summary['output_segments']}, "
                    f"nonhiking_rejected={summary['nonhiking_rejected_points']}, "
                    f"status={summary['status']}"
                )
            except Exception as exc:
                failed += 1
                print(f"[FAILED] {csv_path.name}: {exc}", file=sys.stderr)
                summary_rows.append(failed_summary_row(csv_path.name, str(exc)))
    finally:
        dem_sampler.close()

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cleaning_summary.csv", index=False,
        encoding="utf-8-sig", lineterminator="\n"
    )
    pd.DataFrame(manifest_rows, columns=MANIFEST_FIELDS).to_csv(
        output_dir / "segment_manifest.csv", index=False,
        encoding="utf-8-sig", lineterminator="\n"
    )

    for path, fields in (
        (output_dir / "rejected_points.csv", REJECT_FIELDS),
        (output_dir / "split_events.csv", SPLIT_EVENT_FIELDS),
        (output_dir / "elevation_audit.csv", ELEVATION_AUDIT_FIELDS),
        (output_dir / "all_cleaned_points.csv", ORIGINAL_FIELDS + CLEAN_FIELDS),
    ):
        if not path.exists():
            pd.DataFrame(columns=fields).to_csv(
                path, index=False, encoding="utf-8-sig", lineterminator="\n"
            )

    if cfg.write_gpkg:
        try:
            write_gpkg_layers(
                line_rows,
                output_dir / "cleaned_hiking_tracks.gpkg",
                metric_crs,
            )
        except Exception as exc:
            print(f"[WARNING] GPKG output failed; CSV output is complete: {exc}", file=sys.stderr)

    successful_rows = [row for row in summary_rows if row.get("status") != "FAILED"]
    total_segments = int(sum(safe_float(row.get("output_segments", 0)) for row in successful_rows))
    rejected_points = int(sum(safe_float(row.get("nonhiking_rejected_points", 0)) for row in successful_rows))

    print("\nProcessing complete")
    print(f"Files succeeded: {len(csv_files) - failed}, failed: {failed}")
    print(f"Final hiking segments: {total_segments}")
    print(f"Points rejected for speed inconsistent with hiking: {rejected_points}")
    print(f"Output directory: {output_dir}")
    print("QGIS Points to Path: group by clean_track_id, order by clean_point_index")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
