#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer activity areas and coherent through-routes. Manuscript Equations 1-4 define directional entropy, path efficiency, spatial elongation and activity intensity."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

try:
    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import LineString, Point, box
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"""Missing dependencies. Run:
conda install -c conda-forge pandas numpy geopandas shapely pyproj pyogrio
Original error: {exc}"""
    ) from exc


@dataclass(frozen=True)
class ActivityConfig:
    cell_size_m: float = 10.0
    direction_bins: int = 12
    local_window_points: int = 5
    endpoint_points: int = 3

    min_cell_points: int = 20
    min_cell_tracks: int = 3
    entropy_min: float = 0.62
    endpoint_fraction_min: float = 0.08
    low_efficiency_threshold: float = 0.45
    low_efficiency_fraction_min: float = 0.35
    low_speed_threshold_mps: float = 0.35
    low_speed_fraction_min: float = 0.20

    stable_corridor_dominant_fraction_min: float = 0.55
    stable_corridor_efficiency_min: float = 0.60
    stable_corridor_low_efficiency_fraction_max: float = 0.25
    stable_corridor_endpoint_fraction_max: float = 0.10

    # Genuine curves can also show high directional entropy and low path efficiency within a local grid cell.
    # Protect curved roads using point-cloud elongation and continuous-trajectory evidence to avoid confusing them with parking lots.
    curve_corridor_track_min: int = 3
    curve_corridor_elongation_min: float = 0.48
    curve_corridor_efficiency_min: float = 0.30
    curve_corridor_endpoint_fraction_max: float = 0.15
    curve_corridor_low_speed_fraction_max: float = 0.30
    curve_bridge_radius_cells: int = 2
    curve_bridge_min_stable_neighbors: int = 2
    curve_corridor_weight: float = 0.90

    min_zone_cells: int = 2
    internal_activity_weight: float = 0.10
    boundary_activity_weight: float = 0.65
    coherent_passage_weight: float = 0.85

    # v0.5 raw-track-structure priority: an activity zone may contain the true road.
    # Only continuous gate-to-gate passages supported by several original tracks are
    # promoted; local loops/wandering activity remain suppressed.
    activity_corridor_min_tracks: int = 3
    activity_corridor_gate_tolerance_m: float = 10.0
    activity_corridor_max_axis_spread_deg: float = 30.0
    activity_corridor_min_efficiency: float = 0.55
    activity_corridor_weight: float = 0.95

    # v0.5: an activity zone is allowed to suppress wandering evidence, but it may
    # not erase the dominant road.  Build one simple high-support backbone from
    # clustered boundary gates and the activity-cell support graph.
    activity_backbone_min_tracks: int = 3
    activity_backbone_gate_cluster_m: float = 15.0
    activity_backbone_weight: float = 1.0
    activity_backbone_promote_radius_m: float = 5.0

    normal_weight: float = 1.0
    mainroad_min_weight: float = 0.35
    coherent_point_efficiency_min: float = 0.60
    coherent_point_angle_max_deg: float = 25.0

    min_output_points: int = 3
    min_output_length_m: float = 8.0
    max_output_link_m: float = 25.0
    tile_size_m: float = 560.0
    clear_output: bool = False
    write_gpkg: bool = True


def validate_config(cfg: ActivityConfig) -> None:
    if cfg.cell_size_m <= 0 or cfg.tile_size_m <= 0:
        raise ValueError('cell_size_m and tile_size_m must be greater than 0')
    if cfg.direction_bins < 4:
        raise ValueError('direction_bins must be >= 4')
    if cfg.activity_corridor_min_tracks < 1:
        raise ValueError('activity_corridor_min_tracks must be >= 1')
    if cfg.activity_backbone_min_tracks < 1:
        raise ValueError('activity_backbone_min_tracks must be >= 1')
    if cfg.activity_backbone_gate_cluster_m <= 0:
        raise ValueError('activity_backbone_gate_cluster_m must be > 0')
    if cfg.activity_backbone_promote_radius_m <= 0:
        raise ValueError('activity_backbone_promote_radius_m must be > 0')
    if cfg.activity_corridor_gate_tolerance_m <= 0:
        raise ValueError('activity_corridor_gate_tolerance_m must be > 0')
    if not 0 <= cfg.activity_corridor_max_axis_spread_deg <= 90:
        raise ValueError('activity_corridor_max_axis_spread_deg must be in [0,90]')
    for name in (
        "entropy_min", "endpoint_fraction_min", "low_efficiency_fraction_min",
        "low_speed_fraction_min", "stable_corridor_dominant_fraction_min",
        "stable_corridor_efficiency_min", "mainroad_min_weight",
        "curve_corridor_elongation_min", "curve_corridor_efficiency_min",
        "curve_corridor_endpoint_fraction_max",
        "curve_corridor_low_speed_fraction_max", "curve_corridor_weight",
        "activity_corridor_min_efficiency", "activity_corridor_weight",
        "activity_backbone_weight",
    ):
        value = float(getattr(cfg, name))
        if not 0 <= value <= 1:
            raise ValueError(f"""{name} must be in [0,1]""")


def unit_axis_angles(xy: np.ndarray) -> np.ndarray:
    'Per-point undirected road-axis angle in [0, pi); forward and reverse tracks share a direction.'
    n = len(xy)
    output = np.full(n, np.nan, dtype=float)
    if n < 2:
        return output
    vector = np.zeros_like(xy, dtype=float)
    vector[0] = xy[1] - xy[0]
    vector[-1] = xy[-1] - xy[-2]
    if n > 2:
        vector[1:-1] = xy[2:] - xy[:-2]
    length = np.linalg.norm(vector, axis=1)
    valid = length > 1e-9
    output[valid] = np.mod(np.arctan2(vector[valid, 1], vector[valid, 0]), math.pi)
    return output


# Eq. 2: straight-line displacement divided by accumulated path length
# in a symmetric observation window; stationary windows receive zero efficiency.
def local_efficiency(xy: np.ndarray, half_window: int) -> np.ndarray:
    'Use endpoint distance divided by cumulative path length in a fixed point window to diagnose local detours and stationary activity.'
    n = len(xy)
    output = np.ones(n, dtype=float)
    if n < 3:
        return output
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    index = np.arange(n)
    left = np.maximum(0, index - int(half_window))
    right = np.minimum(n - 1, index + int(half_window))
    path = cumulative[right] - cumulative[left]
    chord = np.linalg.norm(xy[right] - xy[left], axis=1)
    valid = path > 1e-6
    output[valid] = np.clip(chord[valid] / path[valid], 0.0, 1.0)
    output[~valid] = 0.0
    return output


def add_point_diagnostics(frame: pd.DataFrame, cfg: ActivityConfig) -> pd.DataFrame:
    frame = frame.copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_axis_angle_rad"] = np.nan
    frame["local_path_efficiency"] = 1.0
    frame["activity_endpoint_point"] = False

    for _, indices in frame.groupby("clean_track_id", sort=False).groups.items():
        pos = np.asarray(list(indices), dtype=int)
        group = frame.loc[pos].sort_values("clean_point_index", kind="mergesort")
        ordered = group.index.to_numpy(dtype=int)
        xy = group[["x_m", "y_m"]].to_numpy(dtype=float)
        frame.loc[ordered, "_axis_angle_rad"] = unit_axis_angles(xy)
        frame.loc[ordered, "local_path_efficiency"] = local_efficiency(
            xy, cfg.local_window_points
        )
        e = min(cfg.endpoint_points, max(1, len(ordered) // 2))
        endpoint_indices = np.r_[ordered[:e], ordered[-e:]]
        frame.loc[np.unique(endpoint_indices), "activity_endpoint_point"] = True

    if "speed_mps" in frame.columns:
        speed = pd.to_numeric(frame["speed_mps"], errors="coerce").to_numpy(dtype=float)
    else:
        speed = np.full(len(frame), np.nan, dtype=float)
    frame["activity_low_speed_point"] = np.isfinite(speed) & (speed <= cfg.low_speed_threshold_mps)
    frame["activity_low_efficiency_point"] = (
        pd.to_numeric(frame["local_path_efficiency"], errors="coerce").fillna(1.0)
        <= cfg.low_efficiency_threshold
    )
    return frame


def contribution_series(frame: pd.DataFrame) -> pd.Series:
    if {"source_file", "track_id"}.issubset(frame.columns):
        return (
            frame["source_file"].fillna("").astype(str)
            + "||" + frame["track_id"].fillna("").astype(str)
        )
    return frame["clean_track_id"].astype(str)


def aggregate_cells(frame: pd.DataFrame, cfg: ActivityConfig) -> Tuple[pd.DataFrame, float, float]:
    x = pd.to_numeric(frame["x_m"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["y_m"], errors="coerce").to_numpy(dtype=float)
    origin_x = math.floor(float(np.nanmin(x)) / cfg.cell_size_m) * cfg.cell_size_m
    origin_y = math.floor(float(np.nanmin(y)) / cfg.cell_size_m) * cfg.cell_size_m
    col = np.floor((x - origin_x) / cfg.cell_size_m).astype(np.int64)
    row = np.floor((y - origin_y) / cfg.cell_size_m).astype(np.int64)
    frame["activity_cell_col"] = col
    frame["activity_cell_row"] = row
    frame["activity_cell_id"] = np.char.add(
        np.char.add(col.astype(str), "_"), row.astype(str)
    )

    # Compute local point-cloud covariance relative to the grid-cell center to avoid cancellation when squaring large UTM coordinates.
    cell_center_x = origin_x + (col.astype(float) + 0.5) * cfg.cell_size_m
    cell_center_y = origin_y + (row.astype(float) + 0.5) * cfg.cell_size_m
    dx = x - cell_center_x
    dy = y - cell_center_y
    frame["_cell_dx"] = dx
    frame["_cell_dy"] = dy
    frame["_cell_dx2"] = dx * dx
    frame["_cell_dy2"] = dy * dy
    frame["_cell_dxdy"] = dx * dy
    tile_origin_x = math.floor(float(np.nanmin(x)) / cfg.tile_size_m) * cfg.tile_size_m
    tile_origin_y = math.floor(float(np.nanmin(y)) / cfg.tile_size_m) * cfg.tile_size_m
    tile_col = np.floor((x - tile_origin_x) / cfg.tile_size_m).astype(np.int64)
    tile_row = np.floor((y - tile_origin_y) / cfg.tile_size_m).astype(np.int64)
    frame["activity_tile_origin_x_m"] = float(tile_origin_x)
    frame["activity_tile_origin_y_m"] = float(tile_origin_y)
    frame["activity_tile_size_m"] = float(cfg.tile_size_m)
    frame["activity_tile_col"] = tile_col
    frame["activity_tile_row"] = tile_row
    frame["activity_tile_id"] = np.char.add(
        np.char.add(tile_col.astype(str), "_"), tile_row.astype(str)
    )
    frame["_contribution_id"] = contribution_series(frame)

    angles = pd.to_numeric(frame["_axis_angle_rad"], errors="coerce").to_numpy(dtype=float)
    bins = np.full(len(frame), -1, dtype=np.int16)
    valid_angle = np.isfinite(angles)
    bins[valid_angle] = np.floor(angles[valid_angle] / math.pi * cfg.direction_bins).astype(np.int16)
    bins = np.clip(bins, -1, cfg.direction_bins - 1)
    frame["_direction_bin"] = bins

    group = frame.groupby(["activity_cell_col", "activity_cell_row"], sort=False)
    metrics = group.agg(
        point_count=("activity_cell_id", "size"),
        track_count=("_contribution_id", "nunique"),
        endpoint_count=("activity_endpoint_point", "sum"),
        low_efficiency_count=("activity_low_efficiency_point", "sum"),
        low_speed_count=("activity_low_speed_point", "sum"),
        median_efficiency=("local_path_efficiency", "median"),
        mean_cell_dx=("_cell_dx", "mean"),
        mean_cell_dy=("_cell_dy", "mean"),
        mean_cell_dx2=("_cell_dx2", "mean"),
        mean_cell_dy2=("_cell_dy2", "mean"),
        mean_cell_dxdy=("_cell_dxdy", "mean"),
    ).reset_index()

    direction = frame.loc[frame["_direction_bin"] >= 0].groupby(
        ["activity_cell_col", "activity_cell_row", "_direction_bin"], sort=False
    ).size().rename("count").reset_index()
    if direction.empty:
        direction_wide = pd.DataFrame(columns=["activity_cell_col", "activity_cell_row"])
    else:
        direction_wide = direction.pivot_table(
            index=["activity_cell_col", "activity_cell_row"],
            columns="_direction_bin", values="count", fill_value=0, aggfunc="sum"
        ).reset_index()
    metrics = metrics.merge(
        direction_wide, on=["activity_cell_col", "activity_cell_row"], how="left"
    )
    bin_columns = [column for column in metrics.columns if isinstance(column, (int, np.integer))]
    if bin_columns:
        counts = metrics[bin_columns].fillna(0).to_numpy(dtype=float)
        total = counts.sum(axis=1)
        probability = np.divide(counts, total[:, None], out=np.zeros_like(counts), where=total[:, None] > 0)
        log_probability = np.zeros_like(probability)
        positive_probability = probability > 0
        log_probability[positive_probability] = np.log(probability[positive_probability])
        # Eq. 1: normalize heading-bin entropy by log(number of bins).
        entropy = -np.sum(probability * log_probability, axis=1)
        entropy /= max(math.log(cfg.direction_bins), 1e-12)
        dominant_bin_index = np.argmax(counts, axis=1)
        dominant_count = counts[np.arange(len(counts)), dominant_bin_index]
        dominant_fraction = np.divide(dominant_count, total, out=np.zeros_like(total), where=total > 0)
        dominant_bin = np.asarray([int(bin_columns[i]) for i in dominant_bin_index], dtype=int)
        metrics = metrics.drop(columns=bin_columns)
    else:
        entropy = np.zeros(len(metrics), dtype=float)
        dominant_fraction = np.zeros(len(metrics), dtype=float)
        dominant_bin = np.full(len(metrics), -1, dtype=int)

    points = metrics["point_count"].to_numpy(dtype=float)
    metrics["direction_entropy"] = entropy
    metrics["dominant_direction_fraction"] = dominant_fraction
    metrics["dominant_direction_bin"] = dominant_bin
    metrics["endpoint_fraction"] = metrics["endpoint_count"].to_numpy(dtype=float) / np.maximum(points, 1)
    metrics["low_efficiency_fraction"] = metrics["low_efficiency_count"].to_numpy(dtype=float) / np.maximum(points, 1)
    metrics["low_speed_fraction"] = metrics["low_speed_count"].to_numpy(dtype=float) / np.maximum(points, 1)

    # Point-cloud elongation: 1 is nearly linear; 0 is approximately areal.
    mean_dx = metrics["mean_cell_dx"].to_numpy(dtype=float)
    mean_dy = metrics["mean_cell_dy"].to_numpy(dtype=float)
    cov_xx = np.maximum(
        metrics["mean_cell_dx2"].to_numpy(dtype=float) - mean_dx * mean_dx, 0.0
    )
    cov_yy = np.maximum(
        metrics["mean_cell_dy2"].to_numpy(dtype=float) - mean_dy * mean_dy, 0.0
    )
    cov_xy = metrics["mean_cell_dxdy"].to_numpy(dtype=float) - mean_dx * mean_dy
    trace = cov_xx + cov_yy
    discriminant = np.sqrt(np.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy ** 2, 0.0))
    major = 0.5 * (trace + discriminant)
    minor = 0.5 * (trace - discriminant)
    metrics["spatial_elongation"] = np.divide(
        major - minor,
        np.maximum(major, 1e-9),
        out=np.zeros_like(major),
        where=major > 1e-9,
    )
    metrics["spatial_major_sigma_m"] = np.sqrt(np.maximum(major, 0.0))
    metrics["spatial_minor_sigma_m"] = np.sqrt(np.maximum(minor, 0.0))

    evidence = np.maximum.reduce([
        np.clip(metrics["endpoint_fraction"].to_numpy(float) / max(cfg.endpoint_fraction_min, 1e-6), 0, 1),
        np.clip(metrics["low_efficiency_fraction"].to_numpy(float) / max(cfg.low_efficiency_fraction_min, 1e-6), 0, 1),
        np.clip(metrics["low_speed_fraction"].to_numpy(float) / max(cfg.low_speed_fraction_min, 1e-6), 0, 1),
    ])
    density = np.minimum(1.0, metrics["track_count"].to_numpy(float) / max(cfg.min_cell_tracks * 2.0, 1.0))
    metrics["activity_score"] = np.clip(
        0.45 * metrics["direction_entropy"].to_numpy(float)
        + 0.35 * evidence
        + 0.20 * density,
        0.0, 1.0,
    )

    stable = (
        (metrics["track_count"] >= cfg.min_cell_tracks)
        & (metrics["dominant_direction_fraction"] >= cfg.stable_corridor_dominant_fraction_min)
        & (metrics["median_efficiency"] >= cfg.stable_corridor_efficiency_min)
        & (metrics["low_efficiency_fraction"] <= cfg.stable_corridor_low_efficiency_fraction_max)
        & (metrics["endpoint_fraction"] <= cfg.stable_corridor_endpoint_fraction_max)
    )
    candidate = (
        (metrics["point_count"] >= cfg.min_cell_points)
        & (metrics["track_count"] >= cfg.min_cell_tracks)
        & (metrics["direction_entropy"] >= cfg.entropy_min)
        & (
            (metrics["endpoint_fraction"] >= cfg.endpoint_fraction_min)
            | (metrics["low_efficiency_fraction"] >= cfg.low_efficiency_fraction_min)
            | (metrics["low_speed_fraction"] >= cfg.low_speed_fraction_min)
        )
    )

    # Curve protection allows higher directional entropy but requires a linear point cloud, repeated traversal by multiple trajectories,
    # and endpoint and low-speed rates below the high values typical of parking-lot interiors.
    curve = (
        candidate
        & (metrics["track_count"] >= cfg.curve_corridor_track_min)
        & (metrics["spatial_elongation"] >= cfg.curve_corridor_elongation_min)
        & (metrics["median_efficiency"] >= cfg.curve_corridor_efficiency_min)
        & (metrics["endpoint_fraction"] <= cfg.curve_corridor_endpoint_fraction_max)
        & (metrics["low_speed_fraction"] <= cfg.curve_corridor_low_speed_fraction_max)
    )

    # Treat a high-entropy cell between two or more stable road cells as a curve transition.
    stable_cells = {
        (int(c), int(r))
        for c, r in metrics.loc[
            stable, ["activity_cell_col", "activity_cell_row"]
        ].itertuples(index=False, name=None)
    }
    bridge = np.zeros(len(metrics), dtype=bool)
    bridge_eligible = (
        candidate
        & ~stable
        & ~curve
        & (metrics["track_count"] >= cfg.curve_corridor_track_min)
        & (metrics["spatial_elongation"] >= max(0.25, cfg.curve_corridor_elongation_min * 0.65))
        & (metrics["median_efficiency"] >= max(0.20, cfg.curve_corridor_efficiency_min * 0.80))
        & (metrics["endpoint_fraction"] <= min(0.20, cfg.curve_corridor_endpoint_fraction_max + 0.05))
        & (metrics["low_speed_fraction"] <= min(0.40, cfg.curve_corridor_low_speed_fraction_max + 0.10))
    )
    radius = max(1, int(cfg.curve_bridge_radius_cells))
    for index in np.flatnonzero(bridge_eligible.to_numpy(dtype=bool)):
        col_i = int(metrics.at[index, "activity_cell_col"])
        row_i = int(metrics.at[index, "activity_cell_row"])
        neighbor_count = 0
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if dc == 0 and dr == 0:
                    continue
                if (col_i + dc, row_i + dr) in stable_cells:
                    neighbor_count += 1
        bridge[index] = neighbor_count >= int(cfg.curve_bridge_min_stable_neighbors)

    protected = stable.to_numpy(dtype=bool) | curve.to_numpy(dtype=bool) | bridge
    metrics["stable_road_corridor"] = stable
    metrics["curve_road_corridor"] = curve
    metrics["curve_bridge_corridor"] = bridge
    metrics["protected_road_corridor"] = protected
    metrics["activity_candidate_raw"] = candidate
    metrics["activity_suppression_cell"] = candidate.to_numpy(dtype=bool) & ~protected
    metrics["cell_center_x"] = origin_x + (metrics["activity_cell_col"].to_numpy(float) + 0.5) * cfg.cell_size_m
    metrics["cell_center_y"] = origin_y + (metrics["activity_cell_row"].to_numpy(float) + 0.5) * cfg.cell_size_m
    tile_origin_x = math.floor(float(np.nanmin(x)) / cfg.tile_size_m) * cfg.tile_size_m
    tile_origin_y = math.floor(float(np.nanmin(y)) / cfg.tile_size_m) * cfg.tile_size_m
    metrics["activity_tile_origin_x_m"] = float(tile_origin_x)
    metrics["activity_tile_origin_y_m"] = float(tile_origin_y)
    metrics["activity_tile_size_m"] = float(cfg.tile_size_m)
    metrics["activity_tile_col"] = np.floor(
        (metrics["cell_center_x"].to_numpy(float) - tile_origin_x) / cfg.tile_size_m
    ).astype(np.int64)
    metrics["activity_tile_row"] = np.floor(
        (metrics["cell_center_y"].to_numpy(float) - tile_origin_y) / cfg.tile_size_m
    ).astype(np.int64)
    metrics["activity_tile_id"] = (
        metrics["activity_tile_col"].astype(str) + "_" + metrics["activity_tile_row"].astype(str)
    )
    return metrics, origin_x, origin_y


def label_sparse_components(metrics: pd.DataFrame, cfg: ActivityConfig) -> pd.DataFrame:
    metrics = metrics.copy()
    cells = {
        (int(row.activity_cell_col), int(row.activity_cell_row)): index
        for index, row in metrics.loc[metrics["activity_suppression_cell"]].iterrows()
    }
    zone_by_index: Dict[int, int] = {}
    zone_id = 0
    for cell, index in cells.items():
        if index in zone_by_index:
            continue
        queue = [cell]
        component: List[int] = []
        zone_id += 1
        zone_by_index[index] = zone_id
        while queue:
            col, row = queue.pop()
            idx = cells[(col, row)]
            component.append(idx)
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    neighbor = (col + dc, row + dr)
                    nidx = cells.get(neighbor)
                    if nidx is not None and nidx not in zone_by_index:
                        zone_by_index[nidx] = zone_id
                        queue.append(neighbor)
        if len(component) < cfg.min_zone_cells:
            for idx in component:
                zone_by_index[idx] = 0

    metrics["activity_zone_id"] = 0
    for idx, zid in zone_by_index.items():
        metrics.at[idx, "activity_zone_id"] = int(zid)
    metrics["activity_zone_id"] = metrics["activity_zone_id"].astype(int)
    metrics["activity_suppression_cell"] = metrics["activity_zone_id"] > 0

    occupied = {
        (int(row.activity_cell_col), int(row.activity_cell_row)): index
        for index, row in metrics.iterrows()
    }
    boundary = np.zeros(len(metrics), dtype=bool)
    for index, row in metrics.loc[metrics["activity_suppression_cell"]].iterrows():
        col, r = int(row.activity_cell_col), int(row.activity_cell_row)
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                nidx = occupied.get((col + dc, r + dr))
                if nidx is None:
                    continue
                nrow = metrics.loc[nidx]
                if (
                    not bool(nrow["activity_suppression_cell"])
                    and int(nrow["track_count"]) >= 2
                    and float(nrow["dominant_direction_fraction"]) >= 0.40
                ):
                    boundary[index] = True
                    break
            if boundary[index]:
                break
    metrics["activity_boundary_cell"] = boundary
    return metrics


def axis_angle_difference_deg(angle: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.abs(angle - target) % math.pi
    difference = np.minimum(difference, math.pi - difference)
    return np.degrees(difference)


def attach_weights(frame: pd.DataFrame, metrics: pd.DataFrame, cfg: ActivityConfig) -> pd.DataFrame:
    columns = [
        "activity_cell_col", "activity_cell_row", "activity_zone_id",
        "activity_suppression_cell", "activity_boundary_cell", "stable_road_corridor",
        "curve_road_corridor", "curve_bridge_corridor", "protected_road_corridor",
        "activity_score", "direction_entropy", "dominant_direction_fraction",
        "spatial_elongation", "spatial_major_sigma_m", "spatial_minor_sigma_m",
        "dominant_direction_bin", "endpoint_fraction", "low_efficiency_fraction",
        "low_speed_fraction", "track_count", "point_count",
    ]
    frame = frame.merge(
        metrics[columns], on=["activity_cell_col", "activity_cell_row"], how="left"
    )
    frame["activity_zone_id"] = frame["activity_zone_id"].fillna(0).astype(int)
    for col in (
        "activity_suppression_cell", "activity_boundary_cell", "stable_road_corridor",
        "curve_road_corridor", "curve_bridge_corridor", "protected_road_corridor",
    ):
        frame[col] = frame[col].fillna(False).astype(bool)

    weight = np.full(len(frame), cfg.normal_weight, dtype=float)
    suppression = frame["activity_suppression_cell"].to_numpy(bool)
    boundary = frame["activity_boundary_cell"].to_numpy(bool)
    stable = frame["stable_road_corridor"].to_numpy(bool)
    curve = frame["curve_road_corridor"].to_numpy(bool)
    curve_bridge = frame["curve_bridge_corridor"].to_numpy(bool)
    protected = frame["protected_road_corridor"].to_numpy(bool)
    weight[suppression] = cfg.internal_activity_weight
    weight[suppression & boundary] = cfg.boundary_activity_weight
    weight[curve | curve_bridge] = np.maximum(
        weight[curve | curve_bridge], cfg.curve_corridor_weight
    )
    weight[stable] = cfg.normal_weight

    angle = pd.to_numeric(frame["_axis_angle_rad"], errors="coerce").to_numpy(float)
    dominant_bin = pd.to_numeric(frame["dominant_direction_bin"], errors="coerce").fillna(-1).to_numpy(int)
    target = (dominant_bin + 0.5) * math.pi / cfg.direction_bins
    diff = axis_angle_difference_deg(angle, target)
    coherent_point = (
        suppression
        & np.isfinite(angle)
        & (dominant_bin >= 0)
        & (frame["dominant_direction_fraction"].fillna(0).to_numpy(float) >= 0.40)
        & (frame["local_path_efficiency"].fillna(0).to_numpy(float) >= cfg.coherent_point_efficiency_min)
        & (diff <= cfg.coherent_point_angle_max_deg)
    )
    weight[coherent_point] = np.maximum(weight[coherent_point], cfg.coherent_passage_weight)

    frame["road_extraction_weight"] = np.clip(weight, 0.0, 1.0)
    frame["mainroad_candidate"] = frame["road_extraction_weight"] >= cfg.mainroad_min_weight
    reason = np.full(len(frame), "normal", dtype=object)
    reason[suppression] = "activity_internal_suppressed"
    reason[suppression & boundary] = "activity_boundary_retained"
    reason[coherent_point] = "coherent_passage_restored"
    reason[curve] = "curve_road_corridor"
    reason[curve_bridge] = "curve_bridge_corridor"
    reason[stable] = "stable_road_corridor"
    frame["road_extraction_reason"] = reason
    return frame


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if float(np.linalg.norm(v)) <= 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)


def _axis_difference_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 180.0
    d = abs(float(a) - float(b)) % 180.0
    return float(min(d, 180.0 - d))


def detect_activity_through_corridors(
    frame: pd.DataFrame,
    cfg: ActivityConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recover multi-track gate-to-gate road passages inside activity zones.

    The activity label is intentionally *not* a road veto.  A passage is accepted
    only when one original contribution enters a non-zero zone and later leaves it
    in point order, its gate-to-gate route is efficient, and several independent
    contributions agree on the same pair of gates/axis.  This makes a real through
    road positive evidence while leaving local parking/yard loops suppressed.
    """
    out = frame.copy()
    out["activity_through_corridor"] = False
    out["activity_corridor_id"] = 0
    out["activity_corridor_track_support"] = 0
    out["activity_corridor_passage_support"] = 0
    out["activity_corridor_confidence"] = 0.0
    if out.empty or "activity_zone_id" not in out.columns:
        return out, pd.DataFrame(), pd.DataFrame()

    contribution_field = "source_clean_track_id" if "source_clean_track_id" in out.columns else "clean_track_id"
    candidates: List[Dict[str, object]] = []
    for contribution, group in out.groupby(contribution_field, sort=False):
        group = group.sort_values("clean_point_index", kind="mergesort")
        idx = group.index.to_numpy(dtype=int)
        xy = group[["x_m", "y_m"]].to_numpy(dtype=float)
        zones = pd.to_numeric(group["activity_zone_id"], errors="coerce").fillna(0).to_numpy(dtype=int)
        n = len(group)
        i = 0
        while i < n:
            zid = int(zones[i])
            if zid <= 0:
                i += 1
                continue
            j = i + 1
            while j < n and int(zones[j]) == zid:
                j += 1
            # A true through passage has observed points immediately outside both gates.
            if i > 0 and j < n and int(zones[i - 1]) != zid and int(zones[j]) != zid:
                route_xy = xy[i - 1 : j + 1]
                plen = path_length(route_xy)
                chord = float(np.linalg.norm(route_xy[-1] - route_xy[0]))
                efficiency = chord / max(plen, 1e-9)
                axis = _axis_angle_deg(route_xy[0], route_xy[-1])
                if (
                    plen > 1e-6
                    and chord > 1e-6
                    and efficiency >= float(cfg.activity_corridor_min_efficiency)
                    and np.isfinite(axis)
                ):
                    candidates.append({
                        "zone_id": zid,
                        "contribution_id": str(contribution),
                        "inside_indices": idx[i:j].tolist(),
                        "gate_a": route_xy[0].copy(),
                        "gate_b": route_xy[-1].copy(),
                        "axis_deg": float(axis),
                        "efficiency": float(efficiency),
                        "length_m": float(plen),
                        "geometry": LineString(route_xy),
                    })
            i = j

    accepted_rows: List[Dict[str, object]] = []
    rejected_rows: List[Dict[str, object]] = []
    next_corridor_id = 1
    for zid in sorted({int(c["zone_id"]) for c in candidates}):
        remaining = [c for c in candidates if int(c["zone_id"]) == zid]
        clusters: List[List[Dict[str, object]]] = []
        for cand in remaining:
            best = None
            best_metric = float("inf")
            for ci, members in enumerate(clusters):
                ref = members[0]
                axis_diff = _axis_difference_deg(float(cand["axis_deg"]), float(ref["axis_deg"]))
                if axis_diff > float(cfg.activity_corridor_max_axis_spread_deg):
                    continue
                a1 = np.asarray(cand["gate_a"], float); b1 = np.asarray(cand["gate_b"], float)
                a0 = np.asarray(ref["gate_a"], float); b0 = np.asarray(ref["gate_b"], float)
                direct = 0.5 * (np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0))
                reverse = 0.5 * (np.linalg.norm(a1 - b0) + np.linalg.norm(b1 - a0))
                gate_metric = float(min(direct, reverse))
                if gate_metric > float(cfg.activity_corridor_gate_tolerance_m):
                    continue
                metric = axis_diff + gate_metric
                if metric < best_metric:
                    best_metric = metric; best = ci
            if best is None:
                clusters.append([cand])
            else:
                clusters[best].append(cand)

        for members in clusters:
            by_contribution: Dict[str, Dict[str, object]] = {}
            for cand in members:
                cid = str(cand["contribution_id"])
                if cid not in by_contribution or float(cand["efficiency"]) > float(by_contribution[cid]["efficiency"]):
                    by_contribution[cid] = cand
            unique = list(by_contribution.values())
            track_support = len(unique)
            passage_support = len(members)
            axes = np.asarray([float(c["axis_deg"]) for c in unique], dtype=float)
            # Axis spread around the first representative, valid because clustering
            # already handles the 0/180 wrap.
            spread = float(max([_axis_difference_deg(float(axes[0]), float(x)) for x in axes] or [0.0]))
            mean_eff = float(np.mean([float(c["efficiency"]) for c in unique])) if unique else 0.0
            base = {
                "activity_zone_id": int(zid),
                "track_support": int(track_support),
                "passage_support": int(passage_support),
                "axis_spread_deg": float(spread),
                "mean_efficiency": float(mean_eff),
            }
            if track_support < int(cfg.activity_corridor_min_tracks):
                rejected_rows.append({**base, "reject_reason": "insufficient_distinct_tracks"})
                continue
            if spread > float(cfg.activity_corridor_max_axis_spread_deg):
                rejected_rows.append({**base, "reject_reason": "axis_spread_too_large"})
                continue

            # Choose the most efficient observed passage as a diagnostic centreline;
            # topology remains the original point sequence rather than a straight chord.
            medoid = max(unique, key=lambda c: (float(c["efficiency"]), float(c["length_m"])))
            confidence = float(min(1.0, 0.5 * min(track_support / max(cfg.activity_corridor_min_tracks, 1), 1.0) + 0.5 * mean_eff))
            corridor_id = next_corridor_id; next_corridor_id += 1
            mark_indices = sorted({int(i) for c in unique for i in c["inside_indices"]})
            if mark_indices:
                out.loc[mark_indices, "activity_through_corridor"] = True
                out.loc[mark_indices, "activity_corridor_id"] = int(corridor_id)
                out.loc[mark_indices, "activity_corridor_track_support"] = int(track_support)
                out.loc[mark_indices, "activity_corridor_passage_support"] = int(passage_support)
                out.loc[mark_indices, "activity_corridor_confidence"] = float(confidence)
                if "road_extraction_weight" in out.columns:
                    current = pd.to_numeric(out.loc[mark_indices, "road_extraction_weight"], errors="coerce").fillna(0).to_numpy(float)
                    out.loc[mark_indices, "road_extraction_weight"] = np.maximum(current, float(cfg.activity_corridor_weight))
                else:
                    out.loc[mark_indices, "road_extraction_weight"] = float(cfg.activity_corridor_weight)
                out.loc[mark_indices, "mainroad_candidate"] = True
                out.loc[mark_indices, "road_extraction_reason"] = "activity_through_corridor"
            accepted_rows.append({
                **base,
                "activity_corridor_id": int(corridor_id),
                "confidence": float(confidence),
                "contribution_ids": "|".join(sorted(by_contribution)),
                "geometry": medoid["geometry"],
            })

    return out, pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows)


def _cluster_activity_gates(gates: List[Dict[str, object]], tolerance_m: float) -> List[Dict[str, object]]:
    """Greedy spatial clustering for activity-zone boundary observations."""
    clusters: List[List[Dict[str, object]]] = []
    for gate in gates:
        p = np.asarray(gate["xy"], dtype=float)
        best = None
        best_d = float("inf")
        for i, members in enumerate(clusters):
            member_xy = np.asarray([m["xy"] for m in members], dtype=float)
            center = np.mean(member_xy, axis=0)
            d = float(np.linalg.norm(p - center))
            max_member_d = float(np.max(np.linalg.norm(member_xy - p[None, :], axis=1)))
            # Complete-link guard prevents a chain of partial track endpoints from
            # merging the two opposite road gates into one giant cluster.
            if max_member_d <= float(tolerance_m) and d < best_d:
                best = i; best_d = d
        if best is None:
            clusters.append([gate])
        else:
            clusters[best].append(gate)
    out = []
    for members in clusters:
        xy = np.asarray([m["xy"] for m in members], dtype=float)
        contributions = {str(m["contribution_id"]) for m in members}
        transition_contrib = {str(m["contribution_id"]) for m in members if not bool(m.get("partial", False))}
        out.append({
            "xy": np.median(xy, axis=0),
            "track_support": len(contributions),
            "transition_support": len(transition_contrib),
            "contribution_ids": contributions,
            "observations": len(members),
        })
    return out


def _activity_zone_gate_observations(zone_frame: pd.DataFrame, zone_id: int) -> List[Dict[str, object]]:
    contribution_field = "_contribution_id" if "_contribution_id" in zone_frame.columns else "clean_track_id"
    gates: List[Dict[str, object]] = []
    # Use the complete frame slice passed by the caller, where points from this zone
    # and neighbouring outside points coexist.  A run endpoint at the beginning/end
    # of a contribution is also a legitimate partial gate: activity data often starts
    # or stops inside a parking/yard area.
    for cid, group in zone_frame.groupby(contribution_field, sort=False):
        group = group.sort_values("clean_point_index", kind="mergesort")
        z = pd.to_numeric(group["activity_zone_id"], errors="coerce").fillna(0).to_numpy(int)
        xy = group[["x_m", "y_m"]].to_numpy(float)
        inside = z == int(zone_id)
        n = len(group)
        i = 0
        while i < n:
            if not inside[i]:
                i += 1; continue
            j = i + 1
            while j < n and inside[j]:
                j += 1
            gates.append({"xy": xy[i].copy(), "contribution_id": str(cid), "partial": i == 0})
            if j - 1 != i:
                gates.append({"xy": xy[j - 1].copy(), "contribution_id": str(cid), "partial": j == n})
            i = j
    return gates


def extract_activity_zone_backbones(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: ActivityConfig,
    origin_x: float,
    origin_y: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one cycle-free dominant road backbone per supported activity zone.

    Unlike the previous gate-to-gate rule, no single original trajectory must traverse
    the complete zone.  Boundary observations select the two dominant opposite gates,
    while a weighted graph of occupied activity cells supplies the simple path between
    them.  Only points close to that path are promoted, so wandering/loop activity is
    still suppressed.
    """
    out = frame.copy()
    out["activity_backbone_corridor"] = False
    out["activity_backbone_synthetic"] = False
    out["activity_backbone_id"] = 0
    out["activity_backbone_track_support"] = 0
    out["activity_backbone_confidence"] = 0.0
    if out.empty or metrics.empty or "activity_zone_id" not in out.columns:
        return out, pd.DataFrame(), pd.DataFrame()

    contribution_field = "_contribution_id" if "_contribution_id" in out.columns else "clean_track_id"
    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    next_id = 1
    zone_ids = sorted(int(x) for x in pd.to_numeric(metrics.get("activity_zone_id", 0), errors="coerce").fillna(0).unique() if int(x) > 0)
    for zid in zone_ids:
        zmask = pd.to_numeric(out["activity_zone_id"], errors="coerce").fillna(0).to_numpy(int) == zid
        zpoints = out.loc[zmask].copy()
        if zpoints.empty:
            continue
        global_tracks = int(zpoints[contribution_field].astype(str).nunique())
        if global_tracks < int(cfg.activity_backbone_min_tracks):
            rejected.append({"activity_zone_id": zid, "reject_reason": "insufficient_zone_tracks", "track_support": global_tracks})
            continue

        zmetrics = metrics.loc[pd.to_numeric(metrics["activity_zone_id"], errors="coerce").fillna(0).astype(int).eq(zid)].copy()
        cells = {(int(r.activity_cell_col), int(r.activity_cell_row)) for r in zmetrics.itertuples()}
        if len(cells) < 2:
            rejected.append({"activity_zone_id": zid, "reject_reason": "insufficient_zone_cells", "track_support": global_tracks})
            continue

        cell_support: Dict[Tuple[int, int], int] = {}
        cell_point_xy: Dict[Tuple[int, int], np.ndarray] = {}
        for key, grp in zpoints.groupby(["activity_cell_col", "activity_cell_row"], sort=False):
            cell = (int(key[0]), int(key[1]))
            cell_support[cell] = int(grp[contribution_field].astype(str).nunique())
            cell_point_xy[cell] = np.median(grp[["x_m", "y_m"]].to_numpy(float), axis=0)

        graph = nx.Graph()
        for cell in cells:
            graph.add_node(cell)
        for c, r in cells:
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nb = (c + dc, r + dr)
                    if nb not in cells or graph.has_edge((c, r), nb):
                        continue
                    step = float(cfg.cell_size_m) * (math.sqrt(2.0) if dc and dr else 1.0)
                    support = max(1.0, 0.5 * (cell_support.get((c, r), 1) + cell_support.get(nb, 1)))
                    graph.add_edge((c, r), nb, weight=step / support)

        gate_observations = _activity_zone_gate_observations(out, zid)
        strong_gate_observations = [g for g in gate_observations if not bool(g.get("partial", False))]
        strong_clusters = _cluster_activity_gates(strong_gate_observations, cfg.activity_backbone_gate_cluster_m)
        gates = strong_clusters if len(strong_clusters) >= 2 else _cluster_activity_gates(gate_observations, cfg.activity_backbone_gate_cluster_m)
        # Attach each gate cluster to the nearest occupied zone cell.
        for gate in gates:
            p = np.asarray(gate["xy"], dtype=float)
            gate["cell"] = min(cells, key=lambda cell: float(np.linalg.norm(cell_point_xy.get(cell, np.array([
                origin_x + (cell[0] + 0.5) * cfg.cell_size_m,
                origin_y + (cell[1] + 0.5) * cfg.cell_size_m,
            ])) - p)))

        pair = None
        pair_score = -float("inf")
        if len(gates) >= 2:
            for i in range(len(gates)):
                for j in range(i + 1, len(gates)):
                    if gates[i]["cell"] == gates[j]["cell"]:
                        continue
                    dist = float(np.linalg.norm(np.asarray(gates[i]["xy"]) - np.asarray(gates[j]["xy"])))
                    transition_support = min(int(gates[i].get("transition_support", 0)), int(gates[j].get("transition_support", 0)))
                    score = dist * (1.0 + min(int(gates[i]["track_support"]), int(gates[j]["track_support"]))) + 4.0 * transition_support
                    if score > pair_score:
                        pair_score = score; pair = (gates[i], gates[j])

        # Fallback for datasets whose trajectories all start/end inside the zone:
        # choose the farthest pair of occupied boundary cells, still requiring the
        # whole zone to have enough distinct original contributions.
        if pair is None:
            boundary_cells = []
            for cell in cells:
                c, r = cell
                if any((c + dc, r + dr) not in cells for dc, dr in ((1,0),(-1,0),(0,1),(0,-1))):
                    boundary_cells.append(cell)
            best = None; best_d = -1.0
            for i, a in enumerate(boundary_cells):
                for b in boundary_cells[i+1:]:
                    pa = cell_point_xy.get(a); pb = cell_point_xy.get(b)
                    if pa is None or pb is None:
                        continue
                    d = float(np.linalg.norm(pa - pb))
                    if d > best_d:
                        best_d = d; best = (a, b, pa, pb)
            if best is not None:
                a, b, pa, pb = best
                pair = (
                    {"xy": pa, "cell": a, "track_support": cell_support.get(a, 1), "contribution_ids": set()},
                    {"xy": pb, "cell": b, "track_support": cell_support.get(b, 1), "contribution_ids": set()},
                )
        if pair is None:
            rejected.append({"activity_zone_id": zid, "reject_reason": "no_distinct_backbone_gates", "track_support": global_tracks})
            continue

        ga, gb = pair
        try:
            path_cells = nx.shortest_path(graph, ga["cell"], gb["cell"], weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            rejected.append({"activity_zone_id": zid, "reject_reason": "cell_graph_disconnected", "track_support": global_tracks})
            continue
        if len(path_cells) < 2:
            rejected.append({"activity_zone_id": zid, "reject_reason": "backbone_path_too_short", "track_support": global_tracks})
            continue

        coords: List[Tuple[float, float]] = [tuple(map(float, ga["xy"]))]
        for cell in path_cells:
            p = cell_point_xy.get(cell)
            if p is None:
                p = np.array([
                    origin_x + (cell[0] + 0.5) * cfg.cell_size_m,
                    origin_y + (cell[1] + 0.5) * cfg.cell_size_m,
                ], dtype=float)
            q = tuple(map(float, p))
            if np.linalg.norm(np.asarray(q) - np.asarray(coords[-1])) > 0.25:
                coords.append(q)
        qb = tuple(map(float, gb["xy"]))
        if np.linalg.norm(np.asarray(qb) - np.asarray(coords[-1])) > 0.25:
            coords.append(qb)
        if len(coords) < 2:
            continue
        geom = LineString(coords)
        if geom.length <= 1e-6 or not geom.is_simple or geom.is_ring:
            rejected.append({"activity_zone_id": zid, "reject_reason": "non_simple_backbone_geometry", "track_support": global_tracks})
            continue

        radius = max(0.5, float(cfg.activity_backbone_promote_radius_m))
        zone_indices = out.index[zmask].to_numpy(int)
        distances = np.asarray([float(Point(float(out.at[i, "x_m"]), float(out.at[i, "y_m"])).distance(geom)) for i in zone_indices])
        mark_indices = zone_indices[distances <= radius + 1e-9]
        marked_tracks = int(out.loc[mark_indices, contribution_field].astype(str).nunique()) if len(mark_indices) else 0
        if marked_tracks < int(cfg.activity_backbone_min_tracks):
            rejected.append({"activity_zone_id": zid, "reject_reason": "insufficient_backbone_track_support", "track_support": marked_tracks})
            continue

        confidence = float(min(1.0, 0.55 + 0.10 * min(marked_tracks, 4) + 0.05 * min(len(path_cells), 4)))
        bid = next_id; next_id += 1
        out.loc[mark_indices, "activity_backbone_corridor"] = True
        out.loc[mark_indices, "activity_backbone_id"] = bid
        out.loc[mark_indices, "activity_backbone_track_support"] = marked_tracks
        out.loc[mark_indices, "activity_backbone_confidence"] = confidence
        current = pd.to_numeric(out.loc[mark_indices, "road_extraction_weight"], errors="coerce").fillna(0.0).to_numpy(float)
        out.loc[mark_indices, "road_extraction_weight"] = np.maximum(current, float(cfg.activity_backbone_weight))
        out.loc[mark_indices, "mainroad_candidate"] = True
        out.loc[mark_indices, "road_extraction_reason"] = "activity_zone_main_backbone"
        accepted.append({
            "activity_backbone_id": bid,
            "activity_zone_id": zid,
            "track_support": marked_tracks,
            "zone_track_support": global_tracks,
            "gate_a_track_support": int(ga.get("track_support", 0)),
            "gate_b_track_support": int(gb.get("track_support", 0)),
            "path_cells": len(path_cells),
            "length_m": float(geom.length),
            "confidence": confidence,
            "geometry": geom,
        })

    return out, pd.DataFrame(accepted), pd.DataFrame(rejected)


def build_road_candidates(frame: pd.DataFrame, cfg: ActivityConfig, activity_backbones: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    for clean_id, group in frame.sort_values(
        ["clean_track_id", "clean_point_index"], kind="mergesort"
    ).groupby("clean_track_id", sort=False):
        group = group.reset_index(drop=True)
        keep = group["mainroad_candidate"].to_numpy(bool)
        if not keep.any():
            continue
        xy = group[["x_m", "y_m"]].to_numpy(float)
        original_order = pd.to_numeric(group["clean_point_index"], errors="coerce").to_numpy(float)
        retained = np.flatnonzero(keep)
        start = 0
        part_no = 0
        for i in range(1, len(retained) + 1):
            split = i == len(retained)
            if not split:
                prev, current = retained[i - 1], retained[i]
                gap_index = original_order[current] - original_order[prev] > 1.0
                gap_distance = float(np.linalg.norm(xy[current] - xy[prev])) > cfg.max_output_link_m
                split = gap_index or gap_distance
            if not split:
                continue
            indices = retained[start:i]
            start = i
            if len(indices) < cfg.min_output_points:
                continue
            part = group.iloc[indices].copy()
            if path_length(part[["x_m", "y_m"]].to_numpy(float)) < cfg.min_output_length_m:
                continue
            part_no += 1
            part["source_clean_track_id"] = str(clean_id)
            part["clean_track_id"] = f"{clean_id}__road{part_no:03d}"
            part["clean_point_index"] = np.arange(1, len(part) + 1, dtype=int)
            output.append(part)
    if output:
        result = pd.concat(output, ignore_index=True)
    else:
        result = pd.DataFrame(columns=list(frame.columns) + ["source_clean_track_id"])

    # v0.5: activity backbones remain diagnostics only.  They are NOT injected as
    # synthetic trajectories into S5, because a guessed gate-to-gate scaffold can
    # dominate KDE and create a road that no original contribution actually followed.
    # Real activity-zone points may still be promoted by the evidence-based flags
    # above, but downstream density is always computed from observed trajectories.
    return result


def write_zone_gpkg(
    metrics: pd.DataFrame,
    output_path: Path,
    crs: str,
    origin_x: float,
    origin_y: float,
    cfg: ActivityConfig,
) -> None:
    if output_path.exists():
        output_path.unlink()
    zone_records: List[Dict[str, object]] = []
    for zone_id, group in metrics.loc[metrics["activity_zone_id"] > 0].groupby("activity_zone_id"):
        cells = [
            box(
                origin_x + int(row.activity_cell_col) * cfg.cell_size_m,
                origin_y + int(row.activity_cell_row) * cfg.cell_size_m,
                origin_x + (int(row.activity_cell_col) + 1) * cfg.cell_size_m,
                origin_y + (int(row.activity_cell_row) + 1) * cfg.cell_size_m,
            )
            for row in group.itertuples()
        ]
        geometry = unary_union(cells)
        zone_records.append({
            "activity_zone_id": int(zone_id),
            "cell_count": int(len(group)),
            "area_m2": float(geometry.area),
            "mean_activity_score": float(group["activity_score"].mean()),
            "mean_direction_entropy": float(group["direction_entropy"].mean()),
            "max_track_count": int(group["track_count"].max()),
            "geometry": geometry,
        })
    if zone_records:
        gpd.GeoDataFrame(zone_records, geometry="geometry", crs=crs).to_file(
            output_path, layer="activity_suppression_zones", driver="GPKG"
        )
    for layer_name, mask_column in (
        ("stable_road_corridors", "stable_road_corridor"),
        ("protected_road_corridors", "protected_road_corridor"),
    ):
        corridor = metrics.loc[metrics[mask_column]]
        if corridor.empty:
            continue
        records = []
        for row in corridor.itertuples():
            geometry = box(
                origin_x + int(row.activity_cell_col) * cfg.cell_size_m,
                origin_y + int(row.activity_cell_row) * cfg.cell_size_m,
                origin_x + (int(row.activity_cell_col) + 1) * cfg.cell_size_m,
                origin_y + (int(row.activity_cell_row) + 1) * cfg.cell_size_m,
            )
            corridor_type = (
                "stable"
                if bool(getattr(row, "stable_road_corridor", False))
                else "curve"
                if bool(getattr(row, "curve_road_corridor", False))
                else "curve_bridge"
            )
            records.append({
                "cell_col": int(row.activity_cell_col),
                "cell_row": int(row.activity_cell_row),
                "track_count": int(row.track_count),
                "dominant_fraction": float(row.dominant_direction_fraction),
                "spatial_elongation": float(row.spatial_elongation),
                "corridor_type": corridor_type,
                "geometry": geometry,
            })
        mode = "a" if output_path.exists() else "w"
        gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_file(
            output_path, layer=layer_name, driver="GPKG", mode=mode
        )


def run_activity_filter(input_csv: Path, output_dir: Path, cfg: ActivityConfig) -> Dict[str, object]:
    validate_config(cfg)
    input_csv = input_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if cfg.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[ACTIVITY 1/6] Reading cleaned tracks...', flush=True)
    frame = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
    required = {"clean_track_id", "clean_point_index", "x_m", "y_m", "metric_crs"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"""Input is missing fields: {missing}""")
    for col in ("x_m", "y_m", "clean_point_index"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[
        np.isfinite(frame["x_m"]) & np.isfinite(frame["y_m"])
        & frame["clean_track_id"].notna() & frame["clean_point_index"].notna()
    ].copy().reset_index(drop=True)
    crs_values = frame["metric_crs"].dropna().astype(str).unique().tolist()
    if len(crs_values) != 1:
        raise ValueError(f"""metric_crs is not unique: {crs_values}""")
    crs = CRS.from_user_input(crs_values[0])
    if not crs.is_projected:
        raise ValueError('Activity-zone analysis requires a projected CRS')

    print('[ACTIVITY 2/6] Computing local directions, endpoints and path efficiency...', flush=True)
    frame = add_point_diagnostics(frame, cfg)

    print('[ACTIVITY 3/6] Aggregating grid direction entropy and activity evidence...', flush=True)
    metrics, origin_x, origin_y = aggregate_cells(frame, cfg)
    metrics = label_sparse_components(metrics, cfg)

    print('[ACTIVITY 4/6] Assigning road-extraction weights to track points...', flush=True)
    frame = attach_weights(frame, metrics, cfg)
    frame, activity_corridors, activity_corridor_rejected = detect_activity_through_corridors(frame, cfg)
    frame, activity_backbones, activity_backbone_rejected = extract_activity_zone_backbones(
        frame, metrics, cfg, origin_x, origin_y
    )

    print('[ACTIVITY 5/6] Generating continuous main-road candidate tracks...', flush=True)
    road = build_road_candidates(frame, cfg, activity_backbones=activity_backbones)

    drop_internal = [
        "_axis_angle_rad", "_direction_bin", "_contribution_id", "_row_order",
        "_cell_dx", "_cell_dy", "_cell_dx2", "_cell_dy2", "_cell_dxdy",
    ]
    frame.drop(columns=drop_internal, errors="ignore").to_csv(
        output_dir / "all_cleaned_points_activity.csv", index=False, encoding="utf-8-sig"
    )
    road.drop(columns=drop_internal, errors="ignore").to_csv(
        output_dir / "road_candidate_points.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(output_dir / "activity_cell_metrics.csv", index=False, encoding="utf-8-sig")
    activity_corridors.to_csv(output_dir / "activity_through_corridors.csv", index=False, encoding="utf-8-sig")
    activity_corridor_rejected.to_csv(output_dir / "activity_corridor_rejected_candidates.csv", index=False, encoding="utf-8-sig")
    activity_backbones.to_csv(output_dir / "activity_zone_backbones.csv", index=False, encoding="utf-8-sig")
    activity_backbone_rejected.to_csv(output_dir / "activity_backbone_rejected_candidates.csv", index=False, encoding="utf-8-sig")

    if cfg.write_gpkg:
        try:
            write_zone_gpkg(
                metrics, output_dir / "activity_zones.gpkg", crs.to_string(), origin_x, origin_y, cfg
            )
            if not activity_corridors.empty and "geometry" in activity_corridors.columns:
                corridor_gdf = gpd.GeoDataFrame(activity_corridors.copy(), geometry="geometry", crs=crs.to_string())
                corridor_gdf.to_file(
                    output_dir / "activity_zones.gpkg", layer="activity_through_corridors",
                    driver="GPKG", mode="a"
                )
            if not activity_backbones.empty and "geometry" in activity_backbones.columns:
                backbone_gdf = gpd.GeoDataFrame(activity_backbones.copy(), geometry="geometry", crs=crs.to_string())
                backbone_gdf.to_file(
                    output_dir / "activity_zones.gpkg", layer="activity_zone_backbones",
                    driver="GPKG", mode="a"
                )
        except Exception as exc:
            print(f"""[WARNING] Failed to write activity_zones.gpkg: {exc}""", flush=True)

    report = {
        "input": str(input_csv),
        "input_rows": int(len(frame)),
        "input_tracks": int(frame["clean_track_id"].nunique()),
        "metric_crs": crs.to_string(),
        "activity_grid_origin_x": float(origin_x),
        "activity_grid_origin_y": float(origin_y),
        "activity_tile_size_m": float(cfg.tile_size_m),
        "activity_tile_origin_x_m": float(frame["activity_tile_origin_x_m"].iloc[0]),
        "activity_tile_origin_y_m": float(frame["activity_tile_origin_y_m"].iloc[0]),
        "activity_cells": int(metrics["activity_suppression_cell"].sum()),
        "activity_zones": int(metrics["activity_zone_id"].max()),
        "stable_corridor_cells": int(metrics["stable_road_corridor"].sum()),
        "curve_corridor_cells": int(metrics["curve_road_corridor"].sum()),
        "curve_bridge_cells": int(metrics["curve_bridge_corridor"].sum()),
        "protected_corridor_cells": int(metrics["protected_road_corridor"].sum()),
        "activity_through_corridors": int(len(activity_corridors)),
        "activity_corridor_rejected_candidates": int(len(activity_corridor_rejected)),
        "activity_through_corridor_rows": int(frame["activity_through_corridor"].sum()),
        "activity_zone_backbones": int(len(activity_backbones)),
        "activity_backbone_rejected_candidates": int(len(activity_backbone_rejected)),
        "activity_backbone_rows": int(frame["activity_backbone_corridor"].sum()),
        "road_candidate_rows": int(len(road)),
        "road_candidate_tracks": int(road["clean_track_id"].nunique()) if not road.empty else 0,
        "retained_row_ratio": float(len(road) / max(len(frame), 1)),
        "config": asdict(cfg),
        "outputs": {
            "annotated_all": str(output_dir / "all_cleaned_points_activity.csv"),
            "road_candidates": str(output_dir / "road_candidate_points.csv"),
            "cell_metrics": str(output_dir / "activity_cell_metrics.csv"),
            "activity_through_corridors": str(output_dir / "activity_through_corridors.csv"),
            "activity_corridor_rejected": str(output_dir / "activity_corridor_rejected_candidates.csv"),
            "activity_zone_backbones": str(output_dir / "activity_zone_backbones.csv"),
            "activity_backbone_rejected": str(output_dir / "activity_backbone_rejected_candidates.csv"),
            "zones": str(output_dir / "activity_zones.gpkg"),
        },
        "note": 'v0.5: Downweight disorganized activity-zone tracks while prioritizing acyclic backbones in zones with sufficient original-track support to preserve main-road continuity.',
    }
    (output_dir / "activity_zone_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print('[ACTIVITY 6/6] Complete', flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    defaults = ActivityConfig()
    parser = argparse.ArgumentParser(
        description='Detect area-based activity zones such as parking areas and plazas, and export road candidate tracks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="S2 all_cleaned_points.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size-m", type=float, default=defaults.cell_size_m)
    parser.add_argument("--direction-bins", type=int, default=defaults.direction_bins)
    parser.add_argument("--local-window-points", type=int, default=defaults.local_window_points)
    parser.add_argument("--min-cell-points", type=int, default=defaults.min_cell_points)
    parser.add_argument("--min-cell-tracks", type=int, default=defaults.min_cell_tracks)
    parser.add_argument("--entropy-min", type=float, default=defaults.entropy_min)
    parser.add_argument("--endpoint-fraction-min", type=float, default=defaults.endpoint_fraction_min)
    parser.add_argument("--low-efficiency-fraction-min", type=float, default=defaults.low_efficiency_fraction_min)
    parser.add_argument("--low-speed-fraction-min", type=float, default=defaults.low_speed_fraction_min)
    parser.add_argument("--stable-dominant-fraction-min", type=float, default=defaults.stable_corridor_dominant_fraction_min)
    parser.add_argument("--curve-track-min", type=int, default=defaults.curve_corridor_track_min)
    parser.add_argument("--curve-elongation-min", type=float, default=defaults.curve_corridor_elongation_min)
    parser.add_argument("--curve-efficiency-min", type=float, default=defaults.curve_corridor_efficiency_min)
    parser.add_argument("--curve-endpoint-fraction-max", type=float, default=defaults.curve_corridor_endpoint_fraction_max)
    parser.add_argument("--curve-low-speed-fraction-max", type=float, default=defaults.curve_corridor_low_speed_fraction_max)
    parser.add_argument("--curve-bridge-radius-cells", type=int, default=defaults.curve_bridge_radius_cells)
    parser.add_argument("--curve-bridge-min-stable-neighbors", type=int, default=defaults.curve_bridge_min_stable_neighbors)
    parser.add_argument("--curve-corridor-weight", type=float, default=defaults.curve_corridor_weight)
    parser.add_argument("--internal-activity-weight", type=float, default=defaults.internal_activity_weight)
    parser.add_argument("--boundary-activity-weight", type=float, default=defaults.boundary_activity_weight)
    parser.add_argument("--coherent-passage-weight", type=float, default=defaults.coherent_passage_weight)
    parser.add_argument("--activity-corridor-min-tracks", type=int, default=defaults.activity_corridor_min_tracks)
    parser.add_argument("--activity-corridor-gate-tolerance-m", type=float, default=defaults.activity_corridor_gate_tolerance_m)
    parser.add_argument("--activity-corridor-max-axis-spread-deg", type=float, default=defaults.activity_corridor_max_axis_spread_deg)
    parser.add_argument("--activity-corridor-min-efficiency", type=float, default=defaults.activity_corridor_min_efficiency)
    parser.add_argument("--activity-corridor-weight", type=float, default=defaults.activity_corridor_weight)
    parser.add_argument("--activity-backbone-min-tracks", type=int, default=defaults.activity_backbone_min_tracks)
    parser.add_argument("--activity-backbone-gate-cluster-m", type=float, default=defaults.activity_backbone_gate_cluster_m)
    parser.add_argument("--activity-backbone-weight", type=float, default=defaults.activity_backbone_weight)
    parser.add_argument("--activity-backbone-promote-radius-m", type=float, default=defaults.activity_backbone_promote_radius_m)
    parser.add_argument("--mainroad-min-weight", type=float, default=defaults.mainroad_min_weight)
    parser.add_argument("--min-zone-cells", type=int, default=defaults.min_zone_cells)
    parser.add_argument("--max-output-link-m", type=float, default=defaults.max_output_link_m)
    parser.add_argument("--tile-size-m", type=float, default=defaults.tile_size_m)
    parser.add_argument("--no-gpkg", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = ActivityConfig()
    cfg = ActivityConfig(
        cell_size_m=float(args.cell_size_m),
        direction_bins=int(args.direction_bins),
        local_window_points=int(args.local_window_points),
        min_cell_points=int(args.min_cell_points),
        min_cell_tracks=int(args.min_cell_tracks),
        entropy_min=float(args.entropy_min),
        endpoint_fraction_min=float(args.endpoint_fraction_min),
        low_efficiency_fraction_min=float(args.low_efficiency_fraction_min),
        low_speed_fraction_min=float(args.low_speed_fraction_min),
        stable_corridor_dominant_fraction_min=float(args.stable_dominant_fraction_min),
        curve_corridor_track_min=int(args.curve_track_min),
        curve_corridor_elongation_min=float(args.curve_elongation_min),
        curve_corridor_efficiency_min=float(args.curve_efficiency_min),
        curve_corridor_endpoint_fraction_max=float(args.curve_endpoint_fraction_max),
        curve_corridor_low_speed_fraction_max=float(args.curve_low_speed_fraction_max),
        curve_bridge_radius_cells=int(args.curve_bridge_radius_cells),
        curve_bridge_min_stable_neighbors=int(args.curve_bridge_min_stable_neighbors),
        curve_corridor_weight=float(args.curve_corridor_weight),
        internal_activity_weight=float(args.internal_activity_weight),
        boundary_activity_weight=float(args.boundary_activity_weight),
        coherent_passage_weight=float(args.coherent_passage_weight),
        activity_corridor_min_tracks=int(args.activity_corridor_min_tracks),
        activity_corridor_gate_tolerance_m=float(args.activity_corridor_gate_tolerance_m),
        activity_corridor_max_axis_spread_deg=float(args.activity_corridor_max_axis_spread_deg),
        activity_corridor_min_efficiency=float(args.activity_corridor_min_efficiency),
        activity_corridor_weight=float(args.activity_corridor_weight),
        activity_backbone_min_tracks=int(args.activity_backbone_min_tracks),
        activity_backbone_gate_cluster_m=float(args.activity_backbone_gate_cluster_m),
        activity_backbone_weight=float(args.activity_backbone_weight),
        activity_backbone_promote_radius_m=float(args.activity_backbone_promote_radius_m),
        mainroad_min_weight=float(args.mainroad_min_weight),
        min_zone_cells=int(args.min_zone_cells),
        max_output_link_m=float(args.max_output_link_m),
        tile_size_m=float(args.tile_size_m),
        write_gpkg=not bool(args.no_gpkg),
        clear_output=bool(args.clear_output),
    )
    try:
        run_activity_filter(args.input, args.output, cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
