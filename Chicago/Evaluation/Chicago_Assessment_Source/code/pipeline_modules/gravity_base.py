#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal module: topology-preserving robust gravity adjustment, called by S4.

Input
-----
Road candidate points supplied by S4.

Method
------
1. Use independent tracks as gravity sources to avoid self-contraction.
2. Resample each track at fixed distances to reduce sampling-rate differences.
3. Allow interactions only with consistent directions, small along-track offsets and reasonable lateral distances.
4. Detect hairpin legs that are spatially close but far apart along the track; physical-side codes prevent attraction across legs.
5. Estimate lateral targets with independent-track weighted medians and move points only along the current track normal.
6. Apply stricter cumulative displacement limits at sharp bends and hairpins.
7. Diagnose multi-track support for every adjacent edge after adjustment without deleting single-track edges.
8. Freeze points with insufficient multi-track support at their original positions and topology.
9. Retain all valid input points, all clean_track_id values and all adjacent edges for direct KDE input.
10. Export a high-confidence multi-track subset for diagnostics only, not as the default KDE input.

Dependencies
------------
pandas numpy scipy pyproj geopandas shapely
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception:  # pragma: no cover
    gpd = None
    LineString = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PACKAGE_ROOT / "outputs" / "S2_Clean" / "all_cleaned_points.csv"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "S4_Gravity"


@dataclass
class GravityConfig:
    projected_crs: str = "EPSG:32647"
    sample_spacing_m: float = 2.0
    iterations: int = 4
    search_radius_m: float = 15.0
    max_axis_angle_deg: float = 28.0
    max_along_offset_m: float = 7.0
    max_lateral_offset_m: float = 13.0
    min_support_tracks: int = 2
    gravity_alpha: float = 0.65
    max_step_m: float = 2.0
    max_total_move_m: float = 8.0
    max_total_move_hairpin_m: float = 5.0
    displacement_smooth_sigma_points: float = 1.0

    hairpin_pair_radius_m: float = 20.0
    hairpin_min_along_separation_m: float = 30.0
    hairpin_axis_angle_deg: float = 35.0
    hairpin_max_elevation_diff_m: float = 25.0

    edge_check_spacing_m: float = 2.0
    edge_support_radius_m: float = 8.0
    edge_axis_angle_deg: float = 30.0
    edge_min_other_tracks: int = 2
    edge_min_supported_fraction: float = 0.45
    short_edge_keep_m: float = 4.0
    short_edge_min_other_tracks: int = 1
    min_output_points: int = 3
    min_output_length_m: float = 4.0

    # Single-trajectory protection: edge support is diagnostic only; never delete edges during gravity concentration.
    preserve_all_input_edges: bool = True
    write_supported_subset_diagnostic: bool = True

    clear_output: bool = True
    write_gpkg: bool = True


def validate_config(cfg: GravityConfig) -> None:
    positive = [
        "sample_spacing_m", "search_radius_m", "max_along_offset_m",
        "max_lateral_offset_m", "max_step_m", "max_total_move_m",
        "max_total_move_hairpin_m", "hairpin_pair_radius_m",
        "hairpin_min_along_separation_m", "edge_check_spacing_m",
        "edge_support_radius_m",
    ]
    for name in positive:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"""{name} must be a positive finite number""")
    if cfg.iterations < 1:
        raise ValueError('iterations must be >= 1')
    if cfg.min_support_tracks < 1 or cfg.edge_min_other_tracks < 1:
        raise ValueError('Support track-count thresholds must be >= 1')
    if not 0 < cfg.edge_min_supported_fraction <= 1:
        raise ValueError('edge_min_supported_fraction must be in (0, 1]')
    crs = CRS.from_user_input(cfg.projected_crs)
    if not crs.is_projected:
        raise ValueError('projected_crs must be a projected CRS')


def cumulative_distance(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.array([], dtype=float)
    if len(xy) == 1:
        return np.zeros(1, dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def unit_tangents(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    tangent = np.zeros((n, 2), dtype=float)
    if n < 2:
        return tangent
    if n == 2:
        tangent[:] = xy[1] - xy[0]
    else:
        tangent[0] = xy[1] - xy[0]
        tangent[-1] = xy[-1] - xy[-2]
        tangent[1:-1] = xy[2:] - xy[:-2]
    lengths = np.linalg.norm(tangent, axis=1)
    valid = lengths > 1e-9
    tangent[valid] /= lengths[valid, None]
    return tangent


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(cumulative, cumulative[-1] * 0.5)])


def select_elevation_column(df: pd.DataFrame) -> Optional[str]:
    for name in ("elevation_clean_m", "elevation_m", "dem_elevation_m"):
        if name in df.columns:
            return name
    return None


def prepare_input(path: Path, cfg: GravityConfig) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"clean_track_id", "clean_point_index", "source_file", "track_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"""Input is missing required fields: {missing}""")

    for column in ("clean_point_index",):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    use_xy = {"x_m", "y_m"}.issubset(df.columns)
    if use_xy:
        x = pd.to_numeric(df["x_m"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["y_m"], errors="coerce").to_numpy(float)
    elif {"longitude", "latitude"}.issubset(df.columns):
        lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(float)
        transformer = Transformer.from_crs("EPSG:4326", cfg.projected_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    else:
        raise ValueError('Input must include x_m/y_m or longitude/latitude')

    valid = (
        np.isfinite(x) & np.isfinite(y) &
        df["clean_track_id"].notna().to_numpy() &
        df["clean_point_index"].notna().to_numpy()
    )
    df = df.loc[valid].copy().reset_index(drop=True)
    df["x_m"] = x[valid]
    df["y_m"] = y[valid]
    df["clean_track_id"] = df["clean_track_id"].astype(str)
    df["_contribution_id"] = (
        df["source_file"].fillna("").astype(str)
        + "||" + df["track_id"].fillna("").astype(str)
    )
    df = df.sort_values(
        ["clean_track_id", "clean_point_index"], kind="mergesort"
    ).reset_index(drop=True)

    duplicates = df.duplicated(["clean_track_id", "clean_point_index"], keep=False)
    if duplicates.any():
        raise ValueError(
            f"""Found {int(duplicates.sum())} duplicate clean_track_id + clean_point_index rows; track order cannot be determined reliably"""
        )

    return df


def detect_hairpin_codes(
    xy: np.ndarray,
    elevation: Optional[np.ndarray],
    cfg: GravityConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    'Identify self-approaching legs with parallel road axes that are close in space but far apart along the track.'
    n = len(xy)
    codes = np.zeros(n, dtype=np.int8)
    pair_distance = np.full(n, np.nan, dtype=float)
    if n < 4:
        return codes, pair_distance

    along = cumulative_distance(xy)
    tangents = unit_tangents(xy)
    tree = cKDTree(xy)
    neighbors = tree.query_ball_point(xy, r=cfg.hairpin_pair_radius_m, workers=-1)
    cosine_limit = math.cos(math.radians(cfg.hairpin_axis_angle_deg))

    for index, local_ids in enumerate(neighbors):
        best_distance = math.inf
        best_index = -1
        for candidate in local_ids:
            if candidate == index:
                continue
            if abs(along[candidate] - along[index]) < cfg.hairpin_min_along_separation_m:
                continue
            if abs(float(np.dot(tangents[index], tangents[candidate]))) < cosine_limit:
                continue
            if elevation is not None:
                e0, e1 = elevation[index], elevation[candidate]
                if (
                    np.isfinite(e0) and np.isfinite(e1)
                    and abs(e0 - e1) > cfg.hairpin_max_elevation_diff_m
                ):
                    continue
            distance = float(np.linalg.norm(xy[candidate] - xy[index]))
            if 2.0 < distance < best_distance:
                best_distance = distance
                best_index = candidate

        if best_index < 0:
            continue
        delta = xy[best_index] - xy[index]
        # Encode physical legs by the lower/higher absolute-coordinate side so reversing a trajectory does not flip the encoding.
        if abs(delta[0]) >= abs(delta[1]):
            codes[index] = -1 if xy[index, 0] < xy[best_index, 0] else 1
        else:
            codes[index] = -1 if xy[index, 1] < xy[best_index, 1] else 1
        pair_distance[index] = best_distance

    return codes, pair_distance


def annotate_hairpins(df: pd.DataFrame, cfg: GravityConfig) -> pd.DataFrame:
    elevation_column = select_elevation_column(df)
    df["gravity_hairpin_leg_code"] = np.int8(0)
    df["gravity_hairpin_pair_distance_m"] = np.nan
    for _, indices in df.groupby("clean_track_id", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        xy = df.loc[positions, ["x_m", "y_m"]].to_numpy(float)
        elevation = (
            pd.to_numeric(df.loc[positions, elevation_column], errors="coerce").to_numpy(float)
            if elevation_column else None
        )
        codes, distances = detect_hairpin_codes(xy, elevation, cfg)
        df.loc[positions, "gravity_hairpin_leg_code"] = codes
        df.loc[positions, "gravity_hairpin_pair_distance_m"] = distances
    df["gravity_hairpin_leg_code"] = (
        df["gravity_hairpin_leg_code"].fillna(0).astype(np.int8)
    )
    df["gravity_hairpin_protected"] = df["gravity_hairpin_leg_code"].ne(0)
    return df


def resample_all_tracks(
    df: pd.DataFrame,
    spacing_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    contributions: List[str] = []
    hairpin_codes: List[int] = []

    for _, group in df.groupby("clean_track_id", sort=False):
        group = group.sort_values("clean_point_index")
        xy = group[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
        distance = cumulative_distance(xy)
        if len(xy) < 2 or distance[-1] <= 1e-9:
            continue
        samples = np.arange(0.0, distance[-1], spacing_m)
        if len(samples) == 0 or samples[-1] < distance[-1] - 1e-9:
            samples = np.r_[samples, distance[-1]]
        sampled_xy = np.c_[
            np.interp(samples, distance, xy[:, 0]),
            np.interp(samples, distance, xy[:, 1]),
        ]
        sampled_tangent = unit_tangents(sampled_xy)

        right = np.searchsorted(distance, samples, side="left")
        right = np.clip(right, 0, len(group) - 1)
        left = np.clip(right - 1, 0, len(group) - 1)
        nearest = np.where(
            np.abs(distance[right] - samples) <= np.abs(distance[left] - samples),
            right,
            left,
        )
        code_values = group["gravity_hairpin_leg_code"].to_numpy(np.int8)[nearest]

        points.append(sampled_xy)
        tangents.append(sampled_tangent)
        contributions.extend([group["_contribution_id"].iloc[0]] * len(samples))
        hairpin_codes.extend(code_values.tolist())

    if not points:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.array([], dtype=object),
            np.array([], dtype=np.int8),
        )
    return (
        np.vstack(points),
        np.vstack(tangents),
        np.asarray(contributions, dtype=object),
        np.asarray(hairpin_codes, dtype=np.int8),
    )


def current_tangents_and_curvature(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    tangents = np.zeros((len(df), 2), dtype=float)
    curvature = np.zeros(len(df), dtype=float)
    for _, indices in df.groupby("clean_track_id", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        xy = df.loc[positions, ["gravity_x_m", "gravity_y_m"]].to_numpy(float)
        local_tangent = unit_tangents(xy)
        tangents[positions] = local_tangent
        if len(local_tangent) > 2:
            dots = np.abs(np.sum(local_tangent[1:] * local_tangent[:-1], axis=1))
            curvature[positions[1:]] = np.degrees(
                np.arccos(np.clip(dots, 0.0, 1.0))
            )
    return tangents, curvature


def gravity_adjust(df: pd.DataFrame, cfg: GravityConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    original_xy = df[["x_m", "y_m"]].to_numpy(float)
    df["gravity_original_x_m"] = original_xy[:, 0]
    df["gravity_original_y_m"] = original_xy[:, 1]
    df["gravity_x_m"] = original_xy[:, 0]
    df["gravity_y_m"] = original_xy[:, 1]

    iteration_rows: List[Dict[str, float]] = []
    cosine_limit = math.cos(math.radians(cfg.max_axis_angle_deg))
    max_support_seen = np.zeros(len(df), dtype=np.int16)
    ever_move_eligible = np.zeros(len(df), dtype=bool)

    for iteration in range(1, cfg.iterations + 1):
        start = time.time()
        tangents, curvature = current_tangents_and_curvature(df)
        sample_xy, sample_tangent, sample_contribution, sample_hairpin = (
            resample_all_tracks(df, cfg.sample_spacing_m)
        )
        if len(sample_xy) == 0:
            raise RuntimeError('No resampled points available for gravity adjustment')
        tree = cKDTree(sample_xy)
        query_xy = df[["gravity_x_m", "gravity_y_m"]].to_numpy(float)

        # K-nearest-neighbor queries are more stable and substantially faster than per-point query_ball_point calls.
        k_neighbors = min(64, max(16, len(sample_xy)))
        distances, neighbor_ids = tree.query(
            query_xy,
            k=k_neighbors,
            distance_upper_bound=cfg.search_radius_m,
            workers=-1,
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbor_ids = neighbor_ids[:, None]

        raw_move = np.zeros(len(df), dtype=float)
        support_count = np.zeros(len(df), dtype=np.int16)
        cross_leg_rejected = np.zeros(len(df), dtype=np.int16)

        for index in range(len(df)):
            tangent = tangents[index]
            if np.linalg.norm(tangent) < 0.5:
                continue
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            contribution_id = df["_contribution_id"].iat[index]
            target_leg = int(df["gravity_hairpin_leg_code"].iat[index])
            best_by_track: Dict[str, Tuple[float, float, float, float]] = {}

            for distance, sample_index in zip(distances[index], neighbor_ids[index]):
                if not np.isfinite(distance) or sample_index >= len(sample_xy):
                    continue
                if sample_contribution[sample_index] == contribution_id:
                    continue
                axis_dot = abs(float(np.dot(tangent, sample_tangent[sample_index])))
                if axis_dot < cosine_limit:
                    continue
                delta = sample_xy[sample_index] - query_xy[index]
                along_offset = abs(float(np.dot(delta, tangent)))
                lateral_offset = float(np.dot(delta, normal))
                if along_offset > cfg.max_along_offset_m:
                    continue
                if abs(lateral_offset) > cfg.max_lateral_offset_m:
                    continue
                candidate_leg = int(sample_hairpin[sample_index])
                if target_leg != 0 and candidate_leg != target_leg:
                    cross_leg_rejected[index] += 1
                    continue

                source_id = str(sample_contribution[sample_index])
                score = float(distance) + 0.3 * along_offset
                current = best_by_track.get(source_id)
                candidate = (score, lateral_offset, axis_dot, float(distance))
                if current is None or score < current[0]:
                    best_by_track[source_id] = candidate

            if len(best_by_track) < cfg.min_support_tracks:
                continue

            lateral_values = np.asarray([value[1] for value in best_by_track.values()])
            direction_values = np.asarray([value[2] for value in best_by_track.values()])
            distance_values = np.asarray([value[3] for value in best_by_track.values()])
            weights = (
                np.exp(-((distance_values / max(cfg.search_radius_m * 0.55, 1e-6)) ** 2))
                * np.maximum(
                    0.05,
                    (direction_values - cosine_limit) / max(1.0 - cosine_limit, 1e-9),
                )
            )
            raw_move[index] = weighted_median(lateral_values, weights)
            support_count[index] = len(best_by_track)

        # Smooth lateral displacement along the trajectory to avoid individual-point jitter.
        smoothed_move = raw_move.copy()
        for _, indices in df.groupby("clean_track_id", sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            valid_weight = (support_count[positions] >= cfg.min_support_tracks).astype(float)
            numerator = gaussian_filter1d(
                raw_move[positions] * valid_weight,
                cfg.displacement_smooth_sigma_points,
                mode="nearest",
            )
            denominator = gaussian_filter1d(
                valid_weight,
                cfg.displacement_smooth_sigma_points,
                mode="nearest",
            )
            smoothed_move[positions] = np.where(
                denominator > 0.15,
                numerator / np.maximum(denominator, 1e-6),
                0.0,
            )

        # Key correction: strictly freeze points with insufficient support.
        # Do not allow smoothed displacements from adjacent multitrajectory points to propagate into single-trajectory sections.
        move_eligible = support_count >= cfg.min_support_tracks
        smoothed_move[~move_eligible] = 0.0
        max_support_seen = np.maximum(max_support_seen, support_count)
        ever_move_eligible |= move_eligible

        # Reduce movement at sharp bends and strictly constrain all movement to the normal direction.
        curvature_factor = np.clip(1.0 - curvature / 120.0, 0.30, 1.0)
        requested_move = np.clip(
            cfg.gravity_alpha * smoothed_move * curvature_factor,
            -cfg.max_step_m,
            cfg.max_step_m,
        )
        current_xy = query_xy
        normals = np.c_[-tangents[:, 1], tangents[:, 0]]
        proposed_xy = current_xy + requested_move[:, None] * normals

        total_displacement = proposed_xy - original_xy
        total_distance = np.linalg.norm(total_displacement, axis=1)
        total_limit = np.where(
            df["gravity_hairpin_protected"].to_numpy(bool),
            cfg.max_total_move_hairpin_m,
            cfg.max_total_move_m,
        )
        over_limit = total_distance > total_limit
        proposed_xy[over_limit] = (
            original_xy[over_limit]
            + total_displacement[over_limit]
            / total_distance[over_limit, None]
            * total_limit[over_limit, None]
        )

        step_distance = np.linalg.norm(proposed_xy - current_xy, axis=1)
        df["gravity_x_m"] = proposed_xy[:, 0]
        df["gravity_y_m"] = proposed_xy[:, 1]
        df["gravity_support_track_count"] = support_count
        df["gravity_cross_leg_rejected_count"] = cross_leg_rejected

        iteration_rows.append(
            {
                "iteration": iteration,
                "elapsed_s": time.time() - start,
                "moved_point_count": int(np.count_nonzero(step_distance > 1e-4)),
                "mean_step_move_m": float(step_distance.mean()),
                "median_step_move_m": float(np.median(step_distance)),
                "p95_step_move_m": float(np.percentile(step_distance, 95)),
                "max_step_move_m": float(step_distance.max(initial=0.0)),
                "support_ge_min_fraction": float(
                    np.mean(support_count >= cfg.min_support_tracks)
                ),
                "cross_leg_rejected_total": int(cross_leg_rejected.sum()),
            }
        )
        print(
            f"[GRAVITY {iteration}/{cfg.iterations}] "
            f"moved={iteration_rows[-1]['moved_point_count']:,}, "
            f"mean={iteration_rows[-1]['mean_step_move_m']:.3f} m, "
            f"p95={iteration_rows[-1]['p95_step_move_m']:.3f} m, "
            f"support={iteration_rows[-1]['support_ge_min_fraction']:.1%}"
        )

    final_xy = df[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
    df["gravity_displacement_m"] = np.linalg.norm(final_xy - original_xy, axis=1)
    df["gravity_max_support_track_count"] = max_support_seen
    df["gravity_ever_multitrack_supported"] = ever_move_eligible
    df["gravity_single_track_preserved"] = ~ever_move_eligible

    # Audit: points that never meet the multitrajectory-support threshold must remain exactly in place.
    frozen = ~ever_move_eligible
    if frozen.any():
        frozen_error = np.linalg.norm(final_xy[frozen] - original_xy[frozen], axis=1)
        if float(frozen_error.max(initial=0.0)) > 1e-8:
            raise AssertionError(
                f"""Single-track preservation failed: under-supported points moved; maximum displacement={float(frozen_error.max()):.9f} m"""
            )
    return df, pd.DataFrame(iteration_rows)


def evaluate_edges(
    df: pd.DataFrame,
    cfg: GravityConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    sample_xy, sample_tangent, sample_contribution, sample_hairpin = (
        resample_all_tracks(df, min(cfg.sample_spacing_m, 1.5))
    )
    tree = cKDTree(sample_xy)
    cosine_limit = math.cos(math.radians(cfg.edge_axis_angle_deg))
    rows: List[Dict[str, object]] = []
    reliable_by_track: Dict[str, np.ndarray] = {}

    for track_id, group in df.groupby("clean_track_id", sort=False):
        group = group.sort_values("clean_point_index")
        xy = group[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
        leg = group["gravity_hairpin_leg_code"].to_numpy(np.int8)
        contribution = group["_contribution_id"].iloc[0]
        reliable: List[bool] = []

        for edge_index in range(len(xy) - 1):
            start_xy, end_xy = xy[edge_index], xy[edge_index + 1]
            vector = end_xy - start_xy
            length = float(np.linalg.norm(vector))
            if length <= 1e-9:
                reliable.append(False)
                rows.append(
                    {
                        "clean_track_id": track_id,
                        "edge_index": edge_index,
                        "edge_length_m": length,
                        "supported_fraction": 0.0,
                        "median_other_track_count": 0.0,
                        "reliable": False,
                    }
                )
                continue

            tangent = vector / length
            distances_along = np.arange(0.0, length, cfg.edge_check_spacing_m)
            distances_along = np.r_[distances_along, length]
            edge_samples = start_xy + distances_along[:, None] * tangent
            distances, neighbor_ids = tree.query(
                edge_samples,
                k=min(40, max(16, len(sample_xy))),
                distance_upper_bound=cfg.edge_support_radius_m,
                workers=1,
            )
            if distances.ndim == 1:
                distances = distances[:, None]
                neighbor_ids = neighbor_ids[:, None]

            edge_leg = int(leg[edge_index]) if leg[edge_index] == leg[edge_index + 1] else 0
            other_track_counts: List[int] = []
            for row_index in range(len(edge_samples)):
                supported_tracks = set()
                for distance, sample_index in zip(
                    distances[row_index], neighbor_ids[row_index]
                ):
                    if not np.isfinite(distance) or sample_index >= len(sample_xy):
                        continue
                    if sample_contribution[sample_index] == contribution:
                        continue
                    if (
                        abs(float(np.dot(tangent, sample_tangent[sample_index])))
                        < cosine_limit
                    ):
                        continue
                    if edge_leg != 0 and int(sample_hairpin[sample_index]) != edge_leg:
                        continue
                    supported_tracks.add(str(sample_contribution[sample_index]))
                other_track_counts.append(len(supported_tracks))

            counts = np.asarray(other_track_counts, dtype=int)
            supported_fraction = float(
                np.mean(counts >= cfg.edge_min_other_tracks)
            )
            median_count = float(np.median(counts))
            edge_reliable = supported_fraction >= cfg.edge_min_supported_fraction
            if (
                length <= cfg.short_edge_keep_m
                and median_count >= cfg.short_edge_min_other_tracks
            ):
                edge_reliable = True

            reliable.append(edge_reliable)
            rows.append(
                {
                    "clean_track_id": track_id,
                    "edge_index": edge_index,
                    "edge_length_m": length,
                    "supported_fraction": supported_fraction,
                    "median_other_track_count": median_count,
                    "reliable": bool(edge_reliable),
                }
            )

        reliable_by_track[str(track_id)] = np.asarray(reliable, dtype=bool)

    return pd.DataFrame(rows), reliable_by_track


def split_reliable_parts(
    df: pd.DataFrame,
    reliable_by_track: Dict[str, np.ndarray],
    cfg: GravityConfig,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for track_id, group in df.groupby("clean_track_id", sort=False):
        group = group.sort_values("clean_point_index")
        reliable = reliable_by_track[str(track_id)]
        position = 0
        part_number = 0
        while position < len(reliable):
            while position < len(reliable) and not reliable[position]:
                position += 1
            if position >= len(reliable):
                break
            start = position
            while position < len(reliable) and reliable[position]:
                position += 1
            end = position  # edges start..end-1 -> points start..end
            part = group.iloc[start : end + 1].copy()
            xy = part[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            if len(part) < cfg.min_output_points or length < cfg.min_output_length_m:
                continue
            part_number += 1
            part["gravity_parent_clean_track_id"] = str(track_id)
            part["clean_track_id"] = f"{track_id}__grav{part_number:03d}"
            part["clean_point_index"] = np.arange(1, len(part) + 1)
            part["x_m"] = part["gravity_x_m"].to_numpy(float)
            part["y_m"] = part["gravity_y_m"].to_numpy(float)
            part["metric_crs"] = cfg.projected_crs
            parts.append(part)

    if not parts:
        raise RuntimeError('No valid track segments remain after support constraints')
    output = pd.concat(parts, ignore_index=True)
    return output



def attach_edge_diagnostics_to_points(
    df: pd.DataFrame,
    edge_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    'Attach edge-support diagnostics to endpoints without deleting points or edges.'
    output = df.copy()
    n = len(output)
    prev_reliable = np.ones(n, dtype=bool)
    next_reliable = np.ones(n, dtype=bool)
    prev_fraction = np.full(n, np.nan, dtype=float)
    next_fraction = np.full(n, np.nan, dtype=float)
    prev_count = np.full(n, np.nan, dtype=float)
    next_count = np.full(n, np.nan, dtype=float)

    row_position = {index: pos for pos, index in enumerate(output.index)}
    for track_id, group in output.groupby("clean_track_id", sort=False):
        ordered = group.sort_values("clean_point_index")
        positions = np.asarray([row_position[i] for i in ordered.index], dtype=int)
        diagnostics = edge_diagnostics.loc[
            edge_diagnostics["clean_track_id"].astype(str).eq(str(track_id))
        ].sort_values("edge_index")
        if len(positions) <= 1 or diagnostics.empty:
            continue
        for row in diagnostics.itertuples(index=False):
            edge_index = int(row.edge_index)
            if edge_index < 0 or edge_index + 1 >= len(positions):
                continue
            p0, p1 = positions[edge_index], positions[edge_index + 1]
            reliable = bool(row.reliable)
            fraction = float(row.supported_fraction)
            count = float(row.median_other_track_count)
            next_reliable[p0] = reliable
            prev_reliable[p1] = reliable
            next_fraction[p0] = fraction
            prev_fraction[p1] = fraction
            next_count[p0] = count
            prev_count[p1] = count

    output["gravity_prev_edge_multitrack_reliable"] = prev_reliable
    output["gravity_next_edge_multitrack_reliable"] = next_reliable
    output["gravity_prev_edge_supported_fraction"] = prev_fraction
    output["gravity_next_edge_supported_fraction"] = next_fraction
    output["gravity_prev_edge_median_other_tracks"] = prev_count
    output["gravity_next_edge_median_other_tracks"] = next_count
    output["gravity_incident_edge_any_low_support"] = ~(prev_reliable & next_reliable)
    output["gravity_preservation_policy"] = "preserve_all_input_edges"
    return output


def finalize_preserving_all(
    adjusted: pd.DataFrame,
    edge_diagnostics: pd.DataFrame,
    cfg: GravityConfig,
) -> pd.DataFrame:
    """
    Write gravity-adjusted coordinates to the standard coordinate fields, retaining all input points and original track topology.

    Preserve clean_track_id, clean_point_index, row count and adjacency.
    """
    final = attach_edge_diagnostics_to_points(adjusted, edge_diagnostics)
    final["x_m"] = final["gravity_x_m"].to_numpy(float)
    final["y_m"] = final["gravity_y_m"].to_numpy(float)
    final["metric_crs"] = cfg.projected_crs

    if {"longitude", "latitude"}.issubset(final.columns):
        transformer = Transformer.from_crs(cfg.projected_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(
            final["x_m"].to_numpy(float), final["y_m"].to_numpy(float)
        )
        final["longitude"] = np.asarray(lon, dtype=float)
        final["latitude"] = np.asarray(lat, dtype=float)

    # Never renumber or split clean_track_id.
    return final


def write_gpkg(
    original_df: pd.DataFrame,
    adjusted_df: pd.DataFrame,
    output_path: Path,
    crs: str,
) -> None:
    if gpd is None or LineString is None:
        print('[WARN] geopandas/shapely unavailable; skipping GPKG output', file=sys.stderr)
        return
    if output_path.exists():
        output_path.unlink()

    def line_frame(frame: pd.DataFrame, adjusted: bool):
        records = []
        group_field = "clean_track_id"
        for track_id, group in frame.groupby(group_field, sort=False):
            group = group.sort_values("clean_point_index")
            columns = ["x_m", "y_m"]
            xy = group[columns].to_numpy(float)
            if len(xy) < 2:
                continue
            records.append(
                {
                    "clean_track_id": str(track_id),
                    "source_file": str(group["source_file"].iloc[0]),
                    "point_count": int(len(group)),
                    "geometry": LineString(xy),
                }
            )
        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    line_frame(original_df, False).to_file(
        output_path, layer="original_tracks", driver="GPKG"
    )
    line_frame(adjusted_df, True).to_file(
        output_path, layer="adjusted_all_tracks", driver="GPKG"
    )


def run(input_path: Path, output_dir: Path, cfg: GravityConfig) -> Dict[str, object]:
    validate_config(cfg)
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if cfg.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    original = prepare_input(input_path, cfg)
    print(
        f"[INPUT] rows={len(original):,}, tracks={original.clean_track_id.nunique():,}, "
        f"independent={original._contribution_id.nunique():,}"
    )
    working = annotate_hairpins(original.copy(), cfg)
    adjusted, iteration_log = gravity_adjust(working, cfg)
    edge_diagnostics, reliable_by_track = evaluate_edges(adjusted, cfg)

    # Final output: retain all input points and all adjacent edges.
    final = finalize_preserving_all(adjusted, edge_diagnostics, cfg)

    # The high-confidence multitrajectory subset is diagnostic only and must never be the default KDE input.
    supported_subset = None
    if cfg.write_supported_subset_diagnostic:
        try:
            supported_subset = split_reliable_parts(adjusted, reliable_by_track, cfg)
            supported_subset.drop(columns=["_contribution_id"], errors="ignore").to_csv(
                diagnostics_dir / "multitrack_supported_subset_only.csv",
                index=False, encoding="utf-8-sig"
            )
        except RuntimeError:
            supported_subset = None

    output_csv = output_dir / "all_cleaned_points.csv"
    final.drop(columns=["_contribution_id"], errors="ignore").to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    iteration_log.to_csv(
        diagnostics_dir / "gravity_iteration_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edge_diagnostics.to_csv(
        diagnostics_dir / "edge_support_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjusted.drop(columns=["_contribution_id"], errors="ignore").to_csv(
        diagnostics_dir / "all_adjusted_points_preserved.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if cfg.write_gpkg:
        write_gpkg(
            original,
            final,
            output_dir / "gravity_adjusted_tracks.gpkg",
            cfg.projected_crs,
        )

    removed_edges = edge_diagnostics.loc[~edge_diagnostics["reliable"]]

    if len(final) != len(original):
        raise AssertionError(f"""Point preservation failed: input {len(original)}, output {len(final)}""")
    if final["clean_track_id"].nunique() != original["clean_track_id"].nunique():
        raise AssertionError('Track-segment preservation failed: clean_track_id count changed')

    report = {
        "input": str(input_path),
        "output": str(output_csv),
        "input_rows": int(len(original)),
        "input_tracks": int(original["clean_track_id"].nunique()),
        "independent_trajectories": int(original["_contribution_id"].nunique()),
        "hairpin_protected_points": int(working["gravity_hairpin_protected"].sum()),
        "hairpin_protected_tracks": int(
            working.loc[working["gravity_hairpin_protected"], "clean_track_id"].nunique()
        ),
        "mean_total_move_m": float(adjusted["gravity_displacement_m"].mean()),
        "median_total_move_m": float(adjusted["gravity_displacement_m"].median()),
        "p95_total_move_m": float(adjusted["gravity_displacement_m"].quantile(0.95)),
        "max_total_move_m": float(adjusted["gravity_displacement_m"].max()),
        "all_edges": int(len(edge_diagnostics)),
        # Legacy-field compatibility: final output no longer removes any edges.
        "removed_edges": 0,
        "removed_edge_length_m": 0.0,
        "diagnostic_low_support_edges": int(len(removed_edges)),
        "diagnostic_low_support_edge_length_m": float(removed_edges["edge_length_m"].sum()),
        "single_track_preserved_points": int(adjusted["gravity_single_track_preserved"].sum()),
        "output_rows": int(len(final)),
        "output_tracks": int(final["clean_track_id"].nunique()),
        "row_retention_ratio": float(len(final) / max(len(original), 1)),
        "track_retention_ratio": float(final["clean_track_id"].nunique() / max(original["clean_track_id"].nunique(), 1)),
        "supported_subset_rows_diagnostic": int(len(supported_subset)) if supported_subset is not None else 0,
        "config": asdict(cfg),
    }
    (output_dir / "gravity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] output_rows={len(final):,}/{len(original):,}, "
        f"output_tracks={final.clean_track_id.nunique():,}/{original.clean_track_id.nunique():,}, "
        f"deleted_edges=0, diagnostic_low_support_edges={len(removed_edges):,}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Topology-preserving robust gravity adjustment retaining all single-track segments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--projected-crs", default="EPSG:32647")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--sample-spacing-m", type=float, default=2.0)
    parser.add_argument("--search-radius-m", type=float, default=15.0)
    parser.add_argument("--min-support-tracks", type=int, default=2)
    parser.add_argument("--edge-support-radius-m", type=float, default=8.0)
    parser.add_argument("--edge-min-other-tracks", type=int, default=2)
    parser.add_argument("--edge-min-supported-fraction", type=float, default=0.45)
    parser.add_argument("--no-gpkg", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = GravityConfig(
        projected_crs=args.projected_crs,
        sample_spacing_m=args.sample_spacing_m,
        iterations=args.iterations,
        search_radius_m=args.search_radius_m,
        min_support_tracks=args.min_support_tracks,
        edge_support_radius_m=args.edge_support_radius_m,
        edge_min_other_tracks=args.edge_min_other_tracks,
        edge_min_supported_fraction=args.edge_min_supported_fraction,
        write_gpkg=not args.no_gpkg,
    )
    try:
        run(args.input, args.output, cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
