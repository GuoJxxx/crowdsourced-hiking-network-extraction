#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多轨迹 U 形弯检测、聚类与共识中心线构建工具。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


@dataclass
class UTurnConfig:
    resample_spacing_m: float = 2.0
    window_lengths_m: Tuple[float, ...] = (40, 50, 60, 70, 80, 90, 100, 110, 120, 140)
    min_heading_reversal_deg: float = 115.0
    min_signed_turn_deg: float = 110.0
    max_path_efficiency: float = 0.68
    min_turn_consistency: float = 0.45
    max_chord_m: float = 45.0
    min_bulge_m: float = 5.0
    event_overlap_ratio: float = 0.45
    event_center_separation_m: float = 35.0

    cluster_apex_radius_m: float = 30.0
    cluster_shape_rms_m: float = 40.0
    min_cluster_tracks: int = 3
    max_cluster_apex_p90_m: float = 28.0
    max_cluster_median_shape_rms_m: float = 20.0
    min_cluster_mean_score: float = 0.78
    max_cluster_median_efficiency: float = 0.45
    min_cluster_median_length_m: float = 50.0

    consensus_points: int = 101
    consensus_trim_quantile: float = 0.85
    consensus_smooth_sigma_points: float = 1.5
    apex_core_fraction: float = 0.06

    anchor_alpha: float = 0.90
    anchor_max_move_m: float = 18.0
    anchor_taper_fraction: float = 0.15
    protection_buffer_m: float = 12.0

    # v0.5 raw-track structure validation. First-stage heading reversal is only
    # a candidate generator; protection requires two parallel/opposite legs and
    # an apex-concentrated turn supported by distinct original trajectories.
    min_distinct_original_tracks: int = 3
    max_leg_axis_difference_deg: float = 25.0
    min_opposite_direction_score: float = 0.70
    min_apex_turn_concentration: float = 0.50
    min_leg_separation_m: float = 4.0
    max_leg_separation_m: float = 35.0
    leg_fraction: float = 0.28

    # v0.5: require a genuine single-apex hairpin, not merely a large reversal.
    min_leg_straightness: float = 0.86
    min_leg_longitudinal_overlap_ratio: float = 0.45
    max_secondary_turn_peak_ratio: float = 0.60
    activity_min_distinct_original_tracks: int = 4
    activity_min_apex_turn_concentration: float = 0.68
    activity_max_secondary_turn_peak_ratio: float = 0.50
    activity_min_leg_straightness: float = 0.90
    activity_min_leg_overlap_ratio: float = 0.55


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
    tangent[0] = xy[1] - xy[0]
    tangent[-1] = xy[-1] - xy[-2]
    if n > 2:
        tangent[1:-1] = xy[2:] - xy[:-2]
    lengths = np.linalg.norm(tangent, axis=1)
    valid = lengths > 1e-9
    tangent[valid] /= lengths[valid, None]
    return tangent


def circular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def point_line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    delta = points - start
    cross = np.abs(vector[0] * delta[:, 1] - vector[1] * delta[:, 0])
    return cross / length


def resample_curve(xy: np.ndarray, sample_count: int) -> np.ndarray:
    distance = cumulative_distance(xy)
    if len(xy) < 2 or distance[-1] <= 1e-9:
        return np.repeat(xy[:1], sample_count, axis=0)
    samples = np.linspace(0.0, distance[-1], sample_count)
    return np.c_[
        np.interp(samples, distance, xy[:, 0]),
        np.interp(samples, distance, xy[:, 1]),
    ]


def orientation_invariant_rms(a: np.ndarray, b: np.ndarray) -> Tuple[float, bool]:
    direct = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    reverse = float(np.sqrt(np.mean(np.sum((a[::-1] - b) ** 2, axis=1))))
    return (reverse, True) if reverse < direct else (direct, False)


def detect_track_events(
    track_id: str,
    group: pd.DataFrame,
    cfg: UTurnConfig,
) -> List[Dict[str, object]]:
    group = group.sort_values("clean_point_index")
    xy = group[["x_m", "y_m"]].to_numpy(float)
    original_distance = cumulative_distance(xy)
    if len(xy) < 5 or original_distance[-1] < min(cfg.window_lengths_m):
        return []

    samples = np.arange(0.0, original_distance[-1], cfg.resample_spacing_m)
    if samples.size == 0 or samples[-1] < original_distance[-1] - 1e-9:
        samples = np.r_[samples, original_distance[-1]]
    resampled = np.c_[
        np.interp(samples, original_distance, xy[:, 0]),
        np.interp(samples, original_distance, xy[:, 1]),
    ]
    tangents = unit_tangents(resampled)
    headings = np.degrees(np.arctan2(tangents[:, 1], tangents[:, 0]))
    headings_unwrapped = np.degrees(np.unwrap(np.radians(headings)))
    absolute_heading_change = np.abs(np.diff(headings_unwrapped))

    candidates: List[Dict[str, object]] = []
    for start_index in range(max(0, len(samples) - 8)):
        for window_length in cfg.window_lengths_m:
            end_index = int(np.searchsorted(samples, samples[start_index] + window_length))
            if end_index >= len(samples) or end_index - start_index < 8:
                continue
            length_m = float(samples[end_index] - samples[start_index])
            chord_m = float(np.linalg.norm(resampled[end_index] - resampled[start_index]))
            efficiency = chord_m / max(length_m, 1e-9)
            signed_turn = float(headings_unwrapped[end_index] - headings_unwrapped[start_index])
            total_turn = float(absolute_heading_change[start_index:end_index].sum())
            consistency = abs(signed_turn) / max(total_turn, 1e-9)
            reversal = circular_difference_deg(headings[end_index], headings[start_index])
            if reversal < cfg.min_heading_reversal_deg:
                continue
            if abs(signed_turn) < cfg.min_signed_turn_deg:
                continue
            if efficiency > cfg.max_path_efficiency:
                continue
            if consistency < cfg.min_turn_consistency:
                continue
            if chord_m > cfg.max_chord_m:
                continue

            curve = resampled[start_index : end_index + 1]
            distances_to_chord = point_line_distance(
                curve, resampled[start_index], resampled[end_index]
            )
            local_apex = int(np.argmax(distances_to_chord))
            bulge_m = float(distances_to_chord[local_apex])
            if bulge_m < cfg.min_bulge_m:
                continue
            apex_index = start_index + local_apex

            score = (
                0.32 * reversal / 180.0
                + 0.18 * min(abs(signed_turn), 220.0) / 220.0
                + 0.20 * consistency
                + 0.18 * (1.0 - efficiency)
                + 0.12 * min(bulge_m / 25.0, 1.0)
            )
            candidates.append(
                {
                    "track_id": str(track_id),
                    "start_distance_m": float(samples[start_index]),
                    "end_distance_m": float(samples[end_index]),
                    "length_m": length_m,
                    "chord_m": chord_m,
                    "path_efficiency": efficiency,
                    "signed_turn_deg": signed_turn,
                    "heading_reversal_deg": reversal,
                    "turn_consistency": consistency,
                    "bulge_m": bulge_m,
                    "score": float(score),
                    "apex_x": float(resampled[apex_index, 0]),
                    "apex_y": float(resampled[apex_index, 1]),
                    "curve": curve.copy(),
                    "curve_fixed": resample_curve(curve, cfg.consensus_points),
                }
            )

    kept: List[Dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda value: float(value["score"]), reverse=True):
        duplicate = False
        for existing in kept:
            overlap = max(
                0.0,
                min(float(candidate["end_distance_m"]), float(existing["end_distance_m"]))
                - max(float(candidate["start_distance_m"]), float(existing["start_distance_m"])),
            )
            minimum_length = max(
                min(float(candidate["length_m"]), float(existing["length_m"])), 1e-9
            )
            center_distance = abs(
                (float(candidate["start_distance_m"]) + float(candidate["end_distance_m"])) * 0.5
                - (float(existing["start_distance_m"]) + float(existing["end_distance_m"])) * 0.5
            )
            if (
                overlap / minimum_length >= cfg.event_overlap_ratio
                or center_distance < cfg.event_center_separation_m
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_events(df: pd.DataFrame, cfg: UTurnConfig) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for track_id, group in df.groupby("clean_track_id", sort=False):
        # v0.5 activity backbones are derived continuity scaffolds.  They must never
        # count as independent original trajectories or create U-turn protection.
        if "activity_backbone_synthetic" in group.columns:
            flag = group["activity_backbone_synthetic"]
            if not pd.api.types.is_bool_dtype(flag):
                flag = flag.fillna("").astype(str).str.lower().isin({"1", "true", "yes", "y"})
            if bool(flag.any()):
                continue

        ordered = group.sort_values("clean_point_index", kind="mergesort")
        local = detect_track_events(str(track_id), ordered, cfg)
        if "source_clean_track_id" in ordered.columns and ordered["source_clean_track_id"].notna().any():
            contribution = str(ordered["source_clean_track_id"].dropna().astype(str).iloc[0])
        elif "_contribution_id" in ordered.columns and ordered["_contribution_id"].notna().any():
            contribution = str(ordered["_contribution_id"].dropna().astype(str).iloc[0])
        else:
            contribution = str(track_id)

        xy = ordered[["x_m", "y_m"]].to_numpy(float)
        dist = cumulative_distance(xy)
        if "activity_through_corridor" in ordered.columns:
            through = ordered["activity_through_corridor"]
            if not pd.api.types.is_bool_dtype(through):
                through = through.fillna("").astype(str).str.lower().isin({"1", "true", "yes", "y"})
            through_arr = through.to_numpy(bool)
        else:
            through_arr = np.zeros(len(ordered), dtype=bool)
        if "activity_zone_id" in ordered.columns:
            zone_arr = pd.to_numeric(ordered["activity_zone_id"], errors="coerce").fillna(0).to_numpy(int) > 0
        else:
            zone_arr = np.zeros(len(ordered), dtype=bool)

        for event in local:
            lo = float(event["start_distance_m"]); hi = float(event["end_distance_m"])
            sel = (dist >= lo - 1e-9) & (dist <= hi + 1e-9)
            event["original_contribution_id"] = contribution
            event["activity_through_corridor_fraction"] = float(through_arr[sel].mean()) if sel.any() else 0.0
            event["activity_zone_fraction"] = float(zone_arr[sel].mean()) if sel.any() else 0.0
        events.extend(local)
    for event_id, event in enumerate(events, start=1):
        event["event_id"] = event_id
    return events


def build_consensus(events: List[Dict[str, object]], cfg: UTurnConfig) -> Dict[str, object]:
    curves = [np.asarray(event["curve_fixed"], dtype=float) for event in events]
    count = len(curves)
    pairwise = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distance, _ = orientation_invariant_rms(curves[i], curves[j])
            pairwise[i, j] = pairwise[j, i] = distance
    medoid_index = int(np.argmin(pairwise.sum(axis=1)))
    medoid = curves[medoid_index]

    aligned: List[np.ndarray] = []
    reverse_flags: List[bool] = []
    for curve in curves:
        _, reverse = orientation_invariant_rms(curve, medoid)
        aligned.append(curve[::-1].copy() if reverse else curve.copy())
        reverse_flags.append(reverse)
    stack = np.stack(aligned)
    preliminary = np.median(stack, axis=0)
    curve_rms = np.sqrt(np.mean(np.sum((stack - preliminary[None, :, :]) ** 2, axis=2), axis=1))
    threshold = float(np.quantile(curve_rms, cfg.consensus_trim_quantile))
    keep = curve_rms <= threshold + 1e-9
    if int(keep.sum()) < max(3, math.ceil(count * 0.5)):
        keep[:] = True
    consensus = np.median(stack[keep], axis=0)
    consensus[:, 0] = gaussian_filter1d(
        consensus[:, 0], cfg.consensus_smooth_sigma_points, mode="nearest"
    )
    consensus[:, 1] = gaussian_filter1d(
        consensus[:, 1], cfg.consensus_smooth_sigma_points, mode="nearest"
    )
    distance_to_chord = point_line_distance(consensus, consensus[0], consensus[-1])
    apex_index = int(np.argmax(distance_to_chord))
    apex_fraction = apex_index / max(len(consensus) - 1, 1)

    return {
        "events": events,
        "consensus": consensus,
        "medoid_index": medoid_index,
        "reverse_flags": reverse_flags,
        "keep_flags": keep,
        "curve_rms_m": curve_rms,
        "median_curve_rms_m": float(np.median(curve_rms)),
        "p90_curve_rms_m": float(np.percentile(curve_rms, 90)),
        "apex_fraction": float(apex_fraction),
        "consensus_length_m": float(cumulative_distance(consensus)[-1]),
        "consensus_chord_m": float(np.linalg.norm(consensus[-1] - consensus[0])),
        "consensus_bulge_m": float(distance_to_chord[apex_index]),
    }


def _axis_difference_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    aa = math.degrees(math.atan2(float(a[1]), float(a[0]))) % 180.0
    bb = math.degrees(math.atan2(float(b[1]), float(b[0]))) % 180.0
    d = abs(aa - bb) % 180.0
    return float(min(d, 180.0 - d))


def _directed_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _curve_straightness(xy: np.ndarray) -> float:
    arr = np.asarray(xy, dtype=float)
    if len(arr) < 2:
        return 0.0
    path = float(cumulative_distance(arr)[-1])
    chord = float(np.linalg.norm(arr[-1] - arr[0]))
    return float(np.clip(chord / max(path, 1e-9), 0.0, 1.0))


def _longitudinal_overlap_ratio(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n <= 1e-9:
        return 0.0
    u = axis / n
    pa = np.asarray(a, dtype=float) @ u
    pb = np.asarray(b, dtype=float) @ u
    a0, a1 = float(np.min(pa)), float(np.max(pa))
    b0, b1 = float(np.min(pb)), float(np.max(pb))
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-9, min(a1 - a0, b1 - b0))
    return float(np.clip(overlap / denom, 0.0, 1.0))


def validate_uturn_structure(
    events: List[Dict[str, object]],
    consensus: np.ndarray,
    cfg: UTurnConfig,
) -> Dict[str, object]:
    """Validate a physical single-apex hairpin rather than any reversing curve."""
    curve = np.asarray(consensus, dtype=float)
    contributions = {
        str(e.get("original_contribution_id", e.get("track_id", ""))) for e in events
        if not str(e.get("original_contribution_id", e.get("track_id", ""))).startswith("activity_backbone_")
    }
    distinct = len({c for c in contributions if c})
    base = {
        "structure_valid": False,
        "distinct_original_track_support": int(distinct),
        "leg_axis_difference_deg": float("inf"),
        "opposite_direction_score": 0.0,
        "apex_turn_concentration": 0.0,
        "leg_separation_m": float("nan"),
        "start_leg_straightness": 0.0,
        "end_leg_straightness": 0.0,
        "leg_longitudinal_overlap_ratio": 0.0,
        "secondary_turn_peak_ratio": float("inf"),
        "activity_zone_fraction": 0.0,
        "reject_reason": "",
    }
    if distinct < int(cfg.min_distinct_original_tracks):
        base["reject_reason"] = "insufficient_distinct_original_tracks"
        return base
    if len(curve) < 9:
        base["reject_reason"] = "insufficient_curve_points"
        return base

    n = len(curve)
    leg_n = max(4, min(n // 3, int(round(n * float(cfg.leg_fraction)))))
    start_leg = curve[:leg_n]
    end_leg = curve[-leg_n:]
    start_vec = start_leg[-1] - start_leg[0]
    end_vec = end_leg[-1] - end_leg[0]
    if float(np.linalg.norm(start_vec)) <= 1e-9 or float(np.linalg.norm(end_vec)) <= 1e-9:
        base["reject_reason"] = "degenerate_leg"
        return base

    axis_diff = _axis_difference_from_vectors(start_vec, end_vec)
    directed = _directed_angle_deg(start_vec, end_vec)
    opposite_score = float(np.clip(1.0 - abs(180.0 - directed) / 180.0, 0.0, 1.0))
    start_straight = _curve_straightness(start_leg)
    end_straight = _curve_straightness(end_leg)
    common_axis = start_vec - end_vec  # end leg travels in the opposite direction
    overlap_ratio = _longitudinal_overlap_ratio(start_leg, end_leg, common_axis)

    m = max(5, leg_n)
    a = resample_curve(start_leg, m)
    b = resample_curve(end_leg[::-1], m)
    separation = np.linalg.norm(a - b, axis=1)
    leg_sep = float(np.median(separation))

    tang = unit_tangents(curve)
    headings = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    turn_mass = np.abs(np.diff(headings))
    total_turn = float(turn_mass.sum())
    distances_to_chord = point_line_distance(curve, curve[0], curve[-1])
    apex_index = int(np.argmax(distances_to_chord))
    # A true hairpin has one concentrated turn core.  A loop/roundabout distributes
    # comparable curvature around the ring, producing a large secondary peak.
    half = max(3, int(round(n * 0.14)))
    lo = max(0, apex_index - half)
    hi = min(len(turn_mass), apex_index + half)
    apex_mass = float(turn_mass[lo:hi].sum())
    concentration = apex_mass / max(total_turn, 1e-9)
    smooth_turn = gaussian_filter1d(turn_mass, sigma=max(0.8, n / 120.0), mode="nearest") if len(turn_mass) else turn_mass
    if len(smooth_turn):
        apex_lo = max(0, lo); apex_hi = min(len(smooth_turn), hi)
        apex_peak = float(np.max(smooth_turn[apex_lo:apex_hi])) if apex_hi > apex_lo else float(np.max(smooth_turn))
        outside = np.r_[smooth_turn[:apex_lo], smooth_turn[apex_hi:]]
        secondary_peak = float(np.max(outside)) if len(outside) else 0.0
        secondary_ratio = secondary_peak / max(apex_peak, 1e-9)
    else:
        secondary_ratio = float("inf")

    activity_fraction = float(np.median([
        float(e.get("activity_zone_fraction", 0.0) or 0.0) for e in events
    ])) if events else 0.0

    base.update({
        "leg_axis_difference_deg": float(axis_diff),
        "opposite_direction_score": float(opposite_score),
        "apex_turn_concentration": float(concentration),
        "leg_separation_m": float(leg_sep),
        "start_leg_straightness": float(start_straight),
        "end_leg_straightness": float(end_straight),
        "leg_longitudinal_overlap_ratio": float(overlap_ratio),
        "secondary_turn_peak_ratio": float(secondary_ratio),
        "activity_zone_fraction": float(activity_fraction),
        "total_turn_deg": float(math.degrees(total_turn)),
    })

    # Activity zones are intrinsically noisy; require stronger structure *before*
    # ordinary checks so the diagnostic clearly reports this semantic veto.
    if activity_fraction >= 0.50:
        if (
            distinct < int(cfg.activity_min_distinct_original_tracks)
            or concentration < float(cfg.activity_min_apex_turn_concentration)
            or secondary_ratio > float(cfg.activity_max_secondary_turn_peak_ratio)
            or min(start_straight, end_straight) < float(cfg.activity_min_leg_straightness)
            or overlap_ratio < float(cfg.activity_min_leg_overlap_ratio)
        ):
            base["reject_reason"] = "activity_zone_structure_too_weak"
            return base

    if axis_diff > float(cfg.max_leg_axis_difference_deg):
        base["reject_reason"] = "legs_not_parallel"; return base
    if opposite_score < float(cfg.min_opposite_direction_score):
        base["reject_reason"] = "legs_not_opposite"; return base
    if min(start_straight, end_straight) < float(cfg.min_leg_straightness):
        base["reject_reason"] = "leg_not_straight"; return base
    if overlap_ratio < float(cfg.min_leg_longitudinal_overlap_ratio):
        base["reject_reason"] = "insufficient_leg_overlap"; return base
    if not (float(cfg.min_leg_separation_m) <= leg_sep <= float(cfg.max_leg_separation_m)):
        base["reject_reason"] = "leg_separation_out_of_range"; return base
    if concentration < float(cfg.min_apex_turn_concentration):
        base["reject_reason"] = "turn_not_apex_concentrated"; return base
    if secondary_ratio > float(cfg.max_secondary_turn_peak_ratio):
        base["reject_reason"] = "turn_not_single_apex"; return base

    corridor_fraction = float(np.median([
        float(e.get("activity_through_corridor_fraction", 0.0) or 0.0) for e in events
    ])) if events else 0.0
    if corridor_fraction >= 0.75 and concentration < max(0.68, float(cfg.min_apex_turn_concentration)):
        base["reject_reason"] = "activity_through_corridor_conflict"; return base
    base["structure_valid"] = True
    base["reject_reason"] = ""
    return base


def cluster_events(events: List[Dict[str, object]], cfg: UTurnConfig, return_rejected: bool = False):
    if not events:
        return ([], []) if return_rejected else []
    ordered = sorted(events, key=lambda value: float(value["score"]), reverse=True)
    clusters: List[List[Dict[str, object]]] = []

    for event in ordered:
        best_cluster = None
        best_score = math.inf
        event_curve = np.asarray(event["curve_fixed"], dtype=float)
        for cluster_index, members in enumerate(clusters):
            medoid_event = members[0]
            apex_distance = math.hypot(
                float(event["apex_x"]) - float(medoid_event["apex_x"]),
                float(event["apex_y"]) - float(medoid_event["apex_y"]),
            )
            if apex_distance > cfg.cluster_apex_radius_m:
                continue
            shape_distance, _ = orientation_invariant_rms(
                event_curve, np.asarray(medoid_event["curve_fixed"], dtype=float)
            )
            if shape_distance > cfg.cluster_shape_rms_m:
                continue
            score = apex_distance + 0.35 * shape_distance
            if score < best_score:
                best_score = score
                best_cluster = cluster_index
        if best_cluster is None:
            clusters.append([event])
        else:
            clusters[best_cluster].append(event)

    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for members in clusters:
        # One original contribution can be split into several clean_track_id parts;
        # structural support must count it once.
        best_by_track: Dict[str, Dict[str, object]] = {}
        for event in members:
            track_id = str(event.get("original_contribution_id", event["track_id"]))
            if track_id not in best_by_track or float(event["score"]) > float(best_by_track[track_id]["score"]):
                best_by_track[track_id] = event
        unique_events = list(best_by_track.values())
        if len(unique_events) < cfg.min_cluster_tracks:
            continue
        consensus_info = build_consensus(unique_events, cfg)
        apex_coordinates = np.asarray(
            [[float(event["apex_x"]), float(event["apex_y"])] for event in unique_events],
            dtype=float,
        )
        apex_center = np.median(apex_coordinates, axis=0)
        apex_spread = np.linalg.norm(apex_coordinates - apex_center, axis=1)
        mean_score = float(np.mean([float(event["score"]) for event in unique_events]))
        median_efficiency = float(np.median([float(event["path_efficiency"]) for event in unique_events]))
        median_length = float(np.median([float(event["length_m"]) for event in unique_events]))
        if float(np.percentile(apex_spread, 90)) > cfg.max_cluster_apex_p90_m:
            continue
        if float(consensus_info["median_curve_rms_m"]) > cfg.max_cluster_median_shape_rms_m:
            continue
        if mean_score < cfg.min_cluster_mean_score:
            continue
        if median_efficiency > cfg.max_cluster_median_efficiency:
            continue
        if median_length < cfg.min_cluster_median_length_m:
            continue

        structure = validate_uturn_structure(unique_events, np.asarray(consensus_info["consensus"], dtype=float), cfg)
        if not bool(structure["structure_valid"]):
            rejected.append({
                **structure,
                "track_support": int(len(unique_events)),
                "apex_x": float(apex_center[0]),
                "apex_y": float(apex_center[1]),
                "mean_score": mean_score,
                "median_path_efficiency": median_efficiency,
                "median_event_length_m": median_length,
            })
            continue

        accepted.append(
            {
                **consensus_info,
                **structure,
                "track_support": int(len(unique_events)),
                "apex_x": float(apex_center[0]),
                "apex_y": float(apex_center[1]),
                "apex_spread_p90_m": float(np.percentile(apex_spread, 90)),
                "mean_score": mean_score,
                "median_path_efficiency": median_efficiency,
                "median_event_length_m": median_length,
            }
        )

    accepted.sort(key=lambda value: (-int(value["track_support"]), -float(value["mean_score"])))
    for cluster_id, cluster in enumerate(accepted, start=1):
        cluster["cluster_id"] = cluster_id
        for event, reverse in zip(cluster["events"], cluster["reverse_flags"]):
            event["cluster_id"] = cluster_id
            event["reverse_to_consensus"] = bool(reverse)
    return (accepted, rejected) if return_rejected else accepted


def detect_and_cluster(df: pd.DataFrame, cfg: UTurnConfig) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    events = detect_events(df, cfg)
    clusters = cluster_events(events, cfg)
    return events, clusters


def annotate_points(
    df: pd.DataFrame,
    clusters: List[Dict[str, object]],
    cfg: UTurnConfig,
) -> pd.DataFrame:
    result = df.copy()
    result["gravity_uturn_cluster_id"] = 0
    result["gravity_uturn_track_support"] = 0
    result["gravity_uturn_normalized_position"] = np.nan
    result["gravity_uturn_consensus_x_m"] = np.nan
    result["gravity_uturn_consensus_y_m"] = np.nan
    result["gravity_uturn_leg_code"] = 0
    result["gravity_uturn_anchor_weight"] = 0.0

    grouped = {
        str(track_id): group.sort_values("clean_point_index")
        for track_id, group in result.groupby("clean_track_id", sort=False)
    }
    for cluster in clusters:
        consensus = np.asarray(cluster["consensus"], dtype=float)
        sample_count = len(consensus)
        apex_fraction = float(cluster["apex_fraction"])
        cluster_id = int(cluster["cluster_id"])
        support = int(cluster["track_support"])
        for event in cluster["events"]:
            track_id = str(event["track_id"])
            if track_id not in grouped:
                continue
            group = grouped[track_id]
            positions = group.index.to_numpy(int)
            xy = group[["x_m", "y_m"]].to_numpy(float)
            distance = cumulative_distance(xy)
            start_distance = float(event["start_distance_m"])
            end_distance = float(event["end_distance_m"])
            selected = (distance >= start_distance) & (distance <= end_distance)
            if not selected.any():
                continue
            local_t = (distance[selected] - start_distance) / max(end_distance - start_distance, 1e-9)
            consensus_t = 1.0 - local_t if bool(event.get("reverse_to_consensus", False)) else local_t
            position = np.clip(consensus_t, 0.0, 1.0) * (sample_count - 1)
            lower = np.floor(position).astype(int)
            upper = np.minimum(lower + 1, sample_count - 1)
            fraction = position - lower
            target = (
                consensus[lower] * (1.0 - fraction[:, None])
                + consensus[upper] * fraction[:, None]
            )
            selected_positions = positions[selected]
            result.loc[selected_positions, "gravity_uturn_cluster_id"] = cluster_id
            result.loc[selected_positions, "gravity_uturn_track_support"] = support
            result.loc[selected_positions, "gravity_uturn_normalized_position"] = consensus_t
            result.loc[selected_positions, "gravity_uturn_consensus_x_m"] = target[:, 0]
            result.loc[selected_positions, "gravity_uturn_consensus_y_m"] = target[:, 1]

            core_half = cfg.apex_core_fraction
            leg_codes = np.where(
                consensus_t < apex_fraction - core_half,
                -1,
                np.where(consensus_t > apex_fraction + core_half, 1, 2),
            )
            result.loc[selected_positions, "gravity_uturn_leg_code"] = leg_codes.astype(int)

    return result


def apply_consensus_anchor(df: pd.DataFrame, cfg: UTurnConfig) -> pd.DataFrame:
    result = df.copy()
    if "gravity_x_m" not in result.columns:
        result["gravity_x_m"] = result["x_m"].to_numpy(float)
        result["gravity_y_m"] = result["y_m"].to_numpy(float)
    mask = result["gravity_uturn_cluster_id"].to_numpy(int) > 0
    if not mask.any():
        return result

    current = result.loc[mask, ["gravity_x_m", "gravity_y_m"]].to_numpy(float)
    target = result.loc[mask, ["gravity_uturn_consensus_x_m", "gravity_uturn_consensus_y_m"]].to_numpy(float)
    normalized = result.loc[mask, "gravity_uturn_normalized_position"].to_numpy(float)
    taper = np.minimum.reduce(
        [
            np.ones_like(normalized),
            normalized / max(cfg.anchor_taper_fraction, 1e-6),
            (1.0 - normalized) / max(cfg.anchor_taper_fraction, 1e-6),
        ]
    )
    taper = np.clip(taper, 0.0, 1.0)
    weight = cfg.anchor_alpha * taper
    delta = (target - current) * weight[:, None]
    distance = np.linalg.norm(delta, axis=1)
    over = distance > cfg.anchor_max_move_m
    if over.any():
        delta[over] *= (cfg.anchor_max_move_m / distance[over])[:, None]
    adjusted = current + delta
    result.loc[mask, "gravity_x_m"] = adjusted[:, 0]
    result.loc[mask, "gravity_y_m"] = adjusted[:, 1]
    result.loc[mask, "gravity_uturn_anchor_weight"] = weight
    return result


def events_table(events: List[Dict[str, object]]) -> pd.DataFrame:
    fields = [
        "event_id", "cluster_id", "track_id", "original_contribution_id", "activity_through_corridor_fraction", "activity_zone_fraction", "start_distance_m", "end_distance_m",
        "length_m", "chord_m", "path_efficiency", "signed_turn_deg",
        "heading_reversal_deg", "turn_consistency", "bulge_m", "score",
        "apex_x", "apex_y", "reverse_to_consensus",
    ]
    rows = []
    for event in events:
        rows.append({field: event.get(field, 0 if field == "cluster_id" else None) for field in fields})
    return pd.DataFrame(rows)


def clusters_table(clusters: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        rows.append(
            {
                "cluster_id": int(cluster["cluster_id"]),
                "track_support": int(cluster["track_support"]),
                "apex_x": float(cluster["apex_x"]),
                "apex_y": float(cluster["apex_y"]),
                "apex_spread_p90_m": float(cluster["apex_spread_p90_m"]),
                "mean_score": float(cluster["mean_score"]),
                "median_path_efficiency": float(cluster["median_path_efficiency"]),
                "median_event_length_m": float(cluster["median_event_length_m"]),
                "median_curve_rms_m": float(cluster["median_curve_rms_m"]),
                "p90_curve_rms_m": float(cluster["p90_curve_rms_m"]),
                "consensus_length_m": float(cluster["consensus_length_m"]),
                "consensus_chord_m": float(cluster["consensus_chord_m"]),
                "consensus_bulge_m": float(cluster["consensus_bulge_m"]),
                "apex_fraction": float(cluster["apex_fraction"]),
                "structure_valid": bool(cluster.get("structure_valid", True)),
                "distinct_original_track_support": int(cluster.get("distinct_original_track_support", cluster["track_support"])),
                "leg_axis_difference_deg": float(cluster.get("leg_axis_difference_deg", float("nan"))),
                "opposite_direction_score": float(cluster.get("opposite_direction_score", float("nan"))),
                "apex_turn_concentration": float(cluster.get("apex_turn_concentration", float("nan"))),
                "leg_separation_m": float(cluster.get("leg_separation_m", float("nan"))),
                "start_leg_straightness": float(cluster.get("start_leg_straightness", float("nan"))),
                "end_leg_straightness": float(cluster.get("end_leg_straightness", float("nan"))),
                "leg_longitudinal_overlap_ratio": float(cluster.get("leg_longitudinal_overlap_ratio", float("nan"))),
                "secondary_turn_peak_ratio": float(cluster.get("secondary_turn_peak_ratio", float("nan"))),
                "activity_zone_fraction": float(cluster.get("activity_zone_fraction", 0.0)),
                "reject_reason": str(cluster.get("reject_reason", "")),
            }
        )
    return pd.DataFrame(rows)


def config_dict(cfg: UTurnConfig) -> Dict[str, object]:
    value = asdict(cfg)
    value["window_lengths_m"] = list(cfg.window_lengths_m)
    return value
