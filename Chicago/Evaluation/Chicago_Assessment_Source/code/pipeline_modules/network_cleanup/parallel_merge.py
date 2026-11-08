from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString

from .models import NetworkData, clean_linestring, compute_degrees, reindex_network
from .optimizer import edge_confidence
from .parallel import detect_parallel_edges
from .tile_io import RasterBlockSampler


def _resample_normalized(geom: LineString, count: int) -> np.ndarray:
    count = max(2, int(count))
    return np.asarray([
        [geom.interpolate(float(t), normalized=True).x, geom.interpolate(float(t), normalized=True).y]
        for t in np.linspace(0.0, 1.0, count)
    ], dtype=float)


def _orient_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
    reverse = np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0])
    return b if same <= reverse else b[::-1].copy()


def _sample_parallel_profile(
    geom_a: LineString,
    geom_b: LineString,
    sampler: RasterBlockSampler,
    count: int = 31,
) -> Dict[str, float]:
    a = _resample_normalized(geom_a, count)
    b = _orient_pair(a, _resample_normalized(geom_b, count))
    midpoint = 0.5 * (a + b)
    kde_a = sampler.values_xy("kde", a[:, 0], a[:, 1], default=0.0)
    kde_b = sampler.values_xy("kde", b[:, 0], b[:, 1], default=0.0)
    kde_mid = sampler.values_xy("kde", midpoint[:, 0], midpoint[:, 1], default=0.0)
    support_a = sampler.values_xy("support", a[:, 0], a[:, 1], default=0.0)
    support_b = sampler.values_xy("support", b[:, 0], b[:, 1], default=0.0)
    side_mean = 0.5 * (float(np.mean(kde_a)) + float(np.mean(kde_b)))
    midpoint_ratio = float(np.mean(kde_mid) / max(side_mean, 1e-9))
    valley_fraction = float(np.mean(kde_mid < 0.55 * np.maximum(kde_a, kde_b)))
    return {
        "kde_mean_a": float(np.mean(kde_a)),
        "kde_mean_b": float(np.mean(kde_b)),
        "kde_mid_mean": float(np.mean(kde_mid)),
        "midpoint_to_side_kde_ratio": midpoint_ratio,
        "midpoint_valley_fraction": valley_fraction,
        "support_mean_a_profile": float(np.mean(support_a)),
        "support_mean_b_profile": float(np.mean(support_b)),
    }


def _weighted_geometry(strong: Dict[str, Any], weak: Dict[str, Any], strong_weight: float = 0.68) -> LineString:
    count = max(8, min(160, int(max(strong["length_m"], weak["length_m"]) / 3.0) + 2))
    a = _resample_normalized(strong["geometry"], count)
    b = _orient_pair(a, _resample_normalized(weak["geometry"], count))
    coords = float(strong_weight) * a + (1.0 - float(strong_weight)) * b
    # Preserve topology at the retained edge endpoints.
    coords[0] = a[0]
    coords[-1] = a[-1]
    geom = clean_linestring(coords)
    return geom if geom is not None else strong["geometry"]


def _edge_score(edge: Dict[str, Any]) -> float:
    confidence = float(edge.get("confidence", edge_confidence(edge)) or 0.0)
    tracks = min(1.0, float(edge.get("distinct_tracks", 0) or 0) / 4.0)
    coverage = min(1.0, float(edge.get("coverage_ratio", 0) or 0) / 0.5)
    support = min(1.0, float(edge.get("support_mean", 0) or 0) / 2.0)
    return 0.45 * confidence + 0.25 * tracks + 0.15 * coverage + 0.15 * support


def _rewire_weak_endpoint(
    net: NetworkData,
    weak_edge_id: int,
    weak_node_id: int,
    strong_nodes: Tuple[int, int],
    endpoint_snap_m: float,
) -> Tuple[bool, Optional[int]]:
    node_lookup = {int(n["node_id"]): n for n in net.nodes}
    weak_node = node_lookup[int(weak_node_id)]
    p = np.asarray([float(weak_node["x"]), float(weak_node["y"])])
    distances = []
    for strong_node_id in strong_nodes:
        n = node_lookup[int(strong_node_id)]
        q = np.asarray([float(n["x"]), float(n["y"])])
        distances.append((float(np.linalg.norm(p - q)), int(strong_node_id), q))
    distance, target, target_xy = min(distances)
    incident_other = [
        e for e in net.edges
        if int(e["edge_id"]) != int(weak_edge_id)
        and int(weak_node_id) in (int(e["u"]), int(e["v"]))
    ]
    if incident_other and distance > float(endpoint_snap_m):
        return False, None
    if distance > float(endpoint_snap_m) and not incident_other:
        # Isolated weak endpoint can disappear without rewiring.
        return True, target
    for edge in incident_other:
        coords = list(edge["geometry"].coords)
        if int(edge["u"]) == int(weak_node_id):
            edge["u"] = int(target)
            coords[0] = (float(target_xy[0]), float(target_xy[1]))
        if int(edge["v"]) == int(weak_node_id):
            edge["v"] = int(target)
            coords[-1] = (float(target_xy[0]), float(target_xy[1]))
        geom = clean_linestring(coords)
        if geom is None or int(edge["u"]) == int(edge["v"]):
            return False, None
        edge["geometry"] = geom
        edge["length_m"] = float(geom.length)
        edge["parallel_rewired"] = True
    return True, target


def auto_merge_parallel_edges(
    network: NetworkData,
    support_path: Path,
    kde_path: Path,
    min_distance_m: float = 1.5,
    max_distance_m: float = 8.0,
    max_angle_deg: float = 15.0,
    min_length_m: float = 25.0,
    min_overlap_ratio: float = 0.55,
    double_track_min: int = 2,
    double_coverage_min: float = 0.22,
    valley_ratio_threshold: float = 0.58,
    valley_fraction_threshold: float = 0.45,
    merge_midpoint_ratio_min: float = 0.72,
    endpoint_snap_m: float = 12.0,
    strong_geometry_weight: float = 0.68,
    max_merges: int = 100_000,
) -> Tuple[NetworkData, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Detect parallel centerlines and automatically merge only safe single-road duplicates."""
    net = network.copy()
    compute_degrees(net.nodes, net.edges)
    candidates = detect_parallel_edges(
        net,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        max_angle_deg=max_angle_deg,
        min_length_m=min_length_m,
        min_overlap_ratio=min_overlap_ratio,
    )
    candidates.sort(key=lambda r: (-float(r["overlap_ratio"]), float(r["distance_m"])))
    decisions: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []
    consumed: set[int] = set()

    with RasterBlockSampler({"support": support_path, "kde": kde_path}, cache_blocks=96) as sampler:
        for candidate in candidates:
            if len(merged) >= int(max_merges):
                break
            a_id, b_id = int(candidate["edge_a"]), int(candidate["edge_b"])
            if a_id in consumed or b_id in consumed:
                continue
            lookup = {int(e["edge_id"]): e for e in net.edges}
            if a_id not in lookup or b_id not in lookup:
                continue
            a, b = lookup[a_id], lookup[b_id]
            profile = _sample_parallel_profile(a["geometry"], b["geometry"], sampler)
            tracks_a = int(a.get("distinct_tracks", 0) or 0)
            tracks_b = int(b.get("distinct_tracks", 0) or 0)
            coverage_a = float(a.get("coverage_ratio", 0) or 0)
            coverage_b = float(b.get("coverage_ratio", 0) or 0)
            protected = bool(a.get("uturn_protected") or b.get("uturn_protected"))
            activity = bool(a.get("activity_zone") or b.get("activity_zone"))
            dual_track = (
                tracks_a >= int(double_track_min)
                and tracks_b >= int(double_track_min)
                and coverage_a >= float(double_coverage_min)
                and coverage_b >= float(double_coverage_min)
            )
            dual_ridge = (
                profile["midpoint_to_side_kde_ratio"] <= float(valley_ratio_threshold)
                and profile["midpoint_valley_fraction"] >= float(valley_fraction_threshold)
            )
            merge_signal = profile["midpoint_to_side_kde_ratio"] >= float(merge_midpoint_ratio_min)
            decision: Dict[str, Any] = {
                **{k: v for k, v in candidate.items() if k != "geometry"},
                **profile,
                "tracks_a": tracks_a,
                "tracks_b": tracks_b,
                "coverage_a": coverage_a,
                "coverage_b": coverage_b,
                "protected": protected,
                "activity_zone": activity,
            }
            if protected:
                decision.update({"decision": "keep_parallel", "reason": "uturn_protected"})
                decisions.append(decision)
                continue
            if dual_track or dual_ridge:
                decision.update({"decision": "keep_parallel", "reason": "dual_track_evidence" if dual_track else "dual_kde_ridges"})
                decisions.append(decision)
                continue
            if not merge_signal:
                decision.update({"decision": "keep_parallel", "reason": "midpoint_kde_not_single_corridor"})
                decisions.append(decision)
                continue
            score_a, score_b = _edge_score(a), _edge_score(b)
            strong, weak = (a, b) if score_a >= score_b else (b, a)
            strong_id, weak_id = int(strong["edge_id"]), int(weak["edge_id"])
            # Work on a trial copy to avoid partial rewiring when topology checks fail.
            trial = net.copy()
            trial_lookup = {int(e["edge_id"]): e for e in trial.edges}
            trial_strong, trial_weak = trial_lookup[strong_id], trial_lookup[weak_id]
            strong_nodes = (int(trial_strong["u"]), int(trial_strong["v"]))
            ok_u, target_u = _rewire_weak_endpoint(trial, weak_id, int(trial_weak["u"]), strong_nodes, endpoint_snap_m)
            ok_v, target_v = _rewire_weak_endpoint(trial, weak_id, int(trial_weak["v"]), strong_nodes, endpoint_snap_m)
            if not ok_u or not ok_v or target_u == target_v:
                decision.update({"decision": "keep_parallel", "reason": "topology_rewire_unsafe", "strong_edge": strong_id, "weak_edge": weak_id})
                decisions.append(decision)
                continue
            trial_lookup = {int(e["edge_id"]): e for e in trial.edges}
            trial_strong, trial_weak = trial_lookup[strong_id], trial_lookup[weak_id]
            new_geom = _weighted_geometry(trial_strong, trial_weak, strong_weight=strong_geometry_weight)
            trial_strong["geometry"] = new_geom
            trial_strong["length_m"] = float(new_geom.length)
            trial_strong["parallel_merged"] = True
            trial_strong["parallel_merged_from"] = weak_id
            trial_strong["source_type"] = "parallel_merged"
            for field in ("match_points", "hmm_sequence_count"):
                trial_strong[field] = int(trial_strong.get(field, 0) or 0) + int(trial_weak.get(field, 0) or 0)
            for field in ("distinct_tracks", "distinct_clean_tracks"):
                trial_strong[field] = max(int(trial_strong.get(field, 0) or 0), int(trial_weak.get(field, 0) or 0))
            for field in ("support_mean", "kde_mean", "coverage_ratio"):
                trial_strong[field] = max(float(trial_strong.get(field, 0) or 0), float(trial_weak.get(field, 0) or 0))
            trial.edges = [e for e in trial.edges if int(e["edge_id"]) != weak_id and int(e["u"]) != int(e["v"])]
            net = trial
            # Keep original IDs stable during this pass; reindex only after all decisions.
            consumed.update((a_id, b_id))
            decision.update({
                "decision": "merged",
                "reason": "single_kde_corridor_and_safe_topology",
                "strong_edge_old": strong_id,
                "weak_edge_old": weak_id,
                "score_a": score_a,
                "score_b": score_b,
                "geometry": new_geom,
            })
            decisions.append(decision)
            merged.append(dict(decision))

    return reindex_network(net), decisions, merged
