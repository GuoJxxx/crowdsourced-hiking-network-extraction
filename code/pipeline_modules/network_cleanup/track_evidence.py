from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from .models import NetworkData


def _query_indices(tree: STRtree, geoms, envelope) -> List[int]:
    result = tree.query(envelope)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in result]
    lookup = {id(g): i for i, g in enumerate(geoms)}
    return [lookup[id(g)] for g in result if id(g) in lookup]


def _bool_series(series: pd.Series, default: bool = True) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).to_numpy(dtype=bool)
    text = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


def match_track_evidence(
    network: NetworkData,
    points_csv: Optional[Path],
    max_distance_m: float = 35.0,
    match_step: int = 5,
    chunksize: int = 250_000,
    write_points_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Scalable nearest-edge trajectory evidence matching.

    The matcher deliberately samples every Nth point. It is used to estimate per-edge
    independent trajectory support, coverage and match distance; it is not intended to
    replace the dedicated map matching stage used in the preceding workflow.
    """
    if points_csv is None or not Path(points_csv).exists():
        summary = pd.DataFrame([{
            "edge_id": int(e["edge_id"]),
            "match_points": 0,
            "distinct_tracks": 0,
            "distinct_clean_tracks": 0,
            "mean_match_dist_m": np.nan,
            "p95_match_dist_m": np.nan,
            "coverage_ratio": 0.0,
        } for e in network.edges])
        return summary, {"enabled": False, "reason": "gravity points not provided"}

    geoms = [e["geometry"] for e in network.edges]
    if not geoms:
        raise ValueError("路网没有可匹配边")
    tree = STRtree(geoms)
    edge_by_id = {int(e["edge_id"]): e for e in network.edges}

    accum: Dict[int, Dict[str, Any]] = {}
    for e in network.edges:
        accum[int(e["edge_id"])] = {
            "count": 0,
            "sum_dist": 0.0,
            "distances": [],
            "tracks": set(),
            "clean_tracks": set(),
            "min_along": float("inf"),
            "max_along": float("-inf"),
        }

    point_rows: List[Dict[str, Any]] = []
    total_seen = total_sampled = total_matched = 0
    step = max(1, int(match_step))

    header = pd.read_csv(points_csv, nrows=0)
    cols = set(header.columns)
    xcol = "gravity_x_m" if "gravity_x_m" in cols else "x_m"
    ycol = "gravity_y_m" if "gravity_y_m" in cols else "y_m"
    order_col = "clean_point_index" if "clean_point_index" in cols else "point_index"
    clean_col = "clean_track_id" if "clean_track_id" in cols else "track_id"
    raw_col = "track_id" if "track_id" in cols else clean_col
    source_col = "source_file" if "source_file" in cols else None
    usecols = [c for c in (xcol, ycol, order_col, clean_col, raw_col, source_col,
                           "road_extraction_weight", "mainroad_candidate") if c and c in cols]

    for chunk in pd.read_csv(points_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        total_seen += len(chunk)
        x = pd.to_numeric(chunk[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(chunk[ycol], errors="coerce").to_numpy(dtype=float)
        order = pd.to_numeric(chunk.get(order_col, pd.Series(np.arange(len(chunk)))), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        valid = np.isfinite(x) & np.isfinite(y) & ((order % step) == 0)
        if "mainroad_candidate" in chunk.columns:
            valid &= _bool_series(chunk["mainroad_candidate"], default=True)
        if "road_extraction_weight" in chunk.columns:
            weights = pd.to_numeric(chunk["road_extraction_weight"], errors="coerce").fillna(0).to_numpy(float)
            valid &= weights >= 0.30
        indices = np.flatnonzero(valid)
        total_sampled += len(indices)

        for idx in indices:
            px, py = float(x[idx]), float(y[idx])
            point = Point(px, py)
            envelope = box(px-max_distance_m, py-max_distance_m, px+max_distance_m, py+max_distance_m)
            candidates = _query_indices(tree, geoms, envelope)
            best_idx, best_dist = None, float("inf")
            for ci in candidates:
                dist = float(point.distance(geoms[ci]))
                if dist < best_dist and dist <= max_distance_m:
                    best_idx, best_dist = ci, dist
            if best_idx is None:
                continue
            edge = network.edges[int(best_idx)]
            eid = int(edge["edge_id"])
            along = float(edge["geometry"].project(point))
            raw_track = str(chunk.iloc[idx][raw_col])
            clean_track = str(chunk.iloc[idx][clean_col])
            contribution = f"{chunk.iloc[idx][source_col]}||{raw_track}" if source_col else raw_track
            a = accum[eid]
            a["count"] += 1
            a["sum_dist"] += best_dist
            # bounded reservoir for p95; deterministic first 5000 points per edge is sufficient for diagnostics
            if len(a["distances"]) < 5000:
                a["distances"].append(best_dist)
            a["tracks"].add(contribution)
            a["clean_tracks"].add(clean_track)
            a["min_along"] = min(a["min_along"], along)
            a["max_along"] = max(a["max_along"], along)
            total_matched += 1
            if write_points_path is not None:
                projected = edge["geometry"].interpolate(along)
                point_rows.append({
                    "clean_track_id": clean_track,
                    "track_id": raw_track,
                    "x": px,
                    "y": py,
                    "matched_edge_id": eid,
                    "match_distance_m": best_dist,
                    "along_m": along,
                    "matched_x": float(projected.x),
                    "matched_y": float(projected.y),
                })

    rows: List[Dict[str, Any]] = []
    for e in network.edges:
        eid = int(e["edge_id"])
        a = accum[eid]
        distances = np.asarray(a["distances"], dtype=float)
        coverage = 0.0
        if a["count"] >= 2 and np.isfinite(a["min_along"]) and np.isfinite(a["max_along"]):
            coverage = (a["max_along"] - a["min_along"]) / max(float(e["length_m"]), 1e-9)
        rows.append({
            "edge_id": eid,
            "match_points": int(a["count"]),
            "distinct_tracks": int(len(a["tracks"])),
            "distinct_clean_tracks": int(len(a["clean_tracks"])),
            "mean_match_dist_m": float(a["sum_dist"] / a["count"]) if a["count"] else np.nan,
            "p95_match_dist_m": float(np.percentile(distances, 95)) if len(distances) else np.nan,
            "coverage_ratio": float(np.clip(coverage, 0.0, 1.0)),
        })

    if write_points_path is not None:
        Path(write_points_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(point_rows).to_csv(write_points_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    report = {
        "enabled": True,
        "points_seen": int(total_seen),
        "sample_step": step,
        "sampled_points": int(total_sampled),
        "matched_points": int(total_matched),
        "matched_ratio": float(total_matched / max(total_sampled, 1)),
        "max_distance_m": float(max_distance_m),
    }
    return summary, report


def attach_track_evidence(network: NetworkData, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    lookup = summary.set_index("edge_id").to_dict("index")
    for e in network.edges:
        e.update(lookup.get(int(e["edge_id"]), {}))
