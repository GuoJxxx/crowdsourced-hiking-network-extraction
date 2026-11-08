from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
from shapely.geometry import LineString, box
from shapely.strtree import STRtree

from .models import NetworkData


def axis_angle(geom: LineString) -> float:
    coords = np.asarray(geom.coords, dtype=float)
    if len(coords) < 2:
        return 0.0
    vec = coords[-1] - coords[0]
    return float(math.atan2(vec[1], vec[0]) % math.pi)


def axis_diff_deg(a: float, b: float) -> float:
    d = abs(a-b) % math.pi
    return float(math.degrees(min(d, math.pi-d)))


def detect_parallel_edges(
    network: NetworkData,
    min_distance_m: float = 1.5,
    max_distance_m: float = 8.0,
    max_angle_deg: float = 15.0,
    min_length_m: float = 25.0,
    min_overlap_ratio: float = 0.45,
    progress_every: int = 1000,
) -> List[Dict[str, Any]]:
    edges = [e for e in network.edges if float(e["length_m"]) >= min_length_m]
    geoms = [e["geometry"] for e in edges]
    tree = STRtree(geoms)
    lookup = {id(g): i for i, g in enumerate(geoms)}
    angles = [axis_angle(g) for g in geoms]
    rows: List[Dict[str, Any]] = []
    seen = set()

    total = len(edges)
    print(f"    [cleanup:parallel] indexed_edges={total:,}", flush=True)
    for i, e in enumerate(edges):
        if progress_every > 0 and (i == 0 or (i + 1) % int(progress_every) == 0 or i + 1 == total):
            print(
                f"    [cleanup:parallel] scanned={i + 1:,}/{total:,}, "
                f"candidate_pairs_kept={len(rows):,}",
                flush=True,
            )
        geom = geoms[i]
        try:
            # Shapely 2.x performs the distance filter inside GEOS and avoids
            # huge bounding-box candidate lists for long, curved edges.
            result = tree.query(geom, predicate="dwithin", distance=max_distance_m)
        except (TypeError, ValueError):
            envelope = box(*geom.buffer(max_distance_m).bounds)
            result = tree.query(envelope)
        if len(result) and isinstance(result[0], (int, np.integer)):
            candidates = [int(x) for x in result]
        else:
            candidates = [lookup[id(g)] for g in result if id(g) in lookup]
        for j in candidates:
            if j <= i or (i, j) in seen:
                continue
            seen.add((i, j))
            other = edges[j]
            if {int(e["u"]), int(e["v"])} & {int(other["u"]), int(other["v"])}:
                continue
            angle = axis_diff_deg(angles[i], angles[j])
            if angle > max_angle_deg:
                continue
            distance = float(geom.distance(geoms[j]))
            if not (min_distance_m <= distance <= max_distance_m):
                continue
            # Projection overlap on the dominant axis of edge i.
            coords1 = np.asarray(geom.coords, dtype=float)
            coords2 = np.asarray(geoms[j].coords, dtype=float)
            direction = coords1[-1]-coords1[0]
            norm = np.linalg.norm(direction)
            if norm <= 1e-9:
                continue
            direction /= norm
            p1 = coords1 @ direction
            p2 = coords2 @ direction
            overlap = max(0.0, min(p1.max(), p2.max())-max(p1.min(), p2.min()))
            ratio = overlap/max(min(float(e["length_m"]), float(other["length_m"])), 1e-9)
            if ratio < min_overlap_ratio:
                continue
            rows.append({
                "edge_a": int(e["edge_id"]),
                "edge_b": int(other["edge_id"]),
                "distance_m": distance,
                "angle_difference_deg": angle,
                "overlap_ratio": float(ratio),
                "source_a": str(e.get("source_type", "")),
                "source_b": str(other.get("source_type", "")),
                "support_mean_a": float(e.get("support_mean", 0) or 0),
                "support_mean_b": float(other.get("support_mean", 0) or 0),
                "recommendation": "review_or_merge" if max(float(e.get("distinct_tracks", 0) or 0), float(other.get("distinct_tracks", 0) or 0)) <= 1 else "keep_parallel_evidence",
                "geometry": geom.interpolate(0.5, normalized=True).buffer(distance/2.0).boundary,
            })
    return rows
