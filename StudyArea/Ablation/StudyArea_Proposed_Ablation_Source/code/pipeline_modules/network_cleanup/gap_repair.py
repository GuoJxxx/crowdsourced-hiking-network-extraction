from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .models import NetworkData, component_ids, compute_degrees, reindex_network
from .tile_io import RasterBlockSampler
from .topology_utils import cycle_closure_allowed, weighted_graph


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = unit(a), unit(b)
    if not a.any() or not b.any():
        return 180.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))


def endpoint_outward_vector(edge: Dict[str, Any], node_id: int, lookahead: int = 3) -> np.ndarray:
    coords = np.asarray(edge["geometry"].coords, dtype=float)
    if int(edge["u"]) == int(node_id):
        j = min(len(coords)-1, max(1, lookahead))
        # away from the edge interior
        return unit(coords[0] - coords[j])
    if int(edge["v"]) == int(node_id):
        j = max(0, len(coords)-1-max(1, lookahead))
        return unit(coords[-1] - coords[j])
    return np.zeros(2, dtype=float)


def sample_straight_evidence(
    a: np.ndarray,
    b: np.ndarray,
    sampler: RasterBlockSampler,
    spacing_m: float = 2.0,
) -> Dict[str, float]:
    length = float(np.linalg.norm(b-a))
    count = max(2, int(math.ceil(length/max(spacing_m, 0.5)))+1)
    t = np.linspace(0.0, 1.0, count)
    xy = a[None, :] * (1-t[:, None]) + b[None, :] * t[:, None]
    support = sampler.values_xy("support", xy[:, 0], xy[:, 1], default=0.0)
    kde = sampler.values_xy("kde", xy[:, 0], xy[:, 1], default=0.0)
    distance = sampler.values_xy("distance", xy[:, 0], xy[:, 1], default=0.0) if "distance" in sampler.datasets else np.zeros(count)
    return {
        "support_mean": float(np.mean(support)),
        "support_positive_fraction": float(np.mean(support > 0)),
        "support_ge2_fraction": float(np.mean(support >= 2)),
        "support_max": float(np.max(support)),
        "kde_mean": float(np.mean(kde)),
        "kde_positive_fraction": float(np.mean(kde > 0)),
        "distance_mean_m": float(np.mean(distance)),
    }


def repair_gaps(
    network: NetworkData,
    support_path,
    kde_path,
    distance_path=None,
    max_gap_m: float = 20.0,
    max_angle_deg: float = 30.0,
    min_support_fraction: float = 0.15,
    min_kde_fraction: float = 0.60,
    allow_same_component: bool = False,
    max_repairs: int = 5000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    net = network.copy()
    compute_degrees(net.nodes, net.edges)
    nodes_by_id = {int(n["node_id"]): n for n in net.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = {}
    for e in net.edges:
        incident.setdefault(int(e["u"]), []).append(e)
        incident.setdefault(int(e["v"]), []).append(e)
    endpoints = [n for n in net.nodes if int(n.get("degree", 0)) == 1]
    if len(endpoints) < 2:
        return net, []

    xy = np.asarray([[float(n["x"]), float(n["y"])] for n in endpoints], dtype=float)
    tree = cKDTree(xy)
    comps = component_ids(net.nodes, net.edges)
    candidates: List[Tuple[float, int, int, Dict[str, float]]] = []

    with RasterBlockSampler({"support": support_path, "kde": kde_path, "distance": distance_path}) as sampler:
        for i, node in enumerate(endpoints):
            nid = int(node["node_id"])
            edge = incident[nid][0]
            outward = endpoint_outward_vector(edge, nid)
            for j in tree.query_ball_point(xy[i], r=float(max_gap_m)):
                if j <= i:
                    continue
                other = endpoints[j]
                oid = int(other["node_id"])
                if not allow_same_component and comps.get(nid) == comps.get(oid):
                    continue
                vector = xy[j] - xy[i]
                length = float(np.linalg.norm(vector))
                if length <= 0.5:
                    continue
                out2 = endpoint_outward_vector(incident[oid][0], oid)
                if angle_deg(outward, vector) > max_angle_deg:
                    continue
                if angle_deg(out2, -vector) > max_angle_deg:
                    continue
                evidence = sample_straight_evidence(xy[i], xy[j], sampler)
                if not (
                    evidence["support_positive_fraction"] >= min_support_fraction
                    or evidence["kde_positive_fraction"] >= min_kde_fraction
                    or evidence["support_max"] >= 2
                ):
                    continue
                confidence = (
                    0.35 * min(1.0, evidence["support_positive_fraction"] / max(min_support_fraction, 1e-6))
                    + 0.25 * min(1.0, evidence["kde_positive_fraction"] / max(min_kde_fraction, 1e-6))
                    + 0.20 * min(1.0, evidence["support_mean"] / 2.0)
                    + 0.20 * (1.0 - length/max_gap_m)
                )
                candidates.append((-confidence, i, j, evidence))

    candidates.sort()
    used_nodes = set()
    repairs: List[Dict[str, Any]] = []
    for neg_conf, i, j, evidence in candidates:
        if len(repairs) >= max_repairs:
            break
        a, b = endpoints[i], endpoints[j]
        u, v = int(a["node_id"]), int(b["node_id"])
        if u in used_nodes or v in used_nodes:
            continue
        geom = LineString([(float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))])
        edge = {
            "edge_id": len(net.edges),
            "u": u,
            "v": v,
            "length_m": float(geom.length),
            "pixel_count": 0,
            "source_type": "gap_repair",
            "primary_fraction": 0.0,
            "auxiliary_fraction": 0.0,
            "rescued_fraction": 0.0,
            "geometry": geom,
            "uturn_core_protected": False,
            "uturn_context_protected": False,
            "uturn_protected": False,
            "activity_zone": False,
            "gap_repair": True,
            "gap_confidence": float(-neg_conf),
            **evidence,
        }
        net.edges.append(edge)
        repairs.append(dict(edge))
        used_nodes.update((u, v))

    net = reindex_network(net)
    return net, repairs


def _bresenham_rc(r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return an 8-connected Bresenham pixel path, including both endpoints."""
    x0, y0 = int(c0), int(r0)
    x1, y1 = int(c1), int(r1)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    rows: List[int] = []
    cols: List[int] = []
    while True:
        rows.append(y0)
        cols.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _xy_to_rc(transform, x: float, y: float) -> Tuple[int, int]:
    col_f, row_f = (~transform) * (float(x), float(y))
    return int(math.floor(row_f)), int(math.floor(col_f))


def _pixel_centers(transform, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


class _UnionFind:
    def __init__(self, values: Sequence[int]):
        self.parent = {int(v): int(v) for v in values}
        self.rank = {int(v): 0 for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def repair_micro_gaps(
    network: NetworkData,
    primary_path,
    auxiliary_path=None,
    rescued_path=None,
    support_path=None,
    kde_path=None,
    distance_path=None,
    max_missing_pixels: int = 0,
    max_endpoint_distance_cells: float = 0.0,
    max_endpoint_distance_m: float = 10.0,
    max_angle_deg: float = 45.0,
    strong_angle_deg: float = 20.0,
    distance_evidence_max_m: float = 8.0,
    require_evidence_for_two_pixels: bool = True,
    long_gap_threshold_m: float = 5.0,
    long_gap_max_angle_deg: float = 25.0,
    long_gap_min_evidence_fraction: float = 0.60,
    allow_long_gap_in_uturn_context: bool = False,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.8,
    cycle_min_detour_excess_m: float = 8.0,
    max_repairs: int = 20000,
    progress_every: int = 1000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Repair short primary-skeleton gaps, defaulting to a physical 10 m ceiling.

    Safety rules:
    - connects degree-1 endpoints only;
    - different components can be stitched; same-component endpoints are allowed only
      when they close a meaningful long-detour road loop;
    - candidate follows an exact Bresenham raster path;
    - the path may contain at most ``max_missing_pixels`` absent primary pixels;
    - endpoint tangents must face each other;
    - gaps longer than ``long_gap_threshold_m`` use tighter direction limits and
      require continuous trajectory/raster evidence along most of the missing span;
    - long repairs inside U-turn context are rejected by default;
    - union-find prevents accidental cross-component duplicate repairs, while
      evidence-backed real road cycles are allowed;
    - each endpoint is used at most once.
    """
    net = network.copy()
    compute_degrees(net.nodes, net.edges)
    nodes_by_id = {int(n["node_id"]): n for n in net.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = {}
    for edge in net.edges:
        incident.setdefault(int(edge["u"]), []).append(edge)
        incident.setdefault(int(edge["v"]), []).append(edge)

    endpoints = [
        node for node in net.nodes
        if int(node.get("degree", 0)) == 1 and len(incident.get(int(node["node_id"]), ())) == 1
    ]
    if len(endpoints) < 2:
        return net, []

    transform = net.transform
    cell_x = float(math.hypot(transform.a, transform.d))
    cell_y = float(math.hypot(transform.b, transform.e))
    cell = max(cell_x, cell_y, 1e-9)
    max_gap_cells_m = float(max_endpoint_distance_cells) * cell if float(max_endpoint_distance_cells) > 0 else float("inf")
    if float(max_endpoint_distance_m) > 0:
        max_gap_m = float(max_endpoint_distance_m)
    elif np.isfinite(max_gap_cells_m):
        max_gap_m = max_gap_cells_m
    else:
        max_gap_m = 10.0
    effective_max_missing = int(max_missing_pixels)
    if effective_max_missing <= 0:
        effective_max_missing = max(1, int(math.ceil(max_gap_m / cell)) - 1)
    xy = np.asarray([[float(n["x"]), float(n["y"])] for n in endpoints], dtype=float)
    rc = np.asarray([_xy_to_rc(transform, x, y) for x, y in xy], dtype=np.int64)
    tree = cKDTree(xy)
    comps = component_ids(net.nodes, net.edges)
    weighted = weighted_graph(net)

    raster_paths = {
        "primary": primary_path,
        "auxiliary": auxiliary_path,
        "rescued": rescued_path,
        "support": support_path,
        "kde": kde_path,
        "distance": distance_path,
    }
    candidates: List[Dict[str, Any]] = []
    checked = 0
    with RasterBlockSampler(raster_paths, cache_blocks=256) as sampler:
        for i, node in enumerate(endpoints):
            nid = int(node["node_id"])
            outward = endpoint_outward_vector(incident[nid][0], nid)
            for j in tree.query_ball_point(xy[i], r=max_gap_m):
                if j <= i:
                    continue
                other = endpoints[j]
                oid = int(other["node_id"])
                same_component = comps.get(nid) == comps.get(oid)
                if same_component and not bool(allow_cycle_closure):
                    continue
                checked += 1
                vector = xy[j] - xy[i]
                distance_m = float(np.linalg.norm(vector))
                if distance_m <= 0.25 * cell:
                    continue
                outward_other = endpoint_outward_vector(incident[oid][0], oid)
                angle_a = angle_deg(outward, vector)
                angle_b = angle_deg(outward_other, -vector)
                if angle_a > float(max_angle_deg) or angle_b > float(max_angle_deg):
                    continue
                is_long_gap = bool(distance_m > float(long_gap_threshold_m))
                if is_long_gap and max(angle_a, angle_b) > float(long_gap_max_angle_deg):
                    continue
                if is_long_gap and not bool(allow_long_gap_in_uturn_context):
                    endpoint_context = bool(
                        node.get("uturn_core_protected") or node.get("uturn_context_protected") or node.get("uturn_protected")
                        or other.get("uturn_core_protected") or other.get("uturn_context_protected") or other.get("uturn_protected")
                    )
                    if endpoint_context:
                        continue

                rows, cols = _bresenham_rc(int(rc[i, 0]), int(rc[i, 1]), int(rc[j, 0]), int(rc[j, 1]))
                if len(rows) < 3:
                    # Adjacent active pixels should already be connected in stage 1.
                    continue
                primary = sampler.values_rc("primary", rows, cols, default=0.0)
                if primary[0] == 0 or primary[-1] == 0:
                    continue
                interior_primary = primary[1:-1] != 0
                missing_mask = ~interior_primary
                missing_count = int(np.count_nonzero(missing_mask))
                if missing_count < 1 or missing_count > effective_max_missing:
                    continue
                # A true micro-gap has active endpoints and only absent primary
                # pixels between them. Existing interior primary pixels indicate a
                # crossing/shortcut candidate and are therefore rejected.
                if missing_count != len(interior_primary):
                    continue
                missing_positions = np.flatnonzero(missing_mask)
                if len(missing_positions) > 1 and np.any(np.diff(missing_positions) != 1):
                    continue

                mr = rows[1:-1][missing_mask]
                mc = cols[1:-1][missing_mask]
                auxiliary = sampler.values_rc("auxiliary", mr, mc, default=0.0)
                rescued = sampler.values_rc("rescued", mr, mc, default=0.0)
                support = sampler.values_rc("support", mr, mc, default=0.0)
                kde = sampler.values_rc("kde", mr, mc, default=0.0)
                distance = sampler.values_rc("distance", mr, mc, default=np.inf)
                endpoint_kde = sampler.values_rc(
                    "kde",
                    np.asarray([rows[0], rows[-1]], dtype=np.int64),
                    np.asarray([cols[0], cols[-1]], dtype=np.int64),
                    default=0.0,
                )
                endpoint_kde_ref = float(np.mean(endpoint_kde)) if len(endpoint_kde) else 0.0
                kde_ratio = float(np.mean(kde) / max(endpoint_kde_ref, 1e-9)) if len(kde) else 0.0
                distance_mask = np.isfinite(distance) & (distance <= float(distance_evidence_max_m))
                kde_mask = (kde >= 0.20 * endpoint_kde_ref) if endpoint_kde_ref > 1e-9 else np.zeros(len(kde), dtype=bool)
                evidence_mask = (auxiliary != 0) | (rescued != 0) | (support > 0) | distance_mask | kde_mask
                evidence_fraction = float(np.mean(evidence_mask)) if len(evidence_mask) else 0.0
                evidence = bool(np.any(evidence_mask) or kde_ratio >= 0.20)
                # Long gaps are allowed up to 10 m, but must show evidence over most
                # of the missing span; a single supporting pixel is not enough.
                if is_long_gap or missing_count > 2:
                    unsupported = ~evidence_mask
                    longest_unsupported = 0; current_unsupported = 0
                    for flag in unsupported.tolist():
                        current_unsupported = current_unsupported + 1 if flag else 0
                        longest_unsupported = max(longest_unsupported, current_unsupported)
                    # v0.5: allow up to approximately 4 consecutive missing pixels on a 2 m raster, with a physical limit of 10 m.
                    # For gaps of 6-10 m, still require overall evidence but allow up to 2 sampled pixels to temporarily lack evidence,
                    # avoiding rejection of genuine continuous roads due to rasterization boundaries or sampling holes.
                    if evidence_fraction < float(long_gap_min_evidence_fraction) or longest_unsupported > 2:
                        continue
                else:
                    very_straight = max(angle_a, angle_b) <= float(strong_angle_deg)
                    if missing_count >= 2 and require_evidence_for_two_pixels and not (evidence or very_straight):
                        continue
                    if missing_count == 1 and not evidence and max(angle_a, angle_b) > float(strong_angle_deg):
                        continue

                coords = _pixel_centers(transform, rows, cols)
                # Use the Bresenham pixel chain only to determine whether a genuine gap exists and to sample evidence.
                # Connect final vector endpoints directly using their coordinates to avoid writing 2 m raster steps into the result.
                geom = LineString([tuple(xy[i]), tuple(xy[j])])
                if geom.length <= 0:
                    continue
                existing_detour_m = float("inf")
                if same_component:
                    closure_ok, existing_detour_m = cycle_closure_allowed(
                        weighted, nid, oid, float(geom.length),
                        min_detour_ratio=cycle_min_detour_ratio,
                        min_detour_excess_m=cycle_min_detour_excess_m,
                    )
                    if not closure_ok:
                        continue
                confidence = float(np.clip(
                    0.40 * (1.0 - max(angle_a, angle_b) / max(float(max_angle_deg), 1e-6))
                    + 0.25 * (1.0 - missing_count / (effective_max_missing + 1.0))
                    + 0.20 * float(evidence_fraction)
                    + 0.15 * (1.0 - distance_m / max(max_gap_m, 1e-6)),
                    0.0, 1.0,
                ))
                candidates.append({
                    "i": i,
                    "j": j,
                    "u": nid,
                    "v": oid,
                    "component_u": int(comps.get(nid, -1)),
                    "component_v": int(comps.get(oid, -1)),
                    "cycle_closure": bool(same_component),
                    "existing_detour_m": float(existing_detour_m),
                    "distance_m": distance_m,
                    "angle_u_deg": float(angle_a),
                    "angle_v_deg": float(angle_b),
                    "missing_pixels": missing_count,
                    "evidence": evidence,
                    "evidence_fraction": evidence_fraction,
                    "long_gap": bool(is_long_gap or missing_count > 2),
                    "auxiliary_hit": bool(np.any(auxiliary != 0)),
                    "rescued_hit": bool(np.any(rescued != 0)),
                    "support_mean": float(np.mean(support)) if len(support) else 0.0,
                    "support_max": float(np.max(support)) if len(support) else 0.0,
                    "support_ge2_fraction": float(np.mean(support >= 2)) if len(support) else 0.0,
                    "kde_mean": float(np.mean(kde)) if len(kde) else 0.0,
                    "kde_max": float(np.max(kde)) if len(kde) else 0.0,
                    "kde_ratio_to_endpoints": kde_ratio,
                    "distance_mean_m": float(np.mean(distance[np.isfinite(distance)])) if np.any(np.isfinite(distance)) else float("inf"),
                    "confidence": confidence,
                    "rows": rows,
                    "cols": cols,
                    "geometry": geom,
                })
            if progress_every > 0 and ((i + 1) % int(progress_every) == 0 or i + 1 == len(endpoints)):
                print(
                    f"    [micro-gap] endpoints={i+1:,}/{len(endpoints):,}, "
                    f"checked_pairs={checked:,}, candidates={len(candidates):,}",
                    flush=True,
                )

    candidates.sort(key=lambda row: (
        -float(row["confidence"]),
        int(row["missing_pixels"]),
        float(row["distance_m"]),
        max(float(row["angle_u_deg"]), float(row["angle_v_deg"])),
    ))
    uf = _UnionFind(set(comps.values()))
    used_nodes: set[int] = set()
    repairs: List[Dict[str, Any]] = []
    next_edge_id = max((int(e["edge_id"]) for e in net.edges), default=-1) + 1
    for candidate in candidates:
        if len(repairs) >= int(max_repairs):
            break
        u, v = int(candidate["u"]), int(candidate["v"])
        if u in used_nodes or v in used_nodes:
            continue
        cu, cv = int(candidate["component_u"]), int(candidate["component_v"])
        geom = candidate["geometry"]
        if cu == cv:
            if not bool(allow_cycle_closure):
                continue
            closure_ok, detour_now = cycle_closure_allowed(
                weighted, u, v, float(geom.length),
                min_detour_ratio=cycle_min_detour_ratio,
                min_detour_excess_m=cycle_min_detour_excess_m,
            )
            if not closure_ok:
                continue
            candidate["cycle_closure"] = True
            candidate["existing_detour_m"] = float(detour_now)
        else:
            if not uf.union(cu, cv):
                continue
            candidate["cycle_closure"] = False
            candidate["existing_detour_m"] = float("inf")
        edge = {
            "edge_id": next_edge_id,
            "u": u,
            "v": v,
            "length_m": float(geom.length),
            "pixel_count": int(len(candidate["rows"])),
            "source_type": "micro_gap_repair",
            "primary_fraction": float((len(candidate["rows"]) - int(candidate["missing_pixels"])) / len(candidate["rows"])),
            "auxiliary_fraction": 1.0 if candidate["auxiliary_hit"] else 0.0,
            "rescued_fraction": 1.0 if candidate["rescued_hit"] else 0.0,
            "support_mean": float(candidate["support_mean"]),
            "support_median": float(candidate["support_mean"]),
            "support_max": float(candidate["support_max"]),
            "support_ge2_fraction": float(candidate["support_ge2_fraction"]),
            "kde_mean": float(candidate["kde_mean"]),
            "kde_median": float(candidate["kde_mean"]),
            "kde_max": float(candidate["kde_max"]),
            "distance_mean_m": float(candidate["distance_mean_m"]),
            "geometry": geom,
            "uturn_core_protected": False,
            "uturn_context_protected": False,
            "uturn_protected": False,
            "activity_zone": False,
            "gap_repair": True,
            "micro_gap_repair": True,
            "micro_gap_missing_pixels": int(candidate["missing_pixels"]),
            "micro_gap_angle_u_deg": float(candidate["angle_u_deg"]),
            "micro_gap_angle_v_deg": float(candidate["angle_v_deg"]),
            "micro_gap_evidence_fraction": float(candidate.get("evidence_fraction", 0.0)),
            "micro_gap_long_repair": bool(candidate.get("long_gap", False)),
            "micro_gap_cycle_closure": bool(candidate.get("cycle_closure", False)),
            "micro_gap_existing_detour_m": float(candidate.get("existing_detour_m", float("inf"))),
            "gap_confidence": float(candidate["confidence"]),
            "confidence": max(0.60, float(candidate["confidence"])),
            "retained_reason": "approved_evidence_constrained_gap_up_to_10m",
            "geometry_origin": "micro_gap_directional_straight_connector_v0_5",
            "max_geometry_shift_m": 0.0,
            "baseline_length_m": float(geom.length),
            "approved_gap_repair": True,
        }
        net.edges.append(edge)
        row = dict(edge)
        row["candidate_distance_m"] = float(candidate["distance_m"])
        row["evidence_present"] = bool(candidate["evidence"])
        repairs.append(row)
        weighted.add_edge(u, v, weight=float(geom.length))
        used_nodes.update((u, v))
        next_edge_id += 1

    net = reindex_network(net)
    print(
        f"    [micro-gap] accepted={len(repairs):,}, candidates={len(candidates):,}, "
        f"max_missing_pixels={effective_max_missing} (requested={int(max_missing_pixels)}), max_distance_m={max_gap_m:.1f}",
        flush=True,
    )
    return net, repairs
