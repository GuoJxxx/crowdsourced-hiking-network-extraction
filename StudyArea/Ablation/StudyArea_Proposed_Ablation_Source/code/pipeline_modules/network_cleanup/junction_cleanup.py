from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .models import NetworkData, compute_degrees, reindex_network
from .optimizer import edge_confidence
from .tile_io import RasterBlockSampler
from .topology_utils import (
    cycle_closure_allowed,
    endpoint_to_edge_cycle_closure_allowed,
    weighted_graph,
    safe_intersection,
)


def _num(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        x = float(obj.get(key, default))
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _axis_angle_deg_local(a: np.ndarray, b: np.ndarray) -> float:
    angle = _angle_deg(a, b)
    return min(float(angle), abs(180.0 - float(angle)))


def _edge_axis_at_distance_local(edge: Dict[str, Any], distance: float, probe_m: float = 4.0) -> Optional[np.ndarray]:
    geom = edge.get("geometry")
    if not isinstance(geom, LineString) or geom.length <= 0:
        return None
    d = float(np.clip(distance, 0.0, float(geom.length)))
    lo = max(0.0, d - float(probe_m))
    hi = min(float(geom.length), d + float(probe_m))
    if hi - lo <= 1e-6:
        return None
    a = geom.interpolate(lo); b = geom.interpolate(hi)
    v = np.asarray([float(b.x - a.x), float(b.y - a.y)], dtype=float)
    return v if np.linalg.norm(v) > 1e-9 else None


def _raw_connector_angles(
    geom: Optional[LineString], source_cont: Optional[np.ndarray],
    target_edge: Dict[str, Any], target_distance: float,
) -> Tuple[float, float]:
    if not isinstance(geom, LineString) or len(geom.coords) < 2 or source_cont is None:
        return 180.0, 180.0
    coords = np.asarray(geom.coords, dtype=float)
    start_v = coords[min(2, len(coords) - 1)] - coords[0]
    end_v = coords[-1] - coords[max(0, len(coords) - 3)]
    source_angle = _angle_deg(source_cont, start_v)
    target_axis = _edge_axis_at_distance_local(target_edge, float(target_distance))
    target_angle = 180.0 if target_axis is None else _axis_angle_deg_local(target_axis, end_v)
    return float(source_angle), float(target_angle)


def junction_candidate_key(candidate: Dict[str, Any]) -> Tuple[float, ...]:
    """Lexicographic evidence ordering for topology repair candidates.

    Higher is better.  Distance and angle are intentionally last-order tie breakers;
    one real original-track transition must beat a merely nearer geometric target.
    """
    direct = int(candidate.get("direct_witness_tracks", candidate.get("raw_witness_tracks", 0)) or 0)
    distinct = int(candidate.get("distinct_raw_tracks", direct) or 0)
    density = int(candidate.get("raw_density_support", 0) or 0)
    evidence = float(candidate.get("evidence_fraction", 0.0) or 0.0)
    support = float(candidate.get("support_fraction", 0.0) or 0.0)
    angle = float(candidate.get("approach_angle_deg", 180.0) or 180.0)
    distance = float(candidate.get("distance_m", candidate.get("connector_length_m", float("inf"))) or float("inf"))
    return (
        1.0 if direct > 0 else 0.0,
        float(direct),
        float(distinct),
        float(density),
        float(evidence),
        float(support),
        -float(angle),
        -float(distance),
    )


def _protected(edge: Dict[str, Any]) -> bool:
    return bool(
        edge.get("uturn_core_protected")
        or edge.get("uturn_protected")
        or edge.get("gap_repair")
        or edge.get("approved_gap_repair")
        or edge.get("junction_repair")
        or edge.get("approved_junction_repair")
    )


def _node_xy(nodes: Dict[int, Dict[str, Any]], nid: int) -> np.ndarray:
    n = nodes[int(nid)]
    return np.asarray([float(n["x"]), float(n["y"])], dtype=float)


def _edge_vector_into(edge: Dict[str, Any], node_id: int, probe_m: float = 8.0) -> Optional[np.ndarray]:
    coords = np.asarray(edge["geometry"].coords, dtype=float)
    if len(coords) < 2:
        return None
    if int(edge["u"]) == int(node_id):
        arr = coords
    elif int(edge["v"]) == int(node_id):
        arr = coords[::-1]
    else:
        return None
    origin = arr[0]
    target = arr[-1]
    travelled = 0.0
    for a, b in zip(arr[:-1], arr[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg <= 1e-9:
            continue
        if travelled + seg >= probe_m:
            f = (probe_m - travelled) / seg
            target = a + np.clip(f, 0.0, 1.0) * (b - a)
            break
        travelled += seg
    vec = np.asarray(target - origin, dtype=float)
    return vec if float(np.linalg.norm(vec)) > 1e-9 else None


def _continuation_vector(edge: Dict[str, Any], node_id: int) -> Optional[np.ndarray]:
    v = _edge_vector_into(edge, node_id)
    return None if v is None else -v


def _sample_line_evidence(
    geom: LineString,
    sampler: Optional[RasterBlockSampler],
    evidence_distance_m: float,
) -> Dict[str, float]:
    if sampler is None or geom.length <= 0:
        return {
            "evidence_fraction": 0.0,
            "support_fraction": 0.0,
            "distance_fraction": 0.0,
            "kde_fraction": 0.0,
        }
    n = max(3, int(math.ceil(float(geom.length) / 1.25)) + 1)
    ds = np.linspace(0.0, float(geom.length), n)
    pts = [geom.interpolate(float(d)) for d in ds]
    xs = np.asarray([p.x for p in pts], dtype=float)
    ys = np.asarray([p.y for p in pts], dtype=float)
    support = (
        sampler.values_xy("support", xs, ys, default=0.0)
        if "support" in sampler.datasets else np.zeros(n)
    )
    distance = (
        sampler.values_xy("distance", xs, ys, default=np.inf)
        if "distance" in sampler.datasets else np.full(n, np.inf)
    )
    kde = (
        sampler.values_xy("kde", xs, ys, default=0.0)
        if "kde" in sampler.datasets else np.zeros(n)
    )
    sm = support > 0
    dm = np.isfinite(distance) & (distance <= float(evidence_distance_m))
    positive = kde[kde > 0]
    kde_ref = float(np.percentile(positive, 20)) if len(positive) else 0.0
    km = kde >= kde_ref if kde_ref > 0 else np.zeros(n, dtype=bool)
    ev = sm | dm | km
    return {
        "evidence_fraction": float(np.mean(ev)),
        "support_fraction": float(np.mean(sm)),
        "distance_fraction": float(np.mean(dm)),
        "kde_fraction": float(np.mean(km)),
    }


def _raw_transition_evidence(
    raw_witness_index,
    connector: LineString,
    side_a_geometries: Sequence[LineString],
    side_b_geometries: Sequence[LineString],
    min_witness_tracks: int,
    side_distance_m: float,
    max_path_m: float,
    max_path_ratio: float,
    corridor_m: float,
    use_raw_subpath_geometry: bool = False,
) -> Tuple[Dict[str, Any], int, int, bool]:
    """Return direct raw-transition evidence and a local negative-evidence veto.

    v0.5 principle: if original trajectories clearly occupy BOTH road sides but no
    motion-sane trajectory actually traverses the proposed connector, raster/KDE
    proximity must not manufacture a topology connection.  If either side lacks raw
    occupancy, the result is inconclusive and raster evidence may still rescue it.
    """
    empty = {
        "witness_tracks": 0, "witness_chunks": 0, "best_track_id": "",
        "best_contribution_id": "", "best_source_path_m": float("inf"),
        "best_snap_a_m": float("inf"), "best_snap_b_m": float("inf"),
        "best_corridor_p90_m": float("inf"), "best_geometry": None,
    }
    idx = raw_witness_index
    if idx is None or not getattr(idx, "enabled", False):
        return empty, 0, 0, False
    if bool(use_raw_subpath_geometry):
        witness = idx.query_transition_path(
            connector, endpoint_snap_m=float(side_distance_m),
            max_path_m=float(max_path_m), max_path_ratio=float(max_path_ratio),
        )
    else:
        witness = idx.query_connector(
            connector, endpoint_snap_m=float(side_distance_m),
            max_path_m=float(max_path_m), max_path_ratio=float(max_path_ratio),
            corridor_m=float(corridor_m),
        )
    a_tracks: Set[str] = set()
    b_tracks: Set[str] = set()
    for geom in side_a_geometries:
        if isinstance(geom, LineString) and geom.length > 0:
            a_tracks.update(idx.tracks_near_geometry(geom, float(side_distance_m)))
    for geom in side_b_geometries:
        if isinstance(geom, LineString) and geom.length > 0:
            b_tracks.update(idx.tracks_near_geometry(geom, float(side_distance_m)))
    threshold = max(1, int(min_witness_tracks))
    negative = (
        len(a_tracks) >= threshold
        and len(b_tracks) >= threshold
        and int(witness.get("witness_tracks", 0) or 0) < threshold
    )
    return witness, int(len(a_tracks)), int(len(b_tracks)), bool(negative)


def _local_best_center(
    mean_xy: np.ndarray,
    sampler: Optional[RasterBlockSampler],
    search_radius_m: float,
    grid_step_m: float = 1.0,
) -> np.ndarray:
    if sampler is None or search_radius_m <= 0:
        return mean_xy.copy()
    offsets = np.arange(-search_radius_m, search_radius_m + 1e-9, max(0.5, grid_step_m))
    gx, gy = np.meshgrid(offsets, offsets)
    xs = mean_xy[0] + gx.ravel()
    ys = mean_xy[1] + gy.ravel()
    rr = np.hypot(xs - mean_xy[0], ys - mean_xy[1])
    keep = rr <= search_radius_m + 1e-9
    xs, ys, rr = xs[keep], ys[keep], rr[keep]
    if len(xs) == 0:
        return mean_xy.copy()
    kde = sampler.values_xy("kde", xs, ys, default=0.0) if "kde" in sampler.datasets else np.zeros(len(xs))
    support = sampler.values_xy("support", xs, ys, default=0.0) if "support" in sampler.datasets else np.zeros(len(xs))
    distance = sampler.values_xy("distance", xs, ys, default=np.inf) if "distance" in sampler.datasets else np.full(len(xs), np.inf)

    def norm(v):
        v = np.asarray(v, dtype=float)
        finite = np.isfinite(v)
        if not finite.any():
            return np.zeros_like(v)
        lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]))
        if hi <= lo + 1e-12:
            out = np.zeros_like(v)
            out[finite] = 1.0 if hi > 0 else 0.0
            return out
        out = np.zeros_like(v)
        out[finite] = (v[finite] - lo) / (hi - lo)
        return out

    kn = norm(kde)
    sn = norm(support)
    dn = np.where(np.isfinite(distance), np.exp(-np.maximum(distance, 0.0) / 8.0), 0.0)
    # Small centre-distance penalty stabilises the hub when evidence is flat.
    score = 0.58 * kn + 0.27 * sn + 0.12 * dn - 0.03 * (rr / max(search_radius_m, 1e-9))
    i = int(np.argmax(score))
    return np.asarray([float(xs[i]), float(ys[i])], dtype=float)


def _split_linestring_at_distance(geom: LineString, d: float) -> Optional[Tuple[LineString, LineString, Tuple[float, float]]]:
    length = float(geom.length)
    if d <= 1e-8 or d >= length - 1e-8:
        return None
    p = geom.interpolate(float(d))
    px, py = float(p.x), float(p.y)
    coords = list(geom.coords)
    left = [coords[0]]
    right: List[Tuple[float, float]] = []
    travelled = 0.0
    inserted = False
    for a, b in zip(coords[:-1], coords[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        seg = math.hypot(bx - ax, by - ay)
        if not inserted and travelled + seg >= d - 1e-9:
            if left[-1] != (px, py):
                left.append((px, py))
            right = [(px, py)]
            if math.hypot(bx - px, by - py) > 1e-9:
                right.append((bx, by))
            inserted = True
        elif not inserted:
            left.append((bx, by))
        else:
            if not right or right[-1] != (bx, by):
                right.append((bx, by))
        travelled += seg
    if not inserted or len(left) < 2 or len(right) < 2:
        return None
    g1, g2 = LineString(left), LineString(right)
    if g1.length <= 0 or g2.length <= 0:
        return None
    return g1, g2, (px, py)


def _new_connector_edge(
    edge_id: int,
    u: int,
    v: int,
    geom: LineString,
    mode: str,
    evidence: Dict[str, float],
    angle_deg: float,
    *,
    audit_max_connector_m: Optional[float] = None,
    audit_max_angle_deg: Optional[float] = None,
    audit_raw_max_angle_deg: Optional[float] = None,
    audit_min_evidence_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    edge = {
        "edge_id": int(edge_id),
        "u": int(u),
        "v": int(v),
        "geometry": geom,
        "length_m": float(geom.length),
        "pixel_count": 0,
        "source_type": "junction_repair",
        "primary_fraction": 0.0,
        "auxiliary_fraction": 0.0,
        "rescued_fraction": 0.0,
        "support_mean": 0.0,
        "support_median": 0.0,
        "support_max": 0.0,
        "support_ge2_fraction": 0.0,
        "kde_mean": 0.0,
        "kde_median": 0.0,
        "kde_max": 0.0,
        "distance_mean_m": 0.0,
        "gap_repair": True,
        "approved_gap_repair": True,
        "junction_repair": True,
        "approved_junction_repair": True,
        "junction_repair_mode": str(mode),
        "junction_repair_angle_deg": float(angle_deg),
        "junction_repair_evidence_fraction": float(evidence.get("evidence_fraction", 0.0)),
        "junction_repair_support_fraction": float(evidence.get("support_fraction", 0.0)),
        "junction_repair_distance_fraction": float(evidence.get("distance_fraction", 0.0)),
        "junction_repair_kde_fraction": float(evidence.get("kde_fraction", 0.0)),
        "uturn_core_protected": False,
        "uturn_context_protected": False,
        "uturn_protected": False,
        "confidence": 0.90,
        "retained_reason": "junction_connectivity_repair_v0_5",
        "geometry_origin": "s7_junction_repair_v0_5",
        "max_geometry_shift_m": 0.0,
        "baseline_length_m": float(geom.length),
    }
    if audit_max_connector_m is not None:
        edge["junction_audit_max_connector_m"] = float(audit_max_connector_m)
    if audit_max_angle_deg is not None:
        edge["junction_audit_max_angle_deg"] = float(audit_max_angle_deg)
    if audit_raw_max_angle_deg is not None:
        edge["junction_audit_raw_max_angle_deg"] = float(audit_raw_max_angle_deg)
    if audit_min_evidence_fraction is not None:
        edge["junction_audit_min_evidence_fraction"] = float(audit_min_evidence_fraction)
    return edge


def _sample_edges(network: NetworkData, spacing_m: float = 3.0):
    rows = []
    xy = []
    for edge in network.edges:
        geom = edge["geometry"]
        length = float(geom.length)
        n = max(2, int(math.ceil(length / max(spacing_m, 0.5))) + 1)
        ds = np.linspace(0.0, length, n)
        for d in ds:
            p = geom.interpolate(float(d))
            xy.append((float(p.x), float(p.y)))
            rows.append((int(edge["edge_id"]), float(d)))
    return np.asarray(xy, dtype=float), rows




def _strtree_query_indices(tree: STRtree, geoms: Sequence[LineString], query_geom) -> List[int]:
    result = tree.query(query_geom)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in result]
    lookup = {id(g): i for i, g in enumerate(geoms)}
    return [lookup[id(g)] for g in result if id(g) in lookup]


def _intersection_only_at_allowed_points(inter, allowed_touch_points: Sequence[Point], tol: float = 1e-6) -> bool:
    if inter.is_empty or not allowed_touch_points:
        return False
    gtype = inter.geom_type
    if gtype == "Point":
        return any(float(inter.distance(p)) <= float(tol) for p in allowed_touch_points)
    if gtype in {"MultiPoint", "GeometryCollection"}:
        parts = list(inter.geoms)
        return bool(parts) and all(
            _intersection_only_at_allowed_points(part, allowed_touch_points, tol)
            for part in parts
        )
    # Any overlapping line/curve/area is more than a modelled endpoint touch.
    return False


def _crosses_unmodelled_existing_edge(
    geom: LineString,
    edge_geoms: Sequence[LineString],
    edge_ids: Sequence[int],
    tree: Optional[STRtree],
    ignore_edge_ids: Set[int],
    allowed_touch_points: Sequence[Point] = (),
) -> bool:
    if tree is None or geom.is_empty:
        return False
    for idx in _strtree_query_indices(tree, edge_geoms, geom):
        eid = int(edge_ids[int(idx)])
        if eid in ignore_edge_ids:
            continue
        other = edge_geoms[int(idx)]
        inter = safe_intersection(
            geom, other, operation="junction_existing_edge_intersection",
            context={"other_edge_id": eid},
        )
        if inter is None:
            return True
        if inter.is_empty:
            continue
        if _intersection_only_at_allowed_points(inter, allowed_touch_points):
            continue
        return True
    return False


def repair_endpoint_to_edge_junctions(
    network: NetworkData,
    support_path=None,
    kde_path=None,
    distance_path=None,
    max_snap_m: float = 8.0,
    node_merge_m: float = 2.5,
    max_approach_angle_deg: float = 65.0,
    min_evidence_fraction: float = 0.25,
    evidence_distance_m: float = 10.0,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.8,
    cycle_min_detour_excess_m: float = 8.0,
    max_rounds: int = 4,
    max_repairs: int = 5000,
    raw_witness_index=None,
    raw_witness_min_tracks: int = 1,
    raw_witness_endpoint_snap_m: float = 6.0,
    raw_witness_max_path_m: float = 60.0,
    raw_witness_max_path_ratio: float = 3.0,
    raw_witness_corridor_m: float = 10.0,
    raw_witness_relaxed_max_angle_deg: float = 75.0,
    raw_parallel_guard_max_axis_angle_deg: float = 15.0,
    raw_parallel_guard_min_cross_angle_deg: float = 55.0,
    raw_parallel_guard_min_side_tracks: int = 1,
    use_raw_subpath_geometry: bool = False,
    require_raw_witness: bool = False,
    raw_target_max_axis_angle_deg: float = 180.0,
    repair_label: str = "endpoint_to_edge",
    include_protected_endpoints: bool = False,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Snap broken branch endpoints to a nearby road edge and split the target edge.

    This repairs the common T/Y-junction failure where the branch stops a few metres
    before the through-road. Different components are merged directly. A same-component
    endpoint may also close an open road loop when the existing graph path is a much
    longer detour than the missing connector.
    """
    net = network.copy()
    records: List[Dict[str, Any]] = []
    raster_paths = {"support": support_path, "kde": kde_path, "distance": distance_path}
    raster_paths = {k: Path(v) for k, v in raster_paths.items() if v is not None and Path(v).exists()}
    sampler = RasterBlockSampler(raster_paths, cache_blocks=128) if raster_paths else None

    try:
        for round_no in range(1, max(1, int(max_rounds)) + 1):
            if len(records) >= int(max_repairs):
                break
            net = reindex_network(net)
            nodes = {int(n["node_id"]): n for n in net.nodes}
            incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            graph = nx.Graph()
            edge_by_id = {}
            for e in net.edges:
                u, v = int(e["u"]), int(e["v"])
                incident[u].append(e); incident[v].append(e)
                graph.add_edge(u, v)
                edge_by_id[int(e["edge_id"])] = e
            edge_geoms = [e["geometry"] for e in net.edges]
            edge_ids_for_tree = [int(e["edge_id"]) for e in net.edges]
            edge_tree = STRtree(edge_geoms) if edge_geoms else None
            weighted = weighted_graph(net)
            comp = {}
            for cid, cc in enumerate(nx.connected_components(graph)):
                for n in cc:
                    comp[int(n)] = cid
            endpoints = [
                n for n, es in incident.items()
                if len(es) == 1 and (bool(include_protected_endpoints) or not _protected(es[0]))
            ]
            if not endpoints:
                break

            sample_xy, sample_meta = _sample_edges(net, spacing_m=max(2.0, max_snap_m / 3.0))
            if len(sample_xy) == 0:
                break
            tree = cKDTree(sample_xy)
            candidates = []
            for ep in endpoints:
                ep_xy = _node_xy(nodes, ep)
                k = min(32, len(sample_xy))
                dists, inds = tree.query(ep_xy, k=k, distance_upper_bound=float(max_snap_m))
                dists = np.atleast_1d(dists); inds = np.atleast_1d(inds)
                best = None
                for sd, si in zip(dists, inds):
                    if not np.isfinite(sd) or int(si) >= len(sample_meta):
                        continue
                    target_id, _ = sample_meta[int(si)]
                    target = edge_by_id.get(int(target_id))
                    if target is None:
                        continue
                    if int(target_id) == int(incident[ep][0]["edge_id"]):
                        continue
                    # A synthetic repair connector is not a stable through-road
                    # target.  Splitting one would copy its original approval
                    # metadata onto two child geometries whose endpoints/evidence
                    # envelope no longer match the approved repair.  This was the
                    # second source of v0.5 final-audit failures.  Real raster-road
                    # edges remain eligible targets, including ordinary U-turn
                    # protected roads; only synthetic repair geometries are skipped.
                    if bool(
                        target.get("gap_repair")
                        or target.get("approved_gap_repair")
                        or target.get("junction_repair")
                        or target.get("approved_junction_repair")
                    ):
                        continue
                    tu, tv = int(target["u"]), int(target["v"])
                    same_component = comp.get(ep) == comp.get(tu)
                    if same_component and not bool(allow_cycle_closure):
                        continue
                    geom = target["geometry"]
                    proj_d = float(geom.project(Point(float(ep_xy[0]), float(ep_xy[1]))))
                    proj = geom.interpolate(proj_d)
                    q = np.asarray([float(proj.x), float(proj.y)], dtype=float)
                    gap = float(np.linalg.norm(q - ep_xy))
                    if gap <= 0.35 or gap > float(max_snap_m):
                        continue
                    cycle_closure = False
                    existing_detour_m = float("inf")
                    if same_component:
                        cycle_closure, existing_detour_m = endpoint_to_edge_cycle_closure_allowed(
                            weighted, ep, tu, tv, proj_d, float(geom.length), gap,
                            min_detour_ratio=cycle_min_detour_ratio,
                            min_detour_excess_m=cycle_min_detour_excess_m,
                        )
                        if not cycle_closure:
                            continue
                    cont = _continuation_vector(incident[ep][0], ep)
                    if cont is None:
                        continue
                    ang = _angle_deg(cont, q - ep_xy)
                    connector = LineString([tuple(ep_xy), tuple(q)])
                    ev = _sample_line_evidence(connector, sampler, evidence_distance_m)
                    raw, raw_side_a_tracks, raw_side_b_tracks, raw_negative_veto = _raw_transition_evidence(
                        raw_witness_index, connector,
                        [incident[ep][0]["geometry"]], [target["geometry"]],
                        min_witness_tracks=max(int(raw_witness_min_tracks), int(raw_parallel_guard_min_side_tracks)),
                        side_distance_m=float(raw_witness_endpoint_snap_m),
                        max_path_m=float(raw_witness_max_path_m),
                        max_path_ratio=float(raw_witness_max_path_ratio),
                        corridor_m=float(raw_witness_corridor_m),
                        use_raw_subpath_geometry=bool(use_raw_subpath_geometry),
                    )
                    if raw_negative_veto:
                        continue
                    raw_source_angle, raw_target_angle = _raw_connector_angles(
                        raw.get("best_geometry"), cont, target, proj_d
                    )
                    accepted_by_raw = (
                        int(raw.get("witness_tracks", 0)) >= int(raw_witness_min_tracks)
                        and raw_source_angle <= float(raw_witness_relaxed_max_angle_deg)
                        and raw_target_angle <= float(raw_target_max_axis_angle_deg)
                    )
                    raster_ok = (
                        ang <= float(max_approach_angle_deg)
                        and ev["evidence_fraction"] >= float(min_evidence_fraction)
                    )
                    if bool(require_raw_witness):
                        if not accepted_by_raw:
                            continue
                    elif not (raster_ok or accepted_by_raw):
                        continue
                    parallel_bridge = False
                    parallel_metrics = {}
                    if raw_witness_index is not None and getattr(raw_witness_index, "enabled", False):
                        from .raw_track_topology import is_transverse_parallel_bridge
                        parallel_bridge, parallel_metrics = is_transverse_parallel_bridge(
                            connector, incident[ep][0], ep, target, proj_d,
                            parallel_axis_max_angle_deg=float(raw_parallel_guard_max_axis_angle_deg),
                            transverse_min_angle_deg=float(raw_parallel_guard_min_cross_angle_deg),
                        )
                        if parallel_bridge and int(raw.get("witness_tracks", 0)) < int(raw_witness_min_tracks):
                            source_side_tracks = raw_witness_index.count_tracks_near_geometry(
                                incident[ep][0]["geometry"], float(raw_witness_endpoint_snap_m)
                            )
                            target_side_tracks = raw_witness_index.count_tracks_near_geometry(
                                target["geometry"], float(raw_witness_endpoint_snap_m)
                            )
                            if (
                                source_side_tracks >= int(raw_parallel_guard_min_side_tracks)
                                and target_side_tracks >= int(raw_parallel_guard_min_side_tracks)
                            ):
                                continue
                    # Raw topology is authoritative.  Use an explainable
                    # lexicographic key instead of one opaque weighted score.
                    rank_fields = {
                        "direct_witness_tracks": int(raw.get("witness_tracks", 0) or 0),
                        "distinct_raw_tracks": int(raw.get("witness_tracks", 0) or 0),
                        "raw_density_support": int(raw_side_a_tracks) + int(raw_side_b_tracks),
                        "evidence_fraction": float(ev.get("evidence_fraction", 0.0)),
                        "support_fraction": float(ev.get("support_fraction", 0.0)),
                        "approach_angle_deg": float(raw_source_angle if accepted_by_raw else ang),
                        "distance_m": float(gap),
                    }
                    rank_key = junction_candidate_key(rank_fields)
                    cand = (
                        rank_key, ep, target_id, proj_d, q, gap, ang, ev, cycle_closure,
                        existing_detour_m, raw, bool(accepted_by_raw), bool(parallel_bridge), parallel_metrics,
                        int(raw_side_a_tracks), int(raw_side_b_tracks), rank_fields
                    )
                    if best is None or rank_key > best[0]:
                        best = cand
                if best is not None:
                    candidates.append(best)

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            used_endpoints: Set[int] = set()
            used_targets: Set[int] = set()
            accepted = []
            # Track component merges caused by candidates already accepted in THIS
            # round.  Testing only against the pre-round graph can otherwise accept
            # two A<->B repairs and create an artificial cycle.
            round_component_uf = _UF(max(comp.values()) + 1 if comp else 0)
            next_node = max(nodes) + 1 if nodes else 0
            next_edge = max(edge_by_id) + 1 if edge_by_id else 0

            cycle_components_used: Set[int] = set()
            for (rank_key, ep, target_id, proj_d, q, gap, ang, ev, cycle_closure, existing_detour_m,
                 raw, accepted_by_raw, parallel_bridge, parallel_metrics, raw_side_a_tracks, raw_side_b_tracks, rank_fields) in candidates:
                if len(records) + len(accepted) >= int(max_repairs):
                    break
                if ep in used_endpoints or target_id in used_targets:
                    continue
                target = edge_by_id.get(int(target_id))
                if target is None:
                    continue
                target_geom = target["geometry"]
                tu, tv = int(target["u"]), int(target["v"])
                target_len = float(target_geom.length)
                source_cid = comp.get(int(ep))
                target_cid = comp.get(int(tu))
                if source_cid is None or target_cid is None:
                    continue
                same_component_before = int(source_cid) == int(target_cid)
                if same_component_before:
                    if not bool(allow_cycle_closure) or int(source_cid) in cycle_components_used:
                        continue
                elif round_component_uf.find(int(source_cid)) == round_component_uf.find(int(target_cid)):
                    # Do not turn two cross-component repairs from the same stale round
                    # into an unverified cycle. It can be reconsidered next round.
                    continue

                # If projection is already very close to an existing target node, use it.
                if proj_d <= float(node_merge_m):
                    junction_node = tu
                    q = _node_xy(nodes, tu)
                    connector = LineString([tuple(_node_xy(nodes, ep)), tuple(q)])
                    split_parts = None
                elif target_len - proj_d <= float(node_merge_m):
                    junction_node = tv
                    q = _node_xy(nodes, tv)
                    connector = LineString([tuple(_node_xy(nodes, ep)), tuple(q)])
                    split_parts = None
                else:
                    split_parts = _split_linestring_at_distance(target_geom, proj_d)
                    if split_parts is None:
                        continue
                    g1, g2, qtuple = split_parts
                    q = np.asarray(qtuple, dtype=float)
                    junction_node = next_node
                    next_node += 1
                    connector = LineString([tuple(_node_xy(nodes, ep)), tuple(q)])
                if connector.length <= 0.35 or connector.length > float(max_snap_m) + float(node_merge_m) + 1e-9:
                    continue

                # If the projection was merged into an existing target node, q may have
                # moved by up to node_merge_m. Re-evaluate the ACTUAL connector rather
                # than approving it with the pre-snap angle/evidence from the projection.
                cont_final = _continuation_vector(incident[ep][0], ep)
                if cont_final is None:
                    continue
                final_vec = np.asarray(q - _node_xy(nodes, ep), dtype=float)
                final_ang = _angle_deg(cont_final, final_vec)
                final_ev = _sample_line_evidence(connector, sampler, evidence_distance_m)
                final_raw, final_side_a_tracks, final_side_b_tracks, final_raw_negative_veto = _raw_transition_evidence(
                    raw_witness_index, connector,
                    [incident[ep][0]["geometry"]], [target["geometry"]],
                    min_witness_tracks=max(int(raw_witness_min_tracks), int(raw_parallel_guard_min_side_tracks)),
                    side_distance_m=float(raw_witness_endpoint_snap_m),
                    max_path_m=float(raw_witness_max_path_m),
                    max_path_ratio=float(raw_witness_max_path_ratio),
                    corridor_m=float(raw_witness_corridor_m),
                    use_raw_subpath_geometry=bool(use_raw_subpath_geometry),
                )
                if final_raw_negative_veto:
                    continue
                final_raw_source_angle, final_raw_target_angle = _raw_connector_angles(
                    final_raw.get("best_geometry"), cont_final, target, proj_d
                )
                final_accepted_by_raw = (
                    int(final_raw.get("witness_tracks", 0)) >= int(raw_witness_min_tracks)
                    and final_raw_source_angle <= float(raw_witness_relaxed_max_angle_deg)
                    and final_raw_target_angle <= float(raw_target_max_axis_angle_deg)
                )
                final_raster_ok = (
                    final_ang <= float(max_approach_angle_deg)
                    and final_ev["evidence_fraction"] >= float(min_evidence_fraction)
                )
                if bool(require_raw_witness):
                    if not final_accepted_by_raw:
                        continue
                elif not (final_raster_ok or final_accepted_by_raw):
                    continue
                if bool(use_raw_subpath_geometry) and final_accepted_by_raw:
                    raw_geom = final_raw.get("best_geometry")
                    if isinstance(raw_geom, LineString) and raw_geom.length > 0:
                        connector = raw_geom
                        final_ang = float(final_raw_source_angle)
                        final_ev = _sample_line_evidence(connector, sampler, evidence_distance_m)
                final_parallel_bridge = False
                final_parallel_metrics = {}
                if raw_witness_index is not None and getattr(raw_witness_index, "enabled", False):
                    from .raw_track_topology import is_transverse_parallel_bridge
                    final_parallel_bridge, final_parallel_metrics = is_transverse_parallel_bridge(
                        connector, incident[ep][0], ep, target, proj_d,
                        parallel_axis_max_angle_deg=float(raw_parallel_guard_max_axis_angle_deg),
                        transverse_min_angle_deg=float(raw_parallel_guard_min_cross_angle_deg),
                    )
                    if final_parallel_bridge and int(final_raw.get("witness_tracks", 0)) < int(raw_witness_min_tracks):
                        source_side_tracks = raw_witness_index.count_tracks_near_geometry(
                            incident[ep][0]["geometry"], float(raw_witness_endpoint_snap_m)
                        )
                        target_side_tracks = raw_witness_index.count_tracks_near_geometry(
                            target["geometry"], float(raw_witness_endpoint_snap_m)
                        )
                        if (
                            source_side_tracks >= int(raw_parallel_guard_min_side_tracks)
                            and target_side_tracks >= int(raw_parallel_guard_min_side_tracks)
                        ):
                            continue
                final_detour_m = float(existing_detour_m)
                if same_component_before:
                    if split_parts is None:
                        closure_ok, final_detour_m = cycle_closure_allowed(
                            weighted, ep, int(junction_node), float(connector.length),
                            min_detour_ratio=cycle_min_detour_ratio,
                            min_detour_excess_m=cycle_min_detour_excess_m,
                        )
                    else:
                        closure_ok, final_detour_m = endpoint_to_edge_cycle_closure_allowed(
                            weighted, ep, tu, tv, proj_d, target_len, float(connector.length),
                            min_detour_ratio=cycle_min_detour_ratio,
                            min_detour_excess_m=cycle_min_detour_excess_m,
                        )
                    if not closure_ok:
                        continue
                source_edge_id = int(incident[ep][0]["edge_id"])
                allowed_target_touches = [Point(float(q[0]), float(q[1]))] if split_parts is None else []
                if _crosses_unmodelled_existing_edge(
                    connector,
                    edge_geoms,
                    edge_ids_for_tree,
                    edge_tree,
                    {source_edge_id, int(target_id)},
                    allowed_touch_points=allowed_target_touches,
                ):
                    continue

                # Candidates in the same round were all generated against the old
                # network.  Guard against two newly accepted connectors crossing or
                # overlapping each other before either has been inserted into the graph.
                conflicts_new_connector = False
                for prev in accepted:
                    inter_new = safe_intersection(
                        connector, prev["connector"], operation="junction_new_connector_intersection",
                        context={"endpoint_node_id": int(ep), "target_edge_id": int(target_id)},
                    )
                    if inter_new is None:
                        conflicts_new_connector = True
                        break
                    if inter_new.is_empty:
                        continue
                    if _intersection_only_at_allowed_points(inter_new, allowed_target_touches):
                        continue
                    conflicts_new_connector = True
                    break
                if conflicts_new_connector:
                    continue

                accepted.append({
                    "ep": ep, "target_id": int(target_id), "junction_node": int(junction_node),
                    "split_parts": split_parts, "connector": connector, "q": q,
                    "gap": float(connector.length),
                    "euclidean_gap_m": float(np.linalg.norm(q - _node_xy(nodes, ep))),
                    "angle": final_ang, "target_angle": float(final_raw_target_angle), "evidence": final_ev,
                    "raw_witness": final_raw,
                    "accepted_by_raw_witness": bool(final_accepted_by_raw),
                    "parallel_bridge_candidate": bool(final_parallel_bridge),
                    "parallel_metrics": final_parallel_metrics,
                    "raw_side_a_tracks": int(final_side_a_tracks),
                    "raw_side_b_tracks": int(final_side_b_tracks),
                    "ranking": {
                        "direct_witness_tracks": int(final_raw.get("witness_tracks", 0) or 0),
                        "distinct_raw_tracks": int(final_raw.get("witness_tracks", 0) or 0),
                        "raw_density_support": int(final_side_a_tracks) + int(final_side_b_tracks),
                        "evidence_fraction": float(final_ev.get("evidence_fraction", 0.0)),
                        "support_fraction": float(final_ev.get("support_fraction", 0.0)),
                        "approach_angle_deg": float(final_raw_source_angle if final_accepted_by_raw else final_ang),
                        "distance_m": float(connector.length),
                    },
                    "cycle_closure": bool(same_component_before),
                    "existing_detour_m": float(final_detour_m),
                    "new_edge_id": next_edge,
                })
                if same_component_before:
                    cycle_components_used.add(int(source_cid))
                else:
                    round_component_uf.union(int(source_cid), int(target_cid))
                next_edge += 3 if split_parts is not None else 1
                used_endpoints.add(ep)
                used_targets.add(int(target_id))

            if not accepted:
                break

            drop_target_ids = {a["target_id"] for a in accepted if a["split_parts"] is not None}
            new_nodes = [dict(n) for n in net.nodes]
            new_edges = [dict(e) for e in net.edges if int(e["edge_id"]) not in drop_target_ids]
            next_eid = max([int(e["edge_id"]) for e in new_edges] or [-1]) + 1

            for a in accepted:
                target = edge_by_id[a["target_id"]]
                jn = int(a["junction_node"])
                if a["split_parts"] is not None:
                    g1, g2, qtuple = a["split_parts"]
                    new_nodes.append({
                        "node_id": jn, "x": float(qtuple[0]), "y": float(qtuple[1]),
                        "geometry": Point(float(qtuple[0]), float(qtuple[1])),
                        "node_type": "intersection", "degree": 3,
                        "junction_repair_node": True,
                    })
                    e1 = dict(target); e2 = dict(target)
                    e1.update({"edge_id": next_eid, "u": int(target["u"]), "v": jn, "geometry": g1, "length_m": float(g1.length)})
                    next_eid += 1
                    e2.update({"edge_id": next_eid, "u": jn, "v": int(target["v"]), "geometry": g2, "length_m": float(g2.length)})
                    next_eid += 1
                    new_edges.extend([e1, e2])
                connector = _new_connector_edge(
                    next_eid, int(a["ep"]), jn, a["connector"],
                    str(repair_label), a["evidence"], float(a["angle"]),
                    audit_max_connector_m=float(max_snap_m) + float(node_merge_m),
                    audit_max_angle_deg=float(max_approach_angle_deg),
                    audit_raw_max_angle_deg=float(raw_witness_relaxed_max_angle_deg),
                    audit_min_evidence_fraction=float(min_evidence_fraction),
                )
                raw_info = a.get("raw_witness", {})
                connector.update({
                    "raw_witness_tracks": int(raw_info.get("witness_tracks", 0) or 0),
                    "raw_witness_track_id": str(raw_info.get("best_track_id", "")),
                    "raw_witness_source_path_m": float(raw_info.get("best_source_path_m", float("inf"))),
                    "accepted_by_raw_witness": bool(a.get("accepted_by_raw_witness", False)),
                    "parallel_bridge_candidate": bool(a.get("parallel_bridge_candidate", False)),
                })
                if str(repair_label).startswith("final_connectivity_rescue"):
                    connector.update({
                        "source_type": "s7_final_connectivity_rescue",
                        "junction_repair": False,
                        "approved_junction_repair": False,
                        "raw_track_gap_repair": True,
                        "final_connectivity_rescue": True,
                        "raw_continuity_class": "raw_subpath_transition",
                        "raw_gap_euclidean_m": float(a.get("euclidean_gap_m", a["connector"].length)),
                        "raw_gap_angle_a_deg": float(a.get("angle", 180.0)),
                        "raw_gap_angle_b_deg": float(a.get("target_angle", 180.0)),
                        "raw_gap_cycle_closure": False,
                        "geometry_origin": "s7_final_connectivity_rescue_v0_5",
                        "retained_reason": "final_raw_track_continuity_rescue",
                    })
                next_eid += 1
                new_edges.append(connector)
                records.append({
                    "round": round_no,
                    "mode": str(repair_label),
                    "endpoint_node": int(a["ep"]),
                    "target_edge_id": int(a["target_id"]),
                    "junction_node": jn,
                    "connector_length_m": float(a["connector"].length),
                    "approach_angle_deg": float(a["angle"]),
                    "target_axis_angle_deg": float(a.get("target_angle", 180.0)),
                    "final_connectivity_rescue": str(repair_label).startswith("final_connectivity_rescue"),
                    "cycle_closure": bool(a.get("cycle_closure", False)),
                    "existing_detour_m": float(a.get("existing_detour_m", float("inf"))),
                    "raw_witness_tracks": int(a.get("raw_witness", {}).get("witness_tracks", 0) or 0),
                    "raw_witness_track_id": str(a.get("raw_witness", {}).get("best_track_id", "")),
                    "raw_witness_source_path_m": float(a.get("raw_witness", {}).get("best_source_path_m", float("inf"))),
                    "accepted_by_raw_witness": bool(a.get("accepted_by_raw_witness", False)),
                    "parallel_bridge_candidate": bool(a.get("parallel_bridge_candidate", False)),
                    "raw_side_a_tracks": int(a.get("raw_side_a_tracks", 0)),
                    "raw_side_b_tracks": int(a.get("raw_side_b_tracks", 0)),
                    "ranking_direct_witness_tracks": int(a.get("ranking", {}).get("direct_witness_tracks", 0)),
                    "ranking_distinct_raw_tracks": int(a.get("ranking", {}).get("distinct_raw_tracks", 0)),
                    "ranking_raw_density_support": int(a.get("ranking", {}).get("raw_density_support", 0)),
                    "ranking_key": repr(junction_candidate_key(a.get("ranking", {}))),
                    **a.get("parallel_metrics", {}),
                    **a["evidence"],
                    "geometry": a["connector"],
                })

            net = NetworkData(
                new_nodes, new_edges, net.crs, net.transform, net.width, net.height, dict(net.metadata)
            )
            net = reindex_network(net)
            print(
                f"    [junction:endpoint-edge] round={round_no}, accepted={len(accepted):,}, total={len(records):,}",
                flush=True,
            )
    finally:
        if sampler is not None:
            sampler.close()
    return reindex_network(net), records


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def repair_endpoint_clusters(
    network: NetworkData,
    support_path=None,
    kde_path=None,
    distance_path=None,
    cluster_radius_m: float = 8.0,
    center_search_m: float = 4.0,
    max_approach_angle_deg: float = 75.0,
    min_evidence_fraction: float = 0.20,
    evidence_distance_m: float = 10.0,
    min_cluster_size: int = 3,
    max_cluster_size: int = 6,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.8,
    cycle_min_detour_excess_m: float = 8.0,
    max_repairs: int = 2000,
    raw_witness_index=None,
    raw_witness_min_tracks: int = 1,
    raw_witness_endpoint_snap_m: float = 6.0,
    raw_witness_max_path_m: float = 60.0,
    raw_witness_max_path_ratio: float = 3.0,
    raw_witness_corridor_m: float = 10.0,
    raw_witness_relaxed_max_angle_deg: float = 85.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Create one junction hub when several disconnected road ends almost meet.

    This targets X/Y intersections where no through-edge survives the skeletonisation.
    Every accepted arm must point toward the hub and have local raster/track evidence.
    Distinct components are merged directly. Multiple arms from the same existing
    component are allowed only when the current graph path between them is a long
    detour relative to the route through the new hub, which preserves genuine road
    junction loops without creating cheap local shortcuts.
    """
    net = reindex_network(network.copy())
    nodes = {int(n["node_id"]): n for n in net.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    graph = nx.Graph()
    for e in net.edges:
        u, v = int(e["u"]), int(e["v"])
        incident[u].append(e); incident[v].append(e)
        graph.add_edge(u, v)
    weighted = weighted_graph(net)
    comp = {}
    for cid, cc in enumerate(nx.connected_components(graph)):
        for n in cc:
            comp[int(n)] = cid
    endpoints = [
        n for n, es in incident.items()
        if len(es) == 1 and not _protected(es[0])
    ]
    if len(endpoints) < 2:
        return net, []

    xy = np.asarray([_node_xy(nodes, n) for n in endpoints], dtype=float)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(float(cluster_radius_m), output_type="set")
    uf = _UF(len(endpoints))
    for i, j in pairs:
        # Geometric clustering must not depend on current graph connectivity. In a
        # fragmented mesh, several broken approaches to one physical junction can
        # already belong to the same component through a long alternative route.
        uf.union(int(i), int(j))
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(endpoints)):
        clusters[uf.find(i)].append(i)

    edge_geoms = [e["geometry"] for e in net.edges]
    edge_ids_for_tree = [int(e["edge_id"]) for e in net.edges]
    edge_tree = STRtree(edge_geoms) if edge_geoms else None

    raster_paths = {"support": support_path, "kde": kde_path, "distance": distance_path}
    raster_paths = {k: Path(v) for k, v in raster_paths.items() if v is not None and Path(v).exists()}
    sampler = RasterBlockSampler(raster_paths, cache_blocks=128) if raster_paths else None
    records = []
    new_nodes = [dict(n) for n in net.nodes]
    new_edges = [dict(e) for e in net.edges]
    next_node = max(nodes) + 1 if nodes else 0
    next_edge = max([int(e["edge_id"]) for e in new_edges] or [-1]) + 1

    try:
        used_endpoints: Set[int] = set()
        for inds in sorted(clusters.values(), key=len, reverse=True):
            if len(records) >= int(max_repairs):
                break
            if len(inds) < max(2, int(min_cluster_size)):
                continue
            eps = [int(endpoints[i]) for i in inds]
            pts = np.vstack([_node_xy(nodes, n) for n in eps])
            # A robust median is less sensitive than the mean to one noisy broken end.
            seed = np.median(pts, axis=0)
            center = _local_best_center(seed, sampler, float(center_search_m), grid_step_m=1.0)

            # Evaluate each arm independently. One noisy endpoint must not veto a
            # junction. v0.5 then filters same-component arms using graph detour
            # evidence instead of discarding them categorically.
            valid_by_component: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for ep, pxy in zip(eps, pts):
                if ep in used_endpoints:
                    continue
                vec = center - pxy
                d = float(np.linalg.norm(vec))
                if d <= 0.35 or d > float(cluster_radius_m) * 1.35:
                    continue
                cont = _continuation_vector(incident[ep][0], ep)
                if cont is None:
                    continue
                ang = _angle_deg(cont, vec)
                geom = LineString([tuple(pxy), tuple(center)])
                ev = _sample_line_evidence(geom, sampler, evidence_distance_m)
                arm_raw = {"witness_tracks": 0, "best_track_id": "", "best_source_path_m": float("inf"), "contribution_ids": set()}
                arm_side_tracks = 0
                if raw_witness_index is not None and getattr(raw_witness_index, "enabled", False):
                    arm_raw = raw_witness_index.query_connector(
                        geom, endpoint_snap_m=float(raw_witness_endpoint_snap_m),
                        max_path_m=float(raw_witness_max_path_m),
                        max_path_ratio=float(raw_witness_max_path_ratio),
                        corridor_m=float(raw_witness_corridor_m),
                    )
                    arm_side_tracks = raw_witness_index.count_tracks_near_geometry(
                        incident[ep][0]["geometry"], float(raw_witness_endpoint_snap_m)
                    )
                    # Raw data is present on this arm but never continues to the proposed
                    # hub: this is explicit negative topology evidence in v0.5.
                    if (
                        int(arm_side_tracks) >= int(raw_witness_min_tracks)
                        and int(arm_raw.get("witness_tracks", 0) or 0) < int(raw_witness_min_tracks)
                    ):
                        continue
                accepted_by_raw = (
                    int(arm_raw.get("witness_tracks", 0) or 0) >= int(raw_witness_min_tracks)
                    and ang <= float(raw_witness_relaxed_max_angle_deg)
                )
                raster_ok = (
                    ang <= float(max_approach_angle_deg)
                    and float(ev.get("evidence_fraction", 0.0)) >= float(min_evidence_fraction)
                )
                if not (accepted_by_raw or raster_ok):
                    continue
                source_edge_id = int(incident[ep][0]["edge_id"])
                if _crosses_unmodelled_existing_edge(
                    geom,
                    edge_geoms,
                    edge_ids_for_tree,
                    edge_tree,
                    {source_edge_id},
                ):
                    continue
                cid = int(comp.get(ep, -1))
                rank_fields = {
                    "direct_witness_tracks": int(arm_raw.get("witness_tracks", 0) or 0),
                    "distinct_raw_tracks": int(len(set(arm_raw.get("contribution_ids", set())))),
                    "raw_density_support": int(arm_side_tracks),
                    "evidence_fraction": float(ev.get("evidence_fraction", 0.0)),
                    "support_fraction": float(ev.get("support_fraction", 0.0)),
                    "approach_angle_deg": float(ang),
                    "distance_m": float(d),
                }
                rank_key = junction_candidate_key(rank_fields)
                valid_by_component[cid].append({
                    "ep": int(ep), "geom": geom, "angle": float(ang),
                    "evidence": ev, "rank_key": rank_key, "ranking": rank_fields, "component": cid,
                    "raw_witness": arm_raw, "raw_side_tracks": int(arm_side_tracks),
                    "accepted_by_raw_witness": bool(accepted_by_raw),
                    "cycle_closure": False, "existing_detour_m": float("inf"),
                })

            arm_rows: List[Dict[str, Any]] = []
            for cid, group in valid_by_component.items():
                group.sort(key=lambda r: r["rank_key"], reverse=True)
                if not group:
                    continue
                # The best arm from each component can participate in merging that
                # component into the shared hub without creating a cycle.
                kept = [group[0]]
                arm_rows.append(group[0])
                for row in group[1:]:
                    if not bool(allow_cycle_closure):
                        continue
                    compatible = False
                    best_detour = float("-inf")
                    for anchor in kept:
                        via_hub = float(row["geom"].length + anchor["geom"].length)
                        ok, detour = cycle_closure_allowed(
                            weighted, int(row["ep"]), int(anchor["ep"]), via_hub,
                            min_detour_ratio=cycle_min_detour_ratio,
                            min_detour_excess_m=cycle_min_detour_excess_m,
                        )
                        if ok:
                            compatible = True
                            best_detour = max(best_detour, float(detour))
                    if compatible:
                        row["cycle_closure"] = True
                        row["existing_detour_m"] = float(best_detour)
                        kept.append(row)
                        arm_rows.append(row)

            arm_rows.sort(key=lambda r: r["rank_key"], reverse=True)
            if len(arm_rows) < int(min_cluster_size):
                continue
            # Do not arbitrarily select a subset from an over-crowded junction.
            # After per-arm filtering, too many still-valid disconnected approaches
            # means the local topology is ambiguous and should be left for stronger
            # trajectory-level evidence rather than forcing a partial shared hub.
            if int(max_cluster_size) > 0 and len(arm_rows) > int(max_cluster_size):
                continue
            if len(records) + len(arm_rows) > int(max_repairs):
                continue

            hub = next_node
            next_node += 1
            new_nodes.append({
                "node_id": hub, "x": float(center[0]), "y": float(center[1]),
                "geometry": Point(float(center[0]), float(center[1])),
                "node_type": "intersection", "degree": len(arm_rows),
                "junction_repair_node": True,
            })
            for arm in arm_rows:
                ep = int(arm["ep"]); geom = arm["geom"]; ang = float(arm["angle"]); ev = arm["evidence"]
                connector = _new_connector_edge(
                    next_edge, ep, hub, geom, "endpoint_cluster", ev, ang,
                    audit_max_connector_m=float(cluster_radius_m) * 1.35,
                    audit_max_angle_deg=float(max_approach_angle_deg),
                    audit_raw_max_angle_deg=float(raw_witness_relaxed_max_angle_deg),
                    audit_min_evidence_fraction=float(min_evidence_fraction),
                )
                raw_info = arm.get("raw_witness", {})
                connector.update({
                    "raw_witness_tracks": int(raw_info.get("witness_tracks", 0) or 0),
                    "raw_witness_track_id": str(raw_info.get("best_track_id", "")),
                    "raw_witness_source_path_m": float(raw_info.get("best_source_path_m", float("inf"))),
                    "accepted_by_raw_witness": bool(arm.get("accepted_by_raw_witness", False)),
                })
                new_edges.append(connector)
                records.append({
                    "mode": "endpoint_cluster",
                    "endpoint_node": int(ep),
                    "junction_node": int(hub),
                    "cluster_size": len(arm_rows),
                    "cluster_size_raw": len(inds),
                    "connector_length_m": float(geom.length),
                    "approach_angle_deg": float(ang),
                    "cycle_closure": bool(arm.get("cycle_closure", False)),
                    "existing_detour_m": float(arm.get("existing_detour_m", float("inf"))),
                    "raw_witness_tracks": int(arm.get("raw_witness", {}).get("witness_tracks", 0) or 0),
                    "raw_witness_track_id": str(arm.get("raw_witness", {}).get("best_track_id", "")),
                    "raw_side_tracks": int(arm.get("raw_side_tracks", 0)),
                    "accepted_by_raw_witness": bool(arm.get("accepted_by_raw_witness", False)),
                    "ranking_direct_witness_tracks": int(arm.get("ranking", {}).get("direct_witness_tracks", 0)),
                    "ranking_distinct_raw_tracks": int(arm.get("ranking", {}).get("distinct_raw_tracks", 0)),
                    "ranking_raw_density_support": int(arm.get("ranking", {}).get("raw_density_support", 0)),
                    "ranking_key": repr(arm.get("rank_key", ())),
                    **ev,
                    "geometry": geom,
                })
                next_edge += 1
                used_endpoints.add(ep)
    finally:
        if sampler is not None:
            sampler.close()

    if records:
        net = NetworkData(new_nodes, new_edges, net.crs, net.transform, net.width, net.height, dict(net.metadata))
        net = reindex_network(net)
    print(f"    [junction:endpoint-cluster] accepted_arms={len(records):,}", flush=True)
    return net, records




def _edge_has_strong_raw_evidence(edge: Dict[str, Any]) -> bool:
    """v0.5 evidence-first veto against deleting a real short road."""
    tracks = int(edge.get("distinct_tracks", 0) or 0)
    support_mean = float(edge.get("support_mean", 0.0) or 0.0)
    support_max = float(edge.get("support_max", 0.0) or 0.0)
    coverage = float(edge.get("coverage_ratio", 0.0) or 0.0)
    kde_mean = float(edge.get("kde_mean", 0.0) or 0.0)
    return bool(
        tracks >= 2
        or coverage >= 0.30
        or support_mean >= 1.20
        or support_max >= 3.0
        or (tracks >= 1 and kde_mean >= 1.0)
    )

def prune_short_isolated_components(
    network: NetworkData,
    hard_max_length_m: float = 25.0,
    weak_max_length_m: float = 60.0,
    weak_max_tracks: int = 0,
    weak_max_confidence: float = 0.42,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove disconnected short components even if raster origin says 'primary'.

    v0.5 preserved every primary component.  That is why a 10–20 m isolated red
    segment could survive forever.  v0.5 uses component-level topology and counts
    every original edge (including parallel multi-edges):
      * <= hard_max_length_m: remove directly (unless it is the only component);
      * <= weak_max_length_m: remove when no matched tracks and confidence is weak.
    """
    net = reindex_network(network.copy())
    graph = nx.Graph()
    for e in net.edges:
        graph.add_edge(int(e["u"]), int(e["v"]))
    comps = list(nx.connected_components(graph))
    if len(comps) <= 1:
        return net, []

    # nx.Graph stores only one edge per node pair.  v0.5 attached edge_id to that
    # graph and silently lost parallel edges, under-counting component length and
    # deleting only one member of a multi-edge component.  Map nodes to components,
    # then collect ALL original network edges without collapsing parallel geometry.
    node_component: Dict[int, int] = {}
    for cid, cc in enumerate(comps):
        for nid in cc:
            node_component[int(nid)] = int(cid)
    component_edges: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in net.edges:
        cid = node_component.get(int(e["u"]))
        if cid is not None:
            component_edges[int(cid)].append(e)

    drop: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    for cid, cc in enumerate(comps):
        edges = component_edges.get(int(cid), [])
        if not edges:
            continue
        length = float(sum(float(e["geometry"].length) for e in edges))
        max_tracks = max(int(e.get("distinct_tracks", 0) or 0) for e in edges)
        confs = [edge_confidence(e) for e in edges]
        max_conf = max(confs) if confs else 0.0
        strong_raw = any(_edge_has_strong_raw_evidence(e) for e in edges)
        if length <= float(hard_max_length_m) and strong_raw:
            continue
        if length <= float(hard_max_length_m):
            reason = "hard_short_isolated_component_weak_evidence"
        elif (
            length <= float(weak_max_length_m)
            and max_tracks <= int(weak_max_tracks)
            and max_conf <= float(weak_max_confidence)
        ):
            reason = "weak_short_isolated_component"
        else:
            continue
        for e in edges:
            drop.add(int(e["edge_id"]))
            row = dict(e)
            row.update({
                "drop_reason": reason,
                "component_length_m": length,
                "component_max_tracks": max_tracks,
                "component_max_confidence": max_conf,
            })
            rows.append(row)

    if not drop:
        return net, []
    net.edges = [e for e in net.edges if int(e["edge_id"]) not in drop]
    return reindex_network(net), rows


def _leaf_chains(network: NetworkData):
    nodes = {int(n["node_id"]): n for n in network.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    degree = defaultdict(int)
    for e in network.edges:
        u, v = int(e["u"]), int(e["v"])
        incident[u].append(e); incident[v].append(e)
        degree[u] += 1; degree[v] += 1
    seen: Set[int] = set()
    chains = []
    for start, d in list(degree.items()):
        if d != 1:
            continue
        first = incident[start][0]
        if int(first["edge_id"]) in seen:
            continue
        path_edges = []
        prev = None
        cur = int(start)
        while True:
            options = [e for e in incident[cur] if int(e["edge_id"]) not in seen and (prev is None or int(e["edge_id"]) != int(prev["edge_id"]))]
            if not options:
                break
            e = options[0]
            eid = int(e["edge_id"])
            seen.add(eid)
            path_edges.append(e)
            nxt = int(e["v"]) if int(e["u"]) == cur else int(e["u"])
            prev = e
            cur = nxt
            if degree[cur] != 2:
                break
        chains.append((int(start), int(cur), path_edges, int(degree.get(cur, 0))))
    return chains


def prune_short_leaf_chains(
    network: NetworkData,
    hard_max_length_m: float = 6.0,
    weak_max_length_m: float = 22.0,
    weak_max_tracks: int = 0,
    weak_max_confidence: float = 0.40,
    max_rounds: int = 4,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove tiny endpoint-to-junction spurs that clutter intersections.

    The whole leaf chain is removed, not just one edge, so a 2 m raster staircase
    cannot survive as several tiny red pieces.
    """
    net = reindex_network(network.copy())
    removed: List[Dict[str, Any]] = []
    for round_no in range(1, max(1, int(max_rounds)) + 1):
        drop: Set[int] = set()
        for start, end, edges, end_degree in _leaf_chains(net):
            if end_degree < 3 or not edges:
                continue
            if any(_protected(e) for e in edges):
                continue
            length = float(sum(float(e["geometry"].length) for e in edges))
            max_tracks = max(int(e.get("distinct_tracks", 0) or 0) for e in edges)
            max_conf = max(edge_confidence(e) for e in edges)
            strong_raw = any(_edge_has_strong_raw_evidence(e) for e in edges)
            if length <= float(hard_max_length_m) and strong_raw:
                continue
            if length <= float(hard_max_length_m):
                reason = "hard_tiny_junction_spur_weak_evidence"
            elif (
                length <= float(weak_max_length_m)
                and max_tracks <= int(weak_max_tracks)
                and max_conf <= float(weak_max_confidence)
            ):
                reason = "weak_short_junction_spur"
            else:
                continue
            for e in edges:
                drop.add(int(e["edge_id"]))
                row = dict(e)
                row.update({
                    "drop_reason": reason,
                    "leaf_chain_length_m": length,
                    "leaf_chain_max_tracks": max_tracks,
                    "leaf_chain_max_confidence": max_conf,
                    "round": round_no,
                })
                removed.append(row)
        if not drop:
            break
        net.edges = [e for e in net.edges if int(e["edge_id"]) not in drop]
        net = reindex_network(net)
    return net, removed


def audit_junction_repairs(
    network: NetworkData,
    max_connector_m: float = 12.0,
    max_angle_deg: float = 75.0,
    min_evidence_fraction: float = 0.20,
) -> List[Dict[str, Any]]:
    """Audit junction connectors against the envelope used when they were created.

    Each v0.5 connector stores its own length, angle and evidence limits. Raw-track
    accepted connectors use the relaxed raw-witness angle limit; raster-only
    connectors use the ordinary approach-angle limit. This keeps final audit
    consistent with the creation gate instead of reinterpreting all repair modes
    through one global threshold.
    """
    rows: List[Dict[str, Any]] = []
    nodes = {int(n["node_id"]): n for n in network.nodes}
    for e in network.edges:
        if not bool(e.get("junction_repair")):
            continue
        geom = e.get("geometry")
        u, v = int(e["u"]), int(e["v"])
        endpoint_ok = False
        if isinstance(geom, LineString) and u in nodes and v in nodes and len(geom.coords) >= 2:
            a, b = geom.coords[0], geom.coords[-1]
            pu = (float(nodes[u]["x"]), float(nodes[u]["y"]))
            pv = (float(nodes[v]["x"]), float(nodes[v]["y"]))
            direct = (
                math.hypot(a[0]-pu[0], a[1]-pu[1]) <= 1e-6
                and math.hypot(b[0]-pv[0], b[1]-pv[1]) <= 1e-6
            )
            reverse = (
                math.hypot(a[0]-pv[0], a[1]-pv[1]) <= 1e-6
                and math.hypot(b[0]-pu[0], b[1]-pu[1]) <= 1e-6
            )
            endpoint_ok = bool(direct or reverse)

        length = float(geom.length) if isinstance(geom, LineString) else float("inf")
        angle = _num(e, "junction_repair_angle_deg", 180.0)
        ev = _num(e, "junction_repair_evidence_fraction", 0.0)
        raw_tracks = int(e.get("raw_witness_tracks", 0) or 0)
        raw_approved = bool(e.get("accepted_by_raw_witness", False) and raw_tracks >= 1)

        length_limit = _num(e, "junction_audit_max_connector_m", float(max_connector_m))
        ordinary_angle_limit = _num(e, "junction_audit_max_angle_deg", float(max_angle_deg))
        raw_angle_limit = _num(e, "junction_audit_raw_max_angle_deg", ordinary_angle_limit)
        evidence_limit = _num(e, "junction_audit_min_evidence_fraction", float(min_evidence_fraction))
        applied_angle_limit = raw_angle_limit if raw_approved else ordinary_angle_limit

        geometry_ok = bool(isinstance(geom, LineString) and geom.length > 0 and geom.is_simple)
        length_ok = bool(np.isfinite(length) and length <= float(length_limit) + 1e-9)
        angle_ok = bool(angle <= float(applied_angle_limit) + 1e-9)
        raster_evidence_ok = bool(ev >= float(evidence_limit) - 1e-12)
        raw_witness_ok = bool(raw_approved)
        evidence_ok = bool(raster_evidence_ok or raw_witness_ok)
        valid = bool(geometry_ok and endpoint_ok and length_ok and angle_ok and evidence_ok)

        failed_checks: List[str] = []
        if not geometry_ok:
            failed_checks.append("geometry")
        if not endpoint_ok:
            failed_checks.append("endpoint")
        if not length_ok:
            failed_checks.append("length")
        if not angle_ok:
            failed_checks.append("angle")
        if not evidence_ok:
            failed_checks.append("evidence")

        rows.append({
            "edge_id": int(e["edge_id"]),
            "junction_repair": True,
            "mode": str(e.get("junction_repair_mode", "")),
            "length_m": length if np.isfinite(length) else 0.0,
            "length_limit_m": float(length_limit),
            "angle_deg": angle,
            "angle_limit_deg": float(applied_angle_limit),
            "ordinary_angle_limit_deg": float(ordinary_angle_limit),
            "raw_angle_limit_deg": float(raw_angle_limit),
            "evidence_fraction": ev,
            "evidence_limit_fraction": float(evidence_limit),
            "raw_witness_tracks": raw_tracks,
            "accepted_by_raw_witness": raw_approved,
            "geometry_ok": geometry_ok,
            "endpoint_ok": endpoint_ok,
            "endpoint_topology_preserved": endpoint_ok,
            "length_ok": length_ok,
            "angle_ok": angle_ok,
            "raster_evidence_ok": raster_evidence_ok,
            "raw_witness_ok": raw_witness_ok,
            "evidence_ok": evidence_ok,
            "failed_checks": ";".join(failed_checks),
            "outside_baseline_m": 0.0 if valid else (length if np.isfinite(length) else 0.0),
            "outside_baseline_fraction": 0.0 if valid else 1.0,
            "audit_reason": "approved_junction_repair_v0_5" if valid else "failed_junction_repair_audit_v0_5",
            "geometry_origin": str(e.get("geometry_origin", "")),
            "geometry": geom,
        })
    return rows


def rollback_failed_junction_repairs(
    network: NetworkData,
    violations: Sequence[Dict[str, Any]],
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove only connectors that fail final v0.5 junction audit.

    If an endpoint-to-edge repair split a through-road, the two split pieces are
    retained; deleting only the synthetic connector restores safe connectivity while
    preserving the original through-road geometry. ``reindex_network`` removes any
    orphan repair nodes and recomputes degrees.
    """
    bad_ids = {int(r["edge_id"]) for r in violations if "edge_id" in r}
    if not bad_ids:
        return reindex_network(network.copy()), []

    by_id = {int(r["edge_id"]): r for r in violations if "edge_id" in r}
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for edge in network.edges:
        eid = int(edge["edge_id"])
        if eid not in bad_ids:
            kept.append(dict(edge))
            continue
        row = dict(by_id.get(eid, {}))
        row.update({
            "rollback_reason": "failed_final_junction_audit_v0_5",
            "rolled_back": True,
            "geometry": edge.get("geometry"),
        })
        rejected.append(row)

    cleaned = NetworkData(
        [dict(n) for n in network.nodes], kept, network.crs, network.transform,
        network.width, network.height, dict(network.metadata),
    )
    return reindex_network(cleaned), rejected
