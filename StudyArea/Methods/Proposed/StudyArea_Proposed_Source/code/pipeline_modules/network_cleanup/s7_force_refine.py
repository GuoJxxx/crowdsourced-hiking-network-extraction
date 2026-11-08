from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.strtree import STRtree

from .models import NetworkData, orient_coords, reindex_network
from .tile_io import RasterBlockSampler
from .topology_utils import cycle_closure_allowed, safe_intersection, weighted_graph


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _num(edge: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(edge.get(key, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _is_core(edge: Dict[str, Any]) -> bool:
    if "uturn_core_protected" in edge:
        return bool(edge.get("uturn_core_protected"))
    return bool(edge.get("uturn_protected"))


def _is_gap(edge: Dict[str, Any]) -> bool:
    return bool(edge.get("gap_repair") or edge.get("approved_gap_repair"))


def _edge_score(edge: Dict[str, Any]) -> float:
    """Raster/track evidence score available before the first HMM pass.

    The score is used only to decide which *existing* branch of a cycle to keep.
    It never creates road geometry.
    """
    support = min(1.0, _num(edge, "support_mean") / 2.0)
    support2 = min(1.0, _num(edge, "support_ge2_fraction") / 0.35)
    primary = min(1.0, _num(edge, "primary_fraction", 1.0))
    distance = _num(edge, "distance_mean_m", 12.0)
    distance_score = math.exp(-max(0.0, distance) / 8.0)
    confidence = min(1.0, _num(edge, "confidence"))
    source = str(edge.get("source_type", "primary"))
    source_prior = {
        "primary": 1.0,
        "mixed": 0.95,
        "support_rescued": 0.90,
        "auxiliary": 0.45,
    }.get(source, 0.65)
    return float(np.clip(
        0.28 * support
        + 0.22 * support2
        + 0.16 * distance_score
        + 0.12 * primary
        + 0.12 * confidence
        + 0.10 * source_prior,
        0.0, 1.0,
    ))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _edge_from_node_vector(edge: Dict[str, Any], node_id: int, probe_m: float = 8.0) -> Optional[np.ndarray]:
    """Vector from a topology node *into* the incident existing edge."""
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
    remaining = max(1.0, float(probe_m))
    target = arr[-1]
    travelled = 0.0
    for a, b in zip(arr[:-1], arr[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg <= 1e-9:
            continue
        if travelled + seg >= remaining:
            f = (remaining - travelled) / seg
            target = a + np.clip(f, 0.0, 1.0) * (b - a)
            break
        travelled += seg
    vec = np.asarray(target - origin, dtype=float)
    return vec if float(np.linalg.norm(vec)) > 1e-9 else None


def _polygon_area_xy(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _cycle_paths_between_gates(cycle: List[int], ga: int, gb: int) -> Tuple[List[int], List[int]]:
    ia, ib = cycle.index(ga), cycle.index(gb)
    if ia > ib:
        ia, ib = ib, ia
        ga, gb = gb, ga
    forward = cycle[ia:ib + 1]
    backward = [cycle[ia]]
    k = ia
    while True:
        k = (k - 1) % len(cycle)
        backward.append(cycle[k])
        if cycle[k] == cycle[ib]:
            break
    return forward, backward


def _arc_edge_options(path_nodes: Sequence[int], pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    out = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        opts = pair_edges.get((min(int(a), int(b)), max(int(a), int(b))), [])
        if not opts:
            return []
        out.append(opts)
    return out


def _representative_arc(
    path_nodes: Sequence[int],
    pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]],
) -> Optional[Tuple[LineString, List[Dict[str, Any]], Set[int]]]:
    groups = _arc_edge_options(path_nodes, pair_edges)
    if not groups:
        return None
    reps: List[Dict[str, Any]] = []
    all_ids: Set[int] = set()
    coords: List[Tuple[float, float]] = []
    for (a, b), options in zip(zip(path_nodes[:-1], path_nodes[1:]), groups):
        all_ids.update(int(e["edge_id"]) for e in options)
        rep = max(options, key=_edge_score)
        reps.append(rep)
        part = orient_coords(rep, int(a), int(b))
        if coords and part and coords[-1] == part[0]:
            part = part[1:]
        coords.extend(part)
    if len(coords) < 2:
        return None
    geom = LineString(coords)
    if geom.length <= 0:
        return None
    return geom, reps, all_ids


def _arc_score(reps: Sequence[Dict[str, Any]]) -> float:
    if not reps:
        return 0.0
    lengths = np.asarray([max(float(e["geometry"].length), 1e-6) for e in reps], dtype=float)
    scores = np.asarray([_edge_score(e) for e in reps], dtype=float)
    return float(np.dot(lengths, scores) / max(float(lengths.sum()), 1e-9))


def _arc_has_core(options_groups: Sequence[Sequence[Dict[str, Any]]]) -> bool:
    return any(_is_core(e) for group in options_groups for e in group)


def _arc_has_approved_gap(options_groups: Sequence[Sequence[Dict[str, Any]]]) -> bool:
    return any(_is_gap(e) for group in options_groups for e in group)


def _resample(line: LineString, n: int) -> np.ndarray:
    n = max(2, int(n))
    ds = np.linspace(0.0, float(line.length), n)
    return np.asarray([(line.interpolate(float(d)).x, line.interpolate(float(d)).y) for d in ds], dtype=float)


def _two_branch_width(a: LineString, b: LineString, gate_xy: np.ndarray) -> Tuple[float, float, float]:
    n = max(9, int(math.ceil(max(a.length, b.length) / 2.0)) + 1)
    aa = _resample(a, n)
    bb = _resample(b, n)
    if np.linalg.norm(aa[0] - gate_xy) > np.linalg.norm(aa[-1] - gate_xy):
        aa = aa[::-1]
    if np.linalg.norm(bb[0] - gate_xy) > np.linalg.norm(bb[-1] - gate_xy):
        bb = bb[::-1]
    d = np.linalg.norm(aa - bb, axis=1)
    return float(np.median(d)), float(np.percentile(d, 90)), float(np.max(d))


# ---------------------------------------------------------------------------
# Shape-agnostic polygon/cycle resolver
# ---------------------------------------------------------------------------

def resolve_polygon_cycles(
    network: NetworkData,
    max_perimeter_m: float = 180.0,
    max_span_m: float = 60.0,
    max_area_m2: float = 1800.0,
    force_perimeter_m: float = 100.0,
    force_span_m: float = 35.0,
    force_area_m2: float = 650.0,
    thin_p90_width_m: float = 14.0,
    thin_max_width_m: float = 20.0,
    weak_branch_ratio: float = 0.82,
    max_rounds: int = 12,
    force_enabled: bool = True,
) -> Tuple[NetworkData, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve arbitrary polygon cycles without assuming diamond/rectangle geometry.

    The topology cycle itself is the object being processed.  A cycle may have any
    number of vertices/sides.

    * 2 external gates:
        - if one route contains U-turn core, keep the core route;
        - if branches are a thin duplicate pair, remove the weaker entire route;
        - within the hard-force envelope, remove the weaker entire route even when
          both sides have similar evidence.
    * 0/1 gate:
        - within the hard-force envelope, remove the isolated/lollipop ring.
    * >=3 gates:
        - within the hard-force envelope, remove the weakest complete arc between
          adjacent gates, preserving all other gate connectivity.

    No new geometry is created here.  The operation only removes redundant existing
    primary edges, so the final road remains a subset of the S6 skeleton.
    """
    net = network.copy()
    decisions: List[Dict[str, Any]] = []
    remaining_rows: List[Dict[str, Any]] = []

    for round_no in range(1, max(1, int(max_rounds)) + 1):
        incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        graph = nx.Graph()
        nodes = {int(n["node_id"]): n for n in net.nodes}
        for edge in net.edges:
            u, v = int(edge["u"]), int(edge["v"])
            incident[u].append(edge); incident[v].append(edge)
            if u == v:
                continue
            pair_edges[(min(u, v), max(u, v))].append(edge)
            graph.add_edge(u, v)

        cycles = nx.cycle_basis(graph)
        if not cycles:
            break

        planned_drop: Set[int] = set()
        occupied: List[Set[int]] = []
        accepted = 0

        # Smaller spatial cycles first; repeated rounds expose nested/overlapping loops.
        enriched = []
        for cyc in cycles:
            pts = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in cyc if n in nodes], dtype=float)
            if len(pts) != len(cyc):
                continue
            area = _polygon_area_xy(pts)
            span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
            enriched.append((span, area, cyc))
        enriched.sort(key=lambda z: (z[0], z[1], len(z[2])))

        for span, area, cycle in enriched:
            reps: List[Dict[str, Any]] = []
            cycle_all_ids: Set[int] = set()
            valid = True
            perimeter = 0.0
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                opts = pair_edges.get((min(a, b), max(a, b)), [])
                if not opts:
                    valid = False
                    break
                cycle_all_ids.update(int(e["edge_id"]) for e in opts)
                rep = max(opts, key=_edge_score)
                reps.append(rep)
                perimeter += float(rep["geometry"].length)
            if not valid or len(reps) < 3:
                continue
            if any(cycle_all_ids & prev for prev in occupied):
                continue

            within_general = (
                perimeter <= float(max_perimeter_m)
                and span <= float(max_span_m)
                and area <= float(max_area_m2)
            )
            hard_force = bool(
                force_enabled
                and perimeter <= float(force_perimeter_m)
                and span <= float(force_span_m)
                and area <= float(force_area_m2)
            )
            if not within_general and not hard_force:
                continue

            external: Dict[int, List[Dict[str, Any]]] = {}
            for nid in cycle:
                ex = [
                    e for e in incident.get(int(nid), [])
                    if int(e["edge_id"]) not in cycle_all_ids
                ]
                if ex:
                    external[int(nid)] = ex
            gates = [n for n in cycle if n in external]
            core_edges = [e for e in reps if _is_core(e)]
            any_core = bool(core_edges)

            drop_ids: Set[int] = set()
            action = "preserve"
            route1_score = route2_score = None
            med_width = p90_width = max_width = None

            if len(gates) == 2:
                ga, gb = int(gates[0]), int(gates[1])
                p1, p2 = _cycle_paths_between_gates(cycle, ga, gb)
                r1 = _representative_arc(p1, pair_edges)
                r2 = _representative_arc(p2, pair_edges)
                if r1 is None or r2 is None:
                    continue
                geom1, reps1, ids1 = r1
                geom2, reps2, ids2 = r2
                groups1 = _arc_edge_options(p1, pair_edges)
                groups2 = _arc_edge_options(p2, pair_edges)
                core1, core2 = _arc_has_core(groups1), _arc_has_core(groups2)
                gap1, gap2 = _arc_has_approved_gap(groups1), _arc_has_approved_gap(groups2)
                route1_score, route2_score = _arc_score(reps1), _arc_score(reps2)
                gate_xy = np.asarray([float(nodes[ga]["x"]), float(nodes[ga]["y"])], dtype=float)
                med_width, p90_width, max_width = _two_branch_width(geom1, geom2, gate_xy)
                thin_duplicate = (
                    p90_width <= float(thin_p90_width_m)
                    and max_width <= float(thin_max_width_m)
                )

                # Never delete approved gap geometry.  U-turn core wins over non-core.
                if core1 and core2:
                    action = "preserve_both_uturn_core"
                elif gap1 and gap2:
                    action = "preserve_both_approved_gap"
                else:
                    if core1 and not core2 and not gap2:
                        drop_ids = ids2
                        action = "keep_uturn_core_route1"
                    elif core2 and not core1 and not gap1:
                        drop_ids = ids1
                        action = "keep_uturn_core_route2"
                    else:
                        # Choose the branch to keep by evidence.  In near ties, shorter
                        # branch wins because digital detours inflate perimeter.
                        key1 = (route1_score, -geom1.length)
                        key2 = (route2_score, -geom2.length)
                        keep1 = key1 >= key2
                        weak = route2_score if keep1 else route1_score
                        strong = route1_score if keep1 else route2_score
                        weak_ids = ids2 if keep1 else ids1
                        weak_gap = gap2 if keep1 else gap1
                        ratio = weak / max(strong, 1e-9)

                        if not weak_gap and (
                            thin_duplicate
                            or hard_force
                            or ratio <= float(weak_branch_ratio)
                        ):
                            drop_ids = weak_ids
                            action = (
                                "two_gate_thin_duplicate_remove_weaker_route"
                                if thin_duplicate else
                                "two_gate_force_remove_weaker_route"
                                if hard_force else
                                "two_gate_evidence_remove_weaker_route"
                            )

            elif len(gates) <= 1:
                # An isolated or lollipop ring has no through-road reason to survive
                # inside the hard-force envelope, unless it carries U-turn core/gap.
                if hard_force and not any_core and not any(_is_gap(e) for e in reps):
                    drop_ids = set(cycle_all_ids)
                    action = "force_remove_isolated_or_lollipop_cycle"

            else:
                # Split the polygon into complete arcs between adjacent gates. Removing
                # one complete arc breaks the cycle without leaving internal ring stubs.
                if hard_force:
                    gate_indices = sorted(cycle.index(g) for g in gates)
                    arcs = []
                    for q, start_i in enumerate(gate_indices):
                        end_i = gate_indices[(q + 1) % len(gate_indices)]
                        path = [cycle[start_i]]
                        k = start_i
                        while True:
                            k = (k + 1) % len(cycle)
                            path.append(cycle[k])
                            if k == end_i:
                                break
                        rep_arc = _representative_arc(path, pair_edges)
                        groups = _arc_edge_options(path, pair_edges)
                        if rep_arc is None or _arc_has_core(groups) or _arc_has_approved_gap(groups):
                            continue
                        geom, arc_reps, arc_ids = rep_arc
                        arcs.append((_arc_score(arc_reps), float(geom.length), arc_ids))
                    if arcs:
                        _, _, weakest_ids = min(arcs, key=lambda x: (x[0], x[1]))
                        drop_ids = weakest_ids
                        action = "force_remove_weakest_gate_to_gate_arc"

            # Never remove protected/gap edges even if they appear in a parallel group.
            if drop_ids:
                allowed = set()
                lookup = {int(e["edge_id"]): e for e in net.edges}
                for eid in drop_ids:
                    e = lookup.get(int(eid))
                    if e is not None and not _is_core(e) and not _is_gap(e):
                        allowed.add(int(eid))
                drop_ids = allowed

            if drop_ids:
                planned_drop |= drop_ids
                occupied.append(cycle_all_ids)
                accepted += 1

            decisions.append({
                "round": round_no,
                "cycle_nodes": len(cycle),
                "gate_count": len(gates),
                "perimeter_m": perimeter,
                "span_m": span,
                "area_m2": area,
                "hard_force": hard_force,
                "uturn_core_edges": len(core_edges),
                "route1_score": route1_score,
                "route2_score": route2_score,
                "median_branch_width_m": med_width,
                "p90_branch_width_m": p90_width,
                "max_branch_width_m": max_width,
                "action": action,
                "removed_edge_ids": ",".join(map(str, sorted(drop_ids))),
            })

        print(
            f"    [polygon-cycle] round={round_no}, cycles={len(cycles):,}, "
            f"resolved={accepted:,}, drop_edges={len(planned_drop):,}",
            flush=True,
        )
        if not planned_drop:
            # Save still-present candidates for diagnosis.
            for cyc in cycles:
                pts = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in cyc if n in nodes], dtype=float)
                if len(pts) != len(cyc):
                    continue
                remaining_rows.append({
                    "round": round_no,
                    "cycle_nodes": len(cyc),
                    "span_m": max(float(np.ptp(pts[:,0])), float(np.ptp(pts[:,1]))),
                    "area_m2": _polygon_area_xy(pts),
                })
            break

        kept = []
        for e in net.edges:
            if int(e["edge_id"]) not in planned_drop:
                kept.append(e)
        net.edges = kept
        net = reindex_network(net)

    return net, decisions, remaining_rows


# ---------------------------------------------------------------------------
# S4-track continuity + strong-direction forced gap repair
# ---------------------------------------------------------------------------

def _sample_line_evidence(
    geom: LineString,
    sampler: Optional[RasterBlockSampler],
    evidence_distance_m: float,
) -> Dict[str, float]:
    if sampler is None or geom.length <= 0:
        return {"evidence_fraction": 0.0, "support_fraction": 0.0, "distance_fraction": 0.0, "kde_fraction": 0.0}
    n = max(3, int(math.ceil(float(geom.length) / 1.5)) + 1)
    ds = np.linspace(0.0, float(geom.length), n)
    pts = [geom.interpolate(float(d)) for d in ds]
    xs = np.asarray([p.x for p in pts], dtype=float)
    ys = np.asarray([p.y for p in pts], dtype=float)

    support = sampler.values_xy("support", xs, ys, default=0.0) if "support" in sampler.datasets else np.zeros(n)
    distance = sampler.values_xy("distance", xs, ys, default=np.inf) if "distance" in sampler.datasets else np.full(n, np.inf)
    kde = sampler.values_xy("kde", xs, ys, default=0.0) if "kde" in sampler.datasets else np.zeros(n)

    support_mask = support > 0
    distance_mask = np.isfinite(distance) & (distance <= float(evidence_distance_m))
    positive = kde[kde > 0]
    kde_ref = float(np.percentile(positive, 20)) if len(positive) else 0.0
    kde_mask = kde >= kde_ref if kde_ref > 0 else np.zeros(n, dtype=bool)
    evidence = support_mask | distance_mask | kde_mask
    return {
        "evidence_fraction": float(np.mean(evidence)),
        "support_fraction": float(np.mean(support_mask)),
        "distance_fraction": float(np.mean(distance_mask)),
        "kde_fraction": float(np.mean(kde_mask)),
    }


def _endpoint_state(network: NetworkData):
    nodes = {int(n["node_id"]): n for n in network.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    graph = nx.Graph()
    for e in network.edges:
        u, v = int(e["u"]), int(e["v"])
        incident[u].append(e); incident[v].append(e)
        graph.add_edge(u, v)
    endpoints = [n for n, es in incident.items() if len(es) == 1]
    return nodes, incident, graph, endpoints


def _continuation_vector(edge: Dict[str, Any], node_id: int) -> Optional[np.ndarray]:
    inward = _edge_from_node_vector(edge, node_id)
    return None if inward is None else -inward


def _candidate_crosses_other_edges(
    geom: LineString,
    network: NetworkData,
    incident_edge_ids: Set[int],
) -> bool:
    geoms = [e["geometry"] for e in network.edges]
    if not geoms:
        return False
    tree = STRtree(geoms)
    result = tree.query(geom)
    for item in result:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
        else:
            try:
                idx = geoms.index(item)
            except ValueError:
                continue
        edge = network.edges[idx]
        if int(edge["edge_id"]) in incident_edge_ids:
            continue
        other = edge["geometry"]
        inter = safe_intersection(
            geom, other, operation="s7_force_candidate_intersection",
            context={"other_edge_id": int(edge["edge_id"])},
        )
        if inter is None:
            return True
        if inter.is_empty:
            continue
        # Touching the candidate very near its endpoints is harmless; an interior
        # crossing/overlap would create an unmodelled junction and is rejected.
        if geom.project(inter.centroid) > 1.0 and geom.length - geom.project(inter.centroid) > 1.0:
            return True
    return False


def repair_s4_track_gaps(
    network: NetworkData,
    gravity_points_path,
    max_endpoint_distance_m: float = 25.0,
    max_track_path_m: float = 35.0,
    endpoint_snap_m: float = 5.0,
    max_angle_deg: float = 50.0,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.5,
    cycle_min_detour_excess_m: float = 6.0,
    max_repairs: int = 5000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Use the actual adjusted S4 trajectory only as a continuity witness.

    It does not rebuild the network from trajectories.  It only reconnects two current
    S7 endpoints when one *same clean_track_id* passes continuously through both.
    """
    path = Path(gravity_points_path) if gravity_points_path is not None else None
    if path is None or not path.is_file() or int(max_repairs) <= 0:
        return network.copy(), []

    net = network.copy()
    nodes, incident, graph, endpoints = _endpoint_state(net)
    weighted = weighted_graph(net)
    if len(endpoints) < 2:
        return net, []
    endpoint_xy = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in endpoints], dtype=float)
    endpoint_tree = cKDTree(endpoint_xy)
    endpoint_index = {n: i for i, n in enumerate(endpoints)}

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"clean_track_id", "clean_point_index", "x_m", "y_m"}
    if not required.issubset(df.columns):
        print(f"    [track-gap] skipped; missing={sorted(required-set(df.columns))}", flush=True)
        return net, []
    for c in ("clean_point_index", "x_m", "y_m"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["x_m"]) & np.isfinite(df["y_m"])].copy()

    candidates = []
    for clean_id, g in df.groupby("clean_track_id", sort=False):
        g = g.sort_values("clean_point_index")
        xy = g[["x_m", "y_m"]].to_numpy(float)
        if len(xy) < 2:
            continue
        d, idx = endpoint_tree.query(xy, k=1, distance_upper_bound=float(endpoint_snap_m))
        matched: Dict[int, Tuple[int, float]] = {}
        for pi, (dist, ei) in enumerate(zip(d, idx)):
            if not np.isfinite(dist) or int(ei) >= len(endpoints):
                continue
            nid = int(endpoints[int(ei)])
            old = matched.get(nid)
            if old is None or float(dist) < old[1]:
                matched[nid] = (pi, float(dist))
        if len(matched) < 2:
            continue

        ordered = sorted((pi, nid, snap) for nid, (pi, snap) in matched.items())
        for (i0, a, snap_a), (i1, b, snap_b) in zip(ordered[:-1], ordered[1:]):
            if a == b or i1 <= i0:
                continue
            same_component = nx.has_path(weighted, a, b)
            if same_component and not bool(allow_cycle_closure):
                continue
            pa, pb = endpoint_xy[endpoint_index[a]], endpoint_xy[endpoint_index[b]]
            euclid = float(np.linalg.norm(pb - pa))
            if euclid > float(max_endpoint_distance_m):
                continue
            sub = xy[i0:i1+1].copy()
            if len(sub) < 2:
                continue
            sub[0], sub[-1] = pa, pb
            seg = np.linalg.norm(np.diff(sub, axis=0), axis=1)
            path_len = float(seg.sum())
            if path_len <= 0 or path_len > float(max_track_path_m):
                continue
            if euclid > 1e-6 and path_len / euclid > 2.5:
                continue

            va = _continuation_vector(incident[a][0], a)
            vb = _continuation_vector(incident[b][0], b)
            ta = sub[min(2, len(sub)-1)] - sub[0]
            tb = sub[max(0, len(sub)-3)] - sub[-1]
            aa = _angle_deg(va, ta) if va is not None else 180.0
            ab = _angle_deg(vb, tb) if vb is not None else 180.0
            if max(aa, ab) > float(max_angle_deg):
                continue

            geom = LineString(sub)
            if not geom.is_simple or geom.length <= 0:
                continue
            simp = geom.simplify(0.35, preserve_topology=True)
            if isinstance(simp, LineString) and simp.is_simple and simp.length > 0:
                co = list(simp.coords)
                co[0], co[-1] = tuple(pa), tuple(pb)
                geom = LineString(co)

            existing_detour_m = float("inf")
            if same_component:
                closure_ok, existing_detour_m = cycle_closure_allowed(
                    weighted, a, b, float(geom.length),
                    min_detour_ratio=cycle_min_detour_ratio,
                    min_detour_excess_m=cycle_min_detour_excess_m,
                )
                if not closure_ok:
                    continue

            candidates.append({
                "score": max(aa, ab) + 0.35 * euclid + 0.1 * path_len + snap_a + snap_b,
                "a": int(a), "b": int(b), "geometry": geom,
                "clean_track_id": str(clean_id),
                "euclidean_m": euclid, "source_path_m": path_len,
                "angle_a_deg": aa, "angle_b_deg": ab,
                "snap_a_m": snap_a, "snap_b_m": snap_b,
                "cycle_closure": bool(same_component),
                "existing_detour_m": float(existing_detour_m),
            })

    candidates.sort(key=lambda r: r["score"])
    used: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    next_eid = max([int(e["edge_id"]) for e in net.edges] or [-1]) + 1
    for c in candidates:
        if len(rows) >= int(max_repairs):
            break
        a, b = int(c["a"]), int(c["b"])
        if a in used or b in used:
            continue
        if nx.has_path(weighted, a, b):
            if not bool(allow_cycle_closure):
                continue
            closure_ok, detour_now = cycle_closure_allowed(
                weighted, a, b, float(c["geometry"].length),
                min_detour_ratio=cycle_min_detour_ratio,
                min_detour_excess_m=cycle_min_detour_excess_m,
            )
            if not closure_ok:
                continue
            c["cycle_closure"] = True
            c["existing_detour_m"] = float(detour_now)
        else:
            c["cycle_closure"] = False
            c["existing_detour_m"] = float("inf")
        incident_ids = {int(incident[a][0]["edge_id"]), int(incident[b][0]["edge_id"])}
        if _candidate_crosses_other_edges(c["geometry"], net, incident_ids):
            continue

        edge = {
            "edge_id": next_eid, "u": a, "v": b, "geometry": c["geometry"],
            "length_m": float(c["geometry"].length), "pixel_count": 0,
            "source_type": "s7_track_gap_repair",
            "primary_fraction": 0.0, "auxiliary_fraction": 0.0, "rescued_fraction": 0.0,
            "support_mean": 0.0, "support_median": 0.0, "support_max": 0.0,
            "support_ge2_fraction": 0.0, "kde_mean": 0.0, "kde_median": 0.0, "kde_max": 0.0,
            "distance_mean_m": 0.0,
            "gap_repair": True, "approved_gap_repair": True,
            "s7_force_gap_repair": True, "s7_gap_mode": "same_s4_track",
            "s7_gap_clean_track_id": c["clean_track_id"],
            "s7_gap_euclidean_m": c["euclidean_m"],
            "s7_gap_source_path_m": c["source_path_m"],
            "s7_gap_angle_a_deg": c["angle_a_deg"],
            "s7_gap_angle_b_deg": c["angle_b_deg"],
            "s7_gap_evidence_fraction": 1.0,
            "s7_gap_cycle_closure": bool(c.get("cycle_closure", False)),
            "s7_gap_existing_detour_m": float(c.get("existing_detour_m", float("inf"))),
            "uturn_core_protected": False, "uturn_context_protected": False, "uturn_protected": False,
            "confidence": 0.88,
            "retained_reason": "same_s4_track_continuity",
            "geometry_origin": "s7_same_track_gap_v0_5",
            "max_geometry_shift_m": 0.0,
            "baseline_length_m": float(c["geometry"].length),
        }
        net.edges.append(edge)
        graph.add_edge(a, b)
        weighted.add_edge(a, b, weight=float(c["geometry"].length))
        used.update((a, b))
        rows.append({**{k:v for k,v in c.items() if k not in ("geometry","score")}, "edge_id": next_eid, "mode": "same_s4_track", "geometry": c["geometry"]})
        next_eid += 1

    print(f"    [track-gap] candidates={len(candidates):,}, accepted={len(rows):,}", flush=True)
    return reindex_network(net), rows


def repair_strong_direction_gaps(
    network: NetworkData,
    support_path=None,
    kde_path=None,
    distance_path=None,
    max_distance_m: float = 15.0,
    max_angle_deg: float = 20.0,
    min_evidence_fraction: float = 0.40,
    evidence_distance_m: float = 10.0,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.8,
    cycle_min_detour_excess_m: float = 8.0,
    max_repairs: int = 5000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Last-resort endpoint repair for different components.

    This is the deliberately 'force within a reasonable range' pass:
    - different components may be stitched;
    - same-component endpoints may close a real road loop only when the existing
      graph route is a much longer detour than the missing connector;
    - no U-turn context endpoint;
    - <= max_distance_m;
    - both endpoint directions are very straight/aligned;
    - trajectory/KDE/support evidence exists along the line;
    - connector may not cross another existing edge.
    """
    net = network.copy()
    nodes, incident, graph, endpoints = _endpoint_state(net)
    weighted = weighted_graph(net)
    if len(endpoints) < 2 or int(max_repairs) <= 0:
        return net, []

    xy = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in endpoints], dtype=float)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(float(max_distance_m), output_type="set")
    comp = {}
    for cid, cc in enumerate(nx.connected_components(graph)):
        for n in cc:
            comp[int(n)] = cid

    raster_paths = {"support": support_path, "kde": kde_path, "distance": distance_path}
    raster_paths = {k: Path(v) for k, v in raster_paths.items() if v is not None and Path(v).exists()}
    sampler = RasterBlockSampler(raster_paths, cache_blocks=128) if raster_paths else None
    candidates = []
    try:
        for i, j in pairs:
            a, b = int(endpoints[i]), int(endpoints[j])
            same_component = comp.get(a) == comp.get(b)
            if same_component and not bool(allow_cycle_closure):
                continue
            if bool(nodes[a].get("uturn_context_protected") or nodes[a].get("uturn_protected")
                    or nodes[b].get("uturn_context_protected") or nodes[b].get("uturn_protected")):
                continue
            pa, pb = xy[i], xy[j]
            d = float(np.linalg.norm(pb - pa))
            if d <= 0.5:
                continue
            va = _continuation_vector(incident[a][0], a)
            vb = _continuation_vector(incident[b][0], b)
            if va is None or vb is None:
                continue
            direction = (pb - pa) / d
            aa = _angle_deg(va, direction)
            ab = _angle_deg(vb, -direction)
            if max(aa, ab) > float(max_angle_deg):
                continue
            geom = LineString([tuple(pa), tuple(pb)])
            closure_ok = False
            existing_detour_m = float("inf")
            if same_component:
                closure_ok, existing_detour_m = cycle_closure_allowed(
                    weighted, a, b, float(geom.length),
                    min_detour_ratio=cycle_min_detour_ratio,
                    min_detour_excess_m=cycle_min_detour_excess_m,
                )
                if not closure_ok:
                    continue
            ev = _sample_line_evidence(geom, sampler, evidence_distance_m)
            if ev["evidence_fraction"] < float(min_evidence_fraction):
                continue
            incident_ids = {int(incident[a][0]["edge_id"]), int(incident[b][0]["edge_id"])}
            if _candidate_crosses_other_edges(geom, net, incident_ids):
                continue
            candidates.append({
                "score": max(aa, ab) + 0.75*d - 5.0*ev["evidence_fraction"],
                "a": a, "b": b, "geometry": geom, "euclidean_m": d,
                "angle_a_deg": aa, "angle_b_deg": ab,
                "cycle_closure": bool(same_component),
                "existing_detour_m": float(existing_detour_m),
                **ev,
            })
    finally:
        if sampler is not None:
            sampler.close()

    candidates.sort(key=lambda r: r["score"])
    used: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    next_eid = max([int(e["edge_id"]) for e in net.edges] or [-1]) + 1
    for c in candidates:
        if len(rows) >= int(max_repairs):
            break
        a, b = int(c["a"]), int(c["b"])
        if a in used or b in used:
            continue
        if nx.has_path(weighted, a, b):
            if not bool(allow_cycle_closure):
                continue
            closure_ok, detour_now = cycle_closure_allowed(
                weighted, a, b, float(c["geometry"].length),
                min_detour_ratio=cycle_min_detour_ratio,
                min_detour_excess_m=cycle_min_detour_excess_m,
            )
            if not closure_ok:
                continue
            c["cycle_closure"] = True
            c["existing_detour_m"] = float(detour_now)
        else:
            c["cycle_closure"] = False
            c["existing_detour_m"] = float("inf")
        edge = {
            "edge_id": next_eid, "u": a, "v": b, "geometry": c["geometry"],
            "length_m": float(c["geometry"].length), "pixel_count": 0,
            "source_type": "s7_forced_direction_gap",
            "primary_fraction": 0.0, "auxiliary_fraction": 0.0, "rescued_fraction": 0.0,
            "support_mean": 0.0, "support_median": 0.0, "support_max": 0.0,
            "support_ge2_fraction": 0.0, "kde_mean": 0.0, "kde_median": 0.0, "kde_max": 0.0,
            "distance_mean_m": 0.0,
            "gap_repair": True, "approved_gap_repair": True,
            "s7_force_gap_repair": True, "s7_gap_mode": "forced_directional",
            "s7_gap_clean_track_id": "",
            "s7_gap_euclidean_m": c["euclidean_m"],
            "s7_gap_source_path_m": float(c["geometry"].length),
            "s7_gap_angle_a_deg": c["angle_a_deg"],
            "s7_gap_angle_b_deg": c["angle_b_deg"],
            "s7_gap_evidence_fraction": c["evidence_fraction"],
            "s7_gap_cycle_closure": bool(c.get("cycle_closure", False)),
            "s7_gap_existing_detour_m": float(c.get("existing_detour_m", float("inf"))),
            "uturn_core_protected": False, "uturn_context_protected": False, "uturn_protected": False,
            "confidence": 0.82,
            "retained_reason": "forced_strong_direction_gap",
            "geometry_origin": "s7_forced_direction_gap_v0_5",
            "max_geometry_shift_m": 0.0,
            "baseline_length_m": float(c["geometry"].length),
        }
        net.edges.append(edge)
        graph.add_edge(a, b)
        weighted.add_edge(a, b, weight=float(c["geometry"].length))
        used.update((a, b))
        rows.append({**{k:v for k,v in c.items() if k not in ("geometry","score")}, "edge_id": next_eid, "mode": "forced_directional", "geometry": c["geometry"]})
        next_eid += 1

    print(f"    [force-gap] candidates={len(candidates):,}, accepted={len(rows):,}", flush=True)
    return reindex_network(net), rows


def audit_s7_force_gaps(
    network: NetworkData,
    same_track_max_endpoint_m: float = 25.0,
    same_track_max_path_m: float = 35.0,
    same_track_max_angle_deg: float = 50.0,
    forced_max_distance_m: float = 15.0,
    forced_max_angle_deg: float = 20.0,
    forced_min_evidence_fraction: float = 0.40,
) -> List[Dict[str, Any]]:
    nodes = {int(n["node_id"]): n for n in network.nodes}
    rows = []
    for edge in network.edges:
        if not bool(edge.get("s7_force_gap_repair")):
            continue
        geom = edge.get("geometry")
        mode = str(edge.get("s7_gap_mode", ""))
        u, v = int(edge["u"]), int(edge["v"])
        endpoint_ok = False
        if isinstance(geom, LineString) and len(geom.coords) >= 2 and u in nodes and v in nodes:
            a, b = geom.coords[0], geom.coords[-1]
            pu = (float(nodes[u]["x"]), float(nodes[u]["y"]))
            pv = (float(nodes[v]["x"]), float(nodes[v]["y"]))
            direct = math.hypot(a[0]-pu[0], a[1]-pu[1]) <= 1e-6 and math.hypot(b[0]-pv[0], b[1]-pv[1]) <= 1e-6
            reverse = math.hypot(a[0]-pv[0], a[1]-pv[1]) <= 1e-6 and math.hypot(b[0]-pu[0], b[1]-pu[1]) <= 1e-6
            endpoint_ok = direct or reverse
        d = _num(edge, "s7_gap_euclidean_m", 1e9)
        path_m = _num(edge, "s7_gap_source_path_m", 1e9)
        aa = _num(edge, "s7_gap_angle_a_deg", 180.0)
        ab = _num(edge, "s7_gap_angle_b_deg", 180.0)
        ev = _num(edge, "s7_gap_evidence_fraction", 0.0)
        origin = str(edge.get("geometry_origin", ""))

        if mode == "same_s4_track":
            valid_rule = (
                bool(str(edge.get("s7_gap_clean_track_id", "")))
                and d <= float(same_track_max_endpoint_m) + 1e-9
                and path_m <= float(same_track_max_path_m) + 1e-9
                and max(aa, ab) <= float(same_track_max_angle_deg) + 1e-9
                and origin == "s7_same_track_gap_v0_5"
            )
        elif mode == "forced_directional":
            valid_rule = (
                d <= float(forced_max_distance_m) + 1e-9
                and max(aa, ab) <= float(forced_max_angle_deg) + 1e-9
                and ev >= float(forced_min_evidence_fraction) - 1e-12
                and origin == "s7_forced_direction_gap_v0_5"
            )
        else:
            valid_rule = False

        valid = bool(
            isinstance(geom, LineString) and geom.length > 0 and geom.is_simple
            and endpoint_ok and valid_rule
        )
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "gap_repair": True,
            "s7_force_gap_repair": True,
            "mode": mode,
            "length_m": float(geom.length) if isinstance(geom, LineString) else 0.0,
            "euclidean_m": d,
            "source_path_m": path_m,
            "angle_a_deg": aa,
            "angle_b_deg": ab,
            "evidence_fraction": ev,
            "endpoint_topology_preserved": endpoint_ok,
            "outside_baseline_m": 0.0 if valid else float(geom.length if isinstance(geom, LineString) else 0.0),
            "outside_baseline_fraction": 0.0 if valid else 1.0,
            "audit_reason": "approved_s7_force_gap" if valid else "failed_s7_force_gap_audit",
            "geometry_origin": origin,
            "geometry": geom,
        })
    return rows
