from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point

from .models import NetworkData, compute_degrees, make_nx_graph, orient_coords, reindex_network
from .optimizer import edge_confidence
from .parallel import detect_parallel_edges
from .tile_io import RasterBlockSampler


def _num(edge: Dict[str, Any], name: str, default: float = 0.0) -> float:
    value = edge.get(name, default)
    try:
        value = float(value)
    except Exception:
        value = default
    return default if not np.isfinite(value) else value


def _uturn_core(edge: Dict[str, Any]) -> bool:
    """Narrow U-turn protection used by geometry edits.

    Missing values can arrive as ``NaN`` after GeoDataFrame/CSV round-trips; Python
    treats ``bool(np.nan)`` as True, which would incorrectly put almost every road
    into conservative U-turn smoothing. Only explicit truthy values count.
    """
    def flag(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    if "uturn_core_protected" in edge:
        return flag(edge.get("uturn_core_protected"))
    return flag(edge.get("uturn_protected"))


def _artifact_score(edge: Dict[str, Any]) -> float:
    """Evidence score for choosing which side of a tiny digital loop to remove.

    Deliberately does *not* boost the wider U-turn context flag; otherwise every
    raster artifact inside that context receives an artificial score floor and can
    never be ranked for removal.
    """
    tracks = _num(edge, "distinct_tracks")
    coverage = _num(edge, "coverage_ratio")
    support = _num(edge, "support_mean")
    support2 = _num(edge, "support_ge2_fraction")
    source = str(edge.get("source_type", "primary"))
    source_prior = {
        "primary": 0.78,
        "mixed": 0.80,
        "support_rescued": 0.82,
        "micro_gap_repair": 0.88,
        "gap_repair": 0.88,
        "auxiliary": 0.35,
    }.get(source, 0.55)
    return float(np.clip(
        0.34 * min(1.0, tracks / 3.0)
        + 0.18 * min(1.0, coverage / 0.45)
        + 0.22 * min(1.0, support / 2.0)
        + 0.16 * min(1.0, support2 / 0.35)
        + 0.10 * source_prior,
        0.0, 1.0,
    ))


def quality_score(edge: Dict[str, Any]) -> float:
    """Evidence score used only for deletion/selection, never for creating geometry."""
    base = edge_confidence(edge)
    tracks = _num(edge, "distinct_tracks")
    coverage = _num(edge, "coverage_ratio")
    support = _num(edge, "support_mean")
    support2 = _num(edge, "support_ge2_fraction")
    source = str(edge.get("source_type", "primary"))
    source_prior = {
        "primary": 0.16,
        "mixed": 0.18,
        "support_rescued": 0.18,
        "auxiliary": 0.00,
    }.get(source, 0.04)
    score = (
        0.58 * base
        + 0.12 * min(1.0, tracks / 3.0)
        + 0.08 * min(1.0, coverage / 0.45)
        + 0.08 * min(1.0, support / 1.5)
        + 0.06 * min(1.0, support2 / 0.25)
        + source_prior
    )
    if edge.get("uturn_protected"):
        score = max(score, 0.90)
    if edge.get("activity_zone") and source == "auxiliary":
        score *= 0.60
    return float(np.clip(score, 0.0, 1.0))


def is_strong(edge: Dict[str, Any]) -> bool:
    return bool(
        edge.get("gap_repair")
        or edge.get("approved_gap_repair")
        or edge.get("approved_directional_collapse")
        or edge.get("uturn_protected")
        or _num(edge, "distinct_tracks") >= 2
        or (_num(edge, "distinct_tracks") >= 1 and _num(edge, "coverage_ratio") >= 0.22)
        or _num(edge, "support_mean") >= 1.20
        or _num(edge, "support_ge2_fraction") >= 0.20
        or quality_score(edge) >= 0.58
    )


def is_weak(edge: Dict[str, Any], threshold: float) -> bool:
    return bool(
        not edge.get("gap_repair")
        and not edge.get("approved_gap_repair")
        and not edge.get("approved_directional_collapse")
        and not edge.get("uturn_protected")
        and _num(edge, "distinct_tracks") <= 0
        and _num(edge, "coverage_ratio") < 0.08
        and _num(edge, "support_mean") < 0.70
        and _num(edge, "support_max") < 2.0
        and _num(edge, "support_ge2_fraction") < 0.05
        and quality_score(edge) < threshold
    )


def _drop_edges(network: NetworkData, drop_ids: Set[int], reason: str) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    dropped: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    for edge in network.edges:
        if int(edge["edge_id"]) in drop_ids:
            row = dict(edge)
            row["drop_reason"] = reason
            dropped.append(row)
        else:
            kept.append(edge)
    out = network.copy()
    out.edges = kept
    return reindex_network(out), dropped


def iterative_dangle_prune(
    network: NetworkData,
    auxiliary_max_m: float = 32.0,
    primary_max_m: float = 12.0,
    rescued_max_m: float = 8.0,
    confidence_threshold: float = 0.38,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove weak terminal branches with a dynamic leaf queue in O(E+V).

    A repeated whole-graph scan can require thousands of iterations for a long chain.
    The queue immediately revisits a node when deletion turns it into a new endpoint.
    """
    edge_lookup = {int(edge["edge_id"]): edge for edge in network.edges}
    incident: Dict[int, Set[int]] = defaultdict(set)
    degree: Dict[int, int] = defaultdict(int)
    for edge_id, edge in edge_lookup.items():
        u, v = int(edge["u"]), int(edge["v"])
        incident[u].add(edge_id)
        incident[v].add(edge_id)
        if u == v:
            degree[u] += 2
        else:
            degree[u] += 1
            degree[v] += 1

    active = set(edge_lookup)
    queue = deque(node_id for node_id, value in degree.items() if value == 1)
    queued = set(queue)
    dropped: List[Dict[str, Any]] = []
    examined = 0

    while queue:
        node_id = queue.popleft()
        queued.discard(node_id)
        active_incident = [edge_id for edge_id in incident.get(node_id, ()) if edge_id in active]
        if len(active_incident) != 1:
            continue
        edge_id = active_incident[0]
        edge = edge_lookup[edge_id]
        examined += 1
        if edge.get("uturn_protected") or edge.get("gap_repair") or edge.get("approved_gap_repair"):
            continue
        source = str(edge.get("source_type", "primary"))
        limit = {
            "auxiliary": auxiliary_max_m,
            "primary": primary_max_m,
            "mixed": primary_max_m,
            "support_rescued": rescued_max_m,
        }.get(source, primary_max_m)
        if edge.get("activity_zone") and source == "auxiliary":
            limit *= 1.5
        delete = _num(edge, "length_m") <= limit and is_weak(edge, confidence_threshold)
        if not delete:
            continue

        active.remove(edge_id)
        row = dict(edge)
        row["drop_reason"] = "iterative_low_evidence_dangle"
        dropped.append(row)
        u, v = int(edge["u"]), int(edge["v"])
        for endpoint in (u, v):
            if u == v:
                degree[endpoint] = max(0, degree[endpoint] - 2)
            else:
                degree[endpoint] = max(0, degree[endpoint] - 1)
            if degree[endpoint] == 1 and endpoint not in queued:
                queue.append(endpoint)
                queued.add(endpoint)
        if len(dropped) % 10_000 == 0:
            print(
                f"    [cleanup:dangle] removed={len(dropped):,}, queue={len(queue):,}",
                flush=True,
            )

    kept = [edge for edge in network.edges if int(edge["edge_id"]) in active]
    out = network.copy()
    out.edges = kept
    print(
        f"    [cleanup:dangle] examined={examined:,}, removed={len(dropped):,}",
        flush=True,
    )
    return reindex_network(out), dropped

def remove_tiny_weak_components(
    network: NetworkData,
    max_component_length_m: float = 45.0,
    confidence_threshold: float = 0.40,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    graph = make_nx_graph(network.nodes, network.edges)
    edge_by_id = {int(e["edge_id"]): e for e in network.edges}
    drop_ids: Set[int] = set()
    for nodes in nx.connected_components(graph):
        ids = set()
        for u, v, data in graph.subgraph(nodes).edges(data=True):
            if "edge_id" in data:
                ids.add(int(data["edge_id"]))
        edges = [edge_by_id[eid] for eid in ids if eid in edge_by_id]
        length = sum(_num(e, "length_m") for e in edges)
        if not edges or length > max_component_length_m:
            continue
        if any(e.get("uturn_protected") or is_strong(e) for e in edges):
            continue
        if all(quality_score(e) < confidence_threshold for e in edges):
            drop_ids.update(ids)
    return _drop_edges(network, drop_ids, "tiny_weak_component") if drop_ids else (network, [])


def break_weak_small_loops(
    network: NetworkData,
    max_loop_length_m: float = 85.0,
    confidence_threshold: float = 0.42,
    max_rounds: int = 4,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Batch-break unsupported small loops.

    The previous implementation recomputed ``cycle_basis`` after deleting only one
    edge, which can take hours on a large raster-derived graph.  This version finds
    a batch of non-conflicting weak cycle edges per round and removes them together.
    """
    net = network.copy()
    dropped: List[Dict[str, Any]] = []
    max_rounds = max(1, int(max_rounds))

    for round_no in range(1, max_rounds + 1):
        graph = nx.Graph()
        pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for e in net.edges:
            u, v = int(e["u"]), int(e["v"])
            graph.add_edge(u, v)
            pair_edges[(min(u, v), max(u, v))].append(e)

        cycles = nx.cycle_basis(graph)
        print(
            f"    [cleanup:loops] round={round_no}, edges={len(net.edges):,}, "
            f"cycles={len(cycles):,}",
            flush=True,
        )
        if not cycles:
            break

        selected: Set[int] = set()
        selected_cycle_edge_sets: List[Set[int]] = []
        for cycle in cycles:
            cycle_edges: List[Dict[str, Any]] = []
            cycle_ids: Set[int] = set()
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                options = pair_edges.get((min(a, b), max(a, b)), [])
                if options:
                    edge = min(options, key=quality_score)
                    cycle_edges.append(edge)
                    cycle_ids.add(int(edge["edge_id"]))
            if not cycle_edges:
                continue
            length = sum(_num(e, "length_m") for e in cycle_edges)
            if length > max_loop_length_m:
                continue
            if any(e.get("uturn_protected") or e.get("gap_repair") or e.get("approved_gap_repair") for e in cycle_edges):
                continue
            if sum(1 for e in cycle_edges if is_strong(e)) >= 2:
                continue
            weak_options = [
                e for e in cycle_edges
                if str(e.get("source_type", "")) == "auxiliary"
                or quality_score(e) < confidence_threshold
            ]
            if not weak_options:
                continue
            # Avoid removing several edges from overlapping cycles in one batch.
            if any(cycle_ids & previous for previous in selected_cycle_edge_sets):
                continue
            candidate = int(min(weak_options, key=quality_score)["edge_id"])
            selected.add(candidate)
            selected_cycle_edge_sets.append(cycle_ids)

        print(
            f"    [cleanup:loops] round={round_no}, batch_drop={len(selected):,}",
            flush=True,
        )
        if not selected:
            break
        net, rows = _drop_edges(net, selected, "unsupported_small_loop")
        dropped.extend(rows)

    return net, dropped

def remove_parallel_duplicates(
    network: NetworkData,
    min_distance_m: float = 1.0,
    max_distance_m: float = 6.0,
    max_angle_deg: float = 12.0,
    min_length_m: float = 20.0,
    min_overlap_ratio: float = 0.68,
    score_ratio: float = 0.72,
    progress_every: int = 1000,
) -> Tuple[NetworkData, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep the better existing line; never create a midpoint or replacement line."""
    rows = detect_parallel_edges(
        network,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        max_angle_deg=max_angle_deg,
        min_length_m=min_length_m,
        min_overlap_ratio=min_overlap_ratio,
        progress_every=progress_every,
    )
    edge_lookup = {int(e["edge_id"]): e for e in network.edges}
    drop_ids: Set[int] = set()
    decisions: List[Dict[str, Any]] = []
    ordered = sorted(rows, key=lambda r: (-float(r.get("overlap_ratio", 0)), float(r.get("distance_m", 999))))
    for row in ordered:
        a_id, b_id = int(row["edge_a"]), int(row["edge_b"])
        if a_id in drop_ids or b_id in drop_ids:
            continue
        a, b = edge_lookup.get(a_id), edge_lookup.get(b_id)
        if a is None or b is None:
            continue
        if (a.get("uturn_protected") or b.get("uturn_protected")
                or a.get("gap_repair") or b.get("gap_repair")
                or a.get("approved_gap_repair") or b.get("approved_gap_repair")):
            row["decision"] = "keep_uturn"
            decisions.append(row)
            continue
        both_independent = (
            _num(a, "distinct_tracks") >= 2 and _num(b, "distinct_tracks") >= 2
            and _num(a, "coverage_ratio") >= 0.20 and _num(b, "coverage_ratio") >= 0.20
        )
        if both_independent:
            row["decision"] = "keep_both_independent_evidence"
            decisions.append(row)
            continue
        qa, qb = quality_score(a), quality_score(b)
        weaker, stronger = (a, b) if qa <= qb else (b, a)
        qweak, qstrong = (qa, qb) if qa <= qb else (qb, qa)
        weaker_aux = str(weaker.get("source_type", "")) == "auxiliary"
        weak_tracks = _num(weaker, "distinct_tracks") <= 1 and _num(weaker, "coverage_ratio") < 0.20
        decisively_weaker = qweak <= max(0.25, score_ratio * qstrong)
        if weak_tracks and (weaker_aux or decisively_weaker):
            drop_ids.add(int(weaker["edge_id"]))
            row["decision"] = "drop_weaker_existing_parallel"
            row["dropped_edge_id"] = int(weaker["edge_id"])
            row["kept_edge_id"] = int(stronger["edge_id"])
            row["quality_dropped"] = qweak
            row["quality_kept"] = qstrong
        else:
            row["decision"] = "keep_ambiguous_parallel"
        decisions.append(row)
    if not drop_ids:
        return network, [], decisions
    cleaned, dropped = _drop_edges(network, drop_ids, "duplicate_parallel_weaker_line")
    return cleaned, dropped, decisions


def constrained_simplify(
    network: NetworkData,
    tolerance_m: float = 0.8,
    corridor_m: float = 1.5,
    uturn_tolerance_m: float = 0.0,
    progress_every: int = 10_000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Reduce raster stair-steps with a Hausdorff corridor guarantee.

    The old implementation constructed a polygon buffer and difference for every
    edge.  Hausdorff validation gives the required bidirectional corridor bound with
    much lower memory use and predictable linear progress.
    """
    started = time.perf_counter()
    net = network.copy()
    rows: List[Dict[str, Any]] = []
    changed = 0
    total = len(net.edges)
    for index, edge in enumerate(net.edges, 1):
        original: LineString = edge["geometry"]
        tol = uturn_tolerance_m if edge.get("uturn_protected") else tolerance_m
        if tol <= 0 or len(original.coords) <= 2:
            rows.append({"edge_id": int(edge["edge_id"]), "simplified": False, "reason": "protected_or_short"})
        else:
            candidate = original.simplify(tol, preserve_topology=True)
            shift = float(candidate.hausdorff_distance(original)) if isinstance(candidate, LineString) else float("inf")
            valid = (
                isinstance(candidate, LineString)
                and candidate.length > 0
                and shift <= float(corridor_m) + 1e-9
                and Point(candidate.coords[0]).distance(Point(original.coords[0])) <= 1e-6
                and Point(candidate.coords[-1]).distance(Point(original.coords[-1])) <= 1e-6
            )
            if valid:
                before_vertices = len(original.coords)
                edge["geometry"] = candidate
                edge["length_m"] = float(candidate.length)
                edge["max_geometry_shift_m"] = max(
                    float(edge.get("max_geometry_shift_m", 0.0) or 0.0), shift
                )
                edge["geometry_origin"] = "s5_corridor_simplified"
                changed += 1
                rows.append({
                    "edge_id": int(edge["edge_id"]),
                    "simplified": True,
                    "vertices_before": before_vertices,
                    "vertices_after": len(candidate.coords),
                    "length_before_m": float(original.length),
                    "length_after_m": float(candidate.length),
                    "hausdorff_shift_m": shift,
                })
            else:
                rows.append({"edge_id": int(edge["edge_id"]), "simplified": False, "reason": "corridor_validation_failed", "hausdorff_shift_m": shift})
        if progress_every > 0 and (index % int(progress_every) == 0 or index == total):
            elapsed = time.perf_counter() - started
            print(
                f"    [simplify] processed={index:,}/{total:,}, changed={changed:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    return reindex_network(net), rows

def clean_network(
    network: NetworkData,
    dangle_aux_m: float = 32.0,
    dangle_primary_m: float = 12.0,
    dangle_rescued_m: float = 8.0,
    prune_confidence: float = 0.38,
    component_max_m: float = 45.0,
    loop_max_m: float = 85.0,
    parallel_min_distance_m: float = 1.0,
    parallel_max_distance_m: float = 6.0,
    parallel_max_angle_deg: float = 12.0,
    parallel_min_length_m: float = 20.0,
    parallel_min_overlap_ratio: float = 0.68,
    parallel_score_ratio: float = 0.72,
    loop_max_rounds: int = 4,
    parallel_progress_every: int = 1000,
    skip_parallel_cleanup: bool = False,
) -> Tuple[NetworkData, Dict[str, List[Dict[str, Any]]]]:
    for edge in network.edges:
        edge["confidence"] = quality_score(edge)
    print(f"    [cleanup] start edges={len(network.edges):,}", flush=True)
    net, dangles = iterative_dangle_prune(
        network, dangle_aux_m, dangle_primary_m, dangle_rescued_m, prune_confidence
    )
    print(f"    [cleanup] dangles removed={len(dangles):,}", flush=True)
    net, components = remove_tiny_weak_components(net, component_max_m, prune_confidence)
    print(f"    [cleanup] weak components removed={len(components):,}", flush=True)
    net, loops = break_weak_small_loops(
        net, loop_max_m, prune_confidence + 0.04, max_rounds=loop_max_rounds
    )
    print(f"    [cleanup] loop edges removed={len(loops):,}", flush=True)
    if skip_parallel_cleanup:
        parallels, decisions = [], []
        print("    [cleanup:parallel] skipped by option", flush=True)
    else:
        net, parallels, decisions = remove_parallel_duplicates(
            net,
            parallel_min_distance_m,
            parallel_max_distance_m,
            parallel_max_angle_deg,
            parallel_min_length_m,
            parallel_min_overlap_ratio,
            parallel_score_ratio,
            progress_every=parallel_progress_every,
        )
        print(f"    [cleanup] parallel edges removed={len(parallels):,}", flush=True)
    for edge in net.edges:
        edge["confidence"] = quality_score(edge)
        edge.setdefault("retained_reason", "evidence_supported_or_topologically_required")
    return net, {
        "dangles": dangles,
        "components": components,
        "loops": loops,
        "parallel_dropped": parallels,
        "parallel_decisions": decisions,
    }



def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _edge_outward_vector(edge: Dict[str, Any], node_id: int, probe_m: float = 8.0) -> Optional[np.ndarray]:
    """Stable tangent pointing from a topology node out along an incident edge."""
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
    cumulative = 0.0
    target = arr[-1]
    for a, b in zip(arr[:-1], arr[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg <= 1e-9:
            continue
        if cumulative + seg >= max(1.0, float(probe_m)):
            f = (max(1.0, float(probe_m)) - cumulative) / seg
            target = a + np.clip(f, 0.0, 1.0) * (b - a)
            break
        cumulative += seg
    vec = np.asarray(target - origin, dtype=float)
    return vec if float(np.linalg.norm(vec)) > 1e-9 else None


def _sample_line_xy(line: LineString, spacing_m: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    length = float(line.length)
    if length <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    n = max(2, int(np.ceil(length / max(0.5, float(spacing_m)))) + 1)
    ds = np.linspace(0.0, length, n)
    pts = [line.interpolate(float(d)) for d in ds]
    return np.asarray([p.x for p in pts], dtype=float), np.asarray([p.y for p in pts], dtype=float)


def collapse_directional_digital_loops(
    network: NetworkData,
    primary_path=None,
    support_path=None,
    kde_path=None,
    distance_path=None,
    max_perimeter_m: float = 50.0,
    max_span_m: float = 18.0,
    max_rounds: int = 6,
    max_gate_angle_deg: float = 30.0,
    evidence_distance_m: float = 6.0,
    min_evidence_fraction: float = 0.60,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Replace an evidence-supported raster loop by its road-direction chord.

    This is deliberately narrower than generic loop deletion.  A candidate must:
    * be a compact non-U-turn-core cycle;
    * have exactly two gate nodes, each with exactly one external continuation;
    * have external road tangents aligned with the gate-to-gate chord;
    * have continuous trajectory/raster evidence along that chord.

    The operation removes the complete digital ring and inserts one straight centreline
    between the two gates.  If any test fails, the loop is left for the conservative
    v0.5 weak-side deletion fallback.
    """
    net = network.copy()
    rows: List[Dict[str, Any]] = []
    raster_paths = {"primary": primary_path, "support": support_path, "kde": kde_path, "distance": distance_path}
    raster_paths = {k: v for k, v in raster_paths.items() if v is not None and Path(v).exists()}
    sampler_ctx = RasterBlockSampler(raster_paths, cache_blocks=128) if raster_paths else None
    try:
        for round_no in range(1, max(1, int(max_rounds)) + 1):
            nodes = {int(n["node_id"]): n for n in net.nodes}
            edge_by_id = {int(e["edge_id"]): e for e in net.edges}
            incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            graph = nx.Graph()
            pair_edge: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
            for edge in net.edges:
                eid = int(edge["edge_id"]); u = int(edge["u"]); v = int(edge["v"])
                incident[u].append(edge); incident[v].append(edge)
                if u == v:
                    continue
                graph.add_edge(u, v)
                pair_edge[(min(u, v), max(u, v))].append(edge)
            cycles = nx.cycle_basis(graph)
            remove_ids: Set[int] = set()
            new_edges: List[Dict[str, Any]] = []
            used_nodes: Set[int] = set()
            next_eid = max(edge_by_id, default=-1) + 1
            accepted = 0

            for cycle in sorted(cycles, key=len):
                cycle_edges: List[Dict[str, Any]] = []
                cycle_ids: Set[int] = set()
                valid_cycle = True
                for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                    options = pair_edge.get((min(a, b), max(a, b)), [])
                    # Directional collapse is only for simple pixel rings. Parallel
                    # alternatives are handled by the fallback remover.
                    if len(options) != 1:
                        valid_cycle = False; break
                    edge = options[0]
                    cycle_edges.append(edge); cycle_ids.add(int(edge["edge_id"]))
                if not valid_cycle or len(cycle_edges) < 3 or cycle_ids & remove_ids:
                    continue
                if any(_uturn_core(e) or e.get("gap_repair") or e.get("approved_gap_repair")
                       or e.get("approved_directional_collapse") for e in cycle_edges):
                    continue
                perimeter = float(sum(float(e["geometry"].length) for e in cycle_edges))
                minx = min(e["geometry"].bounds[0] for e in cycle_edges)
                miny = min(e["geometry"].bounds[1] for e in cycle_edges)
                maxx = max(e["geometry"].bounds[2] for e in cycle_edges)
                maxy = max(e["geometry"].bounds[3] for e in cycle_edges)
                span = float(max(maxx - minx, maxy - miny))
                if perimeter > float(max_perimeter_m) or span > float(max_span_m):
                    continue

                external: Dict[int, List[Dict[str, Any]]] = {}
                for nid in cycle:
                    ex = [e for e in incident.get(int(nid), []) if int(e["edge_id"]) not in cycle_ids]
                    if ex:
                        external[int(nid)] = ex
                gates = sorted(external)
                if len(gates) != 2 or any(len(external[g]) != 1 for g in gates):
                    continue
                if any(g in used_nodes for g in gates):
                    continue
                a, b = gates
                if a not in nodes or b not in nodes:
                    continue
                pa = np.asarray([float(nodes[a]["x"]), float(nodes[a]["y"])], dtype=float)
                pb = np.asarray([float(nodes[b]["x"]), float(nodes[b]["y"])], dtype=float)
                chord_vec = pb - pa
                chord_len = float(np.linalg.norm(chord_vec))
                if chord_len <= 1.0 or chord_len > float(max_span_m) * 1.25:
                    continue
                out_a = _edge_outward_vector(external[a][0], a)
                out_b = _edge_outward_vector(external[b][0], b)
                if out_a is None or out_b is None:
                    continue
                # At gate A the cycle/chord direction is opposite to the external
                # outward tangent; at gate B it is aligned with the external tangent.
                angle_a = _vector_angle_deg(out_a, -chord_vec)
                angle_b = _vector_angle_deg(out_b, chord_vec)
                if max(angle_a, angle_b) > float(max_gate_angle_deg):
                    continue
                # Also require the two external road continuations to be mutually
                # collinear rather than a true junction.
                continuation_angle = _vector_angle_deg(out_a, -out_b)
                if continuation_angle > float(max_gate_angle_deg) * 1.15:
                    continue

                chord = LineString([tuple(pa), tuple(pb)])
                xs, ys = _sample_line_xy(chord, spacing_m=1.0)
                primary = np.zeros(len(xs), dtype=float)
                support = np.zeros(len(xs), dtype=float)
                kde = np.zeros(len(xs), dtype=float)
                distance = np.full(len(xs), np.inf, dtype=float)
                if sampler_ctx is not None:
                    primary = sampler_ctx.values_xy("primary", xs, ys, default=0.0)
                    support = sampler_ctx.values_xy("support", xs, ys, default=0.0)
                    kde = sampler_ctx.values_xy("kde", xs, ys, default=0.0)
                    distance = sampler_ctx.values_xy("distance", xs, ys, default=np.inf)
                # A replacement is not allowed to cut through a third primary road.
                # Ignore a tiny neighbourhood at each gate because the original ring
                # necessarily touches primary there.
                if len(primary) > 6 and np.any(primary[2:-2] > 0):
                    continue
                gate_kde_ref = 0.0
                if sampler_ctx is not None and "kde" in sampler_ctx.datasets:
                    gate_kde = sampler_ctx.values_xy("kde", np.asarray([pa[0], pb[0]]), np.asarray([pa[1], pb[1]]), default=0.0)
                    gate_kde_ref = float(np.mean(gate_kde)) if len(gate_kde) else 0.0
                kde_mask = (kde >= 0.15 * gate_kde_ref) if gate_kde_ref > 1e-9 else np.zeros(len(kde), dtype=bool)
                distance_mask = np.isfinite(distance) & (distance <= float(evidence_distance_m))
                support_mask = support > 0
                evidence_mask = support_mask | distance_mask | kde_mask
                evidence_fraction = float(np.mean(evidence_mask)) if len(evidence_mask) else 0.0
                distance_fraction = float(np.mean(distance_mask)) if len(distance_mask) else 0.0
                support_fraction = float(np.mean(support_mask)) if len(support_mask) else 0.0
                kde_ratio = float(np.mean(kde) / max(gate_kde_ref, 1e-9)) if gate_kde_ref > 1e-9 and len(kde) else 0.0
                # At least one independent evidence family must be meaningfully present.
                evidence_family_ok = bool(distance_fraction >= 0.35 or support_fraction >= 0.20 or kde_ratio >= 0.25)
                if evidence_fraction < float(min_evidence_fraction) or not evidence_family_ok:
                    continue

                strongest = max(cycle_edges, key=_artifact_score)
                base = dict(strongest)
                base.update({
                    "edge_id": next_eid,
                    "u": int(a), "v": int(b),
                    "geometry": chord,
                    "length_m": float(chord.length),
                    "pixel_count": max(2, int(np.ceil(chord.length / max(1.0, span / 8.0)))),
                    "source_type": "primary",
                    "primary_fraction": 0.0,
                    "support_mean": float(np.mean(support)) if len(support) else _num(strongest, "support_mean"),
                    "support_median": float(np.median(support)) if len(support) else _num(strongest, "support_median"),
                    "support_max": float(np.max(support)) if len(support) else _num(strongest, "support_max"),
                    "support_ge2_fraction": float(np.mean(support >= 2)) if len(support) else _num(strongest, "support_ge2_fraction"),
                    "kde_mean": float(np.mean(kde)) if len(kde) else _num(strongest, "kde_mean"),
                    "kde_median": float(np.median(kde)) if len(kde) else _num(strongest, "kde_median"),
                    "kde_max": float(np.max(kde)) if len(kde) else _num(strongest, "kde_max"),
                    "distance_mean_m": float(np.mean(distance[np.isfinite(distance)])) if np.any(np.isfinite(distance)) else _num(strongest, "distance_mean_m", 999.0),
                    "uturn_core_protected": False,
                    "uturn_context_protected": bool(any(e.get("uturn_context_protected", e.get("uturn_protected")) for e in cycle_edges)),
                    "uturn_protected": bool(any(e.get("uturn_context_protected", e.get("uturn_protected")) for e in cycle_edges)),
                    "gap_repair": False,
                    "approved_gap_repair": False,
                    "digital_loop_diagonal": True,
                    "approved_directional_collapse": True,
                    "directional_collapse_evidence_fraction": evidence_fraction,
                    "directional_collapse_distance_fraction": distance_fraction,
                    "directional_collapse_support_fraction": support_fraction,
                    "directional_collapse_kde_ratio": kde_ratio,
                    "directional_collapse_gate_angle_a_deg": float(angle_a),
                    "directional_collapse_gate_angle_b_deg": float(angle_b),
                    "directional_collapse_original_perimeter_m": perimeter,
                    "directional_collapse_original_span_m": span,
                    "directional_collapse_removed_edge_ids": ",".join(str(x) for x in sorted(cycle_ids)),
                    "confidence": max(_num(strongest, "confidence", 0.0), 0.72),
                    "retained_reason": "approved_directional_digital_loop_collapse",
                    "geometry_origin": "directional_loop_diagonal_v0_5",
                    "max_geometry_shift_m": 0.0,
                    "baseline_length_m": float(chord.length),
                })
                new_edges.append(base)
                remove_ids |= cycle_ids
                used_nodes.update((a, b))
                rows.append({
                    "round": round_no,
                    "u": a, "v": b,
                    "perimeter_before_m": perimeter,
                    "span_m": span,
                    "replacement_length_m": float(chord.length),
                    "length_saved_m": float(perimeter - chord.length),
                    "gate_angle_a_deg": float(angle_a),
                    "gate_angle_b_deg": float(angle_b),
                    "continuation_angle_deg": float(continuation_angle),
                    "evidence_fraction": evidence_fraction,
                    "distance_evidence_fraction": distance_fraction,
                    "support_fraction": support_fraction,
                    "kde_ratio_to_gates": kde_ratio,
                    "removed_edge_ids": ",".join(str(x) for x in sorted(cycle_ids)),
                    "geometry": chord,
                })
                next_eid += 1; accepted += 1

            print(f"    [stage1:directional-collapse] round={round_no}, cycles={len(cycles):,}, accepted={accepted:,}", flush=True)
            if not remove_ids:
                break
            net.edges = [e for e in net.edges if int(e["edge_id"]) not in remove_ids] + new_edges
            net = reindex_network(net)
    finally:
        if sampler_ctx is not None:
            sampler_ctx.close()
    return net, rows


def remove_digital_artifact_loops(
    network: NetworkData,
    max_perimeter_m: float = 28.0,
    max_span_m: float = 8.0,
    max_rounds: int = 8,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Break compact raster-topology rings while preserving the true U-turn core.

    v0.5 treated the *whole* U-turn context as untouchable, so small 2 m-grid
    diamonds/boxes survived along otherwise straight main roads.  v0.5 keeps the
    narrow consensus-line core immutable, but allows a weak non-core side of a
    compact loop to be removed even when it lies inside the wider U-turn context.

    Only existing edges are deleted; no chord, midpoint or replacement geometry is
    created.  Approved micro-gap edges are never selected for deletion.
    """
    net = network.copy()
    dropped: List[Dict[str, Any]] = []
    for round_no in range(1, max(1, int(max_rounds)) + 1):
        edge_by_id = {int(e["edge_id"]): e for e in net.edges}
        selected: Set[int] = set()

        # Short self-loop edges are digital rings unless they are in the narrow
        # consensus core or are explicit approved gap repairs.
        for eid, edge in edge_by_id.items():
            if int(edge["u"]) != int(edge["v"]):
                continue
            if _uturn_core(edge) or edge.get("gap_repair") or edge.get("approved_gap_repair"):
                continue
            minx, miny, maxx, maxy = edge["geometry"].bounds
            span = max(maxx - minx, maxy - miny)
            if float(edge["geometry"].length) <= max_perimeter_m and span <= max_span_m:
                selected.add(eid)

        # Parallel two-edge rings are invisible to nx.Graph.cycle_basis.
        pair_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for edge in net.edges:
            u, v = int(edge["u"]), int(edge["v"])
            if u != v:
                pair_groups[(min(u, v), max(u, v))].append(edge)
        for group in pair_groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda e: (_artifact_score(e), float(e["geometry"].length)))
            combined = sum(float(e["geometry"].length) for e in ordered[:2])
            minx = min(e["geometry"].bounds[0] for e in group)
            miny = min(e["geometry"].bounds[1] for e in group)
            maxx = max(e["geometry"].bounds[2] for e in group)
            maxy = max(e["geometry"].bounds[3] for e in group)
            span = max(maxx - minx, maxy - miny)
            if combined <= max_perimeter_m and span <= max_span_m:
                # Keep the strongest route; delete only non-core/non-gap alternatives.
                for edge in ordered[:-1]:
                    if (_uturn_core(edge) or edge.get("gap_repair")
                            or edge.get("approved_gap_repair")):
                        continue
                    selected.add(int(edge["edge_id"]))

        graph = nx.Graph()
        pair_edge: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for edge in net.edges:
            u, v = int(edge["u"]), int(edge["v"])
            if u == v:
                continue
            graph.add_edge(u, v)
            pair_edge[(min(u, v), max(u, v))].append(edge)
        cycles = nx.cycle_basis(graph)
        occupied_cycle_edges: List[Set[int]] = []
        for cycle in cycles:
            cycle_edges: List[Dict[str, Any]] = []
            cycle_ids: Set[int] = set()
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                options = pair_edge.get((min(a, b), max(a, b)), [])
                if not options:
                    continue
                # Strongest representative defines the actual cycle geometry; weak
                # duplicates are handled by the parallel-pair pass above.
                edge = max(options, key=_artifact_score)
                cycle_edges.append(edge)
                cycle_ids.add(int(edge["edge_id"]))
            if len(cycle_edges) < 3:
                continue
            perimeter = sum(float(e["geometry"].length) for e in cycle_edges)
            minx = min(e["geometry"].bounds[0] for e in cycle_edges)
            miny = min(e["geometry"].bounds[1] for e in cycle_edges)
            maxx = max(e["geometry"].bounds[2] for e in cycle_edges)
            maxy = max(e["geometry"].bounds[3] for e in cycle_edges)
            span = max(maxx - minx, maxy - miny)
            if perimeter > max_perimeter_m or span > max_span_m:
                continue
            if any(cycle_ids & previous for previous in occupied_cycle_edges):
                continue

            removable = [
                e for e in cycle_edges
                if not _uturn_core(e)
                and not e.get("gap_repair")
                and not e.get("approved_gap_repair")
            ]
            if not removable:
                continue
            # Remove exactly one existing side: weak evidence first, then shortest
            # geometry. This preserves connectivity while collapsing the raster ring.
            candidate = min(
                removable,
                key=lambda e: (_artifact_score(e), float(e["geometry"].length)),
            )
            selected.add(int(candidate["edge_id"]))
            occupied_cycle_edges.append(cycle_ids)

        print(
            f"    [stage1:microloops] round={round_no}, cycles={len(cycles):,}, "
            f"drop={len(selected):,}",
            flush=True,
        )
        if not selected:
            break
        net, rows = _drop_edges(net, selected, "digital_pixel_loop_v0_5")
        dropped.extend(rows)
    return net, dropped


def _chaikin_open(coords: np.ndarray, ratio: float, iterations: int) -> np.ndarray:
    """Endpoint-preserving Chaikin smoothing for an open polyline."""
    out = np.asarray(coords, dtype=float)
    ratio = float(np.clip(ratio, 0.0, 0.49))
    for _ in range(max(0, int(iterations))):
        if len(out) <= 2 or ratio <= 0:
            break
        next_coords = [out[0].copy()]
        for a, b in zip(out[:-1], out[1:]):
            q = (1.0 - ratio) * a + ratio * b
            r = ratio * a + (1.0 - ratio) * b
            next_coords.extend((q, r))
        next_coords.append(out[-1].copy())
        out = np.asarray(next_coords, dtype=float)
    return out


def constrained_smooth_and_simplify(
    network: NetworkData,
    smooth_iterations: int = 1,
    smooth_ratio: float = 0.15,
    smooth_corridor_m: float = 1.5,
    simplify_tolerance_m: float = 0.8,
    uturn_smooth_ratio: float = 0.08,
    uturn_smooth_corridor_m: float = 0.8,
    uturn_simplify_tolerance_m: float = 0.4,
    progress_every: int = 10_000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Smooth raster stair-steps with endpoint and corridor guarantees.

    The function never moves graph nodes.  Each edge is smoothed independently only
    after degree-2 contraction, then optionally Douglas-Peucker simplified.  The final
    candidate must remain within a bidirectional Hausdorff corridor of the original
    raster-derived geometry.  The narrow U-turn core receives smaller movement/tolerance.
    """
    started = time.perf_counter()
    net = network.copy()
    rows: List[Dict[str, Any]] = []
    changed = 0
    total = len(net.edges)
    for index, edge in enumerate(net.edges, 1):
        original: LineString = edge["geometry"]
        coords = np.asarray(original.coords, dtype=float)
        core = _uturn_core(edge)
        if (edge.get("gap_repair") or edge.get("approved_gap_repair")
                or edge.get("approved_directional_collapse")
                or int(edge.get("u", -1)) == int(edge.get("v", -2))
                or len(coords) <= 2):
            rows.append({
                "edge_id": int(edge["edge_id"]),
                "smoothed": False,
                "reason": "gap_directional_collapse_selfloop_or_short",
                "vertices_before": int(len(coords)),
                "vertices_after": int(len(coords)),
                "hausdorff_shift_m": 0.0,
            })
            continue

        ratio = float(uturn_smooth_ratio if core else smooth_ratio)
        corridor = float(uturn_smooth_corridor_m if core else smooth_corridor_m)
        simplify_tol = float(uturn_simplify_tolerance_m if core else simplify_tolerance_m)
        candidate_coords = _chaikin_open(coords, ratio, smooth_iterations)
        candidate = LineString(candidate_coords)
        if simplify_tol > 0 and len(candidate.coords) > 2:
            candidate = candidate.simplify(simplify_tol, preserve_topology=True)

        # Force exact topology-node coordinates after every geometry operation.
        cand_coords = list(candidate.coords)
        if len(cand_coords) >= 2:
            cand_coords[0] = tuple(coords[0])
            cand_coords[-1] = tuple(coords[-1])
            candidate = LineString(cand_coords)

        shift = float(candidate.hausdorff_distance(original)) if isinstance(candidate, LineString) else float("inf")
        valid = (
            isinstance(candidate, LineString)
            and candidate.length > 0
            and candidate.is_simple
            and shift <= max(0.0, corridor) + 1e-9
            and Point(candidate.coords[0]).distance(Point(original.coords[0])) <= 1e-7
            and Point(candidate.coords[-1]).distance(Point(original.coords[-1])) <= 1e-7
        )
        if valid:
            edge["geometry"] = candidate
            edge["length_m"] = float(candidate.length)
            edge["max_geometry_shift_m"] = max(
                float(edge.get("max_geometry_shift_m", 0.0) or 0.0), shift
            )
            edge["geometry_origin"] = "primary_corridor_chain_smoothed_v0_5"
            changed += int(not candidate.equals_exact(original, tolerance=1e-9))
            rows.append({
                "edge_id": int(edge["edge_id"]),
                "smoothed": True,
                "uturn_core": bool(core),
                "vertices_before": int(len(coords)),
                "vertices_after": int(len(candidate.coords)),
                "length_before_m": float(original.length),
                "length_after_m": float(candidate.length),
                "hausdorff_shift_m": shift,
                "corridor_limit_m": corridor,
            })
        else:
            rows.append({
                "edge_id": int(edge["edge_id"]),
                "smoothed": False,
                "uturn_core": bool(core),
                "reason": "corridor_or_topology_validation_failed",
                "vertices_before": int(len(coords)),
                "vertices_after": int(len(coords)),
                "hausdorff_shift_m": shift,
                "corridor_limit_m": corridor,
            })

        if progress_every > 0 and (index % int(progress_every) == 0 or index == total):
            elapsed = time.perf_counter() - started
            print(
                f"    [smooth] processed={index:,}/{total:,}, changed={changed:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    return reindex_network(net), rows



def _final_geometry_chains(network: NetworkData) -> List[Dict[str, Any]]:
    """Return maximal topological chains without changing graph connectivity.

    Unlike the earlier topology contraction, this final geometry pass deliberately
    crosses degree-2 nodes even when the neighbouring edge is a gap repair.  Those
    nodes are geometry vertices, not junction semantics, and keeping them fixed is
    exactly what creates the visible zig-zag seams in raster-derived roads.
    """
    incident: Dict[int, List[int]] = defaultdict(list)
    edge_by_id = {int(e["edge_id"]): e for e in network.edges}
    for eid, edge in edge_by_id.items():
        u, v = int(edge["u"]), int(edge["v"])
        incident[u].append(eid)
        incident[v].append(eid)

    degree = {nid: len(eids) for nid, eids in incident.items()}
    visited: Set[int] = set()
    chains: List[Dict[str, Any]] = []

    def trace(start_node: int, first_eid: int, *, closed_seed: bool = False) -> Dict[str, Any]:
        edge_ids: List[int] = []
        edge_starts: List[int] = []
        nodes = [int(start_node)]
        current_node = int(start_node)
        current_eid = int(first_eid)
        closed = False
        while True:
            if current_eid in visited:
                break
            visited.add(current_eid)
            edge = edge_by_id[current_eid]
            u, v = int(edge["u"]), int(edge["v"])
            if u == v:
                edge_ids.append(current_eid)
                edge_starts.append(current_node)
                nodes.append(current_node)
                closed = True
                break
            next_node = v if u == current_node else u
            edge_ids.append(current_eid)
            edge_starts.append(current_node)
            nodes.append(next_node)
            if next_node == start_node and (closed_seed or len(edge_ids) > 1):
                closed = True
                break
            if degree.get(next_node, 0) != 2:
                break
            options = [eid for eid in incident.get(next_node, []) if eid != current_eid]
            if len(options) != 1 or options[0] in visited:
                break
            current_node = next_node
            current_eid = int(options[0])
        return {
            "edge_ids": edge_ids,
            "edge_starts": edge_starts,
            "nodes": nodes,
            "closed": bool(closed),
        }

    # Open chains are anchored by endpoints/junctions and therefore processed first.
    for start in sorted(n for n, d in degree.items() if d != 2):
        for eid in sorted(incident[start]):
            if eid not in visited:
                chains.append(trace(start, eid))

    # Remaining edges are pure degree-2 cycles.  They may be smoothed as closed
    # curves, but never opened or reconnected here.
    for eid in sorted(edge_by_id):
        if eid in visited:
            continue
        edge = edge_by_id[eid]
        chains.append(trace(int(edge["u"]), eid, closed_seed=True))
    return chains


def _sample_linestring(line: LineString, spacing_m: float, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
    length = float(line.length)
    spacing = max(0.20, float(spacing_m))
    if not np.isfinite(length) or length <= 0:
        return np.asarray([0.0]), np.asarray(line.coords[:1], dtype=float)
    if closed:
        count = max(8, int(np.ceil(length / spacing)))
        distances = np.linspace(0.0, length, count, endpoint=False)
    else:
        count = max(3, int(np.ceil(length / spacing)) + 1)
        distances = np.linspace(0.0, length, count)
    coords = np.asarray([[line.interpolate(float(d)).x, line.interpolate(float(d)).y] for d in distances], dtype=float)
    return distances, coords


def _gaussian_chain_candidate(
    line: LineString,
    *,
    sample_spacing_m: float,
    sigma_m: float,
    closed: bool,
) -> Tuple[LineString, np.ndarray, np.ndarray]:
    distances, coords = _sample_linestring(line, sample_spacing_m, closed)
    if len(coords) < 3 or sigma_m <= 0:
        if closed:
            coords = np.vstack([coords, coords[0]])
            distances = np.r_[distances, float(line.length)]
        return LineString(coords), distances, coords

    if len(distances) > 1:
        step = float(np.median(np.diff(distances)))
    else:
        step = max(0.20, float(sample_spacing_m))
    sigma_samples = max(0.01, float(sigma_m) / max(step, 1e-6))
    mode = "wrap" if closed else "nearest"
    smoothed = np.column_stack([
        gaussian_filter1d(coords[:, 0], sigma=sigma_samples, mode=mode),
        gaussian_filter1d(coords[:, 1], sigma=sigma_samples, mode=mode),
    ])

    if closed:
        smoothed = np.vstack([smoothed, smoothed[0]])
        distances = np.r_[distances, float(line.length)]
    else:
        # Pin both topological anchors without creating a kink at either end.  A
        # linearly varying correction keeps the whole curve C-like while returning
        # the exact graph endpoint coordinates.
        delta_start = coords[0] - smoothed[0]
        delta_end = coords[-1] - smoothed[-1]
        t = np.linspace(0.0, 1.0, len(smoothed))[:, None]
        smoothed = smoothed + (1.0 - t) * delta_start + t * delta_end
        smoothed[0] = coords[0]
        smoothed[-1] = coords[-1]
    return LineString(smoothed), distances, smoothed


def _interp_smoothed_xy(distances: np.ndarray, coords: np.ndarray, distance_m: float) -> np.ndarray:
    d = float(np.clip(distance_m, float(distances[0]), float(distances[-1])))
    return np.asarray([
        np.interp(d, distances, coords[:, 0]),
        np.interp(d, distances, coords[:, 1]),
    ], dtype=float)


def smooth_final_road_chains(
    network: NetworkData,
    *,
    sample_spacing_m: float = 0.5,
    sigma_m: float = 3.0,
    corridor_m: float = 2.5,
    uturn_sigma_m: float = 1.0,
    uturn_corridor_m: float = 0.8,
    min_sigma_m: float = 0.30,
    max_attempts: int = 6,
    progress_every: int = 2_000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Strong final geometry smoothing over complete road chains.

    The graph topology is invariant: edge endpoints retain their node IDs and real
    junctions/endpoints are fixed.  Interior degree-2 nodes may move coherently with
    the smoothed curve so adjacent edges share both position and tangent instead of
    meeting at a raster stair-step. Gap-repair edges participate in the same chain.

    A dense Gaussian arclength representation is used instead of a final
    Douglas-Peucker pass, so the exported polyline remains visually smooth.  Each
    chain is validated against its pre-smoothing geometry and the smoothing strength
    is reduced adaptively until it fits the allowed evidence corridor.
    """
    started = time.perf_counter()
    net = network.copy()
    node_by_id = {int(n["node_id"]): n for n in net.nodes}
    edge_by_id = {int(e["edge_id"]): e for e in net.edges}
    chains = _final_geometry_chains(net)
    rows: List[Dict[str, Any]] = []
    changed_edges = 0

    for chain_index, chain in enumerate(chains, 1):
        edge_ids: List[int] = list(chain["edge_ids"])
        edge_starts: List[int] = list(chain["edge_starts"])
        chain_nodes: List[int] = list(chain["nodes"])
        closed = bool(chain["closed"])
        if not edge_ids:
            continue

        parts: List[np.ndarray] = []
        boundaries = [0.0]
        total_length = 0.0
        malformed = False
        for eid, start_node in zip(edge_ids, edge_starts):
            edge = edge_by_id[eid]
            end_node = int(edge["v"] if int(edge["u"]) == int(start_node) else edge["u"])
            try:
                part = np.asarray(orient_coords(edge, int(start_node), end_node), dtype=float)
            except Exception:
                malformed = True
                break
            if len(part) < 2 or not np.isfinite(part).all():
                malformed = True
                break
            parts.append(part)
            total_length += float(LineString(part).length)
            boundaries.append(total_length)

        if malformed or not parts or total_length <= 0:
            rows.append({
                "chain_id": chain_index, "edge_count": len(edge_ids), "smoothed": False,
                "reason": "malformed_or_zero_length", "closed": closed,
            })
            continue

        joined = np.vstack([part if i == 0 else part[1:] for i, part in enumerate(parts)])
        original = LineString(joined)
        if original.length <= 0 or not original.is_valid:
            rows.append({
                "chain_id": chain_index, "edge_count": len(edge_ids), "smoothed": False,
                "reason": "invalid_original_chain", "closed": closed,
            })
            continue

        core = any(_uturn_core(edge_by_id[eid]) for eid in edge_ids)
        contains_gap = any(
            bool(edge_by_id[eid].get("gap_repair")) or bool(edge_by_id[eid].get("approved_gap_repair"))
            for eid in edge_ids
        )
        target_sigma = float(uturn_sigma_m if core else sigma_m)
        corridor = float(uturn_corridor_m if core else corridor_m)
        accepted = None
        accepted_sigma = 0.0
        accepted_shift = 0.0
        attempts = 0

        # Straight two-point chains do not need smoothing, but multi-edge chains can
        # still benefit because moving their interior degree-2 node removes seams.
        if len(joined) <= 2 and len(edge_ids) == 1:
            rows.append({
                "chain_id": chain_index, "edge_count": 1, "smoothed": False,
                "reason": "straight_two_point_chain", "closed": closed,
                "uturn_core": core, "contains_gap_repair": contains_gap,
            })
            continue

        sigma_try = max(0.0, target_sigma)
        while attempts < max(1, int(max_attempts)) and sigma_try >= max(0.0, float(min_sigma_m)):
            attempts += 1
            try:
                candidate, sample_d, sample_xy = _gaussian_chain_candidate(
                    original,
                    sample_spacing_m=sample_spacing_m,
                    sigma_m=sigma_try,
                    closed=closed,
                )
                shift = float(candidate.hausdorff_distance(original))
            except Exception:
                candidate = None
                shift = float("inf")

            valid = bool(
                isinstance(candidate, LineString)
                and candidate.length > 0
                and candidate.is_simple
                and np.isfinite(shift)
                and shift <= max(0.0, corridor) + 1e-9
                and candidate.length >= original.length * (0.82 if core else 0.70)
                and candidate.length <= original.length * 1.05 + 1e-6
            )
            if valid and not closed:
                valid = (
                    Point(candidate.coords[0]).distance(Point(original.coords[0])) <= 1e-8
                    and Point(candidate.coords[-1]).distance(Point(original.coords[-1])) <= 1e-8
                )
            if valid:
                accepted = (candidate, sample_d, sample_xy)
                accepted_sigma = sigma_try
                accepted_shift = shift
                break
            sigma_try *= 0.62

        if accepted is None:
            rows.append({
                "chain_id": chain_index, "edge_count": len(edge_ids), "smoothed": False,
                "reason": "corridor_or_topology_validation_failed", "closed": closed,
                "uturn_core": core, "contains_gap_repair": contains_gap,
                "target_sigma_m": target_sigma, "corridor_limit_m": corridor,
                "attempts": attempts,
            })
            continue

        candidate, sample_d, sample_xy = accepted
        # Move only degree-2 chain nodes.  Open-chain anchors (degree != 2) were
        # pinned by construction; junction coordinates never move.
        for i, nid in enumerate(chain_nodes):
            if i >= len(boundaries):
                break
            if not closed and (i == 0 or i == len(chain_nodes) - 1):
                continue
            xy = _interp_smoothed_xy(sample_d, sample_xy, boundaries[i])
            node = node_by_id.get(int(nid))
            if node is not None:
                node["x"], node["y"] = float(xy[0]), float(xy[1])
                node["geometry"] = Point(float(xy[0]), float(xy[1]))
                node["final_chain_smoothed_v0_5"] = True

        # Split the continuous smoothed chain back at the same topological edge
        # boundaries.  This preserves every edge ID/attribute while removing visual
        # kinks across degree-2 seams.
        for j, (eid, traversal_start) in enumerate(zip(edge_ids, edge_starts)):
            lo, hi = float(boundaries[j]), float(boundaries[j + 1])
            middle = np.where((sample_d > lo + 1e-9) & (sample_d < hi - 1e-9))[0]
            seg_coords = [_interp_smoothed_xy(sample_d, sample_xy, lo)]
            seg_coords.extend(sample_xy[k] for k in middle)
            seg_coords.append(_interp_smoothed_xy(sample_d, sample_xy, hi))
            clean: List[Tuple[float, float]] = []
            for xy in seg_coords:
                pt = (float(xy[0]), float(xy[1]))
                if not clean or np.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1]) > 1e-9:
                    clean.append(pt)
            if len(clean) < 2:
                continue
            edge = edge_by_id[eid]
            if int(edge["u"]) != int(traversal_start):
                clean.reverse()
            old_geom = edge["geometry"]
            new_geom = LineString(clean)
            edge_shift = float(new_geom.hausdorff_distance(old_geom))
            edge["geometry"] = new_geom
            edge["length_m"] = float(new_geom.length)
            edge["max_geometry_shift_m"] = max(
                float(edge.get("max_geometry_shift_m", 0.0) or 0.0), edge_shift
            )
            edge["geometry_origin"] = "final_chain_gaussian_smoothed_v0_5"
            edge["final_chain_smoothed_v0_5"] = True
            edge["final_smoothing_sigma_m"] = float(accepted_sigma)
            changed_edges += int(not new_geom.equals_exact(old_geom, tolerance=1e-9))

        rows.append({
            "chain_id": chain_index,
            "edge_count": len(edge_ids),
            "smoothed": True,
            "closed": closed,
            "uturn_core": core,
            "contains_gap_repair": contains_gap,
            "length_before_m": float(original.length),
            "length_after_m": float(candidate.length),
            "hausdorff_shift_m": float(accepted_shift),
            "corridor_limit_m": float(corridor),
            "target_sigma_m": float(target_sigma),
            "accepted_sigma_m": float(accepted_sigma),
            "sample_spacing_m": float(sample_spacing_m),
            "attempts": attempts,
        })

        if progress_every > 0 and (chain_index % int(progress_every) == 0 or chain_index == len(chains)):
            elapsed = time.perf_counter() - started
            print(
                f"    [final-smooth:v0.5] chains={chain_index:,}/{len(chains):,}, "
                f"changed_edges={changed_edges:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    # IDs/topology are unchanged in meaning; reindex refreshes node degree metadata
    # and edge lengths using the new geometries.
    return reindex_network(net), rows

def compress_collinear_paths(
    network: NetworkData,
    angle_epsilon: float = 1e-10,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove redundant collinear vertices without changing any line geometry."""
    net = network.copy()
    rows: List[Dict[str, Any]] = []
    for edge in net.edges:
        coords = np.asarray(edge["geometry"].coords, dtype=float)
        if edge.get("gap_repair") or edge.get("approved_gap_repair") or edge.get("approved_directional_collapse"):
            rows.append({
                "edge_id": int(edge["edge_id"]),
                "vertices_before": int(len(coords)),
                "vertices_after": int(len(coords)),
                "geometry_shift_m": 0.0,
                "reason": "approved_gap_or_directional_collapse_preserved",
            })
            continue
        if len(coords) <= 2:
            rows.append({"edge_id": int(edge["edge_id"]), "vertices_before": len(coords), "vertices_after": len(coords)})
            continue
        keep = [0]
        for i in range(1, len(coords) - 1):
            a = coords[i] - coords[keep[-1]]
            b = coords[i + 1] - coords[i]
            cross = abs(float(a[0] * b[1] - a[1] * b[0]))
            dot = float(np.dot(a, b))
            scale = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0)
            if cross <= float(angle_epsilon) * scale and dot > 0:
                continue
            keep.append(i)
        keep.append(len(coords) - 1)
        candidate = LineString(coords[np.asarray(keep, dtype=int)])
        # Exact geometric equality under floating point: both lines must cover each
        # other with effectively zero Hausdorff distance and equal length.
        shift = float(candidate.hausdorff_distance(edge["geometry"]))
        if shift <= 1e-9 and abs(candidate.length - edge["geometry"].length) <= 1e-7:
            edge["geometry"] = candidate
            edge["length_m"] = float(candidate.length)
            edge["geometry_origin"] = "s5_primary_exact_collinear_compressed"
            edge["max_geometry_shift_m"] = 0.0
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "vertices_before": int(len(coords)),
            "vertices_after": int(len(edge["geometry"].coords)),
            "geometry_shift_m": 0.0,
        })
    return reindex_network(net), rows
