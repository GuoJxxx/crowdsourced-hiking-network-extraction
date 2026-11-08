from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .models import NetworkData, reindex_network
from .raw_track_topology import RawTrackWitnessIndex
from .s7_force_refine import (
    _arc_score,
    _cycle_paths_between_gates,
    _polygon_area_xy,
    _representative_arc,
    _two_branch_width,
)
from .raw_track_topology import _synthetic_connector


def _cycle_state(network: NetworkData):
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    graph = nx.Graph()
    nodes = {int(n["node_id"]): n for n in network.nodes}
    for edge in network.edges:
        u, v = int(edge["u"]), int(edge["v"])
        incident[u].append(edge); incident[v].append(edge)
        if u == v:
            continue
        pair_edges[(min(u, v), max(u, v))].append(edge)
        graph.add_edge(u, v)
    return nodes, incident, pair_edges, graph


def _support_key(support: Dict[str, Any], arc_score: float, length_m: float):
    return (
        int(support.get("support_tracks", 0)),
        float(support.get("best_coverage", 0.0)),
        -float(support.get("best_p90_m", float("inf"))),
        float(arc_score),
        -float(length_m),
    )


def resolve_single_corridor_cycles(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    raw_corridor_m: float = 4.0,
    min_route_coverage: float = 0.60,
    min_independent_tracks: int = 1,
    min_full_ring_tracks: int = 2,
    max_perimeter_m: float = 220.0,
    max_span_m: float = 80.0,
    max_area_m2: float = 3000.0,
    max_rounds: int = 8,
    thin_duplicate_max_perimeter_m: float = 800.0,
    thin_duplicate_max_span_m: float = 300.0,
    thin_duplicate_max_area_m2: float = 8000.0,
    thin_duplicate_min_independent_tracks: int = 3,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Collapse two-gate false cycles when raw tracks support only one route family.

    This pass intentionally runs *after* all S7 repair passes.  Unlike the legacy
    polygon resolver, approved repair edges and U-turn-protected edges are not
    automatically immune: approval proves the individual connector was admissible,
    not that two admissible connectors represent two real roads.  A cycle survives
    when independent original-track contributions support both gate-to-gate routes.

    v0.5 treats original trajectories as topology evidence, not just a soft score.
    A loop is preserved only when independent raw route families support the competing
    routes.  When raw trajectories occupy the local corridor but do not support a
    second route, the weaker route is opened/collapsed even when it is all-primary.
    """
    idx = witness_index
    net = reindex_network(network.copy())
    if idx is None or not idx.enabled:
        return net, []

    decisions: List[Dict[str, Any]] = []
    for round_no in range(1, max(1, int(max_rounds)) + 1):
        nodes, incident, pair_edges, graph = _cycle_state(net)

        # Raster tracing can represent a two-route corridor as two distinct geometries
        # with exactly the same topology endpoints.  nx.Graph collapses such parallel
        # edges, so cycle_basis cannot see the loop even though it is visibly a cycle.
        # Resolve these two-edge route pairs explicitly before ordinary cycle_basis.
        parallel_drop: Set[int] = set()
        parallel_rows: List[Dict[str, Any]] = []
        for (ga, gb), options in pair_edges.items():
            if len(options) < 2:
                continue
            group_ids = {int(e["edge_id"]) for e in options}
            if len(options) > 2:
                # Three or more geometries can share exactly the same graph endpoints.
                # nx.Graph hides every one of these visible loops, so treat them as a
                # parallel route family. Keep independently witnessed route families;
                # if raw evidence supports only one family, collapse all duplicates to
                # the single best-supported corridor.
                if not any(int(e["edge_id"]) not in group_ids for e in incident.get(int(ga), [])):
                    continue
                if not any(int(e["edge_id"]) not in group_ids for e in incident.get(int(gb), [])):
                    continue
                supports = [
                    idx.route_support(
                        e["geometry"], corridor_m=float(raw_corridor_m),
                        min_coverage=float(min_route_coverage),
                    )
                    for e in options
                ]
                track_sets = [set(x.get("contribution_ids", set())) for x in supports]
                exclusive_counts = []
                for i, tids in enumerate(track_sets):
                    others: Set[str] = set()
                    for j, other in enumerate(track_sets):
                        if i != j:
                            others.update(other)
                    exclusive_counts.append(len(tids - others))
                independent = [i for i, n in enumerate(exclusive_counts) if n >= int(min_independent_tracks)]
                near_dist = max(float(raw_corridor_m) * 1.5, float(raw_corridor_m) + 1.0)
                near_sets = [idx.tracks_near_geometry(e["geometry"], near_dist) for e in options]
                local_near = set().union(*near_sets) if near_sets else set()
                if not local_near and not any(track_sets):
                    best_i = max(
                        range(len(options)),
                        key=lambda i: _support_key(
                            supports[i], _arc_score([options[i]]), options[i]["geometry"].length
                        ),
                    )
                    drop_ids = [int(e["edge_id"]) for i, e in enumerate(options) if i != best_i]
                    parallel_drop.update(drop_ids)
                    parallel_rows.append({
                        "round": int(round_no), "cycle_representation": "parallel_route_family",
                        "gate_a": int(ga), "gate_b": int(gb), "route_count": int(len(options)),
                        "action": "collapse_parallel_family_without_raw_route_evidence",
                        "kept_route_indices": str(best_i + 1),
                        "removed_edge_ids": ",".join(map(str, sorted(drop_ids))),
                    })
                    continue
                if len(independent) >= 2:
                    keep_indices = set(independent)
                    action = "preserve_independent_parallel_route_families"
                elif len(independent) == 1:
                    keep_indices = {independent[0]}
                    action = "collapse_parallel_route_family"
                else:
                    best_i = max(
                        range(len(options)),
                        key=lambda i: _support_key(
                            supports[i], _arc_score([options[i]]), options[i]["geometry"].length
                        ),
                    )
                    keep_indices = {best_i}
                    action = "collapse_parallel_route_family"
                drop_ids = [int(e["edge_id"]) for i, e in enumerate(options) if i not in keep_indices]
                parallel_drop.update(drop_ids)
                parallel_rows.append({
                    "round": int(round_no), "cycle_representation": "parallel_route_family",
                    "gate_a": int(ga), "gate_b": int(gb), "route_count": int(len(options)),
                    "local_raw_tracks": int(len(local_near)),
                    "independent_route_count": int(len(independent)),
                    "action": action,
                    "kept_route_indices": ",".join(str(i + 1) for i in sorted(keep_indices)),
                    "removed_edge_ids": ",".join(map(str, sorted(drop_ids))),
                })
                continue
            group_ids = {int(e["edge_id"]) for e in options}
            if not any(int(e["edge_id"]) not in group_ids for e in incident.get(int(ga), [])):
                continue
            if not any(int(e["edge_id"]) not in group_ids for e in incident.get(int(gb), [])):
                continue
            e1, e2 = options
            geom1, geom2 = e1["geometry"], e2["geometry"]
            s1 = idx.route_support(geom1, corridor_m=float(raw_corridor_m), min_coverage=float(min_route_coverage))
            s2 = idx.route_support(geom2, corridor_m=float(raw_corridor_m), min_coverage=float(min_route_coverage))
            t1 = set(s1.get("contribution_ids", set())); t2 = set(s2.get("contribution_ids", set()))
            ex1, ex2 = t1 - t2, t2 - t1
            gate_xy = np.asarray([float(nodes[int(ga)]["x"]), float(nodes[int(ga)]["y"])], dtype=float)
            median_width, p90_width, max_width = _two_branch_width(geom1, geom2, gate_xy)
            thin_duplicate = p90_width <= 14.0 and max_width <= 20.0
            score1, score2 = _arc_score([e1]), _arc_score([e2])
            row = {
                "round": int(round_no), "cycle_representation": "parallel_edges",
                "gate_a": int(ga), "gate_b": int(gb),
                "perimeter_m": float(geom1.length + geom2.length),
                "span_m": float(max(
                    max(geom1.bounds[2], geom2.bounds[2]) - min(geom1.bounds[0], geom2.bounds[0]),
                    max(geom1.bounds[3], geom2.bounds[3]) - min(geom1.bounds[1], geom2.bounds[1]),
                )),
                "area_m2": 0.0,
                "route1_raw_tracks": int(len(t1)), "route2_raw_tracks": int(len(t2)),
                "route1_exclusive_tracks": int(len(ex1)), "route2_exclusive_tracks": int(len(ex2)),
                "route1_best_coverage": float(s1.get("best_coverage", 0.0)),
                "route2_best_coverage": float(s2.get("best_coverage", 0.0)),
                "route1_best_p90_m": float(s1.get("best_p90_m", float("inf"))),
                "route2_best_p90_m": float(s2.get("best_p90_m", float("inf"))),
                "route1_evidence_score": float(score1), "route2_evidence_score": float(score2),
                "median_branch_width_m": float(median_width), "p90_branch_width_m": float(p90_width),
                "max_branch_width_m": float(max_width), "thin_duplicate": bool(thin_duplicate),
                "removed_edge_ids": "", "kept_route": 0,
            }
            required_independent = (
                max(int(min_independent_tracks), int(thin_duplicate_min_independent_tracks))
                if thin_duplicate else int(min_independent_tracks)
            )
            row["required_independent_tracks_per_route"] = int(required_independent)
            if len(ex1) >= required_independent and len(ex2) >= required_independent:
                row["action"] = "preserve_independent_dual_route_cycle"
                parallel_rows.append(row)
                continue
            near1 = idx.tracks_near_geometry(geom1, max(float(raw_corridor_m) * 1.5, float(raw_corridor_m) + 1.0))
            near2 = idx.tracks_near_geometry(geom2, max(float(raw_corridor_m) * 1.5, float(raw_corridor_m) + 1.0))
            row["route1_near_raw_tracks"] = int(len(near1))
            row["route2_near_raw_tracks"] = int(len(near2))
            if not t1 and not t2 and not near1 and not near2:
                row["collapse_reason"] = "no_raw_route_family_support"
            if not t1 and not t2 and not near1 and not near2 and thin_duplicate:
                row["collapse_reason"] = "thin_duplicate_without_raw_route_evidence"
            keep1 = _support_key(s1, score1, geom1.length) >= _support_key(s2, score2, geom2.length)
            weak = e2 if keep1 else e1
            weak_is_synthetic = _synthetic_connector(weak)
            row["weak_route_synthetic"] = bool(weak_is_synthetic)
            parallel_drop.add(int(weak["edge_id"]))
            row["action"] = "collapse_single_corridor_false_cycle"
            row["kept_route"] = 1 if keep1 else 2
            row["removed_edge_ids"] = str(int(weak["edge_id"]))
            parallel_rows.append(row)

        decisions.extend(parallel_rows)
        if parallel_drop:
            net.edges = [e for e in net.edges if int(e["edge_id"]) not in parallel_drop]
            net = reindex_network(net)
            print(
                f"    [semantic-cycle:v0.5] round={round_no}, parallel_routes_resolved={len(parallel_drop):,}",
                flush=True,
            )
            continue

        cycles = nx.cycle_basis(graph)
        if not cycles:
            break
        candidates = []
        for cyc in cycles:
            pts = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in cyc if n in nodes], dtype=float)
            if len(pts) != len(cyc):
                continue
            span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
            area = _polygon_area_xy(pts)
            perimeter = 0.0
            cycle_ids: Set[int] = set()
            valid = True
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                opts = pair_edges.get((min(int(a), int(b)), max(int(a), int(b))), [])
                if not opts:
                    valid = False; break
                cycle_ids.update(int(e["edge_id"]) for e in opts)
                perimeter += min(float(e["geometry"].length) for e in opts)
            if not valid:
                continue
            external = {
                int(n): [e for e in incident.get(int(n), []) if int(e["edge_id"]) not in cycle_ids]
                for n in cyc
            }
            gates = [int(n) for n in cyc if external[int(n)]]
            legacy_size_ok = (
                perimeter <= float(max_perimeter_m)
                and span <= float(max_span_m)
                and area <= float(max_area_m2)
            )
            oversize_thin_duplicate = False
            if not legacy_size_ok:
                if len(gates) != 2:
                    continue
                if (
                    perimeter > float(thin_duplicate_max_perimeter_m)
                    or span > float(thin_duplicate_max_span_m)
                    or area > float(thin_duplicate_max_area_m2)
                ):
                    continue
                ga, gb = gates
                p1, p2 = _cycle_paths_between_gates(cyc, ga, gb)
                r1 = _representative_arc(p1, pair_edges)
                r2 = _representative_arc(p2, pair_edges)
                if r1 is None or r2 is None:
                    continue
                gate_xy = np.asarray([float(nodes[ga]["x"]), float(nodes[ga]["y"])], dtype=float)
                _, p90_width, max_width = _two_branch_width(r1[0], r2[0], gate_xy)
                oversize_thin_duplicate = p90_width <= 14.0 and max_width <= 20.0
                if not oversize_thin_duplicate:
                    continue
            candidates.append((span, area, perimeter, cyc, cycle_ids, gates, oversize_thin_duplicate))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2], len(x[3])))

        planned_drop: Set[int] = set()
        occupied: List[Set[int]] = []
        resolved_this_round = 0
        round_rows: List[Dict[str, Any]] = []
        for span, area, perimeter, cyc, cycle_ids, gates, oversize_thin_duplicate in candidates:
            if any(cycle_ids & old for old in occupied):
                continue

            # Two-gate loops are the common "bubble on one road" case.  Here it is
            # safe to remove the whole unsupported gate-to-gate branch because no
            # external road attaches to its internal nodes.
            if len(gates) == 2:
                ga, gb = gates
                p1, p2 = _cycle_paths_between_gates(cyc, ga, gb)
                r1 = _representative_arc(p1, pair_edges)
                r2 = _representative_arc(p2, pair_edges)
                if r1 is None or r2 is None:
                    continue
                geom1, reps1, ids1 = r1
                geom2, reps2, ids2 = r2
                s1 = idx.route_support(geom1, corridor_m=float(raw_corridor_m), min_coverage=float(min_route_coverage))
                s2 = idx.route_support(geom2, corridor_m=float(raw_corridor_m), min_coverage=float(min_route_coverage))
                gate_xy = np.asarray([float(nodes[ga]["x"]), float(nodes[ga]["y"])], dtype=float)
                median_width, p90_width, max_width = _two_branch_width(geom1, geom2, gate_xy)
                thin_duplicate = p90_width <= 14.0 and max_width <= 20.0
                t1 = set(s1.get("contribution_ids", set()))
                t2 = set(s2.get("contribution_ids", set()))
                ex1, ex2 = t1 - t2, t2 - t1
                score1, score2 = _arc_score(reps1), _arc_score(reps2)
                near_dist = max(float(raw_corridor_m) * 1.5, float(raw_corridor_m) + 1.0)
                near1 = idx.tracks_near_geometry(geom1, near_dist)
                near2 = idx.tracks_near_geometry(geom2, near_dist)

                row = {
                    "round": int(round_no), "gate_count": 2, "gate_a": int(ga), "gate_b": int(gb),
                    "perimeter_m": float(perimeter), "span_m": float(span), "area_m2": float(area),
                    "route1_raw_tracks": int(len(t1)), "route2_raw_tracks": int(len(t2)),
                    "route1_near_raw_tracks": int(len(near1)), "route2_near_raw_tracks": int(len(near2)),
                    "route1_exclusive_tracks": int(len(ex1)), "route2_exclusive_tracks": int(len(ex2)),
                    "route1_best_coverage": float(s1.get("best_coverage", 0.0)),
                    "route2_best_coverage": float(s2.get("best_coverage", 0.0)),
                    "route1_best_p90_m": float(s1.get("best_p90_m", float("inf"))),
                    "route2_best_p90_m": float(s2.get("best_p90_m", float("inf"))),
                    "route1_evidence_score": float(score1), "route2_evidence_score": float(score2),
                    "median_branch_width_m": float(median_width), "p90_branch_width_m": float(p90_width),
                    "max_branch_width_m": float(max_width), "thin_duplicate": bool(thin_duplicate),
                    "removed_edge_ids": "", "kept_route": 0,
                }

                row["oversize_thin_duplicate"] = bool(oversize_thin_duplicate)
                required_independent = (
                    max(int(min_independent_tracks), int(thin_duplicate_min_independent_tracks))
                    if thin_duplicate else int(min_independent_tracks)
                )
                row["required_independent_tracks_per_route"] = int(required_independent)
                if len(ex1) >= required_independent and len(ex2) >= required_independent:
                    row["action"] = "preserve_independent_dual_route_cycle"
                    round_rows.append(row)
                    occupied.append(cycle_ids)
                    continue
                if not t1 and not t2 and not near1 and not near2:
                    row["collapse_reason"] = "no_raw_route_family_support"
                if not t1 and not t2 and not near1 and not near2 and thin_duplicate:
                    row["collapse_reason"] = "thin_duplicate_without_raw_route_evidence"

                keep1 = _support_key(s1, score1, geom1.length) >= _support_key(s2, score2, geom2.length)
                weak_ids = ids2 if keep1 else ids1
                weak_reps = reps2 if keep1 else reps1
                row["weak_route_synthetic"] = bool(any(_synthetic_connector(e) for e in weak_reps))
                planned_drop.update(int(x) for x in weak_ids)
                row["action"] = "collapse_single_corridor_false_cycle"
                row["kept_route"] = 1 if keep1 else 2
                row["removed_edge_ids"] = ",".join(map(str, sorted(int(x) for x in weak_ids)))
                round_rows.append(row)
                occupied.append(cycle_ids)
                resolved_this_round += 1
                continue

            # Multi-gate cycles previously escaped semantic cleanup completely.  Split
            # the ring into arcs between adjacent external gates, classify each arc
            # from original-track route continuity, and open only the weakest
            # unsupported arc.  Removing one complete gate-to-gate arc breaks the
            # ring while keeping every external branch connected through the rest of
            # the cycle.  The same logic also handles lollipop/isolated cycles by
            # considering one-edge arcs when there are fewer than two gates.
            gate_nodes = [int(g) for g in gates]
            if len(gate_nodes) >= 3:
                gate_indices = sorted(cyc.index(g) for g in gate_nodes)
                arc_paths = []
                for q, start_i in enumerate(gate_indices):
                    end_i = gate_indices[(q + 1) % len(gate_indices)]
                    path = [cyc[start_i]]
                    k = start_i
                    while True:
                        k = (k + 1) % len(cyc)
                        path.append(cyc[k])
                        if k == end_i:
                            break
                    arc_paths.append(path)
            else:
                # 0/1 external gate: inspect individual cycle edges so breaking the
                # loop never removes the only attachment to the rest of the network.
                arc_paths = [[a, b] for a, b in zip(cyc, cyc[1:] + cyc[:1])]

            arc_rows = []
            near_dist = max(float(raw_corridor_m) * 1.5, float(raw_corridor_m) + 1.0)
            for path in arc_paths:
                rep = _representative_arc(path, pair_edges)
                if rep is None:
                    continue
                geom, reps, ids = rep
                sup = idx.route_support(geom, corridor_m=float(raw_corridor_m), min_coverage=float(min_route_coverage))
                tids = set(sup.get("contribution_ids", set()))
                near = idx.tracks_near_geometry(geom, near_dist)
                arc_rows.append({
                    "path": path, "geom": geom, "reps": reps, "ids": ids, "support": sup,
                    "tracks": tids, "near": near, "score": _arc_score(reps),
                    "synthetic": any(_synthetic_connector(e) for e in reps),
                })
            if not arc_rows:
                continue
            local_near = set().union(*(r["near"] for r in arc_rows))
            supported = [r for r in arc_rows if len(r["tracks"]) >= int(min_independent_tracks)]
            full_ring_tracks = set(arc_rows[0]["tracks"])
            for rr in arc_rows[1:]:
                full_ring_tracks &= set(rr["tracks"])
            row = {
                "round": int(round_no), "gate_count": int(len(gates)),
                "perimeter_m": float(perimeter), "span_m": float(span), "area_m2": float(area),
                "arc_count": int(len(arc_rows)), "supported_arc_count": int(len(supported)),
                "local_raw_tracks": int(len(local_near)),
                "full_ring_raw_tracks": int(len(full_ring_tracks)),
                "required_full_ring_tracks": int(max(1, min_full_ring_tracks)),
                "removed_edge_ids": "", "kept_route": 0,
            }
            # Nearby support on every arc can still be the same jittering road bundle.
            # A real 0/1/multi-gate ring needs repeated original contributions that
            # genuinely cover the complete ring, otherwise open the weakest arc.
            if len(full_ring_tracks) >= int(max(1, min_full_ring_tracks)):
                row["action"] = "preserve_cycle_complete_raw_route_family"
                row["preserve_reason"] = "complete_ring_route_family"
                round_rows.append(row); occupied.append(cycle_ids); continue
            if not local_near and len(gates) >= 2:
                row["no_local_raw_evidence"] = True
                row["collapse_reason"] = "no_complete_ring_route_family"
            if not local_near and len(gates) <= 1:
                row["no_local_raw_evidence"] = True
                row["collapse_reason"] = "no_complete_ring_route_family"

            unsupported = [r for r in arc_rows if len(r["tracks"]) < int(min_independent_tracks)]
            pool = unsupported if unsupported else arc_rows
            weak = min(
                pool,
                key=lambda r: (
                    0 if r["synthetic"] else 1,
                    len(r["tracks"]), len(r["near"]),
                    float(r["support"].get("best_coverage", 0.0)),
                    float(r["score"]), float(r["geom"].length),
                ),
            )
            planned_drop.update(int(x) for x in weak["ids"])
            row["action"] = "open_multigate_false_cycle_remove_unsupported_arc"
            row["collapse_reason"] = "no_complete_ring_route_family"
            row["weak_arc_raw_tracks"] = int(len(weak["tracks"]))
            row["weak_arc_near_raw_tracks"] = int(len(weak["near"]))
            row["weak_arc_synthetic"] = bool(weak["synthetic"])
            row["removed_edge_ids"] = ",".join(map(str, sorted(int(x) for x in weak["ids"])))
            round_rows.append(row)
            occupied.append(cycle_ids)
            resolved_this_round += 1

        decisions.extend(round_rows)
        if not planned_drop:
            break
        net.edges = [e for e in net.edges if int(e["edge_id"]) not in planned_drop]
        net = reindex_network(net)
        print(
            f"    [semantic-cycle:v0.5] round={round_no}, resolved={resolved_this_round:,}, "
            f"drop_edges={len(planned_drop):,}", flush=True,
        )
    return net, decisions
