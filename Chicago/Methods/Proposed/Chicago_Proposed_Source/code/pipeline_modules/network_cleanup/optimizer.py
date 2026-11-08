from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point

from .models import NetworkData, clean_linestring, compute_degrees, orient_coords, reindex_network
from .tile_io import RasterBlockSampler


def edge_confidence(e: Dict[str, Any]) -> float:
    tracks = float(e.get("distinct_tracks", 0) or 0)
    coverage = float(e.get("coverage_ratio", 0) or 0)
    support = float(e.get("support_mean", 0) or 0)
    support2 = float(e.get("support_ge2_fraction", 0) or 0)
    match_dist = float(e.get("mean_match_dist_m", 999) or 999)
    source = str(e.get("source_type", "primary"))
    source_prior = {
        "primary": 0.75,
        "mixed": 0.75,
        "support_rescued": 0.80,
        "gap_repair": float(e.get("gap_confidence", 0.5)),
        "parallel_merged": 0.78,
        "auxiliary": 0.35,
    }.get(source, 0.50)
    score = (
        0.28 * min(1.0, tracks / 4.0)
        + 0.16 * min(1.0, coverage / 0.50)
        + 0.18 * min(1.0, support / 2.0)
        + 0.13 * min(1.0, support2 / 0.35)
        + 0.15 * source_prior
        + 0.10 * max(0.0, 1.0-match_dist/30.0)
    )
    if e.get("uturn_protected"):
        score = max(score, 0.75)
    if e.get("activity_zone") and source == "auxiliary":
        score *= 0.55
    return float(np.clip(score, 0.0, 1.0))


def prune_network(
    network: NetworkData,
    terminal_length_m: float = 18.0,
    confidence_threshold: float = 0.28,
    min_component_length_m: float = 25.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    net = network.copy()
    for e in net.edges:
        e["confidence"] = edge_confidence(e)
        e["retained_reason"] = "initial"
    dropped: List[Dict[str, Any]] = []

    changed = True
    while changed:
        changed = False
        degree = defaultdict(int)
        for e in net.edges:
            degree[int(e["u"])] += 1
            degree[int(e["v"])] += 1
        kept: List[Dict[str, Any]] = []
        for e in net.edges:
            terminal = degree[int(e["u"])] == 1 or degree[int(e["v"])] == 1
            weak = float(e["confidence"]) < confidence_threshold
            source = str(e.get("source_type", "primary"))
            protected = bool(e.get("uturn_protected"))
            strong_raster = float(e.get("support_max", 0) or 0) >= 3 or float(e.get("support_mean", 0) or 0) >= 1.25
            matched = int(e.get("distinct_tracks", 0) or 0) > 0
            delete = (
                terminal
                and float(e["length_m"]) < terminal_length_m
                and weak
                and not protected
                and not strong_raster
                and not matched
                and source not in ("support_rescued", "gap_repair")
            )
            if delete:
                x = dict(e)
                x["drop_reason"] = "short_low_confidence_terminal"
                dropped.append(x)
                changed = True
            else:
                e["retained_reason"] = (
                    "uturn_protected" if protected else
                    "track_supported" if matched else
                    "strong_raster" if strong_raster else
                    "rescued_source" if source == "support_rescued" else
                    "gap_repair" if source == "gap_repair" else
                    "network_continuity"
                )
                kept.append(e)
        net.edges = kept

    # Remove tiny isolated components only when all their edges are weak auxiliary evidence.
    adjacency = defaultdict(list)
    for idx, e in enumerate(net.edges):
        adjacency[int(e["u"])].append(idx)
        adjacency[int(e["v"])].append(idx)
    unvisited = set(range(len(net.edges)))
    remove_ids = set()
    while unvisited:
        start = unvisited.pop()
        comp_edges = {start}
        stack = [start]
        nodes = {int(net.edges[start]["u"]), int(net.edges[start]["v"])}
        while stack:
            eid = stack.pop()
            e = net.edges[eid]
            for nid in (int(e["u"]), int(e["v"])):
                for other in adjacency[nid]:
                    if other in unvisited:
                        unvisited.remove(other)
                        comp_edges.add(other)
                        stack.append(other)
                        nodes.update((int(net.edges[other]["u"]), int(net.edges[other]["v"])))
        length = sum(float(net.edges[eid]["length_m"]) for eid in comp_edges)
        if length < min_component_length_m and all(
            float(net.edges[eid].get("confidence", 0)) < confidence_threshold
            and str(net.edges[eid].get("source_type")) == "auxiliary"
            and not net.edges[eid].get("uturn_protected")
            for eid in comp_edges
        ):
            remove_ids.update(comp_edges)

    if remove_ids:
        kept = []
        for idx, e in enumerate(net.edges):
            if idx in remove_ids:
                x = dict(e); x["drop_reason"] = "tiny_isolated_auxiliary_component"; dropped.append(x)
            else:
                kept.append(e)
        net.edges = kept

    return reindex_network(net), dropped


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b)/(na*nb), -1, 1)))))


def _endpoint_tangent(edge: Dict[str, Any], start_node: int, end_node: int, at_end: bool) -> np.ndarray:
    coords = np.asarray(orient_coords(edge, start_node, end_node), dtype=float)
    if len(coords) < 2:
        return np.asarray([0.0, 0.0], dtype=float)
    if at_end:
        i0 = max(0, len(coords) - 3)
        return coords[-1] - coords[i0]
    i1 = min(len(coords) - 1, 2)
    return coords[i1] - coords[0]


def _merge_chain_attributes(
    chain: List[Dict[str, Any]],
    start_node: int,
    end_node: int,
    geometry: LineString,
    new_edge_id: int,
) -> Dict[str, Any]:
    lengths = np.asarray([
        max(float(edge.get("length_m", edge["geometry"].length)), 1e-9)
        for edge in chain
    ], dtype=float)
    total = float(lengths.sum())

    def weighted(name: str, default: float = 0.0) -> float:
        values = []
        for edge in chain:
            try:
                value = float(edge.get(name, default) or default)
            except Exception:
                value = default
            values.append(value if np.isfinite(value) else default)
        return float(np.dot(np.asarray(values, dtype=float), lengths) / max(total, 1e-9))

    def max_value(name: str, default: float = 0.0) -> float:
        values = []
        for edge in chain:
            try:
                value = float(edge.get(name, default) or default)
            except Exception:
                value = default
            if np.isfinite(value):
                values.append(value)
        return float(max(values)) if values else float(default)

    sources = {str(edge.get("source_type", "primary")) for edge in chain}
    source_type = next(iter(sources)) if len(sources) == 1 else "mixed"
    provenance = []
    for edge in chain:
        raw = edge.get("merged_edge_ids", edge.get("old_edge_id", edge.get("edge_id")))
        provenance.extend(str(raw).split(","))

    new: Dict[str, Any] = {
        "edge_id": int(new_edge_id),
        "u": int(start_node),
        "v": int(end_node),
        "geometry": geometry,
        "length_m": float(geometry.length),
        "pixel_count": int(sum(int(edge.get("pixel_count", 0) or 0) for edge in chain)),
        "source_type": source_type,
        "primary_fraction": weighted("primary_fraction"),
        "auxiliary_fraction": weighted("auxiliary_fraction"),
        "rescued_fraction": weighted("rescued_fraction"),
        "support_mean": weighted("support_mean"),
        "support_median": weighted("support_median"),
        "support_max": max_value("support_max"),
        "support_ge2_fraction": weighted("support_ge2_fraction"),
        "kde_mean": weighted("kde_mean"),
        "kde_median": weighted("kde_median"),
        "kde_max": max_value("kde_max"),
        # Counts cannot be unioned because only aggregate counts are stored.  Max is
        # conservative and avoids inventing evidence during topology contraction.
        "distinct_tracks": max(int(edge.get("distinct_tracks", 0) or 0) for edge in chain),
        "distinct_clean_tracks": max(int(edge.get("distinct_clean_tracks", 0) or 0) for edge in chain),
        "match_points": int(sum(int(edge.get("match_points", 0) or 0) for edge in chain)),
        "hmm_sequence_count": int(sum(int(edge.get("hmm_sequence_count", 0) or 0) for edge in chain)),
        "coverage_ratio": weighted("coverage_ratio"),
        "mean_match_dist_m": weighted("mean_match_dist_m", 30.0),
        "p95_match_dist_m": max_value("p95_match_dist_m"),
        "distance_mean_m": weighted("distance_mean_m"),
        "uturn_core_protected": bool(any(edge.get("uturn_core_protected") for edge in chain)),
        "uturn_context_protected": bool(any(edge.get("uturn_context_protected", edge.get("uturn_protected")) for edge in chain)),
        "uturn_protected": bool(any(edge.get("uturn_context_protected", edge.get("uturn_protected")) for edge in chain)),
        "activity_zone": bool(all(edge.get("activity_zone") for edge in chain)),
        "gap_repair": bool(any(edge.get("gap_repair") for edge in chain)),
        "approved_directional_collapse": bool(any(edge.get("approved_directional_collapse") for edge in chain)),
        "digital_loop_diagonal": bool(any(edge.get("digital_loop_diagonal") for edge in chain)),
        "directional_collapse_evidence_fraction": max_value("directional_collapse_evidence_fraction"),
        "confidence": weighted("confidence"),
        "retained_reason": "degree2_chain_merged",
        "merged_edge_ids": ",".join(provenance),
        "geometry_origin": "s5_skeleton_subset",
        "max_geometry_shift_m": max_value("max_geometry_shift_m"),
        "baseline_length_m": float(sum(float(edge.get("baseline_length_m", edge.get("length_m", 0)) or 0) for edge in chain)),
    }
    # Preserve DEM attributes when this function is called after DEM annotation.
    for name in (
        "elevation_start_m", "elevation_end_m", "elevation_min_m", "elevation_max_m",
        "elevation_gain_m", "elevation_loss_m", "grade_mean_pct", "grade_max_abs_pct",
        "length_3d_m", "dem_valid_fraction",
    ):
        if any(name in edge for edge in chain):
            if name in ("elevation_min_m",):
                vals = [float(e[name]) for e in chain if e.get(name) is not None and np.isfinite(float(e[name]))]
                new[name] = min(vals) if vals else np.nan
            elif name in ("elevation_max_m", "grade_max_abs_pct"):
                vals = [float(e[name]) for e in chain if e.get(name) is not None and np.isfinite(float(e[name]))]
                new[name] = max(vals) if vals else np.nan
            elif name in ("elevation_gain_m", "elevation_loss_m", "length_3d_m"):
                vals = [float(e.get(name, 0) or 0) for e in chain]
                new[name] = float(sum(vals))
            else:
                new[name] = weighted(name, np.nan)
    return new


def merge_degree_two(
    network: NetworkData,
    preserve_turn_deg: float = 55.0,
    progress_every: int = 50_000,
) -> Tuple[NetworkData, int]:
    """Contract all safe degree-2 chains in near-linear time.

    The previous implementation searched the complete graph again after every single
    merge, giving O(E^2) behaviour and appearing to hang on large raster networks.
    This implementation classifies mergeable nodes once, then traces each chain once.
    Pure cycles, U-turn protected edges, self-loops, high-curvature nodes and parallel
    two-edge loops are preserved unchanged.
    """
    started = time.perf_counter()
    net = network.copy()
    nodes = {int(node["node_id"]): dict(node) for node in net.nodes}
    edges = {int(edge["edge_id"]): dict(edge) for edge in net.edges}
    incident: Dict[int, List[int]] = defaultdict(list)
    for edge_id, edge in edges.items():
        u, v = int(edge["u"]), int(edge["v"])
        incident[u].append(edge_id)
        incident[v].append(edge_id)

    print(
        f"    [topology] analyse nodes={len(nodes):,}, edges={len(edges):,}",
        flush=True,
    )

    mergeable: set[int] = set()
    for node_id, edge_ids in incident.items():
        if len(edge_ids) != 2:
            continue
        node = nodes.get(node_id, {})
        node_core = bool(node.get("uturn_core_protected", False))
        if node_core:
            continue
        first, second = edges[edge_ids[0]], edges[edge_ids[1]]
        first_core = bool(first.get("uturn_core_protected", False))
        second_core = bool(second.get("uturn_core_protected", False))
        if (first_core or second_core
                or bool(first.get("gap_repair")) or bool(second.get("gap_repair"))
                or bool(first.get("approved_gap_repair")) or bool(second.get("approved_gap_repair"))
                or bool(first.get("approved_directional_collapse")) or bool(second.get("approved_directional_collapse"))):
            continue
        if int(first["u"]) == int(first["v"]) or int(second["u"]) == int(second["v"]):
            continue
        neighbour_a = int(first["v"] if int(first["u"]) == node_id else first["u"])
        neighbour_b = int(second["v"] if int(second["u"]) == node_id else second["u"])
        # Two parallel edges returning to the same neighbour form a small loop and
        # must not be collapsed into a self-loop.
        if neighbour_a == neighbour_b:
            continue
        try:
            incoming = _endpoint_tangent(first, neighbour_a, node_id, at_end=True)
            outgoing = _endpoint_tangent(second, node_id, neighbour_b, at_end=False)
            turn = vector_angle_deg(incoming, outgoing)
        except Exception:
            continue
        if turn <= float(preserve_turn_deg):
            mergeable.add(node_id)

    print(
        f"    [topology] mergeable_degree2_nodes={len(mergeable):,}",
        flush=True,
    )

    visited: set[int] = set()
    output_edges: List[Dict[str, Any]] = []
    collapsed = 0
    merged_chains = 0
    traversed = 0
    next_edge_id = 0

    def emit_chain(
        chain_ids: List[int],
        start_node: int,
        end_node: int,
        coordinate_parts: List[np.ndarray],
    ) -> None:
        nonlocal collapsed, merged_chains, next_edge_id
        if not chain_ids:
            return
        if len(chain_ids) == 1 or start_node == end_node:
            # Preserve single edges and closed chains exactly.
            for edge_id in chain_ids:
                edge = dict(edges[edge_id])
                edge["edge_id"] = next_edge_id
                edge.setdefault("geometry_origin", "s5_skeleton")
                edge.setdefault("max_geometry_shift_m", 0.0)
                edge.setdefault("baseline_length_m", float(edge.get("length_m", edge["geometry"].length)))
                output_edges.append(edge)
                next_edge_id += 1
            return
        coords = np.vstack([
            part if index == 0 else part[1:]
            for index, part in enumerate(coordinate_parts)
            if len(part)
        ])
        geometry = clean_linestring([tuple(point) for point in coords])
        if geometry is None:
            # Defensive fallback: no topology loss if malformed geometry occurs.
            for edge_id in chain_ids:
                edge = dict(edges[edge_id])
                edge["edge_id"] = next_edge_id
                output_edges.append(edge)
                next_edge_id += 1
            return
        chain = [edges[edge_id] for edge_id in chain_ids]
        output_edges.append(_merge_chain_attributes(
            chain, start_node, end_node, geometry, next_edge_id
        ))
        next_edge_id += 1
        collapsed += len(chain_ids) - 1
        merged_chains += 1

    break_nodes = sorted(set(incident) - mergeable)
    for start_node in break_nodes:
        for first_edge_id in incident[start_node]:
            if first_edge_id in visited:
                continue
            chain_ids: List[int] = []
            parts: List[np.ndarray] = []
            current_node = start_node
            current_edge_id = first_edge_id
            end_node = start_node
            while True:
                if current_edge_id in visited:
                    break
                visited.add(current_edge_id)
                traversed += 1
                edge = edges[current_edge_id]
                next_node = int(edge["v"] if int(edge["u"]) == current_node else edge["u"])
                try:
                    part = np.asarray(orient_coords(edge, current_node, next_node), dtype=float)
                except Exception:
                    part = np.asarray(edge["geometry"].coords, dtype=float)
                chain_ids.append(current_edge_id)
                parts.append(part)
                end_node = next_node
                if next_node not in mergeable:
                    break
                options = [eid for eid in incident[next_node] if eid != current_edge_id]
                if len(options) != 1 or options[0] in visited:
                    break
                current_node = next_node
                current_edge_id = options[0]
            emit_chain(chain_ids, start_node, end_node, parts)
            if progress_every > 0 and traversed % max(1, int(progress_every)) < len(chain_ids):
                elapsed = time.perf_counter() - started
                print(
                    f"    [topology] traversed={traversed:,}/{len(edges):,}, "
                    f"output_edges={len(output_edges):,}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # Remaining edges belong to components composed only of mergeable degree-2
    # nodes (typically closed rings). Preserve them unchanged to avoid loop damage.
    for edge_id in sorted(set(edges) - visited):
        edge = dict(edges[edge_id])
        edge["edge_id"] = next_edge_id
        edge.setdefault("geometry_origin", "s5_skeleton")
        edge.setdefault("max_geometry_shift_m", 0.0)
        edge.setdefault("baseline_length_m", float(edge.get("length_m", edge["geometry"].length)))
        output_edges.append(edge)
        next_edge_id += 1
        traversed += 1

    out = NetworkData(
        list(nodes.values()),
        output_edges,
        net.crs,
        net.transform,
        net.width,
        net.height,
        dict(net.metadata),
    )
    out = reindex_network(out)
    elapsed = time.perf_counter() - started
    print(
        f"    [topology] done edges={len(edges):,}->{len(out.edges):,}, "
        f"merged_chains={merged_chains:,}, collapsed_edges={collapsed:,}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return out, collapsed

def resample_line(geom: LineString, spacing_m: float) -> np.ndarray:
    length = float(geom.length)
    if length <= 0:
        return np.asarray(geom.coords, dtype=float)
    distances = np.arange(0.0, length, max(spacing_m, 0.5))
    if len(distances) == 0 or distances[-1] < length:
        distances = np.r_[distances, length]
    return np.asarray([[geom.interpolate(float(d)).x, geom.interpolate(float(d)).y] for d in distances], dtype=float)


def refine_centerlines(
    network: NetworkData,
    support_path,
    kde_path,
    search_m: float = 4.0,
    spacing_m: float = 2.0,
    offset_step_m: float = 1.0,
    smooth_sigma: float = 0.7,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    net = network.copy()
    diagnostics: List[Dict[str, Any]] = []
    offsets = np.arange(-search_m, search_m+0.01, max(offset_step_m, 0.5))
    with RasterBlockSampler({"support": support_path, "kde": kde_path}) as sampler:
        for e in net.edges:
            geom = e["geometry"]
            if geom.length < max(6.0, 2*spacing_m) or e.get("gap_repair"):
                diagnostics.append({"edge_id": int(e["edge_id"]), "mean_shift_m": 0.0, "max_shift_m": 0.0, "refined": False})
                continue
            coords = resample_line(geom, spacing_m)
            original = coords.copy()
            for i in range(1, len(coords)-1):
                tangent = coords[min(i+1, len(coords)-1)] - coords[max(i-1, 0)]
                norm = np.linalg.norm(tangent)
                if norm <= 1e-9:
                    continue
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)/norm
                candidates = coords[i][None, :] + offsets[:, None]*normal[None, :]
                kde = sampler.values_xy("kde", candidates[:, 0], candidates[:, 1], default=0.0)
                support = sampler.values_xy("support", candidates[:, 0], candidates[:, 1], default=0.0)
                kde_scale = max(float(np.max(kde)), 1e-9)
                score = 0.68*(kde/kde_scale) + 0.24*np.minimum(support/3.0, 1.0) - 0.08*np.abs(offsets)/max(search_m, 1e-9)
                coords[i] = candidates[int(np.argmax(score))]
            if len(coords) >= 5 and smooth_sigma > 0:
                coords[:, 0] = gaussian_filter1d(coords[:, 0], smooth_sigma, mode="nearest")
                coords[:, 1] = gaussian_filter1d(coords[:, 1], smooth_sigma, mode="nearest")
            # preserve exact topology endpoints
            coords[0] = original[0]
            coords[-1] = original[-1]
            refined = clean_linestring([tuple(x) for x in coords])
            shifts = np.linalg.norm(coords-original, axis=1)
            if refined is not None:
                e["geometry"] = refined
                e["length_m"] = float(refined.length)
            diagnostics.append({
                "edge_id": int(e["edge_id"]),
                "mean_shift_m": float(np.mean(shifts)),
                "max_shift_m": float(np.max(shifts)),
                "refined": refined is not None,
            })
    return reindex_network(net), diagnostics


def optimize_junctions(
    network: NetworkData,
    support_path,
    kde_path,
    search_m: float = 5.0,
    grid_step_m: float = 1.0,
    max_shift_m: float = 6.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    net = network.copy()
    compute_degrees(net.nodes, net.edges)
    node_lookup = {int(n["node_id"]): n for n in net.nodes}
    incident = defaultdict(list)
    for e in net.edges:
        incident[int(e["u"])].append(e)
        incident[int(e["v"])].append(e)
    diagnostics: List[Dict[str, Any]] = []

    with RasterBlockSampler({"support": support_path, "kde": kde_path}) as sampler:
        for n in net.nodes:
            nid = int(n["node_id"])
            if int(n.get("degree", 0)) < 3 or n.get("uturn_protected"):
                continue
            center = np.asarray([float(n["x"]), float(n["y"])])
            values = np.arange(-search_m, search_m+0.01, max(grid_step_m, 0.5))
            gx, gy = np.meshgrid(values, values)
            candidates = center[None, :] + np.c_[gx.ravel(), gy.ravel()]
            radius = np.linalg.norm(candidates-center[None, :], axis=1)
            candidates = candidates[radius <= search_m]
            radius = radius[radius <= search_m]
            kde = sampler.values_xy("kde", candidates[:, 0], candidates[:, 1], default=0.0)
            support = sampler.values_xy("support", candidates[:, 0], candidates[:, 1], default=0.0)

            line_cost = np.zeros(len(candidates), dtype=float)
            for e in incident[nid]:
                coords = np.asarray(e["geometry"].coords, dtype=float)
                if int(e["u"]) == nid:
                    other = coords[min(2, len(coords)-1)]
                else:
                    other = coords[max(0, len(coords)-3)]
                direction = other-center
                norm = np.linalg.norm(direction)
                if norm <= 1e-9:
                    continue
                direction /= norm
                rel = candidates-center[None, :]
                perpendicular = np.abs(rel[:, 0]*direction[1]-rel[:, 1]*direction[0])
                line_cost += perpendicular
            line_cost /= max(len(incident[nid]), 1)
            kde_scale = max(float(np.max(kde)), 1e-9)
            score = 0.55*(kde/kde_scale) + 0.20*np.minimum(support/3.0, 1.0) - 0.15*(line_cost/max(search_m, 1e-9)) - 0.10*(radius/max(search_m, 1e-9))
            best = candidates[int(np.argmax(score))]
            shift = float(np.linalg.norm(best-center))
            if shift > max_shift_m:
                best = center + (best-center)*(max_shift_m/shift)
                shift = max_shift_m
            old = center.copy()
            n["x"], n["y"] = float(best[0]), float(best[1])
            n["geometry"] = Point(float(best[0]), float(best[1]))
            for e in incident[nid]:
                coords = list(e["geometry"].coords)
                if int(e["u"]) == nid:
                    coords[0] = (float(best[0]), float(best[1]))
                if int(e["v"]) == nid:
                    coords[-1] = (float(best[0]), float(best[1]))
                geom = clean_linestring(coords)
                if geom is not None:
                    e["geometry"] = geom
                    e["length_m"] = float(geom.length)
            diagnostics.append({"node_id": nid, "old_x": old[0], "old_y": old[1], "new_x": best[0], "new_y": best[1], "shift_m": shift})
    return reindex_network(net), diagnostics
