from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .models import NetworkData, reindex_network
from .tile_io import RasterBlockSampler
from .topology_utils import safe_intersection, weighted_graph
from .junction_cleanup import (
    _angle_deg,
    _continuation_vector,
    _crosses_unmodelled_existing_edge,
    _sample_line_evidence,
)


def connectivity_audit(network: NetworkData) -> Dict[str, Any]:
    """Compact graph-connectivity metrics used before/after S7 repairs."""
    net = reindex_network(network.copy())
    graph = weighted_graph(net)
    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    incident: Dict[int, int] = defaultdict(int)
    for edge in net.edges:
        u, v = int(edge["u"]), int(edge["v"])
        incident[u] += 1
        incident[v] += 1

    total_length = float(sum(float(e.get("length_m", e["geometry"].length)) for e in net.edges))
    comp_lengths: List[float] = []
    for cc in components:
        ccset = set(int(x) for x in cc)
        length = sum(
            float(e.get("length_m", e["geometry"].length))
            for e in net.edges
            if int(e["u"]) in ccset and int(e["v"]) in ccset
        )
        comp_lengths.append(float(length))
    largest_length = max(comp_lengths, default=0.0)
    largest_nodes = max((len(c) for c in components), default=0)
    # Cyclomatic number for the actual road multigraph.  Two distinct route geometries
    # can share the same topology endpoints; nx.Graph collapses that parallel pair and
    # would incorrectly report no cycle.  Use the raw edge count while retaining
    # connected components from the simple graph.
    cyclomatic = max(0, int(len(net.edges) - graph.number_of_nodes() + len(components)))
    return {
        "node_count": int(len(net.nodes)),
        "edge_count": int(len(net.edges)),
        "component_count": int(len(components)),
        "endpoint_count": int(sum(1 for n in graph.nodes if int(incident.get(int(n), 0)) == 1)),
        "junction_node_count": int(sum(1 for n in graph.nodes if int(incident.get(int(n), 0)) >= 3)),
        "largest_component_node_fraction": float(largest_nodes / max(1, graph.number_of_nodes())),
        "largest_component_length_fraction": float(largest_length / max(total_length, 1e-9)),
        "total_length_m": total_length,
        "largest_component_length_m": float(largest_length),
        "cyclomatic_number": int(cyclomatic),
    }


def unresolved_endpoints(network: NetworkData) -> List[Dict[str, Any]]:
    """Return degree-1 nodes with component context for QGIS diagnostics."""
    net = reindex_network(network.copy())
    graph = weighted_graph(net)
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for edge in net.edges:
        incident[int(edge["u"])].append(edge)
        incident[int(edge["v"])].append(edge)
    comp_by_node: Dict[int, int] = {}
    comp_lengths: Dict[int, float] = {}
    for cid, cc in enumerate(nx.connected_components(graph), start=1):
        cset = set(int(n) for n in cc)
        for n in cset:
            comp_by_node[n] = cid
        comp_lengths[cid] = float(sum(
            float(e.get("length_m", e["geometry"].length))
            for e in net.edges
            if int(e["u"]) in cset and int(e["v"]) in cset
        ))
    nodes = {int(n["node_id"]): n for n in net.nodes}
    rows = []
    for nid, edges in incident.items():
        if len(edges) != 1:
            continue
        node = nodes[nid]
        cid = int(comp_by_node.get(nid, -1))
        rows.append({
            "node_id": int(nid),
            "component_id": cid,
            "component_length_m": float(comp_lengths.get(cid, 0.0)),
            "incident_edge_id": int(edges[0]["edge_id"]),
            "x": float(node["x"]),
            "y": float(node["y"]),
            "geometry": Point(float(node["x"]), float(node["y"])),
        })
    return rows


class _ComponentUF:
    def __init__(self, values: Set[int]):
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _new_stitch_edge(edge_id: int, u: int, v: int, geom: LineString, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "edge_id": int(edge_id),
        "u": int(u),
        "v": int(v),
        "geometry": geom,
        "length_m": float(geom.length),
        "pixel_count": 0,
        "source_type": "s7_component_stitching",
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
        "component_stitching_repair": True,
        "component_stitch_distance_m": float(row["distance_m"]),
        "component_stitch_angle_a_deg": float(row["angle_a_deg"]),
        "component_stitch_angle_b_deg": float(row["angle_b_deg"]),
        "component_stitch_evidence_fraction": float(row["evidence_fraction"]),
        "confidence": float(row["confidence"]),
        "retained_reason": "evidence_supported_component_stitch_v0_5",
        "geometry_origin": "s7_component_stitch_v0_5",
        "max_geometry_shift_m": 0.0,
        "baseline_length_m": float(geom.length),
    }


def repair_fragmented_components(
    network: NetworkData,
    support_path=None,
    kde_path=None,
    distance_path=None,
    max_distance_m: float = 28.0,
    max_angle_deg: float = 35.0,
    min_evidence_fraction: float = 0.35,
    evidence_distance_m: float = 10.0,
    long_gap_threshold_m: float = 15.0,
    long_gap_max_angle_deg: float = 22.5,
    long_gap_min_evidence_fraction: float = 0.65,
    max_repairs: int = 10000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Stitch nearby disconnected road components using conservative endpoint evidence.

    This pass is deliberately cross-component only. Same-component loop closure is
    handled by micro-gap/S4/directional/junction passes where stronger mode-specific
    evidence is available.
    """
    net = reindex_network(network.copy())
    if int(max_repairs) <= 0 or float(max_distance_m) <= 0:
        return net, []

    nodes = {int(n["node_id"]): n for n in net.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    graph = weighted_graph(net)
    for e in net.edges:
        incident[int(e["u"])].append(e)
        incident[int(e["v"])].append(e)
    comp: Dict[int, int] = {}
    for cid, cc in enumerate(nx.connected_components(graph)):
        for nid in cc:
            comp[int(nid)] = int(cid)
    endpoints = [nid for nid, edges in incident.items() if len(edges) == 1]
    if len(endpoints) < 2:
        return net, []

    xy = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in endpoints], dtype=float)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(float(max_distance_m), output_type="set")

    edge_geoms = [e["geometry"] for e in net.edges]
    edge_ids = [int(e["edge_id"]) for e in net.edges]
    edge_tree = STRtree(edge_geoms) if edge_geoms else None

    raster_paths = {"support": support_path, "kde": kde_path, "distance": distance_path}
    raster_paths = {k: Path(v) for k, v in raster_paths.items() if v is not None and Path(v).exists()}
    sampler = RasterBlockSampler(raster_paths, cache_blocks=128) if raster_paths else None
    candidates: List[Dict[str, Any]] = []
    try:
        for i, j in pairs:
            a, b = int(endpoints[i]), int(endpoints[j])
            ca, cb = comp.get(a), comp.get(b)
            if ca is None or cb is None or ca == cb:
                continue
            pa, pb = xy[i], xy[j]
            d = float(np.linalg.norm(pb - pa))
            if d <= 0.35:
                continue
            va = _continuation_vector(incident[a][0], a)
            vb = _continuation_vector(incident[b][0], b)
            if va is None or vb is None:
                continue
            direction = (pb - pa) / d
            aa = _angle_deg(va, direction)
            ab = _angle_deg(vb, -direction)
            long_gap = d > float(long_gap_threshold_m)
            angle_limit = float(long_gap_max_angle_deg if long_gap else max_angle_deg)
            evidence_limit = float(long_gap_min_evidence_fraction if long_gap else min_evidence_fraction)
            if max(aa, ab) > angle_limit:
                continue
            geom = LineString([tuple(pa), tuple(pb)])
            ev = _sample_line_evidence(geom, sampler, evidence_distance_m)
            if float(ev["evidence_fraction"]) < evidence_limit:
                continue
            ignore = {int(incident[a][0]["edge_id"]), int(incident[b][0]["edge_id"])}
            if _crosses_unmodelled_existing_edge(geom, edge_geoms, edge_ids, edge_tree, ignore):
                continue
            confidence = float(np.clip(
                0.35 * (1.0 - max(aa, ab) / max(angle_limit, 1e-6))
                + 0.35 * float(ev["evidence_fraction"])
                + 0.20 * (1.0 - d / max(float(max_distance_m), 1e-6))
                + 0.10 * float(ev.get("support_fraction", 0.0)),
                0.0, 1.0,
            ))
            candidates.append({
                "a": a, "b": b,
                "component_a_before": int(ca), "component_b_before": int(cb),
                "distance_m": d,
                "angle_a_deg": float(aa), "angle_b_deg": float(ab),
                "long_gap": bool(long_gap),
                "geometry": geom,
                "confidence": confidence,
                **ev,
            })
    finally:
        if sampler is not None:
            sampler.close()

    candidates.sort(key=lambda r: (
        -float(r["confidence"]),
        -float(r["evidence_fraction"]),
        float(r["distance_m"]),
        max(float(r["angle_a_deg"]), float(r["angle_b_deg"])),
    ))
    uf = _ComponentUF(set(comp.values()))
    used_endpoints: Set[int] = set()
    accepted_geoms: List[LineString] = []
    rows: List[Dict[str, Any]] = []
    next_eid = max([int(e["edge_id"]) for e in net.edges] or [-1]) + 1
    for c in candidates:
        if len(rows) >= int(max_repairs):
            break
        a, b = int(c["a"]), int(c["b"])
        if a in used_endpoints or b in used_endpoints:
            continue
        ca, cb = int(c["component_a_before"]), int(c["component_b_before"])
        # Do not mutate connectivity state until the candidate has passed all
        # geometric conflict checks. A rejected crossing must not make later safe
        # alternatives look as though their components were already connected.
        if uf.find(ca) == uf.find(cb):
            continue
        geom = c["geometry"]
        conflict = False
        for old in accepted_geoms:
            inter = safe_intersection(
                geom, old, operation="component_stitch_connector_intersection",
                context={"endpoint_a": int(a), "endpoint_b": int(b)},
            )
            if inter is None:
                conflict = True
                break
            if not inter.is_empty:
                # Two accepted stitch connectors may share an endpoint but may not
                # cross or overlap in their interiors.
                p = inter.centroid
                da = min(float(geom.project(p)), float(geom.length - geom.project(p)))
                db = min(float(old.project(p)), float(old.length - old.project(p)))
                if da > 1e-6 and db > 1e-6:
                    conflict = True
                    break
        if conflict:
            continue
        if not uf.union(ca, cb):
            continue
        edge = _new_stitch_edge(next_eid, a, b, geom, c)
        net.edges.append(edge)
        used_endpoints.update((a, b))
        accepted_geoms.append(geom)
        row = {k: v for k, v in c.items() if k != "geometry"}
        row.update({"edge_id": int(next_eid), "mode": "component_stitch", "geometry": geom})
        rows.append(row)
        next_eid += 1

    if rows:
        net = reindex_network(net)
    print(f"    [component-stitch] candidates={len(candidates):,}, accepted={len(rows):,}", flush=True)
    return net, rows


def audit_component_stitching(
    network: NetworkData,
    max_distance_m: float,
    max_angle_deg: float,
    min_evidence_fraction: float,
    long_gap_threshold_m: float,
    long_gap_max_angle_deg: float,
    long_gap_min_evidence_fraction: float,
) -> List[Dict[str, Any]]:
    """Audit v0.5 component-stitch connectors against their approval envelope."""
    rows: List[Dict[str, Any]] = []
    for edge in network.edges:
        if not bool(edge.get("component_stitching_repair")):
            continue
        distance_value = edge.get("component_stitch_distance_m")
        angle_a_value = edge.get("component_stitch_angle_a_deg")
        angle_b_value = edge.get("component_stitch_angle_b_deg")
        evidence_value = edge.get("component_stitch_evidence_fraction")
        distance = float(edge.get("length_m", edge["geometry"].length) if distance_value is None else distance_value)
        # 0° is a valid perfect continuation angle; preserve numeric zero.
        angle_a = float(180.0 if angle_a_value is None else angle_a_value)
        angle_b = float(180.0 if angle_b_value is None else angle_b_value)
        evidence = float(0.0 if evidence_value is None else evidence_value)
        is_long = distance > float(long_gap_threshold_m)
        angle_limit = float(long_gap_max_angle_deg if is_long else max_angle_deg)
        evidence_limit = float(long_gap_min_evidence_fraction if is_long else min_evidence_fraction)
        reasons: List[str] = []
        if distance > float(max_distance_m) + 1e-9:
            reasons.append("distance")
        if max(angle_a, angle_b) > angle_limit + 1e-9:
            reasons.append("angle")
        if evidence + 1e-9 < evidence_limit:
            reasons.append("evidence")
        approved = bool(edge.get("approved_gap_repair", edge.get("gap_repair", False)))
        if not approved:
            reasons.append("not_approved")
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "distance_m": distance,
            "angle_a_deg": angle_a,
            "angle_b_deg": angle_b,
            "evidence_fraction": evidence,
            "long_gap": bool(is_long),
            "outside_baseline_fraction": 1.0 if reasons else 0.0,
            "audit_reason": "+".join(reasons) if reasons else "approved_component_stitch",
            "gap_repair": True,
            "component_stitching_repair": True,
            "geometry": edge["geometry"],
        })
    return rows
