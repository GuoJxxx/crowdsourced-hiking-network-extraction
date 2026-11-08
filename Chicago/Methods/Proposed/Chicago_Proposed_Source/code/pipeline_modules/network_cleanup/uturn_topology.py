from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from .models import NetworkData, reindex_network
from .raw_track_topology import (
    RawTrackWitnessIndex,
    _axis_angle_deg,
    _synthetic_connector,
    edge_axis_at_node,
)


def _uturn_protected(edge: Dict[str, Any]) -> bool:
    return bool(
        edge.get("uturn_core_protected")
        or edge.get("uturn_context_protected")
        or edge.get("uturn_protected")
    )


def _choose_axis(edges: List[Dict[str, Any]], node_id: int):
    axes = [edge_axis_at_node(e, node_id) for e in edges]
    axes = [a for a in axes if a is not None]
    return max(axes, key=lambda a: float(np.linalg.norm(a))) if axes else None


def prune_uturn_shortcut_connectors(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    max_connector_m: float = 35.0,
    parallel_axis_max_angle_deg: float = 25.0,
    transverse_min_angle_deg: float = 45.0,
    strict_endpoint_snap_m: float = 4.0,
    strict_corridor_m: float = 3.0,
    strict_max_path_ratio: float = 1.8,
    min_witness_tracks: int = 1,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove synthetic shortcuts across the two arms of a U-turn/hairpin.

    A candidate is removable only when it is synthetic, both endpoint continuations
    belong to a protected U-turn context, the edge lies on a cycle (there is already
    an alternate U-shaped path), and original tracks do not witness the *direct*
    cross-arm transition.  A synthetic cap that is required to keep the U connected
    is therefore preserved.
    """
    idx = witness_index
    net = reindex_network(network.copy())
    if idx is None or not idx.enabled:
        return net, []
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    graph = nx.Graph()
    for e in net.edges:
        u, v = int(e["u"]), int(e["v"])
        incident[u].append(e); incident[v].append(e)
        if u != v:
            graph.add_edge(u, v)

    drop: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    for edge in net.edges:
        if not _synthetic_connector(edge):
            continue
        geom = edge.get("geometry")
        if not isinstance(geom, LineString) or geom.length <= 0 or geom.length > float(max_connector_m):
            continue
        u, v = int(edge["u"]), int(edge["v"])
        left = [e for e in incident[u] if int(e["edge_id"]) != int(edge["edge_id"])]
        right = [e for e in incident[v] if int(e["edge_id"]) != int(edge["edge_id"])]
        left_core = [e for e in left if _uturn_protected(e)]
        right_core = [e for e in right if _uturn_protected(e)]
        if not left_core or not right_core:
            continue

        # Only remove a shortcut/redundant branch, never the sole edge completing the
        # real hairpin. Test alternate connectivity with this edge absent.
        g2 = graph.copy()
        if g2.has_edge(u, v):
            g2.remove_edge(u, v)
        if u not in g2 or v not in g2 or not nx.has_path(g2, u, v):
            continue

        a = _choose_axis(left_core, u); b = _choose_axis(right_core, v)
        c = np.asarray(geom.coords[-1], dtype=float) - np.asarray(geom.coords[0], dtype=float)
        if a is None or b is None or float(np.linalg.norm(c)) <= 1e-9:
            continue
        parallel = _axis_angle_deg(a, b)
        cross_a = _axis_angle_deg(a, c); cross_b = _axis_angle_deg(b, c)
        if not (
            parallel <= float(parallel_axis_max_angle_deg)
            and cross_a >= float(transverse_min_angle_deg)
            and cross_b >= float(transverse_min_angle_deg)
        ):
            continue

        left_tracks = set()
        right_tracks = set()
        for e in left_core:
            left_tracks.update(idx.tracks_near_geometry(e["geometry"], float(strict_endpoint_snap_m)))
        for e in right_core:
            right_tracks.update(idx.tracks_near_geometry(e["geometry"], float(strict_endpoint_snap_m)))
        if not left_tracks or not right_tracks:
            continue

        witness = idx.direct_transition_witness(
            geom,
            endpoint_snap_m=float(strict_endpoint_snap_m),
            max_path_m=max(float(max_connector_m), float(geom.length) * float(strict_max_path_ratio)),
            max_path_ratio=float(strict_max_path_ratio),
            corridor_m=float(strict_corridor_m),
        )
        if int(witness.get("witness_tracks", 0)) >= int(min_witness_tracks):
            continue
        drop.add(int(edge["edge_id"]))
        rows.append({
            "edge_id": int(edge["edge_id"]), "action": "remove_uturn_arm_shortcut",
            "source_type": str(edge.get("source_type", "")), "length_m": float(geom.length),
            "parallel_axis_angle_deg": float(parallel),
            "source_cross_angle_deg": float(cross_a), "target_cross_angle_deg": float(cross_b),
            "raw_direct_witness_tracks": int(witness.get("witness_tracks", 0)),
            "raw_side_a_tracks": int(len(left_tracks)), "raw_side_b_tracks": int(len(right_tracks)),
            "geometry": geom,
        })
    if not drop:
        return net, []
    net.edges = [e for e in net.edges if int(e["edge_id"]) not in drop]
    print(f"    [uturn-shortcut:v0.5] removed={len(rows):,}", flush=True)
    return reindex_network(net), rows
