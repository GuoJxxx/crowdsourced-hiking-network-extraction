from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import networkx as nx
import numpy as np
from shapely import set_precision

from .models import NetworkData


_GEOMETRY_NUMERIC_ANOMALIES: List[Dict[str, Any]] = []
_GEOMETRY_NUMERIC_ANOMALY_LIMIT = 10000


def clear_geometry_numeric_anomalies() -> None:
    """Reset per-run Shapely numeric diagnostics."""
    _GEOMETRY_NUMERIC_ANOMALIES.clear()


def geometry_numeric_anomalies() -> List[Dict[str, Any]]:
    """Return a copy of numeric geometry anomalies captured by safe operations."""
    return [dict(row) for row in _GEOMETRY_NUMERIC_ANOMALIES]


def _geometry_has_finite_coordinates(geom) -> bool:
    if geom is None:
        return False
    try:
        if geom.is_empty:
            return True
    except Exception:
        return False

    gtype = str(getattr(geom, "geom_type", ""))
    try:
        if gtype == "Polygon":
            if not _geometry_has_finite_coordinates(geom.exterior):
                return False
            return all(_geometry_has_finite_coordinates(ring) for ring in geom.interiors)
        if hasattr(geom, "geoms") and gtype not in {"Point", "LineString", "LinearRing"}:
            return all(_geometry_has_finite_coordinates(part) for part in geom.geoms)
        coords = np.asarray(geom.coords, dtype=float)
    except Exception:
        return False
    if coords.size == 0:
        return True
    if coords.ndim != 2 or coords.shape[1] < 2:
        return False
    return bool(np.isfinite(coords[:, :2]).all())


def _record_geometry_numeric_anomaly(
    operation: str, reason: str, left, right, context: Optional[Dict[str, Any]] = None,
) -> None:
    if len(_GEOMETRY_NUMERIC_ANOMALIES) >= _GEOMETRY_NUMERIC_ANOMALY_LIMIT:
        return
    row: Dict[str, Any] = {
        "operation": str(operation or "intersection"),
        "reason": str(reason),
        "action": "reject_candidate",
        "left_geom_type": str(getattr(left, "geom_type", type(left).__name__)),
        "right_geom_type": str(getattr(right, "geom_type", type(right).__name__)),
        "left_finite_coords": bool(_geometry_has_finite_coordinates(left)),
        "right_finite_coords": bool(_geometry_has_finite_coordinates(right)),
    }
    if context:
        for key, value in context.items():
            if key not in row:
                row[str(key)] = value
    _GEOMETRY_NUMERIC_ANOMALIES.append(row)


def safe_intersection(left, right, operation: str = "intersection", context: Optional[Dict[str, Any]] = None):
    """Run Shapely intersection with precision-retry recovery.

    v0.5 no longer equates a GEOS RuntimeWarning with invalid road geometry.  Finite
    inputs are retried on small precision grids; only genuinely unrecoverable
    operations fail closed.  This preserves candidate connectors that were being
    rejected in v0.5 despite finite coordinates.
    """
    left_finite = _geometry_has_finite_coordinates(left)
    right_finite = _geometry_has_finite_coordinates(right)
    if not left_finite or not right_finite:
        parts = []
        if not left_finite:
            parts.append("left_non_finite_coordinates")
        if not right_finite:
            parts.append("right_non_finite_coordinates")
        _record_geometry_numeric_anomaly(operation, "+".join(parts), left, right, context)
        return None

    attempts = [(left, right, "native")]
    for grid in (1e-6, 1e-4, 1e-3):
        try:
            attempts.append((set_precision(left, grid), set_precision(right, grid), f"precision_{grid:g}"))
        except Exception:
            pass

    last_reason = "intersection_failed"
    for attempt_left, attempt_right, label_name in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                result = attempt_left.intersection(attempt_right)
        except Exception as exc:
            last_reason = f"{label_name}_exception:{type(exc).__name__}:{exc}"
            continue
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        if runtime_warnings:
            last_reason = f"{label_name}_runtime_warning:{runtime_warnings[0].message}"
            continue
        if not _geometry_has_finite_coordinates(result):
            last_reason = f"{label_name}_result_non_finite_coordinates"
            continue
        if label_name != "native":
            _record_geometry_numeric_anomaly(operation, f"recovered_by_{label_name}", left, right, {**(context or {}), "action": "retry_recovered"})
            if _GEOMETRY_NUMERIC_ANOMALIES:
                _GEOMETRY_NUMERIC_ANOMALIES[-1]["action"] = "retry_recovered"
        return result

    _record_geometry_numeric_anomaly(operation, last_reason, left, right, context)
    return None


def weighted_graph(network: NetworkData) -> nx.Graph:
    """Return a simple weighted graph using the shortest parallel edge per node pair."""
    graph = nx.Graph()
    for node in network.nodes:
        graph.add_node(int(node["node_id"]))
    for edge in network.edges:
        u, v = int(edge["u"]), int(edge["v"])
        length = float(edge.get("length_m", edge["geometry"].length))
        if graph.has_edge(u, v):
            if length < float(graph[u][v].get("weight", float("inf"))):
                graph[u][v]["weight"] = length
        else:
            graph.add_edge(u, v, weight=length)
    return graph


def shortest_path_length(graph: nx.Graph, a: int, b: int) -> float:
    if int(a) == int(b):
        return 0.0
    try:
        return float(nx.shortest_path_length(graph, int(a), int(b), weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def cycle_closure_allowed(
    graph: nx.Graph,
    a: int,
    b: int,
    candidate_length_m: float,
    min_detour_ratio: float = 1.8,
    min_detour_excess_m: float = 8.0,
) -> Tuple[bool, float]:
    """Allow a same-component connector only when it closes a meaningful missing route.

    A real open road loop has a long existing detour between the two broken ends.
    A duplicate shortcut has an existing route nearly as short as the new connector.
    """
    candidate = max(float(candidate_length_m), 1e-6)
    detour = shortest_path_length(graph, int(a), int(b))
    if not (detour < float("inf")):
        return False, detour
    ratio_ok = detour >= candidate * max(1.0, float(min_detour_ratio))
    excess_ok = detour - candidate >= max(0.0, float(min_detour_excess_m))
    return bool(ratio_ok and excess_ok), float(detour)


def endpoint_to_edge_cycle_closure_allowed(
    graph: nx.Graph,
    endpoint: int,
    target_u: int,
    target_v: int,
    projection_distance_from_u_m: float,
    target_length_m: float,
    connector_length_m: float,
    min_detour_ratio: float = 1.8,
    min_detour_excess_m: float = 8.0,
) -> Tuple[bool, float]:
    """Evaluate loop closure to an interior point on an existing target edge."""
    du = shortest_path_length(graph, endpoint, target_u)
    dv = shortest_path_length(graph, endpoint, target_v)
    along_u = max(0.0, float(projection_distance_from_u_m))
    along_v = max(0.0, float(target_length_m) - along_u)
    options = []
    if du < float("inf"):
        options.append(du + along_u)
    if dv < float("inf"):
        options.append(dv + along_v)
    detour = min(options) if options else float("inf")
    if not (detour < float("inf")):
        return False, detour
    candidate = max(float(connector_length_m), 1e-6)
    ratio_ok = detour >= candidate * max(1.0, float(min_detour_ratio))
    excess_ok = detour - candidate >= max(0.0, float(min_detour_excess_m))
    return bool(ratio_ok and excess_ok), float(detour)
