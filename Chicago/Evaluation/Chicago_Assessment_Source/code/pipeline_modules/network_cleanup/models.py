from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from shapely.geometry import LineString, Point


@dataclass
class NetworkData:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    crs: Any
    transform: Any
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "NetworkData":
        return NetworkData(
            nodes=[dict(n) for n in self.nodes],
            edges=[dict(e) for e in self.edges],
            crs=self.crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            metadata=dict(self.metadata),
        )


def used_node_ids(edges: Iterable[Dict[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for e in edges:
        out.add(int(e["u"]))
        out.add(int(e["v"]))
    return out


def edge_adjacency(edges: Iterable[Dict[str, Any]]) -> Dict[int, List[int]]:
    inc: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        eid = int(e["edge_id"])
        inc[int(e["u"])].append(eid)
        inc[int(e["v"])].append(eid)
    return dict(inc)


def compute_degrees(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[int, int]:
    degree: Dict[int, int] = defaultdict(int)
    for e in edges:
        u, v = int(e["u"]), int(e["v"])
        if u == v:
            degree[u] += 2
        else:
            degree[u] += 1
            degree[v] += 1
    for n in nodes:
        nid = int(n["node_id"])
        d = int(degree.get(nid, 0))
        n["degree"] = d
        n["node_type"] = (
            "isolated" if d == 0 else
            "endpoint" if d == 1 else
            "chain" if d == 2 else
            "intersection"
        )
    return dict(degree)


def make_nx_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    used = used_node_ids(edges)
    for n in nodes:
        nid = int(n["node_id"])
        if nid in used:
            graph.add_node(nid, x=float(n["x"]), y=float(n["y"]))
    for e in edges:
        u, v = int(e["u"]), int(e["v"])
        length = float(e.get("length_m", e["geometry"].length))
        if graph.has_edge(u, v):
            if length < float(graph[u][v].get("weight", float("inf"))):
                graph[u][v].update(weight=length, edge_id=int(e["edge_id"]))
        else:
            graph.add_edge(u, v, weight=length, edge_id=int(e["edge_id"]))
    return graph


def component_ids(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[int, int]:
    graph = make_nx_graph(nodes, edges)
    result: Dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(graph), 1):
        for nid in comp:
            result[int(nid)] = cid
    return result


def orient_coords(edge: Dict[str, Any], start_node: int, end_node: int) -> List[Tuple[float, float]]:
    coords = list(edge["geometry"].coords)
    if int(edge["u"]) == int(start_node) and int(edge["v"]) == int(end_node):
        return [(float(x), float(y)) for x, y in coords]
    if int(edge["v"]) == int(start_node) and int(edge["u"]) == int(end_node):
        return [(float(x), float(y)) for x, y in reversed(coords)]
    raise ValueError("edge does not connect requested nodes")


def clean_linestring(coords: Iterable[Tuple[float, float]]) -> Optional[LineString]:
    clean: List[Tuple[float, float]] = []
    for x, y in coords:
        p = (float(x), float(y))
        if not clean or p != clean[-1]:
            clean.append(p)
    if len(clean) < 2:
        return None
    geom = LineString(clean)
    return geom if geom.length > 0 else None


def reindex_network(network: NetworkData) -> NetworkData:
    used = used_node_ids(network.edges)
    old_nodes = {int(n["node_id"]): dict(n) for n in network.nodes if int(n["node_id"]) in used}
    mapping = {old: new for new, old in enumerate(sorted(old_nodes))}

    nodes: List[Dict[str, Any]] = []
    for old in sorted(old_nodes):
        n = old_nodes[old]
        n["old_node_id"] = old
        n["node_id"] = mapping[old]
        if "geometry" not in n:
            n["geometry"] = Point(float(n["x"]), float(n["y"]))
        nodes.append(n)

    edges: List[Dict[str, Any]] = []
    for new_eid, old_edge in enumerate(network.edges):
        e = dict(old_edge)
        e["old_edge_id"] = int(e.get("edge_id", new_eid))
        e["edge_id"] = new_eid
        e["u"] = mapping[int(e["u"])]
        e["v"] = mapping[int(e["v"])]
        e["length_m"] = float(e["geometry"].length)
        edges.append(e)

    compute_degrees(nodes, edges)
    return NetworkData(nodes, edges, network.crs, network.transform, network.width, network.height, dict(network.metadata))
