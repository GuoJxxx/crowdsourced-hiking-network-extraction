#!/usr/bin/env python3
"""Normalize legacy map-construction outputs to one EPSG:26916 graph format."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


TARGET_CRS = "EPSG:26916"


def read_delimited(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        sep=r"\s*,\s*|\s+",
        engine="python",
        comment="#",
    )


def load_text_graph(vertices_path: Path, edges_path: Path, source_crs: str, swap_xy: bool = False):
    raw_vertices = read_delimited(vertices_path)
    if raw_vertices.shape[1] < 3:
        raise ValueError(f"Vertex file needs at least 3 columns: {vertices_path}")
    vertices = raw_vertices.iloc[:, :3].copy()
    vertices.columns = ["source_id", "x", "y"]
    raw_edges = read_delimited(edges_path)
    if raw_edges.shape[1] == 2:
        edges = raw_edges.copy()
        edges.columns = ["u_source", "v_source"]
        edges.insert(0, "source_edge_id", range(len(edges)))
    elif raw_edges.shape[1] >= 3:
        edges = raw_edges.iloc[:, :3].copy()
        edges.columns = ["source_edge_id", "u_source", "v_source"]
    else:
        raise ValueError(f"Edge file needs at least 2 columns: {edges_path}")
    vertices = vertices.dropna()
    if swap_xy:
        vertices[["x", "y"]] = vertices[["y", "x"]].to_numpy()
    edges = edges.dropna()
    return vertices, edges, source_crs


def load_sqlite_graph(database: Path, source_crs: str):
    with sqlite3.connect(database) as connection:
        vertices = pd.read_sql_query(
            "SELECT id AS source_id, longitude AS x, latitude AS y FROM nodes",
            connection,
        )
        edges = pd.read_sql_query(
            "SELECT id AS source_edge_id, in_node AS u_source, out_node AS v_source FROM edges",
            connection,
        )
    return vertices, edges, source_crs


def load_gpkg_graph(network: Path, layer: str):
    source = gpd.read_file(network, layer=layer)
    if source.crs is None:
        raise ValueError(f"Network layer has no CRS: {network}:{layer}")
    vertex_ids: dict[tuple[float, float], int] = {}
    vertex_rows = []
    edge_rows = []
    for geometry in source.geometry:
        if geometry is None or geometry.is_empty:
            continue
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            chain = []
            for coordinate in part.coords:
                key = (round(float(coordinate[0]), 6), round(float(coordinate[1]), 6))
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertex_ids)
                    vertex_rows.append((vertex_ids[key], key[0], key[1]))
                chain.append(vertex_ids[key])
            for u, v in zip(chain, chain[1:]):
                edge_rows.append((len(edge_rows), u, v))
    vertices = pd.DataFrame(vertex_rows, columns=["source_id", "x", "y"])
    edges = pd.DataFrame(edge_rows, columns=["source_edge_id", "u_source", "v_source"])
    return vertices, edges, str(source.crs)


def build_normalized_graph(
    vertices: pd.DataFrame,
    edges: pd.DataFrame,
    source_crs: str,
):
    vertices = vertices.copy()
    vertices["source_id"] = vertices["source_id"].astype(int)
    vertices = vertices.drop_duplicates("source_id", keep="first").sort_values("source_id")
    vertex_gdf = gpd.GeoDataFrame(
        vertices,
        geometry=[Point(x, y) for x, y in vertices[["x", "y"]].itertuples(index=False)],
        crs=source_crs,
    ).to_crs(TARGET_CRS)

    source_to_xy = {
        int(source_id): (float(point.x), float(point.y))
        for source_id, point in vertex_gdf[["source_id", "geometry"]].itertuples(index=False)
    }
    source_to_id = {
        source_id: index for index, source_id in enumerate(sorted(source_to_xy))
    }
    normalized_nodes = gpd.GeoDataFrame(
        [
            {
                "node_id": source_to_id[source_id],
                "source_id": source_id,
                "x": xy[0],
                "y": xy[1],
                "geometry": Point(xy),
            }
            for source_id, xy in source_to_xy.items()
        ],
        crs=TARGET_CRS,
    )

    seen: set[tuple[int, int]] = set()
    normalized_edges: list[dict[str, object]] = []
    invalid_references = 0
    self_loops = 0
    duplicate_directions = 0
    for source_edge_id, u_source, v_source in edges[
        ["source_edge_id", "u_source", "v_source"]
    ].itertuples(index=False):
        u_source = int(u_source)
        v_source = int(v_source)
        if u_source not in source_to_id or v_source not in source_to_id:
            invalid_references += 1
            continue
        u = source_to_id[u_source]
        v = source_to_id[v_source]
        if u == v:
            self_loops += 1
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            duplicate_directions += 1
            continue
        seen.add(key)
        line = LineString([source_to_xy[u_source], source_to_xy[v_source]])
        if line.length == 0:
            self_loops += 1
            continue
        normalized_edges.append(
            {
                "edge_id": len(normalized_edges),
                "source_edge_id": int(source_edge_id),
                "u": u,
                "v": v,
                "length_m": float(line.length),
                "geometry": line,
            }
        )
    edge_gdf = gpd.GeoDataFrame(normalized_edges, crs=TARGET_CRS)
    diagnostics = {
        "source_vertices": int(len(vertices)),
        "source_edges": int(len(edges)),
        "invalid_edge_references_removed": invalid_references,
        "self_loops_removed": self_loops,
        "reverse_or_duplicate_edges_collapsed": duplicate_directions,
    }
    return normalized_nodes, edge_gdf, diagnostics


def graph_metrics(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> dict[str, object]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(int))
    graph.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))
    active_nodes = {node for edge in graph.edges for node in edge}
    active_graph = graph.subgraph(active_nodes).copy()
    components = list(nx.connected_components(active_graph)) if active_nodes else []
    degrees = dict(active_graph.degree())
    return {
        "crs": TARGET_CRS,
        "nodes": int(len(nodes)),
        "active_nodes": int(len(active_nodes)),
        "edges": int(len(edges)),
        "length_m": float(edges.geometry.length.sum()) if len(edges) else 0.0,
        "connected_components": len(components),
        "largest_component_nodes": max(map(len, components), default=0),
        "endpoints": sum(degree == 1 for degree in degrees.values()),
        "intersections_degree_ge_3": sum(degree >= 3 for degree in degrees.values()),
        "cycle_rank": int(len(edges) - len(active_nodes) + len(components)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["text", "sqlite", "gpkg"], required=True)
    parser.add_argument("--vertices", type=Path)
    parser.add_argument("--edges", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--layer", default="edges")
    parser.add_argument("--source-crs")
    parser.add_argument(
        "--swap-xy",
        action="store_true",
        help="Swap the two coordinate columns (for legacy id,latitude,longitude files).",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.format == "text":
        if args.vertices is None or args.edges is None:
            parser.error("--vertices and --edges are required for text format")
        if args.source_crs is None:
            parser.error("--source-crs is required for text format")
        loaded = load_text_graph(args.vertices, args.edges, args.source_crs, args.swap_xy)
    elif args.format == "sqlite":
        if args.database is None:
            parser.error("--database is required for sqlite format")
        if args.source_crs is None:
            parser.error("--source-crs is required for sqlite format")
        loaded = load_sqlite_graph(args.database, args.source_crs)
    else:
        if args.network is None:
            parser.error("--network is required for gpkg format")
        loaded = load_gpkg_graph(args.network, args.layer)

    nodes, edges, diagnostics = build_normalized_graph(*loaded)
    args.output.mkdir(parents=True, exist_ok=True)
    gpkg = args.output / "road_network.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    edges.to_file(gpkg, layer="edges", driver="GPKG")
    nodes.drop(columns="geometry").to_csv(args.output / "nodes.csv", index=False)
    edge_table = edges.drop(columns="geometry").copy()
    edge_table["wkt"] = edges.geometry.to_wkt()
    edge_table.to_csv(args.output / "edges.csv", index=False)
    with (args.output / "benchmark_vertices.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for node_id, x, y in nodes[["node_id", "x", "y"]].itertuples(index=False):
            handle.write(f"{int(node_id)},{float(x):.6f},{float(y):.6f}\n")
    with (args.output / "benchmark_edges.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for edge_id, u, v in edges[["edge_id", "u", "v"]].itertuples(index=False):
            handle.write(f"{int(edge_id)},{int(u)},{int(v)},1\n")

    result = {
        "method": args.method,
        "source_crs": loaded[2],
        "normalization": {
            "target_crs": TARGET_CRS,
            "direction_handling": "collapse exact reverse/duplicate endpoint pairs for geometric evaluation",
            **diagnostics,
        },
        "graph": graph_metrics(nodes, edges),
        "outputs": {
            "gpkg": str(gpkg.resolve()),
            "nodes_csv": str((args.output / "nodes.csv").resolve()),
            "edges_csv": str((args.output / "edges.csv").resolve()),
            "benchmark_vertices": str((args.output / "benchmark_vertices.txt").resolve()),
            "benchmark_edges": str((args.output / "benchmark_edges.txt").resolve()),
        },
    }
    (args.output / "graph_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
