from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .models import NetworkData, compute_degrees, make_nx_graph


def _clean_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def network_frames(network: NetworkData) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    compute_degrees(network.nodes, network.edges)
    node_rows = []
    for n in network.nodes:
        row = {k: _clean_value(v) for k, v in n.items() if k != "geometry"}
        row["geometry"] = n.get("geometry", Point(float(n["x"]), float(n["y"])))
        node_rows.append(row)
    edge_rows = []
    for e in network.edges:
        row = {k: _clean_value(v) for k, v in e.items() if k not in ("geometry", "_pixels")}
        row["geometry"] = e["geometry"]
        edge_rows.append(row)
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=network.crs)
    edges = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=network.crs)
    intersections = nodes[nodes["degree"] >= 3].copy() if len(nodes) else nodes.copy()
    return nodes, edges, intersections


def write_network_gpkg(network: NetworkData, path: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    nodes, edges, intersections = network_frames(network)
    if len(nodes):
        nodes.to_file(path, layer="nodes", driver="GPKG")
    if len(edges):
        edges.to_file(path, layer="edges", driver="GPKG")
    if len(intersections):
        intersections.to_file(path, layer="intersections", driver="GPKG")
    return nodes, edges


def write_diagnostic_gpkg(path: Path, layers: Dict[str, Tuple[Iterable[Dict[str, Any]], Any]]) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
    written = False
    for layer_name, (records, crs) in layers.items():
        rows = list(records)
        if not rows:
            continue
        clean = []
        for item in rows:
            row = {k: _clean_value(v) for k, v in item.items() if k not in ("_pixels",)}
            clean.append(row)
        if "geometry" not in clean[0]:
            continue
        gdf = gpd.GeoDataFrame(clean, geometry="geometry", crs=crs)
        gdf.to_file(path, layer=layer_name[:63], driver="GPKG", mode="a" if written else "w")
        written = True


def write_sqlite(network: NetworkData, path: Path) -> None:
    """Write routable core tables plus complete JSON attributes for every feature."""
    path = Path(path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("""CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            x REAL,
            y REAL,
            elevation_m REAL,
            degree INTEGER,
            node_type TEXT,
            uturn_protected INTEGER,
            activity_zone INTEGER,
            attributes_json TEXT
        )""")
        cur.execute("""CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            u INTEGER,
            v INTEGER,
            length_m REAL,
            length_3d_m REAL,
            source_type TEXT,
            confidence REAL,
            support_mean REAL,
            support_max REAL,
            kde_mean REAL,
            kde_max REAL,
            distinct_tracks INTEGER,
            coverage_ratio REAL,
            mean_match_dist_m REAL,
            grade_mean_pct REAL,
            grade_max_abs_pct REAL,
            energy_before REAL,
            energy_after REAL,
            energy_improvement REAL,
            uturn_protected INTEGER,
            activity_zone INTEGER,
            gap_repair INTEGER,
            retained_reason TEXT,
            geometry_wkt TEXT,
            attributes_json TEXT
        )""")
        cur.execute("CREATE INDEX idx_edges_u ON edges(u)")
        cur.execute("CREATE INDEX idx_edges_v ON edges(v)")
        cur.execute("CREATE TABLE intersections (node_id INTEGER PRIMARY KEY)")
        cur.execute("CREATE TABLE adjacency (node_id INTEGER, edge_id INTEGER, neighbor_node_id INTEGER, length_m REAL)")
        cur.execute("CREATE INDEX idx_adj_node ON adjacency(node_id)")

        compute_degrees(network.nodes, network.edges)
        for key, value in {
            "crs": str(network.crs),
            "width": network.width,
            "height": network.height,
            **network.metadata,
        }.items():
            cur.execute(
                "INSERT INTO metadata VALUES (?,?)",
                (str(key), json.dumps(value, ensure_ascii=False, default=str)),
            )
        for n in network.nodes:
            attributes = {
                k: _clean_value(v)
                for k, v in n.items()
                if k != "geometry"
            }
            cur.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)", (
                int(n["node_id"]),
                float(n["x"]),
                float(n["y"]),
                float(n.get("elevation_m", 0) or 0) if pd.notna(n.get("elevation_m", None)) else None,
                int(n.get("degree", 0)),
                str(n.get("node_type", "")),
                int(bool(n.get("uturn_protected"))),
                int(bool(n.get("activity_zone"))),
                json.dumps(attributes, ensure_ascii=False, default=str),
            ))
            if int(n.get("degree", 0)) >= 3:
                cur.execute("INSERT INTO intersections VALUES (?)", (int(n["node_id"]),))
        for e in network.edges:
            attributes = {
                k: _clean_value(v)
                for k, v in e.items()
                if k not in ("geometry", "_pixels")
            }
            def nullable_float(key: str):
                value = e.get(key, None)
                return float(value) if value is not None and pd.notna(value) else None
            cur.execute("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                int(e["edge_id"]), int(e["u"]), int(e["v"]), float(e["length_m"]),
                nullable_float("length_3d_m"), str(e.get("source_type", "")),
                float(e.get("confidence", 0) or 0), float(e.get("support_mean", 0) or 0),
                float(e.get("support_max", 0) or 0), float(e.get("kde_mean", 0) or 0),
                float(e.get("kde_max", 0) or 0), int(e.get("distinct_tracks", 0) or 0),
                float(e.get("coverage_ratio", 0) or 0), nullable_float("mean_match_dist_m"),
                nullable_float("grade_mean_pct"), nullable_float("grade_max_abs_pct"),
                nullable_float("energy_before"), nullable_float("energy_after"),
                nullable_float("energy_improvement"), int(bool(e.get("uturn_protected"))),
                int(bool(e.get("activity_zone"))), int(bool(e.get("gap_repair"))),
                str(e.get("retained_reason", "")), e["geometry"].wkt,
                json.dumps(attributes, ensure_ascii=False, default=str),
            ))
            cur.execute("INSERT INTO adjacency VALUES (?,?,?,?)", (
                int(e["u"]), int(e["edge_id"]), int(e["v"]), float(e["length_m"])
            ))
            cur.execute("INSERT INTO adjacency VALUES (?,?,?,?)", (
                int(e["v"]), int(e["edge_id"]), int(e["u"]), float(e["length_m"])
            ))
        conn.commit()
    finally:
        conn.close()


def write_network_bundle(network: NetworkData, output_dir: Path, basename: str = "road_network") -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / f"{basename}.gpkg"
    db = output_dir / f"{basename}.sqlite"
    print(f"    [output] build frames nodes={len(network.nodes):,}, edges={len(network.edges):,}", flush=True)
    print(f"    [output] write GPKG: {gpkg.name}", flush=True)
    nodes, edges = write_network_gpkg(network, gpkg)
    print(f"    [output] write SQLite: {db.name}", flush=True)
    write_sqlite(network, db)
    print("    [output] write CSV tables", flush=True)
    nodes.drop(columns="geometry").to_csv(output_dir / f"{basename}_nodes.csv", index=False, encoding="utf-8-sig")
    edges.drop(columns="geometry").to_csv(output_dir / f"{basename}_edges.csv", index=False, encoding="utf-8-sig")
    print("    [output] bundle complete", flush=True)
    return {
        "gpkg": str(gpkg),
        "sqlite": str(db),
        "nodes_csv": str(output_dir / f"{basename}_nodes.csv"),
        "edges_csv": str(output_dir / f"{basename}_edges.csv"),
    }


def network_metrics(network: NetworkData) -> Dict[str, Any]:
    graph = make_nx_graph(network.nodes, network.edges)
    return {
        "nodes": int(len(network.nodes)),
        "edges": int(len(network.edges)),
        "length_m": float(sum(float(e["length_m"]) for e in network.edges)),
        "components": int(__import__("networkx").number_connected_components(graph)) if graph.number_of_nodes() else 0,
        "endpoints": int(sum(1 for n in network.nodes if int(n.get("degree", 0)) == 1)),
        "intersections": int(sum(1 for n in network.nodes if int(n.get("degree", 0)) >= 3)),
    }
