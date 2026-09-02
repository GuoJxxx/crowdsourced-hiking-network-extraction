from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import RawTrackWitnessIndex, prune_raw_negative_connectors
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles
from pipeline_modules.network_cleanup.junction_cleanup import (
    repair_endpoint_clusters,
    repair_endpoint_to_edge_junctions,
)


def make_network(node_xy, edge_specs, crs="EPSG:3857"):
    nodes = [
        {"node_id": i, "x": float(x), "y": float(y), "geometry": Point(float(x), float(y))}
        for i, (x, y) in enumerate(node_xy)
    ]
    edges = []
    for eid, (u, v, coords, attrs) in enumerate(edge_specs):
        geom = LineString(coords)
        row = {
            "edge_id": eid,
            "u": int(u),
            "v": int(v),
            "geometry": geom,
            "length_m": float(geom.length),
            "source_type": "primary",
            "support_mean": 1.0,
            "support_ge2_fraction": 0.5,
            "confidence": 0.8,
        }
        row.update(attrs)
        edges.append(row)
    return reindex_network(NetworkData(nodes, edges, crs, None, 0, 0, {}))


def meters_to_lonlat(x, y):
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return t.transform(float(x), float(y))


def write_raw_track(folder: Path, name: str, xy):
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["track_id,source_file,segment_id,point_index,raw_point_index,longitude,latitude,time_utc"]
    for i, (x, y) in enumerate(xy, start=1):
        lon, lat = meters_to_lonlat(x, y)
        lines.append(
            f"{name},src_{name}.kml,1,{i},{i},{lon:.12f},{lat:.12f},2026-01-01T00:00:{(i-1)*5:02d}Z"
        )
    (folder / f"{name}__gxtrack.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def graph_of(net):
    return nx.Graph((int(e["u"]), int(e["v"])) for e in net.edges)


def test_wide_single_corridor_cycle_collapses_without_requiring_synthetic_edge(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "upper", [(-5, 0), (0, 0), (5, 10), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-5, 0), (0, 0), (5, 10), (5, -10), (10, 0), (15, 0)],
        [
            (0, 1, [(-5, 0), (0, 0)], {}),
            (1, 2, [(0, 0), (5, 10)], {}),
            (2, 4, [(5, 10), (10, 0)], {}),
            (1, 3, [(0, 0), (5, -10)], {}),
            (3, 4, [(5, -10), (10, 0)], {}),
            (4, 5, [(10, 0), (15, 0)], {}),
        ],
    )
    cleaned, rows = resolve_single_corridor_cycles(
        net, idx, raw_corridor_m=2.0, min_route_coverage=0.65,
        min_independent_tracks=1, max_rounds=3,
    )
    assert any(r["action"] == "collapse_single_corridor_false_cycle" for r in rows)
    assert nx.cycle_basis(graph_of(cleaned)) == []
    assert nx.is_connected(graph_of(cleaned))
    assert any(max(y for _, y in e["geometry"].coords) >= 9.5 for e in cleaned.edges)
    assert not any(min(y for _, y in e["geometry"].coords) <= -9.5 for e in cleaned.edges)


def test_multigate_cycle_is_opened_when_only_one_route_family_has_raw_support(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "main", [(-10, 0), (0, 0), (5, 4), (10, 0), (20, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    # Nodes 1 and 4 are corridor gates; node 3 also has an external spur, so the
    # visible loop has THREE external gates and current release skipped it entirely.
    net = make_network(
        [(-10, 0), (0, 0), (5, 4), (5, -4), (10, 0), (20, 0), (5, -12)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (1, 2, [(0, 0), (5, 4)], {}),
            (2, 4, [(5, 4), (10, 0)], {}),
            (1, 3, [(0, 0), (5, -4)], {}),
            (3, 4, [(5, -4), (10, 0)], {}),
            (4, 5, [(10, 0), (20, 0)], {}),
            (3, 6, [(5, -4), (5, -12)], {}),
        ],
    )
    cleaned, rows = resolve_single_corridor_cycles(
        net, idx, raw_corridor_m=2.0, min_route_coverage=0.60,
        min_independent_tracks=1, max_rounds=4,
    )
    assert nx.cycle_basis(graph_of(cleaned)) == []
    # The side spur must survive even though its attachment node lies on the weak loop arm.
    assert any(6 in (int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert any(r["action"].startswith("open_multigate_false_cycle") for r in rows)


def test_endpoint_to_edge_raw_negative_evidence_vetoes_false_t_connection(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "approach", [(-10, 0), (-5, 0), (0, 0)])
    write_raw_track(raw, "through", [(5, -10), (5, -5), (5, 0), (5, 5), (5, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-10, 0), (0, 0), (5, -10), (5, 10)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(5, -10), (5, 10)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=8.0,
        node_merge_m=1.0,
        max_approach_angle_deg=20.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_witness_endpoint_snap_m=4.0,
        raw_witness_max_path_m=20.0,
        raw_witness_max_path_ratio=2.0,
        raw_witness_corridor_m=3.0,
    )
    assert rows == []
    assert len(repaired.edges) == 2


def test_endpoint_cluster_requires_raw_transition_for_cycle_closing_arm(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "left", [(-20, 0), (-10, 0), (-3, 0)])
    write_raw_track(raw, "right", [(20, 0), (10, 0), (3, 0)])
    write_raw_track(raw, "top", [(0, 20), (0, 10), (0, 3)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-20, 0), (-3, 0), (20, 0), (3, 0), (0, 20), (0, 3)],
        [
            (0, 1, [(-20, 0), (-3, 0)], {}),
            (2, 3, [(20, 0), (3, 0)], {}),
            (4, 5, [(0, 20), (0, 3)], {}),
            (0, 4, [(-20, 0), (0, 20)], {}),
            (4, 2, [(0, 20), (20, 0)], {}),
        ],
    )
    repaired, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=15.0,
        min_evidence_fraction=0.0,
        min_cluster_size=3,
        max_cluster_size=6,
        allow_cycle_closure=True,
        cycle_min_detour_ratio=2.0,
        cycle_min_detour_excess_m=10.0,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_witness_endpoint_snap_m=4.0,
        raw_witness_max_path_m=20.0,
        raw_witness_max_path_ratio=2.0,
        raw_witness_corridor_m=3.0,
    )
    # No original trajectory transitions between these three nearby arms, so the
    # cluster repair is not allowed to manufacture a loop/junction among them.
    assert rows == []
    assert len(repaired.edges) == 5




def test_three_parallel_route_geometries_collapse_to_single_raw_supported_corridor(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "middle", [(-5, 0), (0, 0), (5, 0), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-5, 0), (0, 0), (10, 0), (15, 0)],
        [
            (0, 1, [(-5, 0), (0, 0)], {}),
            (1, 2, [(0, 0), (5, 4), (10, 0)], {}),
            (1, 2, [(0, 0), (5, 0), (10, 0)], {}),
            (1, 2, [(0, 0), (5, -4), (10, 0)], {}),
            (2, 3, [(10, 0), (15, 0)], {}),
        ],
    )
    cleaned, rows = resolve_single_corridor_cycles(
        net, idx, raw_corridor_m=1.5, min_route_coverage=0.70,
        min_independent_tracks=1, max_rounds=4,
    )
    internal = [e for e in cleaned.edges if {int(e["u"]), int(e["v"])} == {1, 2}]
    assert len(internal) == 1
    assert max(abs(y) for _, y in internal[0]["geometry"].coords) < 0.5
    assert any(r.get("action") == "collapse_parallel_route_family" for r in rows)


def test_post_prune_removes_any_synthetic_connector_with_local_raw_negative_evidence(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "a", [(-10, 0), (-5, 0), (0, 0)])
    write_raw_track(raw, "b", [(6, 3), (10, 6), (14, 9)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-10, 0), (0, 0), (6, 3), (14, 9)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (1, 2, [(0, 0), (6, 3)], {
                "source_type": "s7_component_stitching",
                "component_stitching_repair": True,
                "gap_repair": True,
            }),
            (2, 3, [(6, 3), (14, 9)], {}),
        ],
    )
    cleaned, rows = prune_raw_negative_connectors(
        net, idx, max_connector_m=20.0, raw_witness_min_tracks=1,
        raw_side_min_tracks=1, raw_endpoint_snap_m=4.0,
        raw_max_path_ratio=2.0, raw_corridor_m=3.0,
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "remove_raw_negative_false_connector"
    assert len(cleaned.edges) == 2


def test_one_click_ui_uses_version_neutral_title():
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8-sig"))
    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    runner = (ROOT / "code" / "tools" / "run_pipeline_configurable.ps1").read_text(encoding="utf-8-sig")
    cmd1 = (ROOT / "run_S1_to_S7_windows.cmd").read_text(encoding="utf-8-sig")
    cmd2 = (ROOT / "run_S2_to_S7_windows.cmd").read_text(encoding="utf-8-sig")
    assert str(cfg["version"]) == "0.5"
    assert str(schema["version"]) == "0.5"
    assert "Activity-Aware Pipeline" in runner
    assert "S7_v0.5_smooth_network" in runner
    assert "v0.5" not in cmd1
    assert "v0.5" not in cmd2
