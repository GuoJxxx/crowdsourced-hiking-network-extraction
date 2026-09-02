from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.junction_cleanup import repair_endpoint_to_edge_junctions
from pipeline_modules.network_cleanup.raw_track_topology import (
    RawTrackWitnessIndex,
    prune_parallel_false_bridges,
    repair_raw_track_gaps,
    audit_raw_track_gaps,
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


def write_raw_track(folder: Path, name: str, xy, times=None):
    folder.mkdir(parents=True, exist_ok=True)
    if times is None:
        times = [f"2026-01-01T00:00:{i*5:02d}Z" for i in range(len(xy))]
    lines = [
        "track_id,source_file,segment_id,point_index,raw_point_index,longitude,latitude,time_utc"
    ]
    for i, ((x, y), ts) in enumerate(zip(xy, times), start=1):
        lon, lat = meters_to_lonlat(x, y)
        lines.append(f"{name},src.kml,1,{i},{i},{lon:.12f},{lat:.12f},{ts}")
    path = folder / f"{name}__gxtrack.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def test_raw_witness_bridges_one_point_gps_spike_without_using_cleaned_track(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "t1", [(0, 0), (5, 0), (500, 300), (10, 0), (15, 0), (20, 0)])
    idx, report = RawTrackWitnessIndex.from_observed_tracks(
        raw, "EPSG:3857", max_step_m=80.0, max_speed_mps=20.0, isolated_spike_bridge_m=20.0
    )
    q = idx.query_connector(
        LineString([(0, 0), (20, 0)]),
        endpoint_snap_m=2.0,
        max_path_m=30.0,
        max_path_ratio=1.5,
        corridor_m=3.0,
    )
    assert report["enabled"] is True
    assert report["isolated_spikes_removed"] == 1
    assert q["witness_tracks"] == 1
    assert q["best_track_id"] == "t1"
    assert q["best_source_path_m"] == pytest.approx(20.0, abs=0.5)


def test_parallel_raw_tracks_do_not_count_as_cross_transition(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "upper", [(0, 0), (10, 0), (20, 0)])
    write_raw_track(raw, "lower", [(0, 6), (10, 6), (20, 6)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    q = idx.query_connector(
        LineString([(10, 0), (10, 6)]),
        endpoint_snap_m=1.5,
        max_path_m=15.0,
        max_path_ratio=2.0,
        corridor_m=3.0,
    )
    assert q["witness_tracks"] == 0


def test_raw_track_gap_repair_restores_connection_when_cleaned_track_may_be_split(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "through", [(-10, 0), (-5, 0), (0, 0), (5, 0), (10, 0), (15, 0), (20, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-10, 0), (0, 0), (10, 0), (20, 0)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(10, 0), (20, 0)], {}),
        ],
    )
    repaired, rows = repair_raw_track_gaps(
        net,
        idx,
        max_endpoint_distance_m=15.0,
        max_track_path_m=25.0,
        endpoint_snap_m=2.0,
        max_angle_deg=20.0,
        min_witness_tracks=1,
        max_repairs=10,
    )
    assert len(rows) == 1
    assert rows[0]["raw_witness_tracks"] == 1
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in repaired.edges)
    assert nx.number_connected_components(graph) == 1
    assert any(e.get("raw_track_gap_repair") for e in repaired.edges)



def test_raw_track_gap_repair_roundtrip_passes_its_final_audit(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "straight", [(-10, 0), (-5, 0), (0, 0), (5, 0), (10, 0), (15, 0), (20, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-10, 0), (0, 0), (10, 0), (20, 0)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(10, 0), (20, 0)], {}),
        ],
    )
    repaired, rows = repair_raw_track_gaps(
        net, idx,
        max_endpoint_distance_m=15.0,
        max_track_path_m=25.0,
        endpoint_snap_m=2.0,
        max_angle_deg=20.0,
        min_witness_tracks=1,
        max_repairs=10,
    )
    assert len(rows) == 1
    repaired_edge = next(e for e in repaired.edges if e.get("raw_track_gap_repair"))
    assert repaired_edge["raw_gap_angle_a_deg"] == pytest.approx(0.0, abs=1e-9)
    assert repaired_edge["raw_gap_angle_b_deg"] == pytest.approx(0.0, abs=1e-9)

    audit = audit_raw_track_gaps(
        repaired,
        max_endpoint_distance_m=15.0,
        max_track_path_m=25.0,
        max_angle_deg=20.0,
        min_witness_tracks=1,
    )
    violations = [r for r in audit if float(r["outside_baseline_fraction"]) > 1e-9]
    assert violations == []
    assert audit[0]["audit_reason"] == "approved_raw_track_witness"

def test_raw_witness_can_rescue_true_endpoint_to_edge_when_raster_evidence_is_missing(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "turn", [(-10, 0), (-5, 0), (0, 0), (4, 0), (8, 0), (8, 5), (8, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(-10, 0), (0, 0), (8, -10), (8, 10)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(8, -10), (8, 10)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=10.0,
        node_merge_m=1.0,
        max_approach_angle_deg=20.0,
        min_evidence_fraction=0.95,
        max_rounds=1,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_witness_endpoint_snap_m=2.0,
        raw_witness_max_path_m=25.0,
        raw_witness_max_path_ratio=2.5,
        raw_witness_corridor_m=4.0,
        raw_witness_relaxed_max_angle_deg=35.0,
    )
    assert len(rows) == 1
    assert rows[0]["raw_witness_tracks"] >= 1
    assert rows[0]["accepted_by_raw_witness"] is True
    assert len(repaired.edges) == 4  # target split into two plus connector


def test_endpoint_to_edge_rejects_transverse_bridge_between_parallel_roads_without_raw_transition(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "left", [(0, -10), (0, -5), (0, 0), (0, 5), (0, 10)])
    write_raw_track(raw, "right", [(6, -10), (6, -5), (6, 0), (6, 5), (6, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(0, -10), (0, 0), (6, -10), (6, 10)],
        [
            (0, 1, [(0, -10), (0, 0)], {}),
            (2, 3, [(6, -10), (6, 10)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=8.0,
        node_merge_m=1.0,
        max_approach_angle_deg=100.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_parallel_guard_max_axis_angle_deg=15.0,
        raw_parallel_guard_min_cross_angle_deg=55.0,
    )
    assert rows == []
    assert len(repaired.edges) == 2


def test_post_prune_removes_synthetic_parallel_crossbar_without_raw_transition(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "left", [(0, -10), (0, 0), (0, 10)])
    write_raw_track(raw, "right", [(6, -10), (6, 0), (6, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(0, -10), (0, 0), (6, 0), (6, 10)],
        [
            (0, 1, [(0, -10), (0, 0)], {}),
            (1, 2, [(0, 0), (6, 0)], {"source_type": "s7_component_stitching", "component_stitching_repair": True, "gap_repair": True}),
            (2, 3, [(6, 0), (6, 10)], {}),
        ],
    )
    cleaned, rows = prune_parallel_false_bridges(
        net, idx, max_bridge_m=12.0, parallel_axis_max_angle_deg=15.0,
        transverse_min_angle_deg=55.0, raw_witness_min_tracks=1,
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "remove_parallel_false_bridge"
    assert len(cleaned.edges) == 2


def test_post_prune_keeps_parallel_crossbar_when_original_track_really_crosses(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "cross", [(0, -4), (0, 0), (3, 0), (6, 0), (6, 4)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(0, -10), (0, 0), (6, 0), (6, 10)],
        [
            (0, 1, [(0, -10), (0, 0)], {}),
            (1, 2, [(0, 0), (6, 0)], {"source_type": "s7_component_stitching", "component_stitching_repair": True, "gap_repair": True}),
            (2, 3, [(6, 0), (6, 10)], {}),
        ],
    )
    cleaned, rows = prune_parallel_false_bridges(
        net, idx, max_bridge_m=12.0, parallel_axis_max_angle_deg=15.0,
        transverse_min_angle_deg=55.0, raw_witness_min_tracks=1,
    )
    assert rows == []
    assert len(cleaned.edges) == 3


def test_parallel_guard_does_not_delete_when_raw_data_has_no_side_occupancy(tmp_path):
    raw = tmp_path / "observed_tracks"
    # Raw data exists globally, but not on either of the two local road corridors.
    write_raw_track(raw, "far", [(100, 100), (110, 100), (120, 100)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    net = make_network(
        [(0, -10), (0, 0), (6, 0), (6, 10)],
        [
            (0, 1, [(0, -10), (0, 0)], {}),
            (1, 2, [(0, 0), (6, 0)], {"source_type": "s7_component_stitching", "component_stitching_repair": True, "gap_repair": True}),
            (2, 3, [(6, 0), (6, 10)], {}),
        ],
    )
    cleaned, rows = prune_parallel_false_bridges(
        net, idx, max_bridge_m=12.0, parallel_axis_max_angle_deg=15.0,
        transverse_min_angle_deg=55.0, raw_witness_min_tracks=1,
    )
    assert rows == []
    assert len(cleaned.edges) == 3


def test_junction_audit_accepts_raw_witness_rescue():
    from pipeline_modules.network_cleanup.junction_cleanup import audit_junction_repairs

    net = make_network(
        [(0, 0), (5, 0)],
        [(0, 1, [(0, 0), (5, 0)], {
            "source_type": "s7_junction_connector",
            "junction_repair": True,
            "junction_repair_mode": "endpoint_to_edge",
            "junction_repair_angle_deg": 30.0,
            "junction_repair_evidence_fraction": 0.0,
            "accepted_by_raw_witness": True,
            "raw_witness_tracks": 1,
        })],
    )
    rows = audit_junction_repairs(net, max_connector_m=8.0, max_angle_deg=65.0, min_evidence_fraction=0.25)
    assert len(rows) == 1
    assert rows[0]["outside_baseline_fraction"] == 0.0
    assert rows[0]["audit_reason"] == "approved_junction_repair_v0_5"



def test_full_s7_release_gate_raw_track_repair_survives_final_audit(tmp_path):
    import json
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from S7_network_cleanup import parser as s7_parser, run as run_s7

    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[31, 10:21] = 1
    arr[31, 32:46] = 1
    transform = from_origin(0, 64, 1, 1)
    primary = tmp_path / "primary.tif"
    support = tmp_path / "support.tif"
    kde = tmp_path / "kde.tif"
    for path, data, dtype in (
        (primary, arr, "uint8"),
        (support, arr.astype("float32") * 3.0, "float32"),
        (kde, arr.astype("float32"), "float32"),
    ):
        with rasterio.open(
            path, "w", driver="GTiff", height=64, width=64, count=1,
            dtype=dtype, crs="EPSG:3857", transform=transform, nodata=0,
        ) as ds:
            ds.write(data, 1)

    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "straight_e2e", [(x, 32.5) for x in range(11, 46)])
    output = tmp_path / "s7_out"
    args = s7_parser().parse_args([
        "--primary", str(primary),
        "--support", str(support),
        "--kde", str(kde),
        "--raw-tracks", str(raw),
        "--output", str(output),
        "--track-matcher", "none",
        "--artifact-loop-rounds", "1",
        "--junction-edge-rounds", "1",
        "--junction-cluster-max-repairs", "0",
        "--component-stitch-max-repairs", "10",
        "--raw-track-gap-max-repairs", "10",
        "--micro-gap-max-repairs", "10",
        "--s4-track-gap-max-repairs", "0",
        "--force-direction-gap-max-repairs", "0",
        "--topology-progress-every", "1000",
        "--simplify-progress-every", "1000",
    ])
    result = run_s7(args)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    audit = report["stages"]["audit"]
    assert report["version"] == "S7_network_cleanup_v0_5_smooth_network"
    assert report["stages"]["baseline"]["approved_raw_track_gap_repairs"] == 1
    assert report["stages"]["baseline"]["connectivity_before"]["component_count"] == 2
    assert report["stages"]["baseline"]["connectivity_after_repair"]["component_count"] == 1
    assert audit["raw_track_gap_violations"] == 0
    assert audit["micro_gap_violations"] == 0
    assert audit["junction_repair_violations"] == 0
    assert audit["component_stitch_violations"] == 0
    assert audit["directional_collapse_violations"] == 0
    assert audit["outside_baseline_edges"] == 0
    assert result["final"]["connectivity"]["component_count"] == 1
    assert (output / "08_final_clean_network" / "road_network_final_clean.gpkg").exists()

def test_config_and_runner_expose_raw_track_topology_controls():
    import json
    from S7_network_cleanup import parser as s7_parser

    args = s7_parser().parse_args([
        "--primary", "primary.tif", "--support", "support.tif", "--kde", "kde.tif", "--output", "out"
    ])
    assert args.raw_tracks is None
    assert args.raw_track_gap_max_endpoint_distance_m == pytest.approx(40.0)
    assert args.parallel_bridge_axis_max_angle_deg == pytest.approx(15.0)

    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8-sig"))
    assert str(cfg["version"]) == "0.5"
    s7 = cfg["stages"]["S7"]
    assert s7["raw_track_gap_max_endpoint_distance_m"] == pytest.approx(40.0)
    assert s7["parallel_bridge_min_side_tracks"] == 1

    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    assert str(schema["version"]) == "0.5"
    dests = {p["dest"] for p in schema["stages"]["S7"]["parameters"]}
    assert "raw_track_gap_max_endpoint_distance_m" in dests
    assert "parallel_bridge_min_side_tracks" in dests

    runner = (ROOT / "code" / "tools" / "run_pipeline_configurable.ps1").read_text(encoding="utf-8-sig")
    assert "S7_v0.5_smooth_network" in runner
    assert "$a.Add('--raw-tracks');$a.Add($s1Tracks)" in runner
