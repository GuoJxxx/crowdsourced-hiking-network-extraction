from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import (
    RawTrackChunk,
    RawTrackWitnessIndex,
    repair_raw_track_gaps,
)
from pipeline_modules.network_cleanup.junction_cleanup import repair_endpoint_to_edge_junctions
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles


def make_network(node_xy, edge_specs):
    nodes = [
        {"node_id": i, "x": float(x), "y": float(y), "geometry": Point(float(x), float(y))}
        for i, (x, y) in enumerate(node_xy)
    ]
    edges = []
    for eid, (u, v, coords) in enumerate(edge_specs):
        geom = LineString(coords)
        edges.append({
            "edge_id": eid,
            "u": int(u),
            "v": int(v),
            "geometry": geom,
            "length_m": float(geom.length),
            "source_type": "primary",
            "support_mean": 1.0,
            "support_ge2_fraction": 0.8,
            "confidence": 0.9,
        })
    return reindex_network(NetworkData(nodes, edges, "EPSG:3857", None, 0, 0, {}))


def graph_of(net):
    return nx.Graph((int(e["u"]), int(e["v"])) for e in net.edges)


def test_final_endpoint_gap_follows_curved_raw_subpath_instead_of_straight_chord():
    net = make_network(
        [(-10, 0), (0, 0), (10, 0), (20, 0)],
        [
            (0, 1, [(-10, 0), (0, 0)]),
            (2, 3, [(10, 0), (20, 0)]),
        ],
    )
    raw_geom = LineString([(-10, 0), (0, 0), (3, 4), (7, 4), (10, 0), (20, 0)])
    idx = RawTrackWitnessIndex([RawTrackChunk("r", "c", "src", raw_geom, 6)], "EPSG:3857")

    fixed, rows = repair_raw_track_gaps(
        net,
        idx,
        max_endpoint_distance_m=15.0,
        max_track_path_m=30.0,
        endpoint_snap_m=1.0,
        max_angle_deg=65.0,
        min_witness_tracks=1,
        max_path_ratio=3.0,
        corridor_m=1.0,  # old straight-chord witness rejects the 4 m bow
        allow_cycle_closure=False,
        use_raw_subpath_geometry=True,
        repair_mode="final_connectivity_rescue_endpoint_endpoint",
    )

    assert nx.is_connected(graph_of(fixed))
    assert len(rows) == 1
    assert rows[0]["mode"] == "final_connectivity_rescue_endpoint_endpoint"
    connector = next(e for e in fixed.edges if e.get("final_connectivity_rescue"))
    assert max(float(y) for _, y in connector["geometry"].coords) >= 3.9
    assert connector["raw_continuity_class"] == "raw_subpath_transition"
    assert connector["raw_gap_cycle_closure"] is False


def test_final_endpoint_to_edge_rescue_uses_curved_raw_transition_and_never_closes_cycle():
    net = make_network(
        [(-10, 0), (0, 0), (10, -10), (10, 10)],
        [
            (0, 1, [(-10, 0), (0, 0)]),
            (2, 3, [(10, -10), (10, 10)]),
        ],
    )
    raw_geom = LineString([(-10, 0), (0, 0), (3, 4), (7, 4), (10, 0), (10, 10)])
    idx = RawTrackWitnessIndex([RawTrackChunk("r", "c", "src", raw_geom, 6)], "EPSG:3857")

    fixed, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=15.0,
        node_merge_m=1.0,
        max_approach_angle_deg=65.0,
        min_evidence_fraction=1.0,  # no raster rescue: raw witness is required
        allow_cycle_closure=False,
        max_rounds=2,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_witness_endpoint_snap_m=1.5,
        raw_witness_max_path_m=30.0,
        raw_witness_max_path_ratio=3.0,
        raw_witness_corridor_m=1.0,
        raw_witness_relaxed_max_angle_deg=65.0,
        use_raw_subpath_geometry=True,
        require_raw_witness=True,
        raw_target_max_axis_angle_deg=65.0,
        repair_label="final_connectivity_rescue_endpoint_edge",
    )

    assert rows
    assert nx.is_connected(graph_of(fixed))
    assert all(not bool(r.get("cycle_closure")) for r in rows)
    connector = next(e for e in fixed.edges if e.get("junction_repair_mode") == "final_connectivity_rescue_endpoint_edge")
    assert max(float(y) for _, y in connector["geometry"].coords) >= 3.9
    assert connector.get("final_connectivity_rescue") is True


def test_long_thin_two_gate_bubble_is_collapsed_even_beyond_legacy_size_limit():
    net = make_network(
        [(-10, 0), (0, 0), (150, 4), (150, -4), (300, 0), (310, 0)],
        [
            (0, 1, [(-10, 0), (0, 0)]),
            (1, 2, [(0, 0), (150, 4)]),
            (2, 4, [(150, 4), (300, 0)]),
            (1, 3, [(0, 0), (150, -4)]),
            (3, 4, [(150, -4), (300, 0)]),
            (4, 5, [(300, 0), (310, 0)]),
        ],
    )
    # One real raw road runs through the middle; the two extracted arms are a
    # long, narrow duplicate skeleton that the legacy 220 m cycle size cap skipped.
    idx = RawTrackWitnessIndex([
        RawTrackChunk("main", "main", "src", LineString([(-10, 0), (0, 0), (150, 0), (300, 0), (310, 0)]), 5),
    ], "EPSG:3857")

    cleaned, rows = resolve_single_corridor_cycles(
        net,
        idx,
        raw_corridor_m=2.0,
        min_route_coverage=0.60,
        min_independent_tracks=1,
        min_full_ring_tracks=2,
        max_perimeter_m=220.0,
        max_span_m=80.0,
        max_area_m2=3000.0,
        thin_duplicate_max_perimeter_m=800.0,
        thin_duplicate_max_span_m=350.0,
        thin_duplicate_max_area_m2=8000.0,
    )

    assert nx.cycle_basis(graph_of(cleaned)) == []
    assert any(r.get("action") == "collapse_single_corridor_false_cycle" for r in rows)
    assert any(bool(r.get("thin_duplicate")) and float(r.get("perimeter_m", 0)) > 220 for r in rows)


def test_zero_or_one_gate_ring_without_local_raw_evidence_is_opened():
    net = make_network(
        [(-5, 0), (0, 0), (10, 0), (10, 10), (0, 10)],
        [
            (0, 1, [(-5, 0), (0, 0)]),
            (1, 2, [(0, 0), (10, 0)]),
            (2, 3, [(10, 0), (10, 10)]),
            (3, 4, [(10, 10), (0, 10)]),
            (4, 1, [(0, 10), (0, 0)]),
        ],
    )
    # Keep the witness index enabled, but all raw data is far from the ring.
    idx = RawTrackWitnessIndex([
        RawTrackChunk("far", "far", "src", LineString([(100, 100), (110, 100)]), 2)
    ], "EPSG:3857")

    cleaned, rows = resolve_single_corridor_cycles(
        net,
        idx,
        raw_corridor_m=2.0,
        min_route_coverage=0.60,
        min_independent_tracks=1,
        min_full_ring_tracks=2,
        max_perimeter_m=100.0,
        max_span_m=30.0,
        max_area_m2=500.0,
    )

    assert nx.cycle_basis(graph_of(cleaned)) == []
    assert any(r.get("collapse_reason") == "no_complete_ring_route_family" for r in rows)
    assert not any(r.get("action") == "preserve_multigate_cycle_no_local_raw_evidence" for r in rows)


def test_release_ui_and_s7_final_rescue_wiring():
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8-sig"))
    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    runner = (ROOT / "code" / "tools" / "run_pipeline_configurable.ps1").read_text(encoding="utf-8-sig")
    s7 = (ROOT / "code" / "S7_network_cleanup.py").read_text(encoding="utf-8")
    cmd1 = (ROOT / "run_S1_to_S7_windows.cmd").read_bytes()
    cmd2 = (ROOT / "run_S2_to_S7_windows.cmd").read_bytes()

    assert cfg["version"] == "0.5"
    assert schema["version"] == "0.5"
    assert "Activity-Aware Pipeline" in runner
    assert "S7_v0.5_smooth_network" in runner
    assert b"v0.5" not in cmd1 and b"v0.5" not in cmd2
    assert cmd1.startswith(b"@echo off") and cmd2.startswith(b"@echo off")
    assert b"\r\n" in cmd1 and b"\r\n" in cmd2
    assert "final_connectivity_rescue.csv" in s7
    assert "final_cycle_resolution.csv" in s7
    assert "junction_final_rejected.csv" in s7
    assert "rollback_failed_junction_repairs" in s7
    assert "allow_cycle_closure=False" in s7


def test_junction_audit_honors_creation_time_raw_witness_limits():
    from pipeline_modules.network_cleanup.junction_cleanup import audit_junction_repairs

    net = make_network(
        [(0, 0), (5, 0)],
        [(0, 1, [(0, 0), (5, 0)])],
    )
    edge = net.edges[0]
    edge.update({
        "junction_repair": True,
        "approved_junction_repair": True,
        "junction_repair_mode": "endpoint_cluster",
        "junction_repair_angle_deg": 80.0,
        "junction_repair_evidence_fraction": 0.0,
        "accepted_by_raw_witness": True,
        "raw_witness_tracks": 2,
        "junction_audit_max_connector_m": 10.8,
        "junction_audit_max_angle_deg": 75.0,
        "junction_audit_raw_max_angle_deg": 85.0,
        "junction_audit_min_evidence_fraction": 0.20,
    })

    rows = audit_junction_repairs(net, max_connector_m=8.0, max_angle_deg=75.0, min_evidence_fraction=0.20)
    assert len(rows) == 1
    row = rows[0]
    assert row["length_ok"] is True
    assert row["angle_ok"] is True
    assert row["evidence_ok"] is True
    assert row["raw_witness_ok"] is True
    assert row["outside_baseline_fraction"] == 0.0
    assert row["audit_reason"] == "approved_junction_repair_v0_5"


def test_failed_junction_repair_is_diagnosed_and_safely_rolled_back():
    from pipeline_modules.network_cleanup.junction_cleanup import (
        audit_junction_repairs,
        rollback_failed_junction_repairs,
    )

    net = make_network(
        [(0, 0), (10, 0), (20, 0)],
        [
            (0, 1, [(0, 0), (10, 0)]),
            (1, 2, [(10, 0), (20, 0)]),
        ],
    )
    # Mark only the second edge as a synthetic repair and give it a deliberately
    # stricter creation-time length budget than its actual geometry.
    bad = net.edges[1]
    bad.update({
        "junction_repair": True,
        "approved_junction_repair": True,
        "junction_repair_mode": "endpoint_to_edge",
        "junction_repair_angle_deg": 10.0,
        "junction_repair_evidence_fraction": 0.9,
        "junction_audit_max_connector_m": 5.0,
        "junction_audit_max_angle_deg": 65.0,
        "junction_audit_raw_max_angle_deg": 75.0,
        "junction_audit_min_evidence_fraction": 0.25,
    })

    audit = audit_junction_repairs(net)
    violations = [r for r in audit if r["outside_baseline_fraction"] > 0]
    assert len(violations) == 1
    assert violations[0]["length_ok"] is False
    assert violations[0]["endpoint_ok"] is True
    assert "length" in violations[0]["failed_checks"]

    cleaned, rejected = rollback_failed_junction_repairs(net, violations)
    assert len(rejected) == 1
    assert rejected[0]["rollback_reason"] == "failed_final_junction_audit_v0_5"
    assert len(cleaned.edges) == 1
    assert all(not e.get("junction_repair") for e in cleaned.edges)


def test_runner_retries_only_explicit_raster_io_failures():
    runner = (ROOT / "code" / "tools" / "run_pipeline_configurable.ps1").read_text(encoding="utf-8-sig")
    assert "function Test-IsRasterIoFailure" in runner
    assert "[NO-RETRY]" in runner
    assert "Test-IsRasterIoFailure $script:LastPythonOutput" in runner


def test_release_contains_no_legacy_version_labels_or_files():
    import re

    legacy_text = re.compile(r"v5(?:[._]\d+)+|v0[._][12]|v0\.[12]", re.IGNORECASE)
    legacy_name = re.compile(r"(?:v5\d+|v5[_\.]\d+|v0[_\.]?[12])(?:\D|$)", re.IGNORECASE)
    text_suffixes = {".py", ".ps1", ".cmd", ".json", ".md", ".csv", ".txt"}

    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if legacy_name.search(path.name):
            offenders.append(f"filename:{rel}")
        if path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if legacy_text.search(text):
                offenders.append(f"content:{rel}")
    assert offenders == []


def test_endpoint_cluster_persists_raw_witness_metadata_for_final_audit():
    from pipeline_modules.network_cleanup.junction_cleanup import (
        audit_junction_repairs,
        repair_endpoint_clusters,
    )
    from pipeline_modules.network_cleanup.raw_track_topology import RawTrackChunk, RawTrackWitnessIndex

    net = make_network(
        [(-10, 0), (-3, 0), (10, 0), (3, 0), (0, 10), (0, 3)],
        [
            (0, 1, [(-10, 0), (-3, 0)]),
            (2, 3, [(10, 0), (3, 0)]),
            (4, 5, [(0, 10), (0, 3)]),
        ],
    )
    idx = RawTrackWitnessIndex(
        [
            RawTrackChunk("a", "a", "src", LineString([(-10, 0), (-3, 0), (0, 0), (3, 0)]), 4),
            RawTrackChunk("b", "b", "src", LineString([(10, 0), (3, 0), (0, 0), (-3, 0)]), 4),
            RawTrackChunk("c", "c", "src", LineString([(0, 10), (0, 3), (0, 0), (0, -3)]), 4),
        ],
        "EPSG:3857",
    )

    fixed, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=60.0,
        min_evidence_fraction=1.0,
        min_cluster_size=3,
        max_cluster_size=6,
        raw_witness_index=idx,
        raw_witness_min_tracks=1,
        raw_witness_endpoint_snap_m=1.5,
        raw_witness_max_path_m=30.0,
        raw_witness_max_path_ratio=4.0,
        raw_witness_corridor_m=2.0,
        raw_witness_relaxed_max_angle_deg=85.0,
    )

    assert len(rows) == 3
    connectors = [e for e in fixed.edges if e.get("junction_repair_mode") == "endpoint_cluster"]
    assert len(connectors) == 3
    assert all(e.get("accepted_by_raw_witness") is True for e in connectors)
    assert all(int(e.get("raw_witness_tracks", 0)) >= 1 for e in connectors)
    assert all(float(e.get("junction_audit_raw_max_angle_deg", 0)) == 85.0 for e in connectors)
    audit = audit_junction_repairs(fixed)
    assert all(r["outside_baseline_fraction"] == 0.0 for r in audit)


def test_stage_output_directories_are_created_before_execution():
    runner = (ROOT / "code" / "tools" / "run_pipeline_configurable.ps1").read_text(encoding="utf-8-sig")
    s3 = (ROOT / "code" / "S3_activity_zone_filter.py").read_text(encoding="utf-8")
    assert "foreach($stageOut in @($s2,$s3,$s4,$s5,$s6,$s7))" in runner
    assert "New-Item -ItemType Directory -Path $stageOut -Force" in runner
    assert "output_dir.mkdir(parents=True, exist_ok=True)" in s3
