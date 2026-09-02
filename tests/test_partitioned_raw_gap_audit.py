from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import audit_raw_track_gaps


def _net(edges):
    node_xy = [(0.0, 0.0), (10.0, 0.0), (90.0, 0.0)]
    nodes = [
        {"node_id": i, "x": x, "y": y, "geometry": Point(x, y)}
        for i, (x, y) in enumerate(node_xy)
    ]
    return reindex_network(NetworkData(nodes, edges, "EPSG:3857", None, 0, 0, {}))


def test_coverage_rescue_audit_uses_creation_time_limits_not_ordinary_gap_limits():
    geom = LineString([(10.0, 0.0), (40.0, 12.0), (90.0, 0.0)])
    edge = {
        "edge_id": 0,
        "u": 1,
        "v": 2,
        "geometry": geom,
        "length_m": geom.length,
        "raw_track_gap_repair": True,
        "approved_gap_repair": True,
        "raw_track_coverage_rescue": True,
        "coverage_audit": True,
        "raw_gap_euclidean_m": 80.0,
        "raw_witness_source_path_m": 95.0,
        "raw_gap_angle_a_deg": 25.0,
        "raw_gap_angle_b_deg": 20.0,
        "raw_witness_tracks": 2,
        "raw_gap_audit_mode": "final_connectivity_rescue_raw_coverage",
        "raw_gap_audit_max_endpoint_distance_m": 90.0,
        "raw_gap_audit_max_track_path_m": 140.0,
        "raw_gap_audit_max_angle_deg": 70.0,
        "raw_gap_audit_min_witness_tracks": 2,
    }
    net = _net([edge])

    rows = audit_raw_track_gaps(
        net,
        max_endpoint_distance_m=40.0,
        max_track_path_m=65.0,
        max_angle_deg=65.0,
        min_witness_tracks=1,
    )

    assert len(rows) == 1
    assert rows[0]["outside_baseline_fraction"] == 0.0
    assert rows[0]["audit_mode"] == "final_connectivity_rescue_raw_coverage"
    assert rows[0]["endpoint_distance_limit_m"] == 90.0
    assert rows[0]["raw_path_limit_m"] == 140.0


def test_failed_raw_gap_repair_is_safely_rolled_back_only_for_bad_edge():
    from pipeline_modules.network_cleanup.raw_track_topology import rollback_failed_raw_track_gap_repairs

    base = LineString([(0.0, 0.0), (10.0, 0.0)])
    bad = LineString([(10.0, 0.0), (90.0, 0.0)])
    edges = [
        {"edge_id": 0, "u": 0, "v": 1, "geometry": base, "length_m": base.length},
        {
            "edge_id": 1,
            "u": 1,
            "v": 2,
            "geometry": bad,
            "length_m": bad.length,
            "raw_track_gap_repair": True,
            "approved_gap_repair": True,
            "raw_gap_euclidean_m": 80.0,
            "raw_witness_source_path_m": 95.0,
            "raw_gap_angle_a_deg": 10.0,
            "raw_gap_angle_b_deg": 10.0,
            "raw_witness_tracks": 2,
            "raw_gap_audit_mode": "final_connectivity_rescue_raw_coverage",
            "raw_gap_audit_max_endpoint_distance_m": 90.0,
            "raw_gap_audit_max_track_path_m": 90.0,
            "raw_gap_audit_max_angle_deg": 70.0,
            "raw_gap_audit_min_witness_tracks": 2,
        },
    ]
    net = _net(edges)
    audit = audit_raw_track_gaps(net, 40.0, 65.0, 65.0, 1)
    violations = [r for r in audit if r["outside_baseline_fraction"] > 0]
    assert len(violations) == 1
    assert "raw_path" in violations[0]["audit_reason"]

    cleaned, rejected = rollback_failed_raw_track_gap_repairs(net, violations)
    assert len(cleaned.edges) == 1
    assert len(rejected) == 1
    assert rejected[0]["rollback_reason"] == "failed_final_raw_track_gap_audit_v0_5"


def test_zero_buffer_linestring_is_rasterized_directly_not_buffered_to_empty_polygon():
    from S6_route_locked_skeleton import raster_shapes_for_geometries

    line = LineString([(1.0, 1.0), (8.0, 1.0)])
    shapes = raster_shapes_for_geometries([line], buffer_m=0.0)

    assert len(shapes) == 1
    assert shapes[0][0]["type"] == "LineString"

    from rasterio.features import rasterize

    mask = rasterize(
        shapes,
        out_shape=(10, 10),
        transform=Affine.translation(0.0, 10.0) * Affine.scale(1.0, -1.0),
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    assert int(np.count_nonzero(mask)) > 0


def test_s7_rolls_back_failed_raw_gap_repairs_before_final_audit():
    text = (ROOT / "code" / "S7_network_cleanup.py").read_text(encoding="utf-8")
    assert "rollback_failed_raw_track_gap_repairs" in text
    assert "raw_track_gap_final_rejected.csv" in text
    assert "[audit:raw-track-gap:rollback]" in text
