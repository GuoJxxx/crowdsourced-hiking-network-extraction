from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import S3_activity_zone_filter as s3
import S5_directional_kde as s5
from pipeline_modules import uturn_consensus as uc
from pipeline_modules.network_cleanup.junction_cleanup import junction_candidate_key
from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import RawTrackChunk, RawTrackWitnessIndex, repair_raw_track_gaps
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles
import networkx as nx


def _activity_frame() -> pd.DataFrame:
    rows = []
    rid = 0
    # Three independent, nearly coincident west-east traversals across one activity zone.
    for t, y in [("main_a", -0.8), ("main_b", 0.0), ("main_c", 0.8)]:
        for i, x in enumerate([-15.0, -8.0, -2.0, 2.0, 8.0, 15.0], start=1):
            rows.append({
                "clean_track_id": t,
                "clean_point_index": i,
                "x_m": x,
                "y_m": y,
                "activity_zone_id": 1 if -9.0 <= x <= 9.0 else 0,
                "activity_suppression_cell": -9.0 <= x <= 9.0,
                "activity_boundary_cell": False,
                "road_extraction_weight": 0.10 if -9.0 <= x <= 9.0 else 1.0,
                "mainroad_candidate": not (-9.0 <= x <= 9.0),
                "road_extraction_reason": "activity_internal_suppressed" if -9.0 <= x <= 9.0 else "normal",
                "_row_order": rid,
            })
            rid += 1
    return pd.DataFrame(rows)


def test_activity_zone_keeps_multi_track_gate_to_gate_corridor():
    frame = _activity_frame()
    cfg = s3.ActivityConfig(
        activity_corridor_min_tracks=3,
        activity_corridor_gate_tolerance_m=12.0,
        activity_corridor_max_axis_spread_deg=20.0,
        activity_corridor_min_efficiency=0.80,
    )
    annotated, accepted, rejected = s3.detect_activity_through_corridors(frame, cfg)
    assert len(accepted) == 1
    assert accepted.iloc[0]["track_support"] == 3
    inside = annotated[annotated["activity_zone_id"].eq(1)]
    assert inside["activity_through_corridor"].all()
    assert inside["mainroad_candidate"].all()
    assert inside["road_extraction_weight"].min() >= cfg.activity_corridor_weight
    road = s3.build_road_candidates(annotated, cfg)
    assert road.groupby("source_clean_track_id")["clean_track_id"].nunique().max() == 1
    assert rejected.empty


def _resample(points, n=101):
    xy = np.asarray(points, dtype=float)
    return uc.resample_curve(xy, n)


def _event(curve: np.ndarray, contribution: str, score=0.95):
    return {
        "track_id": contribution + "__seg",
        "original_contribution_id": contribution,
        "curve_fixed": curve,
        "curve": curve,
        "score": score,
        "apex_x": float(curve[len(curve)//2, 0]),
        "apex_y": float(curve[len(curve)//2, 1]),
        "path_efficiency": 0.25,
        "length_m": float(uc.cumulative_distance(curve)[-1]),
    }


def test_oval_loop_is_rejected_as_uturn_structure():
    theta = np.linspace(0, 2 * math.pi, 101)
    base = np.c_[15.0 * np.cos(theta), 8.0 * np.sin(theta)]
    events = [_event(base + np.array([0.0, dy]), f"oval_{i}") for i, dy in enumerate([-0.4, 0.0, 0.4, 0.8])]
    cfg = uc.UTurnConfig(
        min_distinct_original_tracks=3,
        max_leg_axis_difference_deg=30.0,
        min_opposite_direction_score=0.65,
        min_apex_turn_concentration=0.50,
        min_leg_separation_m=3.0,
        max_leg_separation_m=40.0,
    )
    consensus = uc.build_consensus(events, cfg)["consensus"]
    result = uc.validate_uturn_structure(events, consensus, cfg)
    assert result["structure_valid"] is False
    assert result["reject_reason"] != ""


def test_parallel_opposite_legs_with_apex_is_valid_uturn_structure():
    base = _resample([(0, 12), (0, 7), (0, 2), (1, -1), (3, -3), (5, -1), (6, 2), (6, 7), (6, 12)])
    events = [_event(base + np.array([dx, 0.0]), f"hairpin_{i}") for i, dx in enumerate([-0.3, -0.1, 0.1, 0.3])]
    cfg = uc.UTurnConfig(
        min_distinct_original_tracks=3,
        max_leg_axis_difference_deg=25.0,
        min_opposite_direction_score=0.70,
        min_apex_turn_concentration=0.45,
        min_leg_separation_m=3.0,
        max_leg_separation_m=20.0,
    )
    consensus = uc.build_consensus(events, cfg)["consensus"]
    result = uc.validate_uturn_structure(events, consensus, cfg)
    assert result["structure_valid"] is True
    assert result["distinct_original_track_support"] == 4
    assert result["leg_axis_difference_deg"] <= 25.0
    assert result["opposite_direction_score"] >= 0.70


def test_s5_activity_through_corridor_overrides_legacy_low_weight():
    df = pd.DataFrame({
        "x_m": [0.0, 1.0], "y_m": [0.0, 0.0],
        "clean_track_id": ["t", "t"], "clean_point_index": [1, 2],
        "road_extraction_weight": [0.10, 0.10],
        "activity_through_corridor": [True, True],
    })
    keep = s5.road_eligible_mask(df, s5.Config())
    assert keep.tolist() == [True, True]


def test_junction_ranking_prefers_direct_dense_raw_branch_over_nearer_sparse_branch():
    near_sparse = {
        "direct_witness_tracks": 0,
        "distinct_raw_tracks": 1,
        "raw_density_support": 2,
        "evidence_fraction": 0.90,
        "approach_angle_deg": 5.0,
        "distance_m": 3.0,
    }
    farther_dense = {
        "direct_witness_tracks": 3,
        "distinct_raw_tracks": 4,
        "raw_density_support": 9,
        "evidence_fraction": 0.55,
        "approach_angle_deg": 12.0,
        "distance_m": 5.0,
    }
    assert junction_candidate_key(farther_dense) > junction_candidate_key(near_sparse)


def test_structure_parameters_wired_and_launcher_title_is_version_neutral():
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8-sig"))
    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    assert cfg["version"] == "0.5"
    assert schema["version"] == "0.5"
    assert cfg["stages"]["S3"]["activity_corridor_min_tracks"] == 3
    assert cfg["stages"]["S4"]["uturn_min_distinct_original_tracks"] == 3
    launcher = (ROOT / "run_S2_to_S7_windows.cmd").read_bytes()
    assert launcher.startswith(b"@echo off")
    assert b"v0.5" not in launcher


def _net(node_xy, specs):
    nodes = [
        {"node_id": i, "x": float(x), "y": float(y), "geometry": Point(float(x), float(y))}
        for i, (x, y) in enumerate(node_xy)
    ]
    edges = []
    for eid, (u, v, coords) in enumerate(specs):
        g = LineString(coords)
        edges.append({
            "edge_id": eid, "u": u, "v": v, "geometry": g, "length_m": float(g.length),
            "source_type": "primary", "support_mean": 1.0, "support_ge2_fraction": 1.0,
            "confidence": 0.9,
        })
    return reindex_network(NetworkData(nodes, edges, "EPSG:3857", None, 0, 0, {}))


def test_direct_raw_continuity_gap_is_restored_and_labelled():
    net = _net(
        [(0, 0), (5, 0), (9, 0), (14, 0)],
        [(0, 1, [(0, 0), (5, 0)]), (2, 3, [(9, 0), (14, 0)])],
    )
    raw = RawTrackWitnessIndex([
        RawTrackChunk("r1", "c1", "src", LineString([(0, 0), (5, 0), (7, 0), (9, 0), (14, 0)]), 5)
    ], "EPSG:3857")
    fixed, rows = repair_raw_track_gaps(
        net, raw, max_endpoint_distance_m=6.0, max_track_path_m=12.0,
        endpoint_snap_m=1.0, max_angle_deg=20.0, corridor_m=1.0,
        min_witness_tracks=1,
    )
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in fixed.edges)
    assert nx.is_connected(graph)
    assert len(rows) == 1
    assert rows[0]["raw_continuity_class"] == "direct_transition"


def test_single_track_one_gate_polygon_loop_is_opened():
    # One lollipop attachment plus a geometric square loop. Only one raw contribution
    # runs around part of the square, which is insufficient to prove a true ring road.
    net = _net(
        [(-5, 0), (0, 0), (10, 0), (10, 10), (0, 10)],
        [
            (0, 1, [(-5, 0), (0, 0)]),
            (1, 2, [(0, 0), (10, 0)]),
            (2, 3, [(10, 0), (10, 10)]),
            (3, 4, [(10, 10), (0, 10)]),
            (4, 1, [(0, 10), (0, 0)]),
        ],
    )
    raw = RawTrackWitnessIndex([
        RawTrackChunk("r1", "c1", "src", LineString([(-5, 0), (0, 0), (10, 0), (10, 4)]), 4)
    ], "EPSG:3857")
    cleaned, rows = resolve_single_corridor_cycles(
        net, raw, raw_corridor_m=1.5, min_route_coverage=0.6,
        min_independent_tracks=1, min_full_ring_tracks=2,
        max_perimeter_m=100.0, max_span_m=30.0, max_area_m2=500.0,
    )
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert nx.cycle_basis(graph) == []
    assert any(r.get("collapse_reason") == "no_complete_ring_route_family" for r in rows)
