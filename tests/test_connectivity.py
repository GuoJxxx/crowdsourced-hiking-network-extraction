from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.s7_force_refine import repair_s4_track_gaps, repair_strong_direction_gaps
from pipeline_modules.network_cleanup.junction_cleanup import repair_endpoint_clusters, repair_endpoint_to_edge_junctions
from pipeline_modules.network_cleanup.cleaner import remove_digital_artifact_loops
from pipeline_modules.network_cleanup.conservative import audit_approved_micro_gaps, audit_approved_directional_collapses
from pipeline_modules.network_cleanup.connectivity_repair import (
    audit_component_stitching,
    connectivity_audit,
    repair_fragmented_components,
)
from S7_network_cleanup import parser as s7_parser


def make_network(node_xy, edge_specs):
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
            "distinct_tracks": 1,
        }
        row.update(attrs)
        edges.append(row)
    return reindex_network(NetworkData(nodes, edges, None, None, 0, 0, {}))


def nx_graph(net):
    g = nx.Graph()
    for e in net.edges:
        g.add_edge(int(e["u"]), int(e["v"]))
    return g


def test_strong_direction_gap_can_close_evidence_backed_same_component_loop():
    # Endpoints 0 and 5 face one another across a 5 m missing segment, but are
    # already connected by an ~80 m detour around the rest of the road loop.
    net = make_network(
        [(0, 0), (-10, 0), (-10, 20), (15, 20), (15, 0), (5, 0)],
        [
            (0, 1, [(0, 0), (-10, 0)], {}),
            (1, 2, [(-10, 0), (-10, 20)], {}),
            (2, 3, [(-10, 20), (15, 20)], {}),
            (3, 4, [(15, 20), (15, 0)], {}),
            (4, 5, [(15, 0), (5, 0)], {}),
        ],
    )
    before_cycles = len(nx.cycle_basis(nx_graph(net)))
    repaired, rows = repair_strong_direction_gaps(
        net,
        max_distance_m=8.0,
        max_angle_deg=10.0,
        min_evidence_fraction=0.0,
        allow_cycle_closure=True,
        cycle_min_detour_ratio=2.0,
        cycle_min_detour_excess_m=10.0,
        max_repairs=10,
    )
    assert len(rows) == 1
    assert rows[0]["cycle_closure"] is True
    assert rows[0]["existing_detour_m"] > 50.0
    assert len(nx.cycle_basis(nx_graph(repaired))) == before_cycles + 1


def test_strong_direction_gap_does_not_duplicate_short_same_component_route():
    # Same geometry at the endpoints, but the existing detour is only 9 m. It is
    # not a missing loop closure and must not receive a duplicate shortcut.
    net = make_network(
        [(0, 0), (-2, 0), (-2, 2), (7, 2), (7, 0), (5, 0)],
        [
            (0, 1, [(0, 0), (-2, 0)], {}),
            (1, 2, [(-2, 0), (-2, 2)], {}),
            (2, 3, [(-2, 2), (7, 2)], {}),
            (3, 4, [(7, 2), (7, 0)], {}),
            (4, 5, [(7, 0), (5, 0)], {}),
        ],
    )
    repaired, rows = repair_strong_direction_gaps(
        net,
        max_distance_m=8.0,
        max_angle_deg=10.0,
        min_evidence_fraction=0.0,
        allow_cycle_closure=True,
        cycle_min_detour_ratio=4.0,
        cycle_min_detour_excess_m=20.0,
        max_repairs=10,
    )
    assert rows == []
    assert len(repaired.edges) == len(net.edges)


def test_endpoint_to_edge_can_close_same_component_open_loop_and_split_target():
    # Endpoint 0 points east toward the vertical target edge 4-5. The target is in
    # the same component only through a long detour. current release rejected this solely
    # because a path already existed; current release should close the missing road segment.
    net = make_network(
        [(0, 0), (-10, 0), (-10, 20), (5, 20), (5, 5), (5, -5)],
        [
            (0, 1, [(0, 0), (-10, 0)], {}),
            (1, 2, [(-10, 0), (-10, 20)], {}),
            (2, 3, [(-10, 20), (5, 20)], {}),
            (3, 4, [(5, 20), (5, 5)], {}),
            (4, 5, [(5, 5), (5, -5)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=7.0,
        node_merge_m=1.0,
        max_approach_angle_deg=10.0,
        min_evidence_fraction=0.0,
        allow_cycle_closure=True,
        cycle_min_detour_ratio=2.0,
        cycle_min_detour_excess_m=10.0,
        max_rounds=1,
    )
    assert len(rows) == 1
    assert rows[0]["cycle_closure"] is True
    # Target edge is split in two, plus one connector: +2 net edges overall.
    assert len(repaired.edges) == len(net.edges) + 2
    assert any(int(n.get("degree", 0)) >= 3 for n in repaired.nodes)


def test_fragmented_component_stitching_reduces_components_without_forcing_all_networks():
    net = make_network(
        [(-10, 0), (0, 0), (9, 0), (20, 0), (100, 0), (110, 0)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(9, 0), (20, 0)], {}),
            (4, 5, [(100, 0), (110, 0)], {}),
        ],
    )
    before = connectivity_audit(net)
    repaired, rows = repair_fragmented_components(
        net,
        max_distance_m=12.0,
        max_angle_deg=15.0,
        min_evidence_fraction=0.0,
        long_gap_threshold_m=8.0,
        long_gap_max_angle_deg=15.0,
        long_gap_min_evidence_fraction=0.0,
        max_repairs=10,
    )
    after = connectivity_audit(repaired)
    assert len(rows) == 1
    assert before["component_count"] == 3
    assert after["component_count"] == 2
    # The unsupported far-away third component is intentionally untouched.
    assert after["component_count"] != 1



def test_same_s4_track_can_close_same_component_loop(tmp_path):
    net = make_network(
        [(0, 0), (-10, 0), (-10, 20), (15, 20), (15, 0), (5, 0)],
        [
            (0, 1, [(0, 0), (-10, 0)], {}),
            (1, 2, [(-10, 0), (-10, 20)], {}),
            (2, 3, [(-10, 20), (15, 20)], {}),
            (3, 4, [(15, 20), (15, 0)], {}),
            (4, 5, [(15, 0), (5, 0)], {}),
        ],
    )
    csv = tmp_path / "gravity.csv"
    csv.write_text(
        "clean_track_id,clean_point_index,x_m,y_m\n"
        "t1,0,0,0\n"
        "t1,1,2.5,0\n"
        "t1,2,5,0\n",
        encoding="utf-8-sig",
    )
    repaired, rows = repair_s4_track_gaps(
        net, csv,
        max_endpoint_distance_m=8.0,
        max_track_path_m=8.0,
        endpoint_snap_m=1.0,
        max_angle_deg=10.0,
        allow_cycle_closure=True,
        cycle_min_detour_ratio=2.0,
        cycle_min_detour_excess_m=10.0,
    )
    assert len(rows) == 1
    assert rows[0]["cycle_closure"] is True
    assert len(nx.cycle_basis(nx_graph(repaired))) == 1


def test_endpoint_cluster_can_restore_same_component_three_arm_junction():
    # Three broken approaches are already connected through a long outer detour.
    # A physical center junction should still be reconstructed and may create cycles.
    net = make_network(
        [
            (-20, 0), (-3, 0),
            (20, 0), (3, 0),
            (0, 20), (0, 3),
        ],
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
    )
    assert len(rows) == 3
    assert sum(bool(r["cycle_closure"]) for r in rows) >= 2
    assert len({r["junction_node"] for r in rows}) == 1
    assert connectivity_audit(repaired)["cyclomatic_number"] >= 2

def test_generic_microloop_removal_requires_both_small_perimeter_and_small_span():
    # A 10 x 10 m loop has perimeter 40 m. If max_span_m=5 it is not a tiny raster
    # artifact, even though its perimeter is below 50 m. current release's OR condition would
    # still open the cycle; current release must preserve it.
    net = make_network(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [
            (0, 1, [(0, 0), (10, 0)], {}),
            (1, 2, [(10, 0), (10, 10)], {}),
            (2, 3, [(10, 10), (0, 10)], {}),
            (3, 0, [(0, 10), (0, 0)], {}),
        ],
    )
    cleaned, dropped = remove_digital_artifact_loops(net, max_perimeter_m=50.0, max_span_m=5.0, max_rounds=1)
    assert dropped == []
    assert len(nx.cycle_basis(nx_graph(cleaned))) == 1


def test_connectivity_audit_reports_largest_component_and_endpoints():
    net = make_network(
        [(0, 0), (10, 0), (20, 0), (100, 0), (105, 0)],
        [
            (0, 1, [(0, 0), (10, 0)], {}),
            (1, 2, [(10, 0), (20, 0)], {}),
            (3, 4, [(100, 0), (105, 0)], {}),
        ],
    )
    m = connectivity_audit(net)
    assert m["component_count"] == 2
    assert m["endpoint_count"] == 4
    assert 0.79 < m["largest_component_length_fraction"] < 0.81
    assert m["cyclomatic_number"] == 0


def test_parser_defaults_preserve_real_cycles_and_enable_stitching():
    p = s7_parser()
    # Required paths are syntactic only for parser default inspection.
    args = p.parse_args([
        "--primary", "p.tif", "--support", "s.tif", "--kde", "k.tif", "--output", "out"
    ])
    assert args.allow_polygon_loop_resolver is False
    assert args.disable_component_stitching is False
    assert args.disable_cycle_closure is False
    assert args.component_stitch_max_m >= 20.0


def test_component_stitch_audit_rejects_edge_outside_approved_thresholds():
    net = make_network(
        [(0, 0), (10, 0)],
        [
            (0, 1, [(0, 0), (10, 0)], {
                "source_type": "s7_component_stitching",
                "gap_repair": True,
                "approved_gap_repair": True,
                "component_stitching_repair": True,
                "component_stitch_distance_m": 10.0,
                "component_stitch_angle_a_deg": 5.0,
                "component_stitch_angle_b_deg": 5.0,
                "component_stitch_evidence_fraction": 0.8,
            }),
        ],
    )
    ok = audit_component_stitching(
        net, max_distance_m=12.0, max_angle_deg=15.0, min_evidence_fraction=0.5,
        long_gap_threshold_m=8.0, long_gap_max_angle_deg=10.0,
        long_gap_min_evidence_fraction=0.7,
    )
    assert len(ok) == 1
    assert ok[0]["outside_baseline_fraction"] == 0.0

    bad_edge = net.edges[0]
    bad_edge["component_stitch_evidence_fraction"] = 0.2
    bad = audit_component_stitching(
        net, max_distance_m=12.0, max_angle_deg=15.0, min_evidence_fraction=0.5,
        long_gap_threshold_m=8.0, long_gap_max_angle_deg=10.0,
        long_gap_min_evidence_fraction=0.7,
    )
    assert bad[0]["outside_baseline_fraction"] == 1.0
    assert "evidence" in bad[0]["audit_reason"]



def test_component_stitch_audit_preserves_zero_degree_angles():
    net = make_network(
        [(0, 0), (10, 0)],
        [
            (0, 1, [(0, 0), (10, 0)], {
                "source_type": "s7_component_stitching",
                "gap_repair": True,
                "approved_gap_repair": True,
                "component_stitching_repair": True,
                "component_stitch_distance_m": 10.0,
                "component_stitch_angle_a_deg": 0.0,
                "component_stitch_angle_b_deg": 0.0,
                "component_stitch_evidence_fraction": 0.8,
            }),
        ],
    )
    rows = audit_component_stitching(
        net, max_distance_m=12.0, max_angle_deg=15.0, min_evidence_fraction=0.5,
        long_gap_threshold_m=20.0, long_gap_max_angle_deg=10.0,
        long_gap_min_evidence_fraction=0.7,
    )
    assert rows[0]["angle_a_deg"] == 0.0
    assert rows[0]["angle_b_deg"] == 0.0
    assert rows[0]["outside_baseline_fraction"] == 0.0


def test_directional_collapse_audit_preserves_zero_degree_gate_angles():
    net = make_network(
        [(0, 0), (10, 0)],
        [
            (0, 1, [(0, 0), (10, 0)], {
                "approved_directional_collapse": True,
                "directional_collapse_evidence_fraction": 0.9,
                "directional_collapse_gate_angle_a_deg": 0.0,
                "directional_collapse_gate_angle_b_deg": 0.0,
                "geometry_origin": "directional_loop_diagonal_v0_5",
            }),
        ],
    )
    rows = audit_approved_directional_collapses(
        net, min_evidence_fraction=0.6, max_gate_angle_deg=30.0
    )
    assert rows[0]["gate_angle_a_deg"] == 0.0
    assert rows[0]["gate_angle_b_deg"] == 0.0
    assert rows[0]["outside_baseline_fraction"] == 0.0

def test_rejected_stitch_crossing_does_not_poison_component_union_state():
    # A-B is the first accepted vertical stitch. The best C-D candidate crosses it
    # and must be rejected. A second C-D endpoint pair above A-B is valid; rejecting
    # the first candidate must not make union-find pretend C and D were already joined.
    net = make_network(
        [
            (0, -20), (0, 4.5),      # component A, endpoint 1 faces up
            (0, 5.5), (0, 30),       # component B, endpoint 2 faces down
            (-1, 5), (-5, 40),       # component C endpoints
            (1, 5), (-1, 40),        # component D endpoints
        ],
        [
            (0, 1, [(0, -20), (0, 4.5)], {}),
            (2, 3, [(0, 5.5), (0, 30)], {}),
            (4, 5, [(-1, 5), (-10, 5), (-10, 40), (-5, 40)], {}),
            (6, 7, [(1, 5), (10, 5), (10, 40), (5, 40), (-1, 40)], {}),
        ],
    )
    repaired, rows = repair_fragmented_components(
        net,
        max_distance_m=6.0,
        max_angle_deg=45.0,
        min_evidence_fraction=0.0,
        long_gap_threshold_m=10.0,
        long_gap_max_angle_deg=45.0,
        long_gap_min_evidence_fraction=0.0,
        max_repairs=10,
    )
    pairs = {frozenset((int(r["a"]), int(r["b"]))) for r in rows}
    assert frozenset((1, 2)) in pairs       # vertical A-B stitch
    assert frozenset((4, 6)) not in pairs   # crossing C-D candidate rejected
    assert frozenset((5, 7)) in pairs       # safe alternative C-D stitch accepted
    assert connectivity_audit(repaired)["component_count"] == 2


def test_default_generic_loop_cleanup_preserves_ten_meter_realistic_cycle():
    # The evidence-aware directional collapse can still handle a raster duplicate,
    # but the geometry-only fallback must be much smaller than a plausible 10x10 m
    # road/path loop. This protects real mesh cycles when no positive artefact proof exists.
    net = make_network(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [
            (0, 1, [(0, 0), (10, 0)], {}),
            (1, 2, [(10, 0), (10, 10)], {}),
            (2, 3, [(10, 10), (0, 10)], {}),
            (3, 0, [(0, 10), (0, 0)], {}),
        ],
    )
    cleaned, dropped = remove_digital_artifact_loops(net)
    assert dropped == []
    assert len(nx.cycle_basis(nx_graph(cleaned))) == 1


def test_strict_long_endpoint_to_edge_stitch_splits_through_road_topologically():
    # This is the long-range component-stitch mode used by S7 after normal 8 m
    # junction repair. A branch may reach a through-road 15 m away, but the result
    # must be a real degree-3 graph node, not a line that merely crosses the road.
    net = make_network(
        [(-10, 0), (0, 0), (15, -10), (15, 10)],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(15, -10), (15, 10)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=20.0,
        node_merge_m=1.0,
        max_approach_angle_deg=10.0,
        min_evidence_fraction=0.0,
        allow_cycle_closure=False,
        max_rounds=1,
        max_repairs=10,
    )
    assert len(rows) == 1
    assert abs(float(rows[0]["connector_length_m"]) - 15.0) < 1e-6
    assert connectivity_audit(repaired)["component_count"] == 1
    assert any(int(n.get("degree", 0)) >= 3 for n in repaired.nodes)



def test_micro_gap_audit_does_not_reaudit_component_stitch_as_micro_gap(tmp_path):
    """Component stitching has its own current release audit envelope and must not be
    reinterpreted as a <=10 m micro-gap merely because it is also marked gap_repair.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    primary = tmp_path / "primary.tif"
    with rasterio.open(
        primary, "w", driver="GTiff", height=32, width=32, count=1,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(-10, 20, 1, 1),
        nodata=0,
    ) as ds:
        ds.write(np.ones((32, 32), dtype=np.uint8), 1)

    net = make_network(
        [(0, 0), (20, 0)],
        [(0, 1, [(0, 0), (20, 0)], {
            "source_type": "s7_component_stitching",
            "gap_repair": True,
            "approved_gap_repair": True,
            "component_stitching_repair": True,
            "component_stitch_distance_m": 20.0,
            "component_stitch_angle_a_deg": 5.0,
            "component_stitch_angle_b_deg": 5.0,
            "component_stitch_evidence_fraction": 0.9,
            "geometry_origin": "s7_component_stitch_v0_5",
        })],
    )
    net.crs = rasterio.crs.CRS.from_epsg(3857)

    rows = audit_approved_micro_gaps(
        net, primary,
        max_length_m=10.0,
        max_angle_deg=45.0,
        long_gap_threshold_m=6.0,
        long_gap_max_angle_deg=30.0,
        long_gap_min_evidence_fraction=0.5,
    )
    assert rows == []


def test_endpoint_to_edge_does_not_split_existing_synthetic_gap_repair():
    """A synthetic gap connector is not a trustworthy through-road target.
    Splitting it creates child edges whose micro-gap approval envelope is no longer
    the original endpoint-to-endpoint repair and caused the current release final audit crash.
    """
    net = make_network(
        [(-5, 0), (0, 0), (5, -5), (5, 5)],
        [
            (0, 1, [(-5, 0), (0, 0)], {}),
            (2, 3, [(5, -5), (5, 5)], {
                "source_type": "micro_gap_repair",
                "gap_repair": True,
                "approved_gap_repair": True,
                "micro_gap_repair": True,
                "micro_gap_missing_pixels": 1,
                "micro_gap_angle_u_deg": 0.0,
                "micro_gap_angle_v_deg": 0.0,
                "micro_gap_evidence_fraction": 1.0,
                "geometry_origin": "micro_gap_directional_straight_connector_v0_5",
            }),
        ],
    )

    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=6.0,
        node_merge_m=0.5,
        max_approach_angle_deg=15.0,
        min_evidence_fraction=0.0,
        allow_cycle_closure=False,
        max_rounds=1,
        max_repairs=10,
    )
    assert rows == []
    assert len(repaired.edges) == len(net.edges)
    assert sum(bool(e.get("micro_gap_repair")) for e in repaired.edges) == 1
