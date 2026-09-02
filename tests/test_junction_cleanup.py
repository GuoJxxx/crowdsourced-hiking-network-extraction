from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest
import networkx as nx
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.junction_cleanup import (
    prune_short_isolated_components,
    repair_endpoint_clusters,
    repair_endpoint_to_edge_junctions,
)


def make_network(node_xy, edge_specs):
    nodes = [
        {"node_id": i, "x": float(x), "y": float(y), "geometry": Point(float(x), float(y))}
        for i, (x, y) in enumerate(node_xy)
    ]
    edges = []
    for eid, (u, v, coords, attrs) in enumerate(edge_specs):
        geom = LineString(coords)
        edge = {
            "edge_id": eid,
            "u": u,
            "v": v,
            "geometry": geom,
            "length_m": float(geom.length),
            "source_type": "primary",
            "distinct_tracks": 0,
            "support_mean": 0.0,
            "support_ge2_fraction": 0.0,
            "mean_match_dist_m": 999.0,
        }
        edge.update(attrs)
        edges.append(edge)
    return reindex_network(NetworkData(nodes, edges, None, None, 0, 0, {}))


def test_two_endpoint_gap_is_not_promoted_to_degree2_cluster_hub():
    net = make_network(
        [(-10, 0), (-3, 0), (3, 0), (10, 0)],
        [
            (0, 1, [(-10, 0), (-3, 0)], {}),
            (2, 3, [(3, 0), (10, 0)], {}),
        ],
    )
    repaired, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=75.0,
        min_evidence_fraction=0.0,
        min_cluster_size=3,
    )
    assert rows == []
    assert len(repaired.edges) == 2


def test_short_isolated_component_counts_all_parallel_edges():
    # Two parallel edges share the same node pair.  Their total component length is
    # ~30 m, so a 25 m hard-component rule must NOT delete either edge.
    net = make_network(
        [(0, 0), (15, 0), (100, 0), (130, 0)],
        [
            (0, 1, [(0, 0), (15, 0)], {}),
            (0, 1, [(0, 0), (15, 0.1)], {}),
            (2, 3, [(100, 0), (130, 0)], {}),
        ],
    )
    cleaned, removed = prune_short_isolated_components(
        net,
        hard_max_length_m=25.0,
        weak_max_length_m=0.0,
    )
    assert removed == []
    assert len(cleaned.edges) == 3


def test_cluster_can_ignore_one_bad_arm_and_keep_three_valid_arms():
    # Three valid approaches point toward the same local hub. A fourth component has
    # one endpoint nearby, but its continuation points away. One bad endpoint must not
    # veto the whole junction.
    net = make_network(
        [
            (-20, 0), (-3, 0),
            (3, 0), (20, 0),
            (0, 20), (0, 3),
            (4, 4), (2, 2), (20, 20),
        ],
        [
            (0, 1, [(-20, 0), (-3, 0)], {}),
            (2, 3, [(3, 0), (20, 0)], {}),
            (4, 5, [(0, 20), (0, 3)], {}),
            (6, 7, [(4, 4), (2, 2)], {}),
            (7, 8, [(2, 2), (20, 20)], {}),
        ],
    )
    repaired, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=60.0,
        min_evidence_fraction=0.0,
        min_cluster_size=3,
        max_cluster_size=6,
    )
    assert len(rows) == 3
    hubs = {r["junction_node"] for r in rows}
    assert len(hubs) == 1
    assert len(repaired.edges) == 8


def test_endpoint_to_edge_rechecks_angle_after_snapping_to_existing_node():
    # Endpoint E=(0,0) points east.  Exact projection on target edge is Q=(5,0),
    # but Q is within node_merge_m of target node N=(5,2).  Snapping E->N changes
    # the connector direction enough to exceed the 10-degree gate, so it must reject.
    net = make_network(
        [
            (-10, 0), (0, 0),     # source component, endpoint at (0,0)
            (5, 2), (10, 0),      # target starts at N=(5,2), bends through exact projection Q=(5,0)
        ],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(5, 2), (5, 0), (10, 0)], {}),
        ],
    )
    # Use a generous first-stage angle to ensure candidate exists; final snap must be
    # rechecked against the same threshold. Here E->N is ~21.8 degrees.
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=8.0,
        node_merge_m=3.0,
        max_approach_angle_deg=10.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
    )
    assert rows == []
    assert len(repaired.edges) == 2


def test_endpoint_to_edge_stops_at_nearer_same_component_edge_instead_of_crossing_it():
    # Current release: allows a strongly justified same-component loop closure. The x=4 edge
    # therefore becomes the correct first topological target. The repair must stop
    # and node there rather than overshooting to the x=8 component.
    net = make_network(
        [
            (0, 0), (-10, 0),
            (4, 10), (4, -10),
            (8, 10), (8, -10),
        ],
        [
            (1, 0, [(-10, 0), (0, 0)], {}),
            (1, 2, [(-10, 0), (4, 10)], {}),
            (2, 3, [(4, 10), (4, -10)], {}),
            (4, 5, [(8, 10), (8, -10)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=8.1,
        node_merge_m=1.0,
        max_approach_angle_deg=20.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
    )
    assert len(rows) == 1
    assert rows[0]["target_edge_id"] == 2
    assert rows[0]["cycle_closure"] is True
    assert float(rows[0]["geometry"].coords[-1][0]) == pytest.approx(4.0)
    # Split target edge + connector => +2 edges total, with no crossing to x=8.
    assert len(repaired.edges) == 6


def test_cluster_hub_cannot_be_drawn_through_unmodelled_existing_edge():
    net = make_network(
        [
            (-10, 0), (-3, 0),
            (3, 0), (10, 0),
            (0, 10), (0, 3),
            (-20, 0), (20, 0),
        ],
        [
            (0, 1, [(-10, 0), (-3, 0)], {}),
            (2, 3, [(3, 0), (10, 0)], {}),
            (4, 5, [(0, 10), (0, 3)], {}),
            # Protected through-edge is not itself an endpoint-cluster candidate.
            (6, 7, [(-20, 0), (20, 0)], {"uturn_protected": True}),
        ],
    )
    repaired, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=75.0,
        min_evidence_fraction=0.0,
        min_cluster_size=3,
    )
    assert rows == []
    assert len(repaired.edges) == 4


def test_endpoint_to_edge_may_touch_other_edges_at_existing_target_junction_node():
    # Snapping to an already-modelled target node is legitimate even when multiple
    # target-component edges are incident there.  The connector only touches those
    # edges at the existing node; that is not an unmodelled crossing.
    net = make_network(
        [
            (-10, 0), (0, 0),
            (5, 0), (5, 10), (12, 0),
        ],
        [
            (0, 1, [(-10, 0), (0, 0)], {}),
            (2, 3, [(5, 0), (5, 10)], {}),
            (2, 4, [(5, 0), (12, 0)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=6.0,
        node_merge_m=2.5,
        max_approach_angle_deg=15.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
    )
    assert len(rows) == 1
    assert rows[0]["junction_node"] == 2
    assert len(repaired.edges) == 4


def test_cluster_with_too_many_valid_arms_is_rejected_as_ambiguous():
    node_xy = []
    edge_specs = []
    for i in range(7):
        a = 2.0 * math.pi * i / 7.0
        outer = (20.0 * math.cos(a), 20.0 * math.sin(a))
        inner = (3.0 * math.cos(a), 3.0 * math.sin(a))
        u = len(node_xy); node_xy.append(outer)
        v = len(node_xy); node_xy.append(inner)
        edge_specs.append((u, v, [outer, inner], {}))
    net = make_network(node_xy, edge_specs)
    repaired, rows = repair_endpoint_clusters(
        net,
        cluster_radius_m=8.0,
        center_search_m=0.0,
        max_approach_angle_deg=75.0,
        min_evidence_fraction=0.0,
        min_cluster_size=3,
        max_cluster_size=6,
    )
    assert rows == []
    assert len(repaired.edges) == 7


def test_endpoint_to_edge_repairs_in_same_round_cannot_create_unmodelled_overlap_or_crossing():
    # Multiple candidates are generated against the pre-repair graph.  They may meet
    # at a shared modelled node, but they must not overlap as duplicate connectors or
    # cross at a point that has no shared node id.
    net = make_network(
        [
            (-10, -15), (0, -5),
            (-10, 15), (0, 5),
            (8, 4), (8, 6),
            (8, -6), (8, -4),
        ],
        [
            (0, 1, [(-10, -15), (0, -5)], {}),
            (2, 3, [(-10, 15), (0, 5)], {}),
            (4, 5, [(8, 4), (8, 6)], {}),
            (6, 7, [(8, -6), (8, -4)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=13.5,
        node_merge_m=0.5,
        max_approach_angle_deg=20.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
    )
    assert len(rows) >= 1
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            inter = a["geometry"].intersection(b["geometry"])
            if inter.is_empty:
                continue
            assert inter.geom_type == "Point"
            a_nodes = {int(a["endpoint_node"]), int(a["junction_node"])}
            b_nodes = {int(b["endpoint_node"]), int(b["junction_node"])}
            assert a_nodes & b_nodes


def test_endpoint_to_edge_same_round_does_not_double_connect_same_components_and_create_cycle():
    net = make_network(
        [(0, 0), (10, 0), (0, 5), (10, 5)],
        [
            (0, 1, [(0, 0), (10, 0)], {}),
            (2, 3, [(0, 5), (10, 5)], {}),
        ],
    )
    repaired, rows = repair_endpoint_to_edge_junctions(
        net,
        max_snap_m=6.0,
        node_merge_m=0.5,
        max_approach_angle_deg=100.0,
        min_evidence_fraction=0.0,
        max_rounds=1,
    )
    graph = nx.Graph()
    for e in repaired.edges:
        graph.add_edge(int(e["u"]), int(e["v"]))
    assert nx.cycle_basis(graph) == []
    assert len(rows) == 1
