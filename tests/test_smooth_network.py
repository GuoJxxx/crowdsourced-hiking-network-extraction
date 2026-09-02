from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.cleaner import smooth_final_road_chains
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
            "support_mean": 8.0,
            "support_ge2_fraction": 0.9,
            "confidence": 0.95,
            "distinct_tracks": 8,
        }
        row.update(attrs)
        edges.append(row)
    return reindex_network(NetworkData(nodes, edges, None, None, 0, 0, {}))


def _turn_energy(net):
    pts = []
    ordered = sorted(net.edges, key=lambda e: int(e["edge_id"]))
    for i, e in enumerate(ordered):
        c = list(e["geometry"].coords)
        if i:
            c = c[1:]
        pts.extend(c)
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 3:
        return 0.0
    v = np.diff(arr, axis=0)
    n = np.linalg.norm(v, axis=1)
    good = n > 1e-8
    v = v[good] / n[good, None]
    if len(v) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(v, axis=0), axis=1)))


def test_final_chain_smoothing_rounds_degree2_zigzags_but_keeps_endpoints():
    net = make_network(
        [(0, 0), (4, 2), (8, -2), (12, 2), (16, 0)],
        [
            (0, 1, [(0, 0), (2, 0), (4, 2)], {}),
            (1, 2, [(4, 2), (6, 2), (8, -2)], {}),
            (2, 3, [(8, -2), (10, -2), (12, 2)], {}),
            (3, 4, [(12, 2), (14, 0), (16, 0)], {}),
        ],
    )
    before_energy = _turn_energy(net)
    before_mid = [(n["x"], n["y"]) for n in net.nodes[1:-1]]
    out, rows = smooth_final_road_chains(
        net,
        sample_spacing_m=0.5,
        sigma_m=3.0,
        corridor_m=3.0,
        uturn_sigma_m=1.0,
        uturn_corridor_m=0.8,
    )
    assert rows and rows[0]["smoothed"] is True
    assert Point(out.nodes[0]["x"], out.nodes[0]["y"]).distance(Point(0, 0)) < 1e-9
    assert Point(out.nodes[-1]["x"], out.nodes[-1]["y"]).distance(Point(16, 0)) < 1e-9
    after_mid = [(n["x"], n["y"]) for n in out.nodes[1:-1]]
    assert after_mid != before_mid
    assert _turn_energy(out) < before_energy * 0.70
    for left, right in zip(out.edges[:-1], out.edges[1:]):
        assert Point(left["geometry"].coords[-1]).distance(Point(right["geometry"].coords[0])) < 1e-9


def test_gap_repair_edges_are_smoothed_as_part_of_the_same_chain():
    net = make_network(
        [(0, 0), (5, 0), (10, 3), (15, 0)],
        [
            (0, 1, [(0, 0), (5, 0)], {}),
            (1, 2, [(5, 0), (7, 0), (10, 3)], {"gap_repair": True, "approved_gap_repair": True}),
            (2, 3, [(10, 3), (12, 3), (15, 0)], {}),
        ],
    )
    original_gap = net.edges[1]["geometry"]
    out, rows = smooth_final_road_chains(
        net, sample_spacing_m=0.5, sigma_m=2.5, corridor_m=3.0,
        uturn_sigma_m=1.0, uturn_corridor_m=0.8,
    )
    assert len(rows) == 1
    assert rows[0]["contains_gap_repair"] is True
    assert rows[0]["smoothed"] is True
    assert not out.edges[1]["geometry"].equals_exact(original_gap, tolerance=1e-9)


def test_uturn_chain_uses_narrow_corridor_and_does_not_shortcut_hairpin():
    coords = [(0, 12), (0, 8), (0, 4), (0, 1), (1, -1), (3, -2), (5, -1), (6, 1), (6, 4), (6, 8), (6, 12)]
    net = make_network(
        [(0, 12), (6, 12)],
        [(0, 1, coords, {"uturn_core_protected": True, "uturn_protected": True})],
    )
    original = net.edges[0]["geometry"]
    out, rows = smooth_final_road_chains(
        net, sample_spacing_m=0.4, sigma_m=3.0, corridor_m=3.0,
        uturn_sigma_m=1.0, uturn_corridor_m=0.8,
    )
    geom = out.edges[0]["geometry"]
    assert rows[0]["uturn_core"] is True
    assert rows[0]["hausdorff_shift_m"] <= 0.8 + 1e-9
    assert geom.length > original.length * 0.85
    assert min(y for x, y in geom.coords) < 0.0
    assert Point(geom.coords[0]).distance(Point(coords[0])) < 1e-9
    assert Point(geom.coords[-1]).distance(Point(coords[-1])) < 1e-9


def test_defaults_do_not_repolygonize_smoothed_curves():
    args = s7_parser().parse_args(["--primary", "p.tif", "--support", "s.tif", "--kde", "k.tif", "--output", "out"])
    assert args.simplify_tolerance_m == 0.0
    assert args.uturn_simplify_tolerance_m == 0.0
    assert args.final_smooth_sample_m <= 0.5
    assert args.final_smooth_sigma_m >= 2.5
    assert args.final_smooth_corridor_m >= 2.0

def test_real_junction_coordinate_and_topology_are_invariant():
    net = make_network(
        [(0, 0), (5, 2), (10, 0), (10, 8), (10, -8)],
        [
            (0, 1, [(0, 0), (3, 0), (5, 2)], {}),
            (1, 2, [(5, 2), (7, 2), (10, 0)], {}),
            (2, 3, [(10, 0), (10, 8)], {}),
            (2, 4, [(10, 0), (10, -8)], {}),
        ],
    )
    junction_before = next(n for n in net.nodes if int(n["degree"]) == 3)
    out, _ = smooth_final_road_chains(
        net, sample_spacing_m=0.5, sigma_m=3.0, corridor_m=3.0,
        uturn_sigma_m=1.0, uturn_corridor_m=0.8,
    )
    junction_after = next(n for n in out.nodes if int(n["degree"]) == 3)
    assert junction_after["x"] == junction_before["x"]
    assert junction_after["y"] == junction_before["y"]
    assert len(out.nodes) == len(net.nodes)
    assert len(out.edges) == len(net.edges)
    assert sorted(int(n["degree"]) for n in out.nodes) == sorted(int(n["degree"]) for n in net.nodes)

def test_nan_uturn_flags_do_not_force_every_road_into_uturn_smoothing():
    net = make_network(
        [(0, 0), (5, 2), (10, 0)],
        [
            (0, 1, [(0, 0), (2, 0), (5, 2)], {"uturn_core_protected": np.nan}),
            (1, 2, [(5, 2), (8, 0), (10, 0)], {"uturn_core_protected": np.nan}),
        ],
    )
    _, rows = smooth_final_road_chains(
        net, sample_spacing_m=0.5, sigma_m=3.0, corridor_m=2.5,
        uturn_sigma_m=1.0, uturn_corridor_m=0.8,
    )
    assert rows[0]["uturn_core"] is False
    assert rows[0]["target_sigma_m"] == 3.0


def test_disable_final_chain_smoothing_is_boolean_flag_in_schema():
    import json
    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    params = {p["dest"]: p for p in schema["stages"]["S7"]["parameters"]}
    meta = params["disable_final_chain_smoothing"]
    assert meta["kind"] == "bool_flag"
    assert meta["flag_value"] is True


def test_default_s7_schema_arguments_parse_without_literal_boolean_values():
    import json
    schema = json.loads((ROOT / "configs" / "parameter_schema.json").read_text(encoding="utf-8-sig"))
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8-sig"))
    args = ["--primary", "p.tif", "--support", "s.tif", "--kde", "k.tif", "--output", "out"]
    for meta in schema["stages"]["S7"]["parameters"]:
        value = cfg["stages"]["S7"].get(meta["dest"], meta.get("default"))
        kind = meta["kind"]
        if kind == "bool_optional":
            args.append(meta["positive_option"] if bool(value) else meta["negative_option"])
        elif kind == "bool_flag":
            if bool(value) == bool(meta.get("flag_value", True)):
                args.append(meta["option"])
        else:
            if value is not None and str(value).strip():
                args.extend([meta["option"], str(value)])
    parsed = s7_parser().parse_args(args)
    assert parsed.disable_final_chain_smoothing is False
    assert "False" not in args
