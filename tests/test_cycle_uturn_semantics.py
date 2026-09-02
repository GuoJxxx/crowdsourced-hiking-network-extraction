from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import RawTrackWitnessIndex
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles
from pipeline_modules.network_cleanup.uturn_topology import prune_uturn_shortcut_connectors


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


def cycle_network():
    # External gates are nodes 1 and 4.  Two approved branches form a false/real cycle.
    return make_network(
        [(-5, 0), (0, 0), (5, 3), (5, -3), (10, 0), (15, 0)],
        [
            (0, 1, [(-5, 0), (0, 0)], {}),
            (1, 2, [(0, 0), (5, 3)], {"gap_repair": True, "approved_gap_repair": True}),
            (2, 4, [(5, 3), (10, 0)], {"gap_repair": True, "approved_gap_repair": True}),
            (1, 3, [(0, 0), (5, -3)], {"gap_repair": True, "approved_gap_repair": True}),
            (3, 4, [(5, -3), (10, 0)], {"gap_repair": True, "approved_gap_repair": True}),
            (4, 5, [(10, 0), (15, 0)], {}),
        ],
    )


def test_route_support_distinguishes_one_supported_branch(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "top", [(-5, 0), (0, 0), (2.5, 1.5), (5, 3), (7.5, 1.5), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    top = LineString([(0, 0), (5, 3), (10, 0)])
    bottom = LineString([(0, 0), (5, -3), (10, 0)])
    a = idx.route_support(top, corridor_m=2.0, min_coverage=0.65)
    b = idx.route_support(bottom, corridor_m=2.0, min_coverage=0.65)
    assert a["support_tracks"] == 1
    assert b["support_tracks"] == 0


def test_semantic_cycle_removes_unsupported_branch_even_when_both_are_approved_gaps(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "top", [(-5, 0), (0, 0), (2.5, 1.5), (5, 3), (7.5, 1.5), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    cleaned, rows = resolve_single_corridor_cycles(
        cycle_network(), idx, raw_corridor_m=2.0, min_route_coverage=0.65,
        min_independent_tracks=1, max_rounds=3,
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "collapse_single_corridor_false_cycle"
    kept = int(rows[0]["kept_route"])
    assert rows[0][f"route{kept}_raw_tracks"] == 1
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert nx.cycle_basis(graph) == []
    assert nx.is_connected(graph)
    # The retained internal branch is the upper raw-track corridor, not the lower artefact.
    assert any(max(y for _, y in e["geometry"].coords) > 2.5 for e in cleaned.edges)
    assert not any(min(y for _, y in e["geometry"].coords) < -2.5 for e in cleaned.edges)


def test_semantic_cycle_preserves_true_two_route_loop_with_independent_raw_tracks(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "top", [(-5, 0), (0, 0), (5, 3), (10, 0), (15, 0)])
    write_raw_track(raw, "bottom", [(-5, 0), (0, 0), (5, -3), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    cleaned, rows = resolve_single_corridor_cycles(
        cycle_network(), idx, raw_corridor_m=2.0, min_route_coverage=0.65,
        min_independent_tracks=1, thin_duplicate_min_independent_tracks=1, max_rounds=3,
    )
    assert any(r["action"] == "preserve_independent_dual_route_cycle" for r in rows)
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert len(nx.cycle_basis(graph)) == 1
    assert len(cleaned.edges) == 6


def test_uturn_shortcut_guard_removes_synthetic_crossbar_but_keeps_single_u(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "hairpin", [(0, 10), (0, 5), (0, 0), (3, -1), (6, 0), (6, 5), (6, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    core = {"uturn_core_protected": True, "uturn_context_protected": True}
    net = make_network(
        [(0, 10), (0, 0), (6, 0), (6, 10)],
        [
            (0, 1, [(0, 10), (0, 0)], core),
            (1, 2, [(0, 0), (3, -1), (6, 0)], core),
            (2, 3, [(6, 0), (6, 10)], core),
            (0, 3, [(0, 10), (6, 10)], {
                **core,
                "source_type": "s7_raw_track_gap_repair",
                "raw_track_gap_repair": True,
                "gap_repair": True,
                "approved_gap_repair": True,
            }),
        ],
    )
    cleaned, rows = prune_uturn_shortcut_connectors(
        net, idx, max_connector_m=12.0, strict_corridor_m=2.0,
        strict_max_path_ratio=1.8, min_witness_tracks=1,
    )
    assert len(rows) == 1
    assert rows[0]["action"] == "remove_uturn_arm_shortcut"
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert len(cleaned.edges) == 3
    assert nx.is_connected(graph)
    assert nx.cycle_basis(graph) == []


def test_uturn_guard_does_not_remove_synthetic_cap_when_it_is_required_for_connectivity(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "hairpin", [(0, 10), (0, 0), (3, -1), (6, 0), (6, 10)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    core = {"uturn_core_protected": True, "uturn_context_protected": True}
    net = make_network(
        [(0, 10), (0, 0), (6, 0), (6, 10)],
        [
            (0, 1, [(0, 10), (0, 0)], core),
            (1, 2, [(0, 0), (3, -1), (6, 0)], {**core, "source_type": "s7_raw_track_gap_repair", "raw_track_gap_repair": True, "gap_repair": True}),
            (2, 3, [(6, 0), (6, 10)], core),
        ],
    )
    cleaned, rows = prune_uturn_shortcut_connectors(net, idx, max_connector_m=12.0)
    assert rows == []
    assert len(cleaned.edges) == 3


def test_semantic_cycle_preserves_wide_all_primary_loop_when_unobserved_branch_is_not_proven_false(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "top", [(-5, 0), (0, 0), (5, 10), (10, 0), (15, 0)])
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
        min_independent_tracks=1, max_rounds=2,
    )
    assert any(r["action"] == "collapse_single_corridor_false_cycle" for r in rows)
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert nx.cycle_basis(graph) == []
    assert nx.is_connected(graph)


def test_full_s7_release_gate_collapses_false_corridor_cycle_and_passes_final_audit(tmp_path):
    import json
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from skimage.draw import line as draw_line
    from S7_network_cleanup import parser as s7_parser, run as run_s7

    arr = np.zeros((64, 64), dtype=np.uint8)
    segments = [
        ((32, 10), (32, 20)),
        ((32, 20), (27, 30)),
        ((27, 30), (32, 40)),
        ((32, 20), (37, 30)),
        ((37, 30), (32, 40)),
        ((32, 40), (32, 50)),
    ]
    for (r0, c0), (r1, c1) in segments:
        rr, cc = draw_line(r0, c0, r1, c1)
        arr[rr, cc] = 1
    transform = from_origin(0, 64, 1, 1)
    primary = tmp_path / "primary_cycle.tif"
    support = tmp_path / "support_cycle.tif"
    kde = tmp_path / "kde_cycle.tif"
    for path, data, dtype in (
        (primary, arr, "uint8"),
        (support, arr.astype("float32") * 3.0, "float32"),
        (kde, arr.astype("float32"), "float32"),
    ):
        with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=1,
                           dtype=dtype, crs="EPSG:3857", transform=transform, nodata=0) as ds:
            ds.write(data, 1)

    # Raw trajectory follows only the upper route between the two gates.
    raw_pixels = []
    for (r0, c0), (r1, c1) in [segments[0], segments[1], segments[2], segments[5]]:
        rr, cc = draw_line(r0, c0, r1, c1)
        for r, c in zip(rr, cc):
            xy = (float(c) + 0.5, 63.5 - float(r))
            if not raw_pixels or xy != raw_pixels[-1]:
                raw_pixels.append(xy)
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "upper_only", raw_pixels)

    output = tmp_path / "s7_out"
    args = s7_parser().parse_args([
        "--primary", str(primary), "--support", str(support), "--kde", str(kde),
        "--raw-tracks", str(raw), "--output", str(output), "--track-matcher", "none",
        "--no-directional-loop-collapse", "--artifact-fallback-loop-max-m", "0",
        "--disable-raw-track-gap-repair", "--disable-micro-gap-repair",
        "--disable-s4-track-gap-repair", "--force-direction-gap-max-repairs", "0",
        "--junction-edge-rounds", "1", "--junction-cluster-max-repairs", "0",
        "--disable-component-stitching", "--topology-progress-every", "1000",
        "--simplify-progress-every", "1000",
    ])
    result = run_s7(args)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    baseline = report["stages"]["baseline"]
    audit = report["stages"]["audit"]
    assert report["version"] == "S7_network_cleanup_v0_5_smooth_network"
    assert baseline["single_corridor_false_cycles_collapsed"] >= 1
    assert baseline["connectivity_before"]["cyclomatic_number"] >= 1
    assert baseline["connectivity_after_repair"]["cyclomatic_number"] == 0
    assert audit["raw_track_gap_violations"] == 0
    assert audit["micro_gap_violations"] == 0
    assert audit["junction_repair_violations"] == 0
    assert audit["component_stitch_violations"] == 0
    assert audit["directional_collapse_violations"] == 0
    assert audit["outside_baseline_edges"] == 0
    assert result["final"]["connectivity"]["cyclomatic_number"] == 0
    assert (output / "08_final_clean_network" / "road_network_final_clean.gpkg").exists()


def test_semantic_cycle_can_collapse_false_loop_inside_uturn_core_instead_of_preserving_all_core_edges(tmp_path):
    raw = tmp_path / "observed_tracks"
    write_raw_track(raw, "main", [(-5, 0), (0, 0), (5, 3), (10, 0), (15, 0)])
    idx, _ = RawTrackWitnessIndex.from_observed_tracks(raw, "EPSG:3857")
    core = {"uturn_core_protected": True, "uturn_context_protected": True}
    net = make_network(
        [(-5, 0), (0, 0), (5, 3), (5, -3), (10, 0), (15, 0)],
        [
            (0, 1, [(-5, 0), (0, 0)], core),
            (1, 2, [(0, 0), (5, 3)], core),
            (2, 4, [(5, 3), (10, 0)], core),
            (1, 3, [(0, 0), (5, -3)], core),
            (3, 4, [(5, -3), (10, 0)], core),
            (4, 5, [(10, 0), (15, 0)], core),
        ],
    )
    cleaned, rows = resolve_single_corridor_cycles(
        net, idx, raw_corridor_m=2.0, min_route_coverage=0.65,
        min_independent_tracks=1, max_rounds=2,
    )
    assert any(r["action"] == "collapse_single_corridor_false_cycle" for r in rows)
    graph = nx.Graph((int(e["u"]), int(e["v"])) for e in cleaned.edges)
    assert nx.cycle_basis(graph) == []
