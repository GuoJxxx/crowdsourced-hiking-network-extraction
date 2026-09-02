from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import S3_activity_zone_filter as s3


def _fragmented_zone_frame():
    rows = []
    # No single track crosses from west to east. West fragments terminate near x=0;
    # east fragments begin near x=0. Dense overlapping cells still define one corridor.
    specs = [
        ("west_a", [-15, -8, -3, 1], -0.8),
        ("west_b", [-15, -8, -2, 2], 0.0),
        ("west_c", [-15, -7, -1, 2], 0.8),
        ("east_a", [-1, 3, 8, 15], -0.7),
        ("east_b", [-2, 2, 8, 15], 0.1),
        ("east_c", [-1, 3, 7, 15], 0.9),
    ]
    for tid, xs, y in specs:
        for i, x in enumerate(xs, 1):
            inside = -9 <= x <= 9
            rows.append({
                "clean_track_id": tid,
                "clean_point_index": i,
                "x_m": float(x),
                "y_m": float(y),
                "metric_crs": "EPSG:3857",
                "activity_zone_id": 1 if inside else 0,
                "activity_cell_col": int(np.floor((x + 20) / 10)),
                "activity_cell_row": 0,
                "road_extraction_weight": 0.10 if inside else 1.0,
                "mainroad_candidate": not inside,
                "road_extraction_reason": "activity_internal_suppressed" if inside else "normal",
                "_contribution_id": tid,
            })
    return pd.DataFrame(rows)


def _metrics_for_frame(frame: pd.DataFrame):
    rows = []
    for (c, r), g in frame[frame.activity_zone_id.eq(1)].groupby(["activity_cell_col", "activity_cell_row"]):
        rows.append({
            "activity_cell_col": int(c),
            "activity_cell_row": int(r),
            "activity_zone_id": 1,
            "track_count": int(g["_contribution_id"].nunique()),
            "point_count": int(len(g)),
            "activity_score": 0.9,
        })
    return pd.DataFrame(rows)


def test_activity_backbone_recovers_fragmented_zone_without_full_gate_to_gate_track():
    frame = _fragmented_zone_frame()
    metrics = _metrics_for_frame(frame)
    cfg = s3.ActivityConfig(
        cell_size_m=10.0,
        activity_backbone_min_tracks=3,
        activity_backbone_gate_cluster_m=12.0,
        activity_backbone_weight=1.0,
    )
    annotated, backbones, rejected = s3.extract_activity_zone_backbones(
        frame, metrics, cfg, origin_x=-20.0, origin_y=0.0
    )
    assert len(backbones) == 1
    assert rejected.empty
    assert int(backbones.iloc[0]["track_support"]) >= 3
    promoted = annotated[annotated["activity_backbone_corridor"]]
    assert not promoted.empty
    assert promoted["mainroad_candidate"].all()
    assert promoted["road_extraction_weight"].min() >= 1.0

    road = s3.build_road_candidates(annotated, cfg, activity_backbones=backbones)
    synthetic = road[road["activity_backbone_synthetic"].fillna(False).astype(bool)]
    assert synthetic.empty
    assert road["activity_backbone_corridor"].fillna(False).astype(bool).any()


def test_activity_backbone_is_simple_path_not_cycle():
    frame = _fragmented_zone_frame()
    # Add noisy loop-like points inside the same activity zone.
    extra = []
    for tid, pts in [
        ("loop1", [(-6, 5), (0, 8), (6, 5), (0, 2), (-6, 5)]),
        ("loop2", [(-5, 5), (0, 7), (5, 5), (0, 3), (-5, 5)]),
    ]:
        for i, (x, y) in enumerate(pts, 1):
            extra.append({
                "clean_track_id": tid, "clean_point_index": i, "x_m": x, "y_m": y,
                "metric_crs": "EPSG:3857", "activity_zone_id": 1,
                "activity_cell_col": int(np.floor((x + 20) / 10)),
                "activity_cell_row": int(np.floor(y / 10)),
                "road_extraction_weight": 0.1, "mainroad_candidate": False,
                "road_extraction_reason": "activity_internal_suppressed", "_contribution_id": tid,
            })
    frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    metrics = _metrics_for_frame(frame)
    cfg = s3.ActivityConfig(activity_backbone_min_tracks=3, activity_backbone_gate_cluster_m=12.0)
    _, backbones, _ = s3.extract_activity_zone_backbones(frame, metrics, cfg, -20.0, 0.0)
    assert len(backbones) == 1
    geom = backbones.iloc[0]["geometry"]
    assert geom.is_simple
    assert not geom.is_ring
