#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'S3->S4->S5->S6->S7 upstream input consistency checks: CRS, fields, 560 m tiles, raster alignment and track order.'
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
import numpy as np
import rasterio
try:
    import geopandas as gpd
except Exception:
    gpd = None


def unique_crs(path: Path) -> str:
    values = set()
    for chunk in pd.read_csv(path, usecols=lambda c: c == "metric_crs", chunksize=250000, low_memory=False):
        values.update(chunk["metric_crs"].dropna().astype(str).unique().tolist())
    if len(values) != 1:
        raise ValueError(f"""{path}: metric_crs is missing or not unique: {sorted(values)}""")
    return next(iter(values))


def check_track_order(path: Path) -> dict:
    frame = pd.read_csv(path, usecols=lambda c: c in {"clean_track_id", "clean_point_index", "x_m", "y_m", "road_extraction_weight"}, low_memory=False)
    required = {"clean_track_id", "clean_point_index", "x_m", "y_m"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"""{path}: missing fields {missing}""")
    duplicates = frame.duplicated(["clean_track_id", "clean_point_index"], keep=False)
    bad_order = 0
    for _, group in frame.groupby("clean_track_id", sort=False):
        order = pd.to_numeric(group["clean_point_index"], errors="coerce").to_numpy(float)
        if np.any(np.diff(order) <= 0):
            bad_order += 1
    return {
        "rows": int(len(frame)),
        "tracks": int(frame.clean_track_id.nunique()),
        "duplicate_order_rows": int(duplicates.sum()),
        "tracks_with_nonincreasing_order": int(bad_order),
        "weight_min": float(pd.to_numeric(frame.get("road_extraction_weight", 1.0), errors="coerce").min()) if len(frame) else None,
    }


def read_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--road-points", type=Path, required=True)
    parser.add_argument("--gravity-points", type=Path, required=True)
    parser.add_argument("--gravity-report", type=Path, required=True)
    parser.add_argument("--kde-report", type=Path, required=True)
    parser.add_argument("--kde", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--distance", type=Path, required=True)
    parser.add_argument("--raw-union", type=Path, required=True)
    parser.add_argument("--activity-zones", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    issues = []
    report = {}
    try:
        road_crs = unique_crs(args.road_points)
        gravity_crs = unique_crs(args.gravity_points)
        report["road_crs"] = road_crs
        report["gravity_crs"] = gravity_crs
        if road_crs != gravity_crs:
            issues.append(f"""S3 and S4 CRS differ: {road_crs} != {gravity_crs}""")
        report["road_order"] = check_track_order(args.road_points)
        report["gravity_order"] = check_track_order(args.gravity_points)
        if report["road_order"]["duplicate_order_rows"]:
            issues.append('road_candidate_points contains duplicate clean_track_id+clean_point_index values')

        gravity_report = read_json(args.gravity_report)
        kde_report = read_json(args.kde_report)
        gt = gravity_report.get("gravity_tile_config", {})
        kt_size = kde_report.get("tile_size_m_actual")
        kt_x = kde_report.get("tile_origin_x")
        kt_y = kde_report.get("tile_origin_y")
        report["gravity_tile"] = gt
        report["kde_tile"] = {"size": kt_size, "origin_x": kt_x, "origin_y": kt_y}
        if gt:
            pairs = [
                (gt.get("gravity_tile_size_m"), kt_size, "tile_size"),
                (gt.get("gravity_tile_origin_x_m"), kt_x, "origin_x"),
                (gt.get("gravity_tile_origin_y_m"), kt_y, "origin_y"),
            ]
            for a, b, name in pairs:
                if a is None or b is None or not math.isclose(float(a), float(b), abs_tol=1e-6):
                    issues.append(f"""S4/S5 {name} differs: {a} != {b}""")

        paths = [args.kde, args.support, args.distance, args.raw_union]
        datasets = [rasterio.open(path) for path in paths]
        try:
            first = datasets[0]
            report["raster"] = {
                "crs": str(first.crs), "width": first.width, "height": first.height,
                "transform": tuple(first.transform),
            }
            for path, ds in zip(paths[1:], datasets[1:]):
                if ds.width != first.width or ds.height != first.height or ds.transform != first.transform or ds.crs != first.crs:
                    issues.append(f"""S5 rasters are not aligned: {path}""")
            if str(first.crs) not in {road_crs, gravity_crs}:
                try:
                    from pyproj import CRS
                    if CRS.from_user_input(first.crs) != CRS.from_user_input(gravity_crs):
                        issues.append(f"""S5 raster CRS differs from track CRS: {first.crs} != {gravity_crs}""")
                except Exception:
                    issues.append(f"""Cannot compare raster and track CRS: {first.crs}, {gravity_crs}""")
        finally:
            for ds in datasets:
                ds.close()

        if args.activity_zones is not None and args.activity_zones.exists() and gpd is not None:
            zones = gpd.read_file(args.activity_zones, layer="activity_suppression_zones")
            report["activity_zone_features"] = int(len(zones))
            report["activity_zone_crs"] = str(zones.crs)
            if zones.crs is not None:
                from pyproj import CRS
                if CRS.from_user_input(zones.crs) != CRS.from_user_input(gravity_crs):
                    issues.append('activity_zones CRS differs from track CRS')

    except Exception as exc:
        issues.append(str(exc))

    report["issues"] = issues
    report["ok"] = not issues
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
