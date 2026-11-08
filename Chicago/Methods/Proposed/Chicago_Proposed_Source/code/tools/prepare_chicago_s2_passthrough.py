#!/usr/bin/env python3
"""Build the S2 table required by S3 without applying hiking-only filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csv_files = sorted(args.input.glob("*__gxtrack.csv"))
    if not csv_files:
        parser.error(f"no *__gxtrack.csv files found in {args.input}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {
            "track_id",
            "point_index",
            "source_x_m",
            "source_y_m",
            "source_crs",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame["clean_track_id"] = frame["track_id"].astype(str)
        frame["clean_point_index"] = pd.to_numeric(frame["point_index"], errors="raise").astype(int)
        frame["clip_part_id"] = 1
        frame["temporal_part_id"] = 1
        frame["spatial_part_id"] = 1
        frame["hiking_part_id"] = 1
        frame["hiking_quality"] = "not_applicable_vehicle_passthrough"
        frame["hiking_confidence"] = 1.0
        frame["x_m"] = pd.to_numeric(frame["source_x_m"], errors="raise")
        frame["y_m"] = pd.to_numeric(frame["source_y_m"], errors="raise")
        frame["metric_crs"] = frame["source_crs"].astype(str)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    metric_values = sorted(combined["metric_crs"].dropna().unique().tolist())
    if metric_values != ["EPSG:26916"]:
        raise ValueError(f"unexpected source CRS values: {metric_values}")
    if not np.isfinite(combined[["x_m", "y_m"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite projected coordinates")

    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        args.output / "all_cleaned_points.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    coordinate_error = np.max(
        np.abs(
            combined[["x_m", "y_m"]].to_numpy(dtype=float)
            - combined[["source_x_m", "source_y_m"]].to_numpy(dtype=float)
        )
    )
    report = {
        "mode": "vehicle_passthrough_no_hiking_filter",
        "files": len(csv_files),
        "tracks": int(combined["clean_track_id"].nunique()),
        "points": len(combined),
        "metric_crs": metric_values[0],
        "coordinate_max_abs_error_m": float(coordinate_error),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
    }
    (args.output / "s2_vehicle_passthrough_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
