#!/usr/bin/env python3
"""Adapt projected Chicago S1-compatible CSVs to the WGS84 input S2 expects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-crs", default="EPSG:26916")
    args = parser.parse_args()

    csv_files = sorted(args.input.glob("*__gxtrack.csv"))
    if not csv_files:
        parser.error(f"no *__gxtrack.csv files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    transformer = Transformer.from_crs(args.source_crs, "EPSG:4326", always_xy=True)
    point_count = 0

    for source_path in csv_files:
        frame = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
        source_x = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(dtype=float)
        source_y = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(source_x).all() and np.isfinite(source_y).all()):
            raise ValueError(f"non-finite source coordinate in {source_path}")

        longitude, latitude = transformer.transform(source_x, source_y)
        frame["source_x_m"] = source_x
        frame["source_y_m"] = source_y
        frame["source_crs"] = args.source_crs
        frame["longitude"] = longitude
        frame["latitude"] = latitude
        frame.to_csv(
            args.output / source_path.name,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        point_count += len(frame)

    summary = {
        "files": len(csv_files),
        "points": point_count,
        "source_crs": args.source_crs,
        "target_crs": "EPSG:4326",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
    }
    (args.output / "coordinate_adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
