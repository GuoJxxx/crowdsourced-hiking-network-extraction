#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full block-by-block raster readability check.

This is a launcher reliability helper only. It does not modify any raster
and does not change any S1-S7 algorithm.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rasterio


def validate_one(path: Path, retries: int, wait_s: float) -> bool:
    for attempt in range(retries + 1):
        try:
            with rasterio.open(path) as ds:
                blocks = 0
                for _, window in ds.block_windows(1):
                    ds.read(1, window=window)
                    blocks += 1
                if blocks == 0:
                    ds.read(1)
            print(
                f"[RASTER CHECK OK] {path} | "
                f"Rasterio={rasterio.__version__} GDAL={rasterio.__gdal_version__}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"[RASTER CHECK FAIL] {path} | attempt={attempt + 1}/{retries + 1} | "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(max(0.0, float(wait_s)))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--wait-s", type=float, default=2.0)
    ap.add_argument("rasters", nargs="+")
    args = ap.parse_args()

    failed = []
    for raw in args.rasters:
        p = Path(raw)
        if not p.is_file():
            print(f"[RASTER CHECK FAIL] file not found: {p}", file=sys.stderr, flush=True)
            failed.append(str(p))
            continue
        if not validate_one(p, max(0, args.retries), max(0.0, args.wait_s)):
            failed.append(str(p))

    if failed:
        print("[RASTER CHECK SUMMARY] FAILED:", file=sys.stderr, flush=True)
        for item in failed:
            print(f"  - {item}", file=sys.stderr, flush=True)
        return 2

    print("[RASTER CHECK SUMMARY] ALL RASTERS READ OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
