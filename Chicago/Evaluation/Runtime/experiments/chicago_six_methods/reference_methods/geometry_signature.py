"""Deterministic geometry signature for comparing independently generated graphs."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np


def canonical_geometry_signature(path: Path, layer: str = "edges",
                                 precision_m: float = 0.001) -> str:
    """Hash rounded XY lines without feature-order or direction sensitivity."""
    if not math.isfinite(precision_m) or precision_m <= 0:
        raise ValueError("Coordinate precision must be finite and positive")
    frame = gpd.read_file(path, layer=layer)
    if frame.crs is None or frame.crs.to_epsg() != 26916:
        raise ValueError("Network must use EPSG:26916")
    quantum = Decimal(str(precision_m))
    lines = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            raise ValueError("Network contains an empty geometry")
        if geometry.geom_type == "LineString":
            members = (geometry,)
        elif geometry.geom_type == "MultiLineString":
            members = geometry.geoms
        else:
            raise ValueError("Network must contain only line geometries")
        for member in members:
            coordinates = np.asarray(member.coords, dtype=float)[:, :2]
            if not np.isfinite(coordinates).all():
                raise ValueError("Network contains nonfinite coordinates")
            rounded = tuple(tuple(float(quantum * round(value / precision_m))
                                  for value in point) for point in coordinates)
            lines.append(min(rounded, rounded[::-1]))
    if not lines:
        raise ValueError("Network contains no lines")
    payload = json.dumps(sorted(lines), separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
