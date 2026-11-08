from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from .models import NetworkData, reindex_network


class DEMSampler:
    """Coordinate-aware DEM sampler. DEM does not need to align with S4/S5 rasters."""

    def __init__(self, dem_path: Optional[Path], network_crs: Any) -> None:
        self.path = Path(dem_path) if dem_path else None
        self.dataset = None
        self.transformer: Optional[Transformer] = None
        self.nodata = None
        if self.path is None or not self.path.exists():
            return
        self.dataset = rasterio.open(self.path)
        self.nodata = self.dataset.nodata
        if self.dataset.crs is None:
            raise ValueError(f"""DEM is missing a CRS: {self.path}""")
        source = CRS.from_user_input(network_crs)
        target = CRS.from_user_input(self.dataset.crs)
        if source != target:
            self.transformer = Transformer.from_crs(source, target, always_xy=True)

    @property
    def enabled(self) -> bool:
        return self.dataset is not None

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def __enter__(self) -> "DEMSampler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def sample_xy(self, xs: np.ndarray, ys: np.ndarray, default: float = np.nan) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        out = np.full(len(xs), float(default), dtype=float)
        if self.dataset is None or len(xs) == 0:
            return out
        valid = np.isfinite(xs) & np.isfinite(ys)
        if not valid.any():
            return out
        tx, ty = xs[valid], ys[valid]
        if self.transformer is not None:
            tx, ty = self.transformer.transform(tx, ty)
            tx, ty = np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)
        coordinates = list(zip(tx.tolist(), ty.tolist()))
        values = np.fromiter(
            (float(v[0]) for v in self.dataset.sample(coordinates, indexes=1, masked=False)),
            dtype=float,
            count=len(coordinates),
        )
        bad = ~np.isfinite(values)
        if self.nodata is not None and np.isfinite(self.nodata):
            bad |= np.isclose(values, float(self.nodata))
        values[bad] = float(default)
        out[np.flatnonzero(valid)] = values
        return out

    def sample_line(self, geom: LineString, spacing_m: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = float(geom.length)
        count = max(2, int(math.ceil(length / max(float(spacing_m), 0.5))) + 1)
        distances = np.linspace(0.0, length, count)
        points = [geom.interpolate(float(d)) for d in distances]
        xs = np.asarray([p.x for p in points], dtype=float)
        ys = np.asarray([p.y for p in points], dtype=float)
        elevations = self.sample_xy(xs, ys)
        return distances, np.c_[xs, ys], elevations


def _dem_metrics(distances: np.ndarray, elevations: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(elevations)
    if valid.sum() < 2:
        return {
            "elevation_start_m": np.nan,
            "elevation_end_m": np.nan,
            "elevation_min_m": np.nan,
            "elevation_max_m": np.nan,
            "elevation_mean_m": np.nan,
            "elevation_gain_m": np.nan,
            "elevation_loss_m": np.nan,
            "grade_mean_pct": np.nan,
            "grade_max_abs_pct": np.nan,
            "length_3d_m": np.nan,
            "dem_valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        }
    d = distances[valid]
    z = elevations[valid]
    order = np.argsort(d)
    d, z = d[order], z[order]
    dd = np.diff(d)
    dz = np.diff(z)
    usable = dd > 1e-6
    grades = np.abs(dz[usable] / dd[usable]) * 100.0 if usable.any() else np.empty(0)
    gain = float(np.sum(np.maximum(dz, 0.0)))
    loss = float(np.sum(np.maximum(-dz, 0.0)))
    length3d = float(np.sum(np.hypot(dd, dz)))
    horizontal = max(float(d[-1] - d[0]), 1e-9)
    return {
        "elevation_start_m": float(z[0]),
        "elevation_end_m": float(z[-1]),
        "elevation_min_m": float(np.min(z)),
        "elevation_max_m": float(np.max(z)),
        "elevation_mean_m": float(np.mean(z)),
        "elevation_gain_m": gain,
        "elevation_loss_m": loss,
        "grade_mean_pct": float((z[-1] - z[0]) / horizontal * 100.0),
        "grade_max_abs_pct": float(np.max(grades)) if len(grades) else 0.0,
        "length_3d_m": length3d,
        "dem_valid_fraction": float(valid.mean()),
    }


def attach_dem_attributes(
    network: NetworkData,
    dem_path: Optional[Path],
    spacing_m: float = 5.0,
    progress_every: int = 10_000,
) -> Dict[str, Any]:
    if dem_path is None or not Path(dem_path).exists():
        return {"enabled": False, "reason": "DEM not provided"}
    valid_edges = 0
    started = time.perf_counter()
    with DEMSampler(dem_path, network.crs) as dem:
        for node in network.nodes:
            z = dem.sample_xy(
                np.asarray([float(node["x"])]),
                np.asarray([float(node["y"])]),
            )[0]
            node["elevation_m"] = float(z) if np.isfinite(z) else np.nan
        total_edges = len(network.edges)
        for index, edge in enumerate(network.edges, 1):
            distances, _, elevations = dem.sample_line(edge["geometry"], spacing_m=spacing_m)
            metrics = _dem_metrics(distances, elevations)
            edge.update(metrics)
            if metrics["dem_valid_fraction"] >= 0.5:
                valid_edges += 1
            if progress_every > 0 and (index % int(progress_every) == 0 or index == total_edges):
                print(
                    f"    [dem] edges={index:,}/{total_edges:,}, valid={valid_edges:,}, "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    return {
        "enabled": True,
        "dem_path": str(dem_path),
        "spacing_m": float(spacing_m),
        "edges_with_dem": int(valid_edges),
        "edge_valid_ratio": float(valid_edges / max(len(network.edges), 1)),
    }


def filter_gap_repairs_by_dem(
    network: NetworkData,
    dem_path: Optional[Path],
    max_grade_pct: float = 45.0,
    spacing_m: float = 2.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove newly repaired gaps whose DEM grade is physically implausible."""
    if dem_path is None or not Path(dem_path).exists():
        return network, []
    net = network.copy()
    rejected: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    with DEMSampler(dem_path, net.crs) as dem:
        for edge in net.edges:
            if str(edge.get("source_type", "")) != "gap_repair":
                kept.append(edge)
                continue
            distances, _, elevations = dem.sample_line(edge["geometry"], spacing_m=spacing_m)
            metrics = _dem_metrics(distances, elevations)
            edge.update(metrics)
            max_grade = metrics.get("grade_max_abs_pct", np.nan)
            valid_fraction = metrics.get("dem_valid_fraction", 0.0)
            if valid_fraction >= 0.5 and np.isfinite(max_grade) and float(max_grade) > float(max_grade_pct):
                row = dict(edge)
                row["reject_reason"] = "dem_grade_exceeds_limit"
                row["max_allowed_grade_pct"] = float(max_grade_pct)
                rejected.append(row)
            else:
                edge["dem_gap_validated"] = bool(valid_fraction >= 0.5)
                kept.append(edge)
    net.edges = kept
    return reindex_network(net), rejected


def candidate_grade_penalty(
    dem: DEMSampler,
    candidates: np.ndarray,
    previous_xy: np.ndarray,
    next_xy: np.ndarray,
    previous_z: float,
    next_z: float,
    max_grade_pct: float,
) -> np.ndarray:
    """Penalty for candidate vertices producing excessive local longitudinal grade."""
    if not dem.enabled or len(candidates) == 0:
        return np.zeros(len(candidates), dtype=float)
    z = dem.sample_xy(candidates[:, 0], candidates[:, 1])
    d0 = np.linalg.norm(candidates - previous_xy[None, :], axis=1)
    d1 = np.linalg.norm(next_xy[None, :] - candidates, axis=1)
    penalty = np.zeros(len(candidates), dtype=float)
    valid0 = np.isfinite(z) & np.isfinite(previous_z) & (d0 > 1e-6)
    valid1 = np.isfinite(z) & np.isfinite(next_z) & (d1 > 1e-6)
    grade0 = np.zeros(len(candidates), dtype=float)
    grade1 = np.zeros(len(candidates), dtype=float)
    grade0[valid0] = np.abs(z[valid0] - previous_z) / d0[valid0] * 100.0
    grade1[valid1] = np.abs(next_z - z[valid1]) / d1[valid1] * 100.0
    excess = np.maximum(grade0 - max_grade_pct, 0.0) + np.maximum(grade1 - max_grade_pct, 0.0)
    penalty = excess / max(float(max_grade_pct), 1e-6)
    penalty[~(valid0 | valid1)] = 0.0
    return penalty
