from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .dem_constraints import DEMSampler, candidate_grade_penalty
from .models import NetworkData, clean_linestring, reindex_network
from .tile_io import RasterBlockSampler


def resample_line(geom: LineString, spacing_m: float) -> np.ndarray:
    length = float(geom.length)
    if length <= 1e-9:
        return np.asarray(geom.coords, dtype=float)
    count = max(2, int(math.ceil(length / max(float(spacing_m), 0.5))) + 1)
    distances = np.linspace(0.0, length, count)
    return np.asarray([[geom.interpolate(float(d)).x, geom.interpolate(float(d)).y] for d in distances], dtype=float)


def _bool_values(series: pd.Series, default: bool = True) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).to_numpy(dtype=bool)
    values = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


def load_trajectory_tree(
    points_csv: Optional[Path],
    sample_step: int = 10,
    max_points: int = 1_500_000,
    chunksize: int = 250_000,
    min_road_weight: float = 0.30,
) -> Tuple[Optional[cKDTree], Dict[str, Any]]:
    if points_csv is None or not Path(points_csv).exists():
        return None, {"enabled": False, "reason": "gravity points not provided"}
    points_csv = Path(points_csv)
    columns = set(pd.read_csv(points_csv, nrows=0).columns)
    xcol = "gravity_x_m" if "gravity_x_m" in columns else "x_m"
    ycol = "gravity_y_m" if "gravity_y_m" in columns else "y_m"
    if xcol not in columns or ycol not in columns:
        return None, {"enabled": False, "reason": "x/y columns missing"}
    order_col = "clean_point_index" if "clean_point_index" in columns else (
        "point_index" if "point_index" in columns else None
    )
    usecols = [xcol, ycol]
    for name in (order_col, "road_extraction_weight", "mainroad_candidate"):
        if name and name in columns:
            usecols.append(name)
    parts: List[np.ndarray] = []
    total_valid = total_sampled = 0
    step = max(1, int(sample_step))
    running = 0
    for chunk in pd.read_csv(points_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        x = pd.to_numeric(chunk[xcol], errors="coerce").to_numpy(float)
        y = pd.to_numeric(chunk[ycol], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if "mainroad_candidate" in chunk.columns:
            valid &= _bool_values(chunk["mainroad_candidate"], default=True)
        if "road_extraction_weight" in chunk.columns:
            weights = pd.to_numeric(chunk["road_extraction_weight"], errors="coerce").fillna(0).to_numpy(float)
            valid &= weights >= float(min_road_weight)
        idx = np.flatnonzero(valid)
        total_valid += len(idx)
        if len(idx):
            # Global deterministic stride, independent of chunk boundaries.
            select = idx[((running + np.arange(len(idx))) % step) == 0]
            running += len(idx)
            if len(select):
                parts.append(np.c_[x[select], y[select]])
                total_sampled += len(select)
    if not parts:
        return None, {"enabled": False, "reason": "no valid trajectory points"}
    xy = np.vstack(parts)
    if len(xy) > int(max_points):
        indices = np.linspace(0, len(xy) - 1, int(max_points), dtype=np.int64)
        xy = xy[indices]
    return cKDTree(xy), {
        "enabled": True,
        "points_valid": int(total_valid),
        "points_sampled_before_cap": int(total_sampled),
        "tree_points": int(len(xy)),
        "sample_step": int(step),
        "max_points": int(max_points),
    }


def _normal(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= 1e-9:
        return np.asarray([0.0, 0.0])
    t = x / n
    return np.asarray([-t[1], t[0]], dtype=float)


def _trajectory_cost(tree: Optional[cKDTree], candidates: np.ndarray, scale_m: float) -> np.ndarray:
    if tree is None or len(candidates) == 0:
        return np.zeros(len(candidates), dtype=float)
    distances, _ = tree.query(candidates, k=1, workers=1)
    return np.minimum(np.asarray(distances, dtype=float) / max(float(scale_m), 1e-6), 3.0)


def _candidate_costs(
    candidates: np.ndarray,
    offsets: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    sampler: RasterBlockSampler,
    trajectory_tree: Optional[cKDTree],
    trajectory_scale_m: float,
    search_m: float,
    weights: Dict[str, float],
    dem: DEMSampler,
    previous_z: float,
    following_z: float,
    dem_max_grade_pct: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    kde = sampler.values_xy("kde", candidates[:, 0], candidates[:, 1], default=0.0)
    support = sampler.values_xy("support", candidates[:, 0], candidates[:, 1], default=0.0)
    local_kde_scale = max(float(np.max(kde)), float(np.percentile(kde, 75)) if len(kde) else 0.0, 1e-9)
    ridge_cost = 1.0 - np.clip(kde / local_kde_scale, 0.0, 1.0)
    support_cost = 1.0 - np.clip(support / 3.0, 0.0, 1.0)
    trajectory_cost = _trajectory_cost(trajectory_tree, candidates, trajectory_scale_m)
    midpoint = 0.5 * (previous + following)
    smooth_cost = np.linalg.norm(candidates - midpoint[None, :], axis=1) / max(float(search_m), 1e-6)
    offset_cost = np.abs(offsets) / max(float(search_m), 1e-6)
    dem_cost = candidate_grade_penalty(
        dem,
        candidates,
        previous,
        following,
        previous_z,
        following_z,
        max_grade_pct=dem_max_grade_pct,
    )
    total = (
        float(weights["ridge"]) * ridge_cost
        + float(weights["support"]) * support_cost
        + float(weights["trajectory"]) * trajectory_cost
        + float(weights["smooth"]) * smooth_cost
        + float(weights["topology"]) * offset_cost
        + float(weights["dem"]) * dem_cost
    )
    return total, {
        "ridge": ridge_cost,
        "support": support_cost,
        "trajectory": trajectory_cost,
        "smooth": smooth_cost,
        "topology": offset_cost,
        "dem": dem_cost,
        "kde": kde,
        "support_value": support,
    }


def _edge_energy(
    coords: np.ndarray,
    sampler: RasterBlockSampler,
    trajectory_tree: Optional[cKDTree],
    trajectory_scale_m: float,
    weights: Dict[str, float],
    dem: DEMSampler,
    dem_max_grade_pct: float,
) -> Dict[str, float]:
    if len(coords) < 3:
        return {"total": 0.0, "ridge": 0.0, "support": 0.0, "trajectory": 0.0, "smooth": 0.0, "topology": 0.0, "dem": 0.0}
    kde = sampler.values_xy("kde", coords[:, 0], coords[:, 1], default=0.0)
    support = sampler.values_xy("support", coords[:, 0], coords[:, 1], default=0.0)
    scale = max(float(np.percentile(kde, 95)), 1e-9)
    ridge = float(np.mean(1.0 - np.clip(kde / scale, 0.0, 1.0)))
    support_cost = float(np.mean(1.0 - np.clip(support / 3.0, 0.0, 1.0)))
    trajectory = float(np.mean(_trajectory_cost(trajectory_tree, coords, trajectory_scale_m)))
    second = coords[:-2] - 2.0 * coords[1:-1] + coords[2:]
    segment = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    smooth = float(np.mean(np.linalg.norm(second, axis=1) / max(float(np.mean(segment)), 1e-6)))
    topology = 0.0
    dem_cost = 0.0
    if dem.enabled:
        elevations = dem.sample_xy(coords[:, 0], coords[:, 1])
        valid = np.isfinite(elevations[:-1]) & np.isfinite(elevations[1:]) & (segment > 1e-6)
        if valid.any():
            grades = np.abs(np.diff(elevations)[valid]) / segment[valid] * 100.0
            dem_cost = float(np.mean(np.maximum(grades - dem_max_grade_pct, 0.0) / max(dem_max_grade_pct, 1e-6)))
    total = (
        weights["ridge"] * ridge
        + weights["support"] * support_cost
        + weights["trajectory"] * trajectory
        + weights["smooth"] * smooth
        + weights["topology"] * topology
        + weights["dem"] * dem_cost
    )
    return {
        "total": float(total),
        "ridge": ridge,
        "support": support_cost,
        "trajectory": trajectory,
        "smooth": smooth,
        "topology": topology,
        "dem": dem_cost,
    }


def optimize_network_energy(
    network: NetworkData,
    support_path: Path,
    kde_path: Path,
    gravity_points: Optional[Path] = None,
    dem_path: Optional[Path] = None,
    iterations: int = 3,
    spacing_m: float = 3.0,
    search_m: float = 5.0,
    offset_step_m: float = 1.0,
    smooth_sigma: float = 0.55,
    trajectory_sample_step: int = 10,
    trajectory_max_points: int = 1_500_000,
    trajectory_scale_m: float = 12.0,
    lambda_ridge: float = 0.32,
    lambda_trajectory: float = 0.25,
    lambda_smooth: float = 0.18,
    lambda_topology: float = 0.08,
    lambda_support: float = 0.12,
    lambda_dem: float = 0.05,
    dem_max_grade_pct: float = 45.0,
    preserve_uturn: bool = True,
) -> Tuple[NetworkData, List[Dict[str, Any]], Dict[str, Any]]:
    """Topology-preserving coordinate descent over all edge centerlines.

    Endpoints are fixed at graph nodes, so graph connectivity is invariant. Internal
    vertices are moved across the local normal to minimise a weighted energy:
    KDE ridge + trajectory proximity + smoothness + support + topology + DEM grade.
    """
    net = network.copy()
    trajectory_tree, trajectory_report = load_trajectory_tree(
        gravity_points,
        sample_step=trajectory_sample_step,
        max_points=trajectory_max_points,
    )
    weights = {
        "ridge": float(lambda_ridge),
        "trajectory": float(lambda_trajectory),
        "smooth": float(lambda_smooth),
        "topology": float(lambda_topology),
        "support": float(lambda_support),
        "dem": float(lambda_dem),
    }
    total_weight = sum(max(v, 0.0) for v in weights.values())
    if total_weight <= 0:
        raise ValueError('Graph optimization weights must sum to more than 0')
    weights = {k: max(v, 0.0) / total_weight for k, v in weights.items()}
    diagnostics: List[Dict[str, Any]] = []
    offsets = np.arange(-float(search_m), float(search_m) + 0.01, max(float(offset_step_m), 0.25))
    node_xy = {int(n["node_id"]): np.asarray([float(n["x"]), float(n["y"])]) for n in net.nodes}

    with RasterBlockSampler({"support": support_path, "kde": kde_path}, cache_blocks=128) as sampler, DEMSampler(dem_path, net.crs) as dem:
        for edge in net.edges:
            original_geom = edge["geometry"]
            coords = resample_line(original_geom, spacing_m=spacing_m)
            if len(coords) < 3:
                diagnostics.append({"edge_id": int(edge["edge_id"]), "optimized": False, "reason": "too_short"})
                continue
            coords[0] = node_xy[int(edge["u"])]
            coords[-1] = node_xy[int(edge["v"])]
            before = _edge_energy(coords, sampler, trajectory_tree, trajectory_scale_m, weights, dem, dem_max_grade_pct)
            total_shift = np.zeros(len(coords), dtype=float)

            # U-turns are not skipped; their search radius is reduced to avoid changing leg topology.
            edge_search = float(search_m) * (0.55 if preserve_uturn and edge.get("uturn_protected") else 1.0)
            edge_offsets = offsets[np.abs(offsets) <= edge_search + 1e-9]
            for _ in range(max(1, int(iterations))):
                if dem.enabled:
                    elevations = dem.sample_xy(coords[:, 0], coords[:, 1])
                else:
                    elevations = np.full(len(coords), np.nan)
                updated = coords.copy()
                for i in range(1, len(coords) - 1):
                    tangent = coords[i + 1] - coords[i - 1]
                    normal = _normal(tangent)
                    if not normal.any():
                        continue
                    candidates = coords[i][None, :] + edge_offsets[:, None] * normal[None, :]
                    costs, _ = _candidate_costs(
                        candidates,
                        edge_offsets,
                        coords[i - 1],
                        coords[i],
                        coords[i + 1],
                        sampler,
                        trajectory_tree,
                        trajectory_scale_m,
                        edge_search,
                        weights,
                        dem,
                        elevations[i - 1],
                        elevations[i + 1],
                        dem_max_grade_pct,
                    )
                    best = int(np.argmin(costs))
                    updated[i] = candidates[best]
                    total_shift[i] += float(np.linalg.norm(updated[i] - coords[i]))
                coords = updated

            if smooth_sigma > 0 and len(coords) > 4:
                x = gaussian_filter1d(coords[:, 0], sigma=float(smooth_sigma), mode="nearest")
                y = gaussian_filter1d(coords[:, 1], sigma=float(smooth_sigma), mode="nearest")
                coords[1:-1, 0] = x[1:-1]
                coords[1:-1, 1] = y[1:-1]
            coords[0] = node_xy[int(edge["u"])]
            coords[-1] = node_xy[int(edge["v"])]
            refined = clean_linestring(coords)
            if refined is None:
                diagnostics.append({"edge_id": int(edge["edge_id"]), "optimized": False, "reason": "invalid_geometry"})
                continue
            after = _edge_energy(coords, sampler, trajectory_tree, trajectory_scale_m, weights, dem, dem_max_grade_pct)
            # Accept only if the aggregate energy does not meaningfully deteriorate.
            accepted = after["total"] <= before["total"] * 1.03 or before["total"] <= 1e-9
            if accepted:
                edge["geometry"] = refined
                edge["length_m"] = float(refined.length)
            edge["energy_before"] = before["total"]
            edge["energy_after"] = after["total"] if accepted else before["total"]
            edge["energy_improvement"] = before["total"] - edge["energy_after"]
            diagnostics.append({
                "edge_id": int(edge["edge_id"]),
                "optimized": bool(accepted),
                "uturn_protected": bool(edge.get("uturn_protected")),
                "point_count": int(len(coords)),
                "mean_accumulated_shift_m": float(np.mean(total_shift[1:-1])) if len(total_shift) > 2 else 0.0,
                "max_accumulated_shift_m": float(np.max(total_shift)) if len(total_shift) else 0.0,
                "energy_before": float(before["total"]),
                "energy_after_candidate": float(after["total"]),
                "energy_after_accepted": float(edge["energy_after"]),
                "energy_improvement": float(edge["energy_improvement"]),
                **{f"before_{k}": float(v) for k, v in before.items() if k != "total"},
                **{f"after_{k}": float(v) for k, v in after.items() if k != "total"},
            })

    optimized = reindex_network(net)
    valid = [d for d in diagnostics if "energy_before" in d]
    report = {
        "method": "coordinate_descent_graph_energy",
        "iterations": int(iterations),
        "spacing_m": float(spacing_m),
        "search_m": float(search_m),
        "offset_step_m": float(offset_step_m),
        "weights_normalized": weights,
        "trajectory": trajectory_report,
        "dem_enabled": bool(dem_path and Path(dem_path).exists()),
        "edges_processed": int(len(valid)),
        "edges_accepted": int(sum(bool(d.get("optimized")) for d in valid)),
        "mean_energy_before": float(np.mean([d["energy_before"] for d in valid])) if valid else 0.0,
        "mean_energy_after": float(np.mean([d["energy_after_accepted"] for d in valid])) if valid else 0.0,
        "total_energy_improvement": float(np.sum([d["energy_improvement"] for d in valid])) if valid else 0.0,
    }
    return optimized, diagnostics, report
