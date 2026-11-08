#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3: track-support-constrained protected thinning at power-of-two thresholds and 2D road geometry extraction.

Input
-----
From the S2 output directory:
- kde_independent_support_sigma*m_uint16_0_1024.tif
- independent_track_support.tif
- raw_trajectory_union.tif
- distance_to_raw_trajectory_m.tif

Method
------
1. Use only 2^k thresholds: 512, 256, 128, 64, 32, 16, 8.
2. Constrain each binary layer to the observed-track support corridor to prevent long unsupported shortcuts from Gaussian diffusion.
3. Protect the preceding higher-threshold skeleton pixels during constrained two-subiteration Zhang-Suen thinning.
4. Remove small connected components and prune short, low-support terminal branches using independent-track support.
5. Export binary and hierarchical skeletons, first-formation thresholds, vector-line GPKG and topology statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import xy as transform_xy
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import thin

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception:  # pragma: no cover
    gpd = None
    LineString = None

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PACKAGE_ROOT / "outputs" / "S5_KDE"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "S6_Skeleton"


@dataclass
class SkeletonConfig:
    max_exponent: int = 9   # 512
    min_exponent: int = 3   # 8
    support_corridor_m: float = 8.0
    min_component_pixels: int = 8
    max_thinning_iterations: int = 512
    prune_short_branch_m: float = 10.0
    prune_mean_support_lt: float = 2.0
    min_component_length_m: float = 12.0
    coverage_distance_m: float = 6.0
    simplify_tolerance_m: float = 1.0
    write_stages: bool = True
    write_gpkg: bool = True
    clear_output: bool = True


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
]


def power_thresholds(cfg: SkeletonConfig, raster_max: int) -> List[int]:
    values = [2 ** exponent for exponent in range(cfg.max_exponent, cfg.min_exponent - 1, -1)]
    return [value for value in values if value <= raster_max]


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask.astype(bool)
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def neighbor_arrays(image: np.ndarray) -> Tuple[np.ndarray, ...]:
    p = np.pad(image.astype(np.uint8), 1, mode="constant")
    p2 = p[:-2, 1:-1]
    p3 = p[:-2, 2:]
    p4 = p[1:-1, 2:]
    p5 = p[2:, 2:]
    p6 = p[2:, 1:-1]
    p7 = p[2:, :-2]
    p8 = p[1:-1, :-2]
    p9 = p[:-2, :-2]
    return p2, p3, p4, p5, p6, p7, p8, p9


def transition_count(neighbors: Tuple[np.ndarray, ...]) -> np.ndarray:
    sequence = neighbors + (neighbors[0],)
    transitions = np.zeros_like(neighbors[0], dtype=np.uint8)
    for first, second in zip(sequence[:-1], sequence[1:]):
        transitions += ((first == 0) & (second == 1)).astype(np.uint8)
    return transitions


def protected_zhang_suen(
    foreground: np.ndarray,
    protected: np.ndarray,
    max_iterations: int,
) -> Tuple[np.ndarray, int, bool]:
    image = (foreground | protected).astype(bool)
    protected = protected.astype(bool)

    for iteration in range(1, max_iterations + 1):
        removed_total = 0
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbor_arrays(image)
        neighbor_sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = transition_count((p2, p3, p4, p5, p6, p7, p8, p9))
        delete1 = (
            image & ~protected
            & (neighbor_sum >= 2) & (neighbor_sum <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        removed_total += int(delete1.sum())
        image[delete1] = False

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbor_arrays(image)
        neighbor_sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = transition_count((p2, p3, p4, p5, p6, p7, p8, p9))
        delete2 = (
            image & ~protected
            & (neighbor_sum >= 2) & (neighbor_sum <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        removed_total += int(delete2.sum())
        image[delete2] = False

        if removed_total == 0:
            return image, iteration, True
    return image, max_iterations, False


def pixel_neighbors(pixel: Tuple[int, int], pixels: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    row, col = pixel
    return [
        (row + dr, col + dc)
        for dr, dc in NEIGHBORS_8
        if (row + dr, col + dc) in pixels
    ]


def degree_map(skeleton: np.ndarray) -> np.ndarray:
    p = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    degree = np.zeros_like(skeleton, dtype=np.uint8)
    for dr, dc in NEIGHBORS_8:
        degree += p[1 + dr : 1 + dr + skeleton.shape[0], 1 + dc : 1 + dc + skeleton.shape[1]]
    degree[~skeleton] = 0
    return degree


def step_length(a: Tuple[int, int], b: Tuple[int, int], cell_size: float) -> float:
    return cell_size * (math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0)


def prune_short_low_support_branches(
    skeleton: np.ndarray,
    support: np.ndarray,
    cell_size: float,
    max_length_m: float,
    mean_support_lt: float,
) -> np.ndarray:
    result = skeleton.copy().astype(bool)
    changed = True
    while changed:
        changed = False
        pixels = set(map(tuple, np.argwhere(result)))
        if not pixels:
            break
        degree = degree_map(result)
        endpoints = [tuple(value) for value in np.argwhere(degree == 1)]
        delete_pixels: Set[Tuple[int, int]] = set()

        for endpoint in endpoints:
            if endpoint not in pixels:
                continue
            path = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while True:
                neighbors = [value for value in pixel_neighbors(current, pixels) if value != previous]
                if not neighbors:
                    break
                if len(neighbors) > 1:
                    break
                nxt = neighbors[0]
                length += step_length(current, nxt, cell_size)
                previous, current = current, nxt
                path.append(current)
                current_degree = int(degree[current])
                if current_degree != 2:
                    break
                if length > max_length_m:
                    break

            terminal_degree = int(degree[current])
            if terminal_degree < 3 or length > max_length_m:
                continue
            values = [support[row, col] for row, col in path[:-1]]
            if values and float(np.mean(values)) < mean_support_lt:
                delete_pixels.update(path[:-1])

        if delete_pixels:
            rows, cols = zip(*delete_pixels)
            result[np.asarray(rows), np.asarray(cols)] = False
            changed = True
    return result


def remove_short_components_by_length(
    skeleton: np.ndarray,
    cell_size: float,
    min_length_m: float,
) -> np.ndarray:
    labels, count = label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    result = skeleton.copy()
    for component in range(1, count + 1):
        mask = labels == component
        pixels = set(map(tuple, np.argwhere(mask)))
        edge_length = 0.0
        for pixel in pixels:
            for neighbor in pixel_neighbors(pixel, pixels):
                if neighbor > pixel:
                    edge_length += step_length(pixel, neighbor, cell_size)
        if edge_length < min_length_m:
            result[mask] = False
    return result


def graph_metrics(skeleton: np.ndarray, support: np.ndarray, cell_size: float) -> Dict[str, float]:
    pixels = set(map(tuple, np.argwhere(skeleton)))
    if not pixels:
        return {
            "skeleton_pixels": 0, "length_m": 0.0, "components": 0,
            "endpoints": 0, "junctions": 0, "cycle_rank": 0,
            "mean_support": 0.0, "median_support": 0.0,
        }
    degree = degree_map(skeleton)
    edges = 0
    length = 0.0
    for pixel in pixels:
        for neighbor in pixel_neighbors(pixel, pixels):
            if neighbor > pixel:
                edges += 1
                length += step_length(pixel, neighbor, cell_size)
    _, components = label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    values = support[skeleton]
    return {
        "skeleton_pixels": int(len(pixels)),
        "length_m": float(length),
        "components": int(components),
        "endpoints": int(np.count_nonzero(degree == 1)),
        "junctions": int(np.count_nonzero(degree >= 3)),
        "cycle_rank": int(max(edges - len(pixels) + components, 0)),
        "mean_support": float(values.mean()) if values.size else 0.0,
        "median_support": float(np.median(values)) if values.size else 0.0,
    }


def trace_segments(skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
    pixels = set(map(tuple, np.argwhere(skeleton)))
    if not pixels:
        return []
    degree = degree_map(skeleton)
    nodes = {pixel for pixel in pixels if int(degree[pixel]) != 2}
    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    segments: List[List[Tuple[int, int]]] = []

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    for node in sorted(nodes):
        for neighbor in pixel_neighbors(node, pixels):
            key = edge_key(node, neighbor)
            if key in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(key)
            previous, current = node, neighbor
            while current not in nodes:
                candidates = [value for value in pixel_neighbors(current, pixels) if value != previous]
                if not candidates:
                    break
                nxt = candidates[0]
                visited_edges.add(edge_key(current, nxt))
                path.append(nxt)
                previous, current = current, nxt
            segments.append(path)

    # Handle closed loops consisting entirely of degree=2 pixels.
    for start in sorted(pixels):
        unvisited = [
            value for value in pixel_neighbors(start, pixels)
            if edge_key(start, value) not in visited_edges
        ]
        if not unvisited:
            continue
        path = [start, unvisited[0]]
        visited_edges.add(edge_key(start, unvisited[0]))
        previous, current = start, unvisited[0]
        while True:
            candidates = [
                value for value in pixel_neighbors(current, pixels)
                if value != previous and edge_key(current, value) not in visited_edges
            ]
            if not candidates:
                break
            nxt = candidates[0]
            visited_edges.add(edge_key(current, nxt))
            path.append(nxt)
            previous, current = current, nxt
            if current == start:
                break
        if len(path) >= 2:
            segments.append(path)
    return segments


def write_gpkg(
    path: Path,
    skeleton: np.ndarray,
    transform,
    crs,
    support: np.ndarray,
    cfg: SkeletonConfig,
) -> None:
    if gpd is None or LineString is None:
        print('[WARN] geopandas/shapely unavailable; skipping GPKG', file=sys.stderr)
        return
    records = []
    for segment_id, pixels in enumerate(trace_segments(skeleton), start=1):
        coords = []
        supports = []
        for row, col in pixels:
            x, y = transform_xy(transform, row, col, offset="center")
            coords.append((float(x), float(y)))
            supports.append(float(support[row, col]))
        if len(coords) < 2:
            continue
        geometry = LineString(coords)
        if cfg.simplify_tolerance_m > 0:
            geometry = geometry.simplify(cfg.simplify_tolerance_m, preserve_topology=True)
        records.append(
            {
                "segment_id": segment_id,
                "length_m": float(geometry.length),
                "mean_support": float(np.mean(supports)),
                "min_support": float(np.min(supports)),
                "geometry": geometry,
            }
        )
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    if path.exists():
        path.unlink()
    frame.to_file(path, layer="road_skeleton_segments", driver="GPKG")


def write_raster(path: Path, array: np.ndarray, reference, nodata=0):
    profile = reference.profile.copy()
    profile.update(
        dtype=str(array.dtype), nodata=nodata, count=1,
        compress="DEFLATE", predictor=3 if array.dtype == np.float32 else 2 if array.dtype.itemsize > 1 else 1,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def write_preview(path: Path, array: np.ndarray, binary: bool = False) -> None:
    if binary:
        image = np.where(array > 0, 0, 255).astype(np.uint8)
    else:
        positive = array[array > 0]
        upper = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        image = np.round((1.0 - np.clip(array / max(upper, 1e-9), 0, 1)) * 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def parse_sigma(path: Path) -> str:
    match = re.search(r"sigma([^m]+)m", path.name)
    return match.group(1) if match else path.stem


def process_one(
    kde_path: Path,
    support_path: Path,
    union_path: Path,
    distance_path: Path,
    output_dir: Path,
    cfg: SkeletonConfig,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(kde_path) as kde_src, rasterio.open(support_path) as support_src, rasterio.open(union_path) as union_src, rasterio.open(distance_path) as distance_src:
        kde = kde_src.read(1)
        support = support_src.read(1)
        union = union_src.read(1) > 0
        distance_to_raw = distance_src.read(1).astype(np.float32)
        for src in (support_src, union_src, distance_src):
            if src.shape != kde_src.shape or not src.transform.almost_equals(kde_src.transform):
                raise ValueError('KDE and support raster dimensions or affine transforms differ')

        if not np.issubdtype(kde.dtype, np.integer):
            raise ValueError(f"""Skeleton input must be uint16 scaled KDE; actual dtype={kde.dtype}""")
        thresholds = power_thresholds(cfg, int(kde.max()))
        if not thresholds:
            raise ValueError('No power-of-two thresholds fall within the KDE value range')

        corridor = distance_to_raw <= cfg.support_corridor_m
        protected = np.zeros_like(kde, dtype=bool)
        level_raster = np.zeros_like(kde, dtype=np.uint16)
        threshold_raster = np.zeros_like(kde, dtype=np.uint16)
        stage_rows = []
        stage_count = len(thresholds)

        for stage_index, threshold in enumerate(thresholds, start=1):
            mask = (kde >= threshold) & corridor
            mask = remove_small_components(mask, cfg.min_component_pixels)
            skeleton, iterations, converged = protected_zhang_suen(
                mask, protected, cfg.max_thinning_iterations
            )
            new_pixels = skeleton & ~protected
            level_value = stage_count - stage_index + 1
            level_raster[new_pixels] = level_value
            threshold_raster[new_pixels] = threshold
            protected = skeleton
            stage_rows.append(
                {
                    "stage": stage_index,
                    "threshold": int(threshold),
                    "foreground_pixels": int(mask.sum()),
                    "skeleton_pixels": int(skeleton.sum()),
                    "new_skeleton_pixels": int(new_pixels.sum()),
                    "thinning_iterations": int(iterations),
                    "converged": bool(converged),
                }
            )
            print(
                f"[{kde_path.name} threshold={threshold}] "
                f"foreground={mask.sum():,}, skeleton={skeleton.sum():,}"
            )
            if cfg.write_stages:
                write_raster(
                    output_dir / f"stage_{stage_index:02d}_threshold_{threshold}_binary.tif",
                    skeleton.astype(np.uint8), kde_src, 0,
                )

        # Multilayer protection may retain a few adjacent pixels; perform one final global thinning pass,
        # removing only redundant pixels from the existing foreground without adding paths.
        protected = thin(protected).astype(bool)
        level_raster[~protected] = 0
        threshold_raster[~protected] = 0

        pruned = prune_short_low_support_branches(
            protected, support, abs(kde_src.transform.a),
            cfg.prune_short_branch_m, cfg.prune_mean_support_lt,
        )
        pruned = remove_short_components_by_length(
            pruned, abs(kde_src.transform.a), cfg.min_component_length_m
        )
        # Remove layer/threshold information for pruned pixels.
        level_raster[~pruned] = 0
        threshold_raster[~pruned] = 0

        binary_path = output_dir / "skeleton_binary.tif"
        level_path = output_dir / "skeleton_level.tif"
        threshold_path = output_dir / "skeleton_first_threshold_uint16.tif"
        write_raster(binary_path, pruned.astype(np.uint8), kde_src, 0)
        write_raster(level_path, level_raster.astype(np.uint16), kde_src, 0)
        write_raster(threshold_path, threshold_raster.astype(np.uint16), kde_src, 0)
        write_preview(output_dir / "skeleton_binary_preview.png", pruned, binary=True)
        write_preview(output_dir / "skeleton_level_preview.png", level_raster, binary=False)

        if cfg.write_gpkg:
            write_gpkg(
                output_dir / "road_skeleton.gpkg", pruned, kde_src.transform,
                kde_src.crs, support, cfg,
            )

        metrics = graph_metrics(pruned, support, abs(kde_src.transform.a))
        distance_to_skeleton = distance_transform_edt(~pruned).astype(np.float32) * abs(kde_src.transform.a)
        raw_pixels = union
        metrics["raw_trajectory_coverage_fraction"] = float(
            np.mean(distance_to_skeleton[raw_pixels] <= cfg.coverage_distance_m)
        ) if raw_pixels.any() else 0.0
        high_support_pixels = support >= 2
        metrics["support_ge2_coverage_fraction"] = float(
            np.mean(distance_to_skeleton[high_support_pixels] <= cfg.coverage_distance_m)
        ) if high_support_pixels.any() else 0.0
        if support.sum() > 0:
            metrics["support_weighted_coverage_fraction"] = float(
                support[distance_to_skeleton <= cfg.coverage_distance_m].sum() / support.sum()
            )
        else:
            metrics["support_weighted_coverage_fraction"] = 0.0
        metrics["skeleton_beyond_raw_6m_fraction"] = float(
            np.mean(distance_to_raw[pruned] > 6.0)
        ) if pruned.any() else 0.0
        metrics["skeleton_beyond_corridor_fraction"] = float(
            np.mean(distance_to_raw[pruned] > cfg.support_corridor_m)
        ) if pruned.any() else 0.0

        report = {
            "kde_path": str(kde_path),
            "thresholds": thresholds,
            "stages": stage_rows,
            "metrics": metrics,
            "config": asdict(cfg),
        }
        (output_dir / "skeleton_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report


def run(input_dir: Path, output_dir: Path, sigma_values: Sequence[float], cfg: SkeletonConfig) -> Dict[str, object]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if cfg.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_path = input_dir / "independent_track_support.tif"
    union_path = input_dir / "raw_trajectory_union.tif"
    distance_path = input_dir / "distance_to_raw_trajectory_m.tif"
    for path in (support_path, union_path, distance_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    results = []
    for sigma in sigma_values:
        suffix = f"{sigma:g}".replace(".", "p")
        candidates = sorted(input_dir.glob(f"kde_independent_support_sigma{suffix}m_uint16_0_*.tif"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"""sigma={sigma:g}m should have exactly one uint16 KDE; found {len(candidates)} files"""
            )
        result = process_one(
            candidates[0], support_path, union_path, distance_path,
            output_dir / f"sigma{suffix}m", cfg,
        )
        result["sigma_m"] = float(sigma)
        results.append(result)

    summary = {
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "results": results, "config": asdict(cfg),
    }
    (output_dir / "skeleton_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Track-support-constrained protected thinning at power-of-two thresholds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sigma-m", type=float, nargs="+", default=[3.0, 4.0, 6.0])
    parser.add_argument("--max-exponent", type=int, default=9)
    parser.add_argument("--min-exponent", type=int, default=3)
    parser.add_argument("--support-corridor-m", type=float, default=8.0)
    parser.add_argument("--no-stages", action="store_true")
    parser.add_argument("--no-gpkg", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SkeletonConfig(
        max_exponent=args.max_exponent,
        min_exponent=args.min_exponent,
        support_corridor_m=args.support_corridor_m,
        write_stages=not args.no_stages,
        write_gpkg=not args.no_gpkg,
    )
    try:
        run(args.input, args.output, args.sigma_m, cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
