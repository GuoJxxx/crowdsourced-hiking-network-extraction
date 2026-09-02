#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 v4：连通性保持 + 单轨迹显式保留的多阈值道路骨架提取。

改进点
------
1. 全局多阈值骨架阶段不删除低支持短支路，也不删除短连通分量；
2. U 形弯/连续回弯替换后，沿原始全局骨架与真实轨迹走廊补回最短连接，避免替换区边界断裂；
3. 根据引力结果中的 gravity_single_track_preserved / gravity_max_support_track_count，
   将单轨迹边显式栅格化并写入全部证据骨架；
4. 方向一致性仅用于“主干/辅助”分类，不再逐像元删除全部证据骨架；
5. primary 与 auxiliary 按完整图段分类，二者并集严格等于 all_evidence。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import label
from scipy.spatial import cKDTree
from shapely.geometry import LineString, mapping
from skimage.morphology import thin

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "seq_v3", HERE / "skeleton_switchback_sequence.py"
)
seq = importlib.util.module_from_spec(SPEC)
sys.modules["seq_v3"] = seq
assert SPEC.loader is not None
SPEC.loader.exec_module(seq)
rm = seq.rm
base = rm.base


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
]


def safe_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.fillna("").astype(str).str.strip().str.lower()
        .isin({"1", "true", "yes", "y"}).to_numpy(dtype=bool)
    )


def pixel_line(points: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Bresenham 像元线。"""
    if len(points) != 2:
        raise ValueError("pixel_line 需要两个端点")
    (r0, c0), (r1, c1) = points
    x0, y0, x1, y1 = c0, r0, c1, r1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out: List[Tuple[int, int]] = []
    while True:
        out.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


def rasterize_singletrack_edges(
    points: pd.DataFrame,
    shape: Tuple[int, int],
    transform,
    max_link_m: float = 25.0,
    transition_edges: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    将“从未获得多轨迹支持”的原始相邻边显式栅格化。

    为保证单轨迹支路能接回主干，在单轨迹区与多轨迹区的边界额外保留
    transition_edges 条相邻边。
    """
    required = {"clean_track_id", "clean_point_index", "x_m", "y_m"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"引力点文件缺少字段：{missing}")

    lines: List[Tuple[dict, int]] = []
    rows: List[Dict[str, object]] = []

    for clean_id, group in points.sort_values(
        ["clean_track_id", "clean_point_index"], kind="mergesort"
    ).groupby("clean_track_id", sort=False):
        group = group.reset_index(drop=True)
        xy = group[["x_m", "y_m"]].to_numpy(dtype=float)
        if len(xy) < 2:
            continue

        if "gravity_single_track_preserved" in group.columns:
            point_single = safe_bool(group["gravity_single_track_preserved"])
        elif "gravity_ever_multitrack_supported" in group.columns:
            point_single = ~safe_bool(group["gravity_ever_multitrack_supported"])
        elif "gravity_max_support_track_count" in group.columns:
            point_single = (
                pd.to_numeric(group["gravity_max_support_track_count"], errors="coerce")
                .fillna(0).to_numpy(dtype=float) < 2
            )
        else:
            # 无诊断字段时保守认为所有边均需保留；后续仍由主/辅助分类区分。
            point_single = np.ones(len(group), dtype=bool)

        edge_single = point_single[:-1] & point_single[1:]
        if transition_edges > 0 and edge_single.any():
            expanded = edge_single.copy()
            idx = np.flatnonzero(edge_single)
            for i in idx:
                lo = max(0, i - transition_edges)
                hi = min(len(expanded), i + transition_edges + 1)
                expanded[lo:hi] = True
            edge_single = expanded

        contribution = (
            str(group.get("source_file", pd.Series([""])).iloc[0])
            + "||" + str(group.get("track_id", pd.Series([""])).iloc[0])
        )

        for i, keep in enumerate(edge_single):
            if not keep:
                continue
            a, b = xy[i], xy[i + 1]
            length = float(np.linalg.norm(b - a))
            if not np.isfinite(length) or length <= 1e-9 or length > max_link_m:
                continue
            geom = LineString([tuple(a), tuple(b)])
            lines.append((mapping(geom), 1))
            rows.append({
                "clean_track_id": clean_id,
                "edge_start_index": int(group.loc[i, "clean_point_index"]),
                "edge_end_index": int(group.loc[i + 1, "clean_point_index"]),
                "length_m": length,
                "contribution_id": contribution,
            })

    if lines:
        mask = rasterize(
            lines,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        mask = thin(mask).astype(bool)
    else:
        mask = np.zeros(shape, dtype=bool)
    return mask, pd.DataFrame(rows)


def neighbors(pixel: Tuple[int, int], allowed: np.ndarray) -> Iterable[Tuple[int, int]]:
    r, c = pixel
    h, w = allowed.shape
    for dr, dc in NEIGHBORS_8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc]:
            # 禁止跨过已有正交像元产生人为对角捷径。
            if dr != 0 and dc != 0:
                if allowed[r + dr, c] and allowed[r, c + dc]:
                    continue
            yield rr, cc


def shortest_scaffold_path(
    starts: Set[Tuple[int, int]],
    targets: Set[Tuple[int, int]],
    scaffold: np.ndarray,
    max_steps: int,
) -> Optional[List[Tuple[int, int]]]:
    """在二值 scaffold 上做多源 BFS，返回不超过 max_steps 的最短路径。"""
    if not starts or not targets:
        return None
    overlap = starts & targets
    if overlap:
        return [next(iter(overlap))]

    q = deque()
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    dist: Dict[Tuple[int, int], int] = {}
    for s in starts:
        if scaffold[s]:
            q.append(s)
            prev[s] = None
            dist[s] = 0

    hit: Optional[Tuple[int, int]] = None
    while q:
        p = q.popleft()
        d = dist[p]
        if d >= max_steps:
            continue
        for nb in neighbors(p, scaffold):
            if nb in prev:
                continue
            prev[nb] = p
            dist[nb] = d + 1
            if nb in targets:
                hit = nb
                q.clear()
                break
            q.append(nb)

    if hit is None:
        return None
    path: List[Tuple[int, int]] = []
    p: Optional[Tuple[int, int]] = hit
    while p is not None:
        path.append(p)
        p = prev[p]
    path.reverse()
    return path


def component_pixel_sets(mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, Set[Tuple[int, int]]]]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sets: Dict[int, Set[Tuple[int, int]]] = {}
    for component in range(1, count + 1):
        sets[component] = set(map(tuple, np.argwhere(labels == component)))
    return labels, sets


def reconnect_on_scaffold(
    skeleton: np.ndarray,
    scaffold: np.ndarray,
    max_connector_m: float,
    cell_size_m: float,
    max_rounds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    仅沿已有全局骨架/真实轨迹种子补回最短缺口。
    不在无证据区域创造直线连接。
    """
    result = skeleton.copy().astype(bool)
    connector = np.zeros_like(result)
    records: List[Dict[str, object]] = []
    max_steps = max(1, int(math.ceil(max_connector_m / cell_size_m)))

    for round_index in range(1, max_rounds + 1):
        labels, components = component_pixel_sets(result)
        if len(components) <= 1:
            break

        # 每个当前分量在 scaffold 中可到达的最近其他分量。
        best: Optional[Tuple[int, int, List[Tuple[int, int]]]] = None
        ids = sorted(components)
        for i, a in enumerate(ids):
            starts = components[a]
            # 搜索时一次只指定一个目标分量，栅格很小，结果更可审计。
            for b in ids[i + 1:]:
                targets = components[b]
                # 欧氏下界大于最大连接距离则跳过。
                aa = np.asarray(list(starts), dtype=float)
                bb = np.asarray(list(targets), dtype=float)
                # 用包围盒间距避免全矩阵。
                dr = max(0.0, max(aa[:, 0].min() - bb[:, 0].max(), bb[:, 0].min() - aa[:, 0].max()))
                dc = max(0.0, max(aa[:, 1].min() - bb[:, 1].max(), bb[:, 1].min() - aa[:, 1].max()))
                if math.hypot(dr, dc) > max_steps:
                    continue
                path = shortest_scaffold_path(starts, targets, scaffold, max_steps)
                if path is None:
                    continue
                if best is None or len(path) < len(best[2]):
                    best = (a, b, path)

        if best is None:
            break
        a, b, path = best
        for p in path:
            result[p] = True
            connector[p] = True
        records.append({
            "round": round_index,
            "component_a": a,
            "component_b": b,
            "connector_pixels": len(path),
            "connector_length_m_upper": max(0, len(path) - 1) * cell_size_m,
        })

    # 不再全局 thin；连接路径本身为 1 像元，原骨架也是 1 像元。
    return result, connector, pd.DataFrame(records)


def sample_alignment_counts(
    skeleton: np.ndarray,
    points: pd.DataFrame,
    transform,
    cfg,
) -> Dict[Tuple[int, int], int]:
    sxy, sang, scid = rm.build_samples(points, cfg)
    if len(sxy) == 0:
        return {}
    tree = cKDTree(sxy)
    pixels = set(map(tuple, np.argwhere(skeleton)))
    counts: Dict[Tuple[int, int], int] = {}

    for p in pixels:
        ns = rm.graph_neighbors_clean(p, pixels)
        if len(ns) >= 2:
            arr = np.asarray([[q[1] - p[1], -(q[0] - p[0])] for q in ns], dtype=float)
            covariance = arr.T @ arr
            values, vectors = np.linalg.eigh(covariance)
            v = vectors[:, int(np.argmax(values))]
        elif len(ns) == 1:
            v = np.asarray([ns[0][1] - p[1], -(ns[0][0] - p[0])], dtype=float)
        else:
            v = np.asarray([1.0, 0.0])
        theta = math.atan2(v[1], v[0])
        x, y = transform * (p[1] + 0.5, p[0] + 0.5)
        candidate_ids = tree.query_ball_point([x, y], cfg.direction_query_radius_m)
        best: Dict[object, Tuple[float, float]] = {}
        for index in candidate_ids:
            distance = float(np.linalg.norm(sxy[index] - [x, y]))
            cid = scid[index]
            difference = abs(math.degrees(math.atan2(
                math.sin(2 * (sang[index] - theta)),
                math.cos(2 * (sang[index] - theta)),
            )) / 2.0)
            if cid not in best or distance < best[cid][0]:
                best[cid] = (distance, difference)
        counts[p] = sum(
            1 for distance, difference in best.values()
            if distance <= cfg.direction_max_nearest_m and difference <= 35.0
        )
    return counts


def classify_complete_segments(
    skeleton: np.ndarray,
    singletrack_seed: np.ndarray,
    points: pd.DataFrame,
    support: np.ndarray,
    transform,
    cfg,
    primary_aligned_mean: float,
    primary_ge2_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """按完整图段分类，不对选中的 primary 再单独细化。"""
    counts = sample_alignment_counts(skeleton, points, transform, cfg)
    paths = rm.trace_clean(skeleton)
    primary = np.zeros_like(skeleton)
    auxiliary = np.zeros_like(skeleton)
    rows: List[Dict[str, object]] = []

    for segment_id, path in enumerate(paths, start=1):
        aligned = np.asarray([counts.get(pixel, 0) for pixel in path], dtype=float)
        supports = np.asarray([support[pixel] for pixel in path], dtype=float)
        explicit_single_fraction = float(np.mean([singletrack_seed[p] for p in path]))
        length_m = sum(
            base.step_length(a, b, abs(transform.a)) for a, b in zip(path[:-1], path[1:])
        )
        aligned_mean = float(aligned.mean()) if len(aligned) else 0.0
        aligned_ge2 = float(np.mean(aligned >= 2)) if len(aligned) else 0.0
        support_mean = float(supports.mean()) if len(supports) else 0.0

        # 显式单轨迹段优先归入辅助层，但绝不删除。
        is_explicit_single = explicit_single_fraction >= 0.25 and aligned_ge2 < 0.5
        is_primary = (
            not is_explicit_single
            and (
                aligned_mean >= primary_aligned_mean
                or aligned_ge2 >= primary_ge2_fraction
                or support_mean >= 2.0
            )
        )
        target = primary if is_primary else auxiliary
        for pixel in path:
            target[pixel] = True

        rows.append({
            "segment_id": segment_id,
            "length_m": length_m,
            "aligned_track_mean": aligned_mean,
            "aligned_ge2_fraction": aligned_ge2,
            "support_mean": support_mean,
            "explicit_singletrack_fraction": explicit_single_fraction,
            "class": "primary" if is_primary else "auxiliary_singletrack",
        })

    # 共享节点写入相邻两层，确保各自矢量化时端点不后退一个像元。
    degree = base.degree_map(skeleton)
    nodes = skeleton & (degree != 2)
    for r, c in map(tuple, np.argwhere(nodes)):
        nearby_primary = any(
            primary[rr, cc]
            for rr, cc in base.pixel_neighbors((r, c), set(map(tuple, np.argwhere(skeleton))))
        )
        nearby_auxiliary = any(
            auxiliary[rr, cc]
            for rr, cc in base.pixel_neighbors((r, c), set(map(tuple, np.argwhere(skeleton))))
        )
        if nearby_primary:
            primary[r, c] = True
        if nearby_auxiliary:
            auxiliary[r, c] = True

    if not np.array_equal(primary | auxiliary, skeleton):
        missing = skeleton & ~(primary | auxiliary)
        auxiliary |= missing
    return primary, auxiliary, pd.DataFrame(rows)


def coverage_fraction(source: np.ndarray, target: np.ndarray, distance_pixels: int = 1) -> float:
    if not source.any():
        return 1.0
    from scipy.ndimage import binary_dilation
    covered = binary_dilation(target, iterations=distance_pixels)
    return float(np.mean(covered[source]))


def run(args: argparse.Namespace) -> None:
    cfg = rm.Config()
    cfg.final_branch_prune_m = 0.0
    cfg.final_branch_support_lt = -1.0
    cfg.min_component_length_m = 0.0

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(args.gravity_points, low_memory=False)
    consensus = gpd.read_file(args.uturn_gpkg, layer="uturn_consensus_lines")

    base_suffix = f"{args.base_sigma_m:g}".replace(".", "p")
    uturn_suffix = f"{args.uturn_sigma_m:g}".replace(".", "p")
    candidates = sorted(args.kde_dir.glob(
        f"kde_independent_support_sigma{base_suffix}m_adaptive_uturn{uturn_suffix}m_uint16_0_*.tif"
    ))
    if not candidates:
        raise FileNotFoundError("未找到自适应 uint16 KDE")
    kde_path = candidates[0]
    support_path = args.kde_dir / "independent_track_support.tif"
    union_path = args.kde_dir / "raw_trajectory_union.tif"
    distance_path = args.kde_dir / "distance_to_raw_trajectory_m.tif"

    global_dir = args.output / "global_multithreshold_unpruned"
    global_cfg = base.SkeletonConfig(
        max_exponent=9,
        min_exponent=args.min_exponent,
        support_corridor_m=args.support_corridor_m,
        min_component_pixels=1,
        prune_short_branch_m=0.0,
        prune_mean_support_lt=-1.0,
        min_component_length_m=0.0,
        write_stages=True,
        write_gpkg=True,
        clear_output=True,
    )
    base.process_one(
        kde_path, support_path, union_path, distance_path, global_dir, global_cfg
    )
    global_path = global_dir / "skeleton_binary.tif"

    with (
        rasterio.open(kde_path) as kde_src,
        rasterio.open(support_path) as support_src,
        rasterio.open(union_path) as union_src,
        rasterio.open(global_path) as global_src,
    ):
        kde = kde_src.read(1)
        support = support_src.read(1)
        raw_union = union_src.read(1) > 0
        global_skeleton = global_src.read(1) > 0
        transform = kde_src.transform
        cell_size = abs(transform.a)

        # 单轨迹边显式写入。
        single_seed, single_edge_table = rasterize_singletrack_edges(
            points,
            global_skeleton.shape,
            transform,
            max_link_m=cfg.max_track_link_m,
            transition_edges=args.singletrack_transition_edges,
        )

        # U 形弯与连续回弯替换；不执行逐像元方向删除。
        after_clusters, inserted_clusters, removed_clusters, cluster_records, cluster_zones = (
            rm.replace_clusters(global_skeleton, points, consensus, kde, support, transform, cfg)
        )
        after_sequences, inserted_sequences, removed_sequences, sequence_records, sequence_zones = (
            seq.replace_pairs(
                after_clusters, points, kde, support, transform, cfg,
                args.min_pair_tracks, args.pair_buffer_m,
            )
        )

        evidence_before_connect = thin(after_sequences | single_seed).astype(bool)
        scaffold = global_skeleton | inserted_clusters | inserted_sequences | single_seed | raw_union
        all_evidence, connectors, connector_table = reconnect_on_scaffold(
            evidence_before_connect,
            scaffold,
            args.max_connector_m,
            cell_size,
        )
        # 仅删除并行冗余像元；thin 保持已有连通关系。
        all_evidence = thin(all_evidence).astype(bool)

        primary, auxiliary, segment_table = classify_complete_segments(
            all_evidence,
            single_seed,
            points,
            support,
            transform,
            cfg,
            args.primary_aligned_mean,
            args.primary_ge2_fraction,
        )
        # 单轨迹辅助层是“显式证据层”，允许与主干在汇入口处重叠。
        # 将单轨迹种子映射到最终一像元骨架附近，避免 thin 后像元平移造成漏标。
        from scipy.ndimage import binary_dilation
        auxiliary |= all_evidence & binary_dilation(single_seed, iterations=2)

        outputs = {
            "skeleton_all_evidence_connected": all_evidence,
            "skeleton_primary_multitrack": primary,
            "skeleton_auxiliary_singletrack": auxiliary,
            "skeleton_singletrack_injected": single_seed,
            "skeleton_connectors_added": connectors,
            "cluster_routes_inserted": inserted_clusters,
            "sequence_routes_inserted": inserted_sequences,
            "cluster_routes_removed": removed_clusters,
            "sequence_routes_removed": removed_sequences,
        }
        for name, array in outputs.items():
            base.write_raster(
                args.output / f"{name}.tif",
                array.astype(np.uint8),
                kde_src,
                0,
            )

        rm.write_clean_gpkg(
            args.output / "road_skeleton_complete.gpkg",
            all_evidence, transform, kde_src.crs, support, cfg,
        )
        rm.write_clean_gpkg(
            args.output / "road_skeleton_primary_multitrack.gpkg",
            primary, transform, kde_src.crs, support, cfg,
        )
        rm.write_clean_gpkg(
            args.output / "road_skeleton_auxiliary_singletrack.gpkg",
            auxiliary, transform, kde_src.crs, support, cfg,
        )

        single_edge_table.to_csv(
            args.output / "singletrack_edges_injected.csv", index=False, encoding="utf-8-sig"
        )
        connector_table.to_csv(
            args.output / "connectivity_repair.csv", index=False, encoding="utf-8-sig"
        )
        segment_table.to_csv(
            args.output / "primary_auxiliary_segments.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([
            {k: v for k, v in record.items() if k != "geometry"}
            for record in cluster_records
        ]).to_csv(args.output / "uturn_route_diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {k: v for k, v in record.items() if k != "geometry"}
            for record in sequence_records
        ]).to_csv(args.output / "sequence_route_diagnostics.csv", index=False, encoding="utf-8-sig")

        _, global_components = label(global_skeleton, structure=np.ones((3, 3), dtype=np.uint8))
        _, before_components = label(evidence_before_connect, structure=np.ones((3, 3), dtype=np.uint8))
        _, final_components = label(all_evidence, structure=np.ones((3, 3), dtype=np.uint8))

        report = {
            "global_unpruned_metrics": base.graph_metrics(global_skeleton, support, cell_size),
            "before_connect_metrics": base.graph_metrics(evidence_before_connect, support, cell_size),
            "all_evidence_metrics": base.graph_metrics(all_evidence, support, cell_size),
            "primary_metrics": base.graph_metrics(primary, support, cell_size),
            "auxiliary_metrics": base.graph_metrics(auxiliary, support, cell_size),
            "global_components": int(global_components),
            "components_before_repair": int(before_components),
            "components_after_repair": int(final_components),
            "singletrack_seed_pixels": int(single_seed.sum()),
            "singletrack_seed_coverage_within_1px": coverage_fraction(single_seed, all_evidence, 1),
            "singletrack_seed_coverage_primary_within_1px": coverage_fraction(single_seed, primary, 1),
            "singletrack_seed_coverage_auxiliary_within_1px": coverage_fraction(single_seed, auxiliary, 1),
            "connector_pixels_added": int(connectors.sum()),
            "connector_count": int(len(connector_table)),
            "uturn_routes": int(len(cluster_records)),
            "sequence_routes": int(len(sequence_records)),
            "partition_exact": bool(np.array_equal(primary | auxiliary, all_evidence)),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="连通性保持与单轨迹显式保留的道路骨架提取",
    )
    parser.add_argument("--kde-dir", type=Path, required=True)
    parser.add_argument("--gravity-points", type=Path, required=True)
    parser.add_argument("--uturn-gpkg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sigma-m", type=float, default=4.0)
    parser.add_argument("--uturn-sigma-m", type=float, default=2.0)
    parser.add_argument("--min-exponent", type=int, default=2, help="2^2=4")
    parser.add_argument("--support-corridor-m", type=float, default=8.0)
    parser.add_argument("--min-pair-tracks", type=int, default=3)
    parser.add_argument("--pair-buffer-m", type=float, default=6.0)
    parser.add_argument("--max-connector-m", type=float, default=24.0)
    parser.add_argument("--singletrack-transition-edges", type=int, default=0)
    parser.add_argument("--primary-aligned-mean", type=float, default=1.75)
    parser.add_argument("--primary-ge2-fraction", type=float, default=0.45)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
