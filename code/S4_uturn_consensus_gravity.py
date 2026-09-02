#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4：活动区感知的 U 形弯共识约束拓扑保护引力集中（560 m 分块流式查询版）。

相较普通局部引力，本脚本先从多条清洗轨迹中识别 U 形弯事件，聚类同一物理弯道，
构建多轨迹共识中心线；随后在普通路段执行法向引力集中，在 U 形弯区域按物理腿
禁止跨腿吸引，并将弯道点稳健锚定到共识曲线。引力阶段不再删除任何单轨迹点或边；
边支持度只作为诊断字段输出，全部原始轨迹拓扑均进入后续 KDE。

本版针对大范围数据做了低内存改造：
1. 引力候选查询按 560 m × 560 m 物理瓦片分批执行，避免一次性生成 N × K 邻居矩阵；
2. 候选样本 KDTree 仍覆盖全域，因此瓦片边界不会截断引力来源，算法结果与原全量查询等价；
3. 输出 gravity_tile_id / gravity_tile_col / gravity_tile_row，便于与 560 m 分块 KDE 对应检查；
4. 正式输出仍保留全部输入点、全部 clean_track_id 和全部相邻边。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "pipeline_modules" / "gravity_base.py"
if not BASE_PATH.is_file():
    raise FileNotFoundError(f"缺少基础模块：{BASE_PATH}")
spec = importlib.util.spec_from_file_location("gravity_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

from pipeline_modules.uturn_consensus import (  # noqa: E402
    UTurnConfig,
    annotate_points,
    apply_consensus_anchor,
    cluster_events,
    clusters_table,
    config_dict,
    detect_events,
    events_table,
)

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
except Exception:  # pragma: no cover
    gpd = None
    LineString = None
    Point = None

DEFAULT_INPUT = Path("outputs/S2_Clean/all_cleaned_points.csv")
DEFAULT_OUTPUT = Path("outputs/S4_Gravity")
DEFAULT_TILE_SIZE_M = 560.0


# =============================================================================
# 输入 CRS 与活动区候选校验
# =============================================================================
def _normalise_crs(value: str) -> str:
    """把 EPSG/WKT/PROJ 表达统一为 pyproj 可识别的标准字符串。"""
    return base.CRS.from_user_input(str(value)).to_string()


def resolve_projected_crs(input_path: Path, requested: str) -> str:
    """解析 --projected-crs。

    - requested=auto：优先继承输入 CSV 中唯一的 metric_crs；
    - 显式指定：若输入已有 metric_crs 且不一致，立即报错；
    - 只有经纬度且 requested=auto：根据轨迹中位经纬度推断本地 UTM。
    """
    input_path = input_path.expanduser().resolve()
    header = pd.read_csv(input_path, encoding="utf-8-sig", nrows=0)
    columns = set(header.columns)

    metric_values: set[str] = set()
    if "metric_crs" in columns:
        for chunk in pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            usecols=["metric_crs"],
            chunksize=250_000,
            low_memory=False,
        ):
            for value in chunk["metric_crs"].dropna().astype(str).str.strip().unique():
                if value:
                    metric_values.add(_normalise_crs(value))
            if len(metric_values) > 1:
                break

    requested_text = str(requested).strip()
    if requested_text.lower() != "auto":
        resolved = _normalise_crs(requested_text)
        if metric_values and metric_values != {resolved}:
            raise ValueError(
                "显式 --projected-crs 与输入 metric_crs 不一致："
                f"requested={resolved}, input={sorted(metric_values)}。"
                "脚本不会只修改 CRS 标签；请使用 --projected-crs auto，"
                "或在上游真正重新投影 x_m/y_m。"
            )
        return resolved

    if len(metric_values) == 1:
        return next(iter(metric_values))
    if len(metric_values) > 1:
        raise ValueError(
            f"输入存在多个 metric_crs：{sorted(metric_values)}，无法自动选择。"
        )

    if {"x_m", "y_m"}.issubset(columns):
        raise ValueError(
            "--projected-crs auto 需要输入包含唯一 metric_crs；"
            "当前已有 x_m/y_m，但没有可用 metric_crs。"
        )

    if {"longitude", "latitude"}.issubset(columns):
        lon_parts = []
        lat_parts = []
        remaining = 200_000
        for chunk in pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            usecols=["longitude", "latitude"],
            chunksize=100_000,
            low_memory=False,
        ):
            lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(float)
            lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(float)
            valid = np.isfinite(lon) & np.isfinite(lat)
            if valid.any():
                lon_parts.append(lon[valid][:remaining])
                lat_parts.append(lat[valid][:remaining])
                remaining -= min(int(valid.sum()), remaining)
            if remaining <= 0:
                break
        if not lon_parts:
            raise ValueError("无法从 longitude/latitude 推断投影坐标系")
        median_lon = float(np.median(np.concatenate(lon_parts)))
        median_lat = float(np.median(np.concatenate(lat_parts)))
        zone = int((median_lon + 180.0) // 6.0) + 1
        epsg = 32600 + zone if median_lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"

    raise ValueError("输入既没有 metric_crs，也没有可用于推断的 longitude/latitude")


def _safe_bool(series: pd.Series, default: bool = True) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).to_numpy(dtype=bool)
    text = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


def apply_activity_input_guard(
    frame: pd.DataFrame,
    min_road_weight: float,
) -> Dict[str, object]:
    """校验 S4 输入确实是 S3 输出的道路候选点。

    S4 的基础 GravityConfig 不包含 min_road_weight。该阈值属于活动区流程
    的输入一致性条件：S3 已负责输出 road_candidate_points.csv，S4 在此
    复核，避免低权重停车场轨迹重新成为引力源。
    """
    threshold = float(min_road_weight)
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("--min-road-weight 必须位于 [0, 1]")

    if "road_extraction_weight" in frame.columns:
        weights = pd.to_numeric(
            frame["road_extraction_weight"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
    else:
        weights = np.ones(len(frame), dtype=float)
        print(
            "[WARNING] 输入缺少 road_extraction_weight；按权重 1.0 处理。"
            "正式活动区流程应使用 S3 的 road_candidate_points.csv。",
            flush=True,
        )

    if "mainroad_candidate" in frame.columns:
        mainroad = _safe_bool(frame["mainroad_candidate"], default=False)
    else:
        mainroad = np.ones(len(frame), dtype=bool)
        print(
            "[WARNING] 输入缺少 mainroad_candidate；仅按 road_extraction_weight 校验。",
            flush=True,
        )

    eligible = np.isfinite(weights) & (weights >= threshold) & mainroad
    existing = frame.get(
        "gravity_query_eligible", pd.Series(True, index=frame.index)
    )
    existing_eligible = _safe_bool(existing, default=True)
    eligible &= existing_eligible

    rejected = int((~eligible).sum())
    if rejected:
        raise ValueError(
            f"S4 输入中仍有 {rejected:,}/{len(frame):,} 行不满足 "
            f"road_extraction_weight >= {threshold:g} 且 mainroad_candidate=True。"
            "请确认输入使用 S3 输出的 road_candidate_points.csv，"
            "不要使用 all_cleaned_points_activity.csv。"
        )

    frame["gravity_query_eligible"] = eligible
    return {
        "min_road_weight": threshold,
        "eligible_rows": int(eligible.sum()),
        "eligible_ratio": float(eligible.mean()) if len(eligible) else 0.0,
        "input_has_road_extraction_weight": "road_extraction_weight" in frame.columns,
        "input_has_mainroad_candidate": "mainroad_candidate" in frame.columns,
    }


# =============================================================================
# 560 m 瓦片工具
# =============================================================================
def infer_tile_origin(
    x: np.ndarray,
    y: np.ndarray,
    tile_size_m: float,
    tile_origin_x: Optional[float],
    tile_origin_y: Optional[float],
) -> Tuple[float, float]:
    """推断 560 m 瓦片左下角原点。可用 KDE 报告中的 bounds[0], bounds[1] 手动指定。"""
    if tile_size_m <= 0:
        raise ValueError("gravity_tile_size_m 必须大于 0")
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("无法推断瓦片原点：没有有效 x/y 坐标")
    ox = (
        float(tile_origin_x)
        if tile_origin_x is not None
        else math.floor(float(np.nanmin(x[valid])) / tile_size_m) * tile_size_m
    )
    oy = (
        float(tile_origin_y)
        if tile_origin_y is not None
        else math.floor(float(np.nanmin(y[valid])) / tile_size_m) * tile_size_m
    )
    return ox, oy


def tile_indices(
    x: np.ndarray,
    y: np.ndarray,
    tile_size_m: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    col = np.floor((x - origin_x) / tile_size_m).astype(np.int64)
    row = np.floor((y - origin_y) / tile_size_m).astype(np.int64)
    return col, row


def tile_id_from_col_row(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.asarray([f"T{int(c):06d}_{int(r):06d}" for c, r in zip(col, row)], dtype=object)


def iter_tile_position_batches(
    x: np.ndarray,
    y: np.ndarray,
    tile_size_m: float,
    origin_x: float,
    origin_y: float,
    max_points_per_batch: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """按物理瓦片返回点位置；稠密瓦片再切成若干小批。"""
    col, row = tile_indices(x, y, tile_size_m, origin_x, origin_y)
    keys = pd.DataFrame(
        {
            "pos": np.arange(len(x), dtype=np.int64),
            "tile_col": col,
            "tile_row": row,
        }
    )
    keys.sort_values(["tile_col", "tile_row", "pos"], inplace=True, kind="mergesort")
    limit = max(1, int(max_points_per_batch))
    for (tile_col, tile_row), group in keys.groupby(["tile_col", "tile_row"], sort=False):
        positions = group["pos"].to_numpy(dtype=np.int64)
        for start in range(0, len(positions), limit):
            yield int(tile_col), int(tile_row), positions[start : start + limit]


def attach_tile_fields(
    frame: pd.DataFrame,
    tile_size_m: float,
    origin_x: float,
    origin_y: float,
    x_field: str = "x_m",
    y_field: str = "y_m",
) -> pd.DataFrame:
    """为输出点附加与 560 m KDE 可对应的瓦片编号。"""
    output = frame.copy()
    x = pd.to_numeric(output[x_field], errors="coerce").to_numpy(float)
    y = pd.to_numeric(output[y_field], errors="coerce").to_numpy(float)
    col, row = tile_indices(x, y, tile_size_m, origin_x, origin_y)
    output["gravity_tile_size_m"] = float(tile_size_m)
    output["gravity_tile_origin_x_m"] = float(origin_x)
    output["gravity_tile_origin_y_m"] = float(origin_y)
    output["gravity_tile_col"] = col.astype(np.int64)
    output["gravity_tile_row"] = row.astype(np.int64)
    output["gravity_tile_id"] = tile_id_from_col_row(col, row)
    return output


def build_tile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "gravity_tile_id" not in frame.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for tile_id, group in frame.groupby("gravity_tile_id", sort=True):
        rows.append(
            {
                "gravity_tile_id": tile_id,
                "gravity_tile_col": int(group["gravity_tile_col"].iloc[0]),
                "gravity_tile_row": int(group["gravity_tile_row"].iloc[0]),
                "point_count": int(len(group)),
                "track_count": int(group["clean_track_id"].nunique()) if "clean_track_id" in group.columns else 0,
                "source_track_count": int(group["track_id"].nunique()) if "track_id" in group.columns else 0,
                "min_x_m": float(pd.to_numeric(group["x_m"], errors="coerce").min()),
                "max_x_m": float(pd.to_numeric(group["x_m"], errors="coerce").max()),
                "min_y_m": float(pd.to_numeric(group["y_m"], errors="coerce").min()),
                "max_y_m": float(pd.to_numeric(group["y_m"], errors="coerce").max()),
                "mean_gravity_displacement_m": float(pd.to_numeric(group.get("gravity_displacement_m", np.nan), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 分块引力核心：只改变查询组织方式，不改变引力判据和位移公式
# =============================================================================
def gravity_adjust_tiled(
    df: pd.DataFrame,
    cfg: "base.GravityConfig",
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    tile_origin_x: Optional[float] = None,
    tile_origin_y: Optional[float] = None,
    query_batch_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    560 m 瓦片流式引力集中。

    与基础模块 base.gravity_adjust 的判据保持一致，区别只是：
    - 原版一次性 tree.query(query_xy) 得到全体点的 N×K 邻居矩阵；
    - 本版按 560 m 瓦片分批 query，同一个全域 KDTree 作为候选源，避免瓦片边界截断。
    """
    original_xy = df[["x_m", "y_m"]].to_numpy(float)
    origin_x, origin_y = infer_tile_origin(
        original_xy[:, 0], original_xy[:, 1], tile_size_m, tile_origin_x, tile_origin_y
    )

    df["gravity_original_x_m"] = original_xy[:, 0]
    df["gravity_original_y_m"] = original_xy[:, 1]
    df["gravity_x_m"] = original_xy[:, 0]
    df["gravity_y_m"] = original_xy[:, 1]

    iteration_rows: List[Dict[str, float]] = []
    cosine_limit = math.cos(math.radians(cfg.max_axis_angle_deg))
    max_support_seen = np.zeros(len(df), dtype=np.int16)
    ever_move_eligible = np.zeros(len(df), dtype=bool)

    for iteration in range(1, cfg.iterations + 1):
        start_time = time.time()
        tangents, curvature = base.current_tangents_and_curvature(df)
        sample_xy, sample_tangent, sample_contribution, sample_hairpin = (
            base.resample_all_tracks(df, cfg.sample_spacing_m)
        )
        if len(sample_xy) == 0:
            raise RuntimeError("没有可用于引力计算的重采样点")
        tree = base.cKDTree(sample_xy)
        query_xy = df[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
        query_eligible = df.get(
            "gravity_query_eligible", pd.Series(True, index=df.index)
        ).fillna(False).to_numpy(bool)

        # 与原版保持一致：K 近邻上限仍为 64/16 逻辑。
        k_neighbors = min(64, max(16, len(sample_xy)))
        raw_move = np.zeros(len(df), dtype=float)
        support_count = np.zeros(len(df), dtype=np.int16)
        cross_leg_rejected = np.zeros(len(df), dtype=np.int16)

        tile_count = 0
        max_batch_points = 0
        for _tile_col, _tile_row, positions in iter_tile_position_batches(
            query_xy[:, 0],
            query_xy[:, 1],
            tile_size_m,
            origin_x,
            origin_y,
            query_batch_size,
        ):
            tile_count += 1
            max_batch_points = max(max_batch_points, int(len(positions)))
            q = query_xy[positions]
            distances, neighbor_ids = tree.query(
                q,
                k=k_neighbors,
                distance_upper_bound=cfg.search_radius_m,
                workers=-1,
            )
            if distances.ndim == 1:
                distances = distances[:, None]
                neighbor_ids = neighbor_ids[:, None]

            for local_index, point_index in enumerate(positions):
                if not query_eligible[point_index]:
                    continue
                tangent = tangents[point_index]
                if np.linalg.norm(tangent) < 0.5:
                    continue
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                contribution_id = df["_contribution_id"].iat[point_index]
                target_leg = int(df["gravity_hairpin_leg_code"].iat[point_index])
                best_by_track: Dict[str, Tuple[float, float, float, float]] = {}

                for distance, sample_index in zip(distances[local_index], neighbor_ids[local_index]):
                    if not np.isfinite(distance) or sample_index >= len(sample_xy):
                        continue
                    if sample_contribution[sample_index] == contribution_id:
                        continue
                    axis_dot = abs(float(np.dot(tangent, sample_tangent[sample_index])))
                    if axis_dot < cosine_limit:
                        continue
                    delta = sample_xy[sample_index] - query_xy[point_index]
                    along_offset = abs(float(np.dot(delta, tangent)))
                    lateral_offset = float(np.dot(delta, normal))
                    if along_offset > cfg.max_along_offset_m:
                        continue
                    if abs(lateral_offset) > cfg.max_lateral_offset_m:
                        continue
                    candidate_leg = int(sample_hairpin[sample_index])
                    if target_leg != 0 and candidate_leg != target_leg:
                        cross_leg_rejected[point_index] += 1
                        continue

                    source_id = str(sample_contribution[sample_index])
                    score = float(distance) + 0.3 * along_offset
                    current = best_by_track.get(source_id)
                    candidate = (score, lateral_offset, axis_dot, float(distance))
                    if current is None or score < current[0]:
                        best_by_track[source_id] = candidate

                if len(best_by_track) < cfg.min_support_tracks:
                    continue

                lateral_values = np.asarray([value[1] for value in best_by_track.values()])
                direction_values = np.asarray([value[2] for value in best_by_track.values()])
                distance_values = np.asarray([value[3] for value in best_by_track.values()])
                weights = (
                    np.exp(-((distance_values / max(cfg.search_radius_m * 0.55, 1e-6)) ** 2))
                    * np.maximum(
                        0.05,
                        (direction_values - cosine_limit) / max(1.0 - cosine_limit, 1e-9),
                    )
                )
                raw_move[point_index] = base.weighted_median(lateral_values, weights)
                support_count[point_index] = len(best_by_track)

        # 沿轨迹方向平滑横向位移，避免单点抖动。
        smoothed_move = raw_move.copy()
        for _, indices in df.groupby("clean_track_id", sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            valid_weight = (
                (support_count[positions] >= cfg.min_support_tracks)
                & query_eligible[positions]
            ).astype(float)
            numerator = base.gaussian_filter1d(
                raw_move[positions] * valid_weight,
                cfg.displacement_smooth_sigma_points,
                mode="nearest",
            )
            denominator = base.gaussian_filter1d(
                valid_weight,
                cfg.displacement_smooth_sigma_points,
                mode="nearest",
            )
            smoothed_move[positions] = np.where(
                denominator > 0.15,
                numerator / np.maximum(denominator, 1e-6),
                0.0,
            )

        # 支持数不足的点严格冻结，避免单轨迹路段被相邻多轨迹位移污染。
        move_eligible = (support_count >= cfg.min_support_tracks) & query_eligible
        smoothed_move[~move_eligible] = 0.0
        max_support_seen = np.maximum(max_support_seen, support_count)
        ever_move_eligible |= move_eligible

        curvature_factor = np.clip(1.0 - curvature / 120.0, 0.30, 1.0)
        requested_move = np.clip(
            cfg.gravity_alpha * smoothed_move * curvature_factor,
            -cfg.max_step_m,
            cfg.max_step_m,
        )
        current_xy = query_xy
        normals = np.c_[-tangents[:, 1], tangents[:, 0]]
        proposed_xy = current_xy + requested_move[:, None] * normals

        total_displacement = proposed_xy - original_xy
        total_distance = np.linalg.norm(total_displacement, axis=1)
        total_limit = np.where(
            df["gravity_hairpin_protected"].to_numpy(bool),
            cfg.max_total_move_hairpin_m,
            cfg.max_total_move_m,
        )
        over_limit = total_distance > total_limit
        proposed_xy[over_limit] = (
            original_xy[over_limit]
            + total_displacement[over_limit]
            / total_distance[over_limit, None]
            * total_limit[over_limit, None]
        )

        step_distance = np.linalg.norm(proposed_xy - current_xy, axis=1)
        df["gravity_x_m"] = proposed_xy[:, 0]
        df["gravity_y_m"] = proposed_xy[:, 1]
        df["gravity_support_track_count"] = support_count
        df["gravity_cross_leg_rejected_count"] = cross_leg_rejected

        iteration_rows.append(
            {
                "iteration": iteration,
                "elapsed_s": time.time() - start_time,
                "moved_point_count": int(np.count_nonzero(step_distance > 1e-4)),
                "mean_step_move_m": float(step_distance.mean()),
                "median_step_move_m": float(np.median(step_distance)),
                "p95_step_move_m": float(np.percentile(step_distance, 95)),
                "max_step_move_m": float(step_distance.max(initial=0.0)),
                "support_ge_min_fraction": float(
                    np.mean(support_count >= cfg.min_support_tracks)
                ),
                "cross_leg_rejected_total": int(cross_leg_rejected.sum()),
                "tile_size_m": float(tile_size_m),
                "tile_count_batches": int(tile_count),
                "max_query_batch_points": int(max_batch_points),
            }
        )
        print(
            f"[GRAVITY {iteration}/{cfg.iterations}] "
            f"moved={iteration_rows[-1]['moved_point_count']:,}, "
            f"mean={iteration_rows[-1]['mean_step_move_m']:.3f} m, "
            f"p95={iteration_rows[-1]['p95_step_move_m']:.3f} m, "
            f"support={iteration_rows[-1]['support_ge_min_fraction']:.1%}, "
            f"tiles={tile_count:,}, max_batch={max_batch_points:,}"
        )

    final_xy = df[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
    df["gravity_displacement_m"] = np.linalg.norm(final_xy - original_xy, axis=1)
    df["gravity_max_support_track_count"] = max_support_seen
    df["gravity_ever_multitrack_supported"] = ever_move_eligible
    df["gravity_single_track_preserved"] = ~ever_move_eligible

    frozen = ~ever_move_eligible
    if frozen.any():
        frozen_error = np.linalg.norm(final_xy[frozen] - original_xy[frozen], axis=1)
        if float(frozen_error.max(initial=0.0)) > 1e-8:
            raise AssertionError(
                "单轨迹保护失败：存在支持不足点发生位移，"
                f"最大位移={float(frozen_error.max()):.9f} m"
            )

    tile_info = {
        "gravity_tiling_enabled": True,
        "gravity_tile_size_m": float(tile_size_m),
        "gravity_tile_origin_x_m": float(origin_x),
        "gravity_tile_origin_y_m": float(origin_y),
        "gravity_query_batch_size": int(query_batch_size),
    }
    return df, pd.DataFrame(iteration_rows), tile_info


def override_uturn_edges(
    adjusted: pd.DataFrame,
    edge_diagnostics: pd.DataFrame,
    reliable_by_track: Dict[str, np.ndarray],
    min_cluster_support: int,
) -> int:
    """多轨迹共识 U 形弯内部边不再按普通局部支持阈值误删。"""
    if "uturn_consensus_override" not in edge_diagnostics.columns:
        edge_diagnostics["uturn_consensus_override"] = False
    else:
        edge_diagnostics["uturn_consensus_override"] = (
            edge_diagnostics["uturn_consensus_override"]
            .fillna(False)
            .astype(bool)
        )
    kept = 0
    row_lookup = {
        (str(row.clean_track_id), int(row.edge_index)): index
        for index, row in edge_diagnostics.iterrows()
    }
    for track_id, group in adjusted.groupby("clean_track_id", sort=False):
        group = group.sort_values("clean_point_index")
        cluster = group["gravity_uturn_cluster_id"].fillna(0).to_numpy(int)
        support = group["gravity_uturn_track_support"].fillna(0).to_numpy(int)
        reliable = reliable_by_track[str(track_id)]
        for edge_index in range(len(reliable)):
            same_cluster = cluster[edge_index] > 0 and cluster[edge_index] == cluster[edge_index + 1]
            enough_support = min(support[edge_index], support[edge_index + 1]) >= min_cluster_support
            if same_cluster and enough_support and not reliable[edge_index]:
                reliable[edge_index] = True
                kept += 1
                row_index = row_lookup.get((str(track_id), edge_index))
                if row_index is not None:
                    edge_diagnostics.at[row_index, "reliable"] = True
                    edge_diagnostics.at[row_index, "uturn_consensus_override"] = True
        reliable_by_track[str(track_id)] = reliable
    edge_diagnostics["uturn_consensus_override"] = (
        edge_diagnostics["uturn_consensus_override"].fillna(False).astype(bool)
    )
    return kept


def write_uturn_gpkg(
    clusters: List[Dict[str, object]],
    output_path: Path,
    crs: str,
    protection_buffer_m: float,
) -> None:
    if gpd is None or LineString is None or Point is None:
        print("[WARN] geopandas/shapely 不可用，跳过 U 形弯 GPKG", file=sys.stderr)
        return
    if output_path.exists():
        output_path.unlink()
    line_records = []
    point_records = []
    zone_records = []
    for cluster in clusters:
        line = LineString(np.asarray(cluster["consensus"], dtype=float))
        common = {
            "cluster_id": int(cluster["cluster_id"]),
            "track_support": int(cluster["track_support"]),
            "mean_score": float(cluster["mean_score"]),
            "median_rms_m": float(cluster["median_curve_rms_m"]),
            "length_m": float(cluster["consensus_length_m"]),
            "chord_m": float(cluster["consensus_chord_m"]),
            "bulge_m": float(cluster["consensus_bulge_m"]),
        }
        line_records.append({**common, "geometry": line})
        point_records.append(
            {
                **common,
                "geometry": Point(float(cluster["apex_x"]), float(cluster["apex_y"])),
            }
        )
        zone_records.append(
            {**common, "geometry": line.buffer(protection_buffer_m, cap_style=2, join_style=2)}
        )
    if not line_records:
        return
    gpd.GeoDataFrame(line_records, geometry="geometry", crs=crs).to_file(
        output_path, layer="uturn_consensus_lines", driver="GPKG"
    )
    gpd.GeoDataFrame(point_records, geometry="geometry", crs=crs).to_file(
        output_path, layer="uturn_apex_points", driver="GPKG"
    )
    gpd.GeoDataFrame(zone_records, geometry="geometry", crs=crs).to_file(
        output_path, layer="uturn_protection_zones", driver="GPKG"
    )


def run(
    input_path: Path,
    output_dir: Path,
    gravity_cfg: "base.GravityConfig",
    uturn_cfg: UTurnConfig,
    gravity_tile_size_m: float = DEFAULT_TILE_SIZE_M,
    gravity_tile_origin_x: Optional[float] = None,
    gravity_tile_origin_y: Optional[float] = None,
    gravity_query_batch_size: int = 50_000,
    min_road_weight: float = 0.35,
) -> Dict[str, object]:
    base.validate_config(gravity_cfg)
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if gravity_cfg.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    original = base.prepare_input(input_path, gravity_cfg)
    activity_guard = apply_activity_input_guard(original, min_road_weight)
    print(
        f"[INPUT] rows={len(original):,}, tracks={original.clean_track_id.nunique():,}, "
        f"independent={original._contribution_id.nunique():,}, "
        f"road_eligible={activity_guard['eligible_ratio']:.1%}, "
        f"crs={gravity_cfg.projected_crs}"
    )

    all_events = detect_events(original, uturn_cfg)
    clusters, rejected_uturn_structures = cluster_events(all_events, uturn_cfg, return_rejected=True)
    print(
        f"[U-TURN] candidates={len(all_events):,}, accepted_clusters={len(clusters):,}, "
        f"supported_tracks={sum(int(c['track_support']) for c in clusters):,}"
    )
    working = annotate_points(original.copy(), clusters, uturn_cfg)
    # 基础引力模块只依赖以下三个急弯字段。这里以可靠的事件物理腿编码替换旧的
    # “任意空间自接近”编码，避免将整段近平行道路误标为回头弯两腿。
    working["gravity_hairpin_leg_code"] = (
        working["gravity_uturn_leg_code"].fillna(0).astype(np.int8)
    )
    working["gravity_hairpin_pair_distance_m"] = np.nan
    working["gravity_hairpin_protected"] = working["gravity_uturn_cluster_id"].gt(0)

    adjusted, iteration_log, tile_info = gravity_adjust_tiled(
        working,
        gravity_cfg,
        tile_size_m=gravity_tile_size_m,
        tile_origin_x=gravity_tile_origin_x,
        tile_origin_y=gravity_tile_origin_y,
        query_batch_size=gravity_query_batch_size,
    )
    adjusted = apply_consensus_anchor(adjusted, uturn_cfg)
    current_xy = adjusted[["gravity_x_m", "gravity_y_m"]].to_numpy(float)
    original_xy = adjusted[["gravity_original_x_m", "gravity_original_y_m"]].to_numpy(float)
    adjusted["gravity_displacement_m"] = np.linalg.norm(current_xy - original_xy, axis=1)

    edge_diagnostics, reliable_by_track = base.evaluate_edges(adjusted, gravity_cfg)
    override_count = override_uturn_edges(
        adjusted, edge_diagnostics, reliable_by_track, uturn_cfg.min_cluster_tracks
    )

    # 正式输出必须保留全部输入点和全部轨迹边；支持度只做诊断。
    final = base.finalize_preserving_all(adjusted, edge_diagnostics, gravity_cfg)
    final = attach_tile_fields(
        final,
        tile_size_m=float(tile_info["gravity_tile_size_m"]),
        origin_x=float(tile_info["gravity_tile_origin_x_m"]),
        origin_y=float(tile_info["gravity_tile_origin_y_m"]),
    )
    adjusted = attach_tile_fields(
        adjusted,
        tile_size_m=float(tile_info["gravity_tile_size_m"]),
        origin_x=float(tile_info["gravity_tile_origin_x_m"]),
        origin_y=float(tile_info["gravity_tile_origin_y_m"]),
        x_field="gravity_x_m",
        y_field="gravity_y_m",
    )
    tile_summary = build_tile_summary(final)

    # 旧的“多轨迹高置信子集”仅作为诊断文件，不作为 KDE 输入。
    supported_subset = None
    if gravity_cfg.write_supported_subset_diagnostic:
        try:
            supported_subset = base.split_reliable_parts(
                adjusted, reliable_by_track, gravity_cfg
            )
            supported_subset.drop(columns=["_contribution_id"], errors="ignore").to_csv(
                diagnostics_dir / "multitrack_supported_subset_only.csv",
                index=False, encoding="utf-8-sig"
            )
        except RuntimeError:
            supported_subset = None

    output_csv = output_dir / "all_cleaned_points.csv"
    final.drop(columns=["_contribution_id"], errors="ignore").to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    iteration_log.to_csv(
        diagnostics_dir / "gravity_iteration_log.csv", index=False, encoding="utf-8-sig"
    )
    edge_diagnostics.to_csv(
        diagnostics_dir / "edge_support_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    adjusted.drop(columns=["_contribution_id"], errors="ignore").to_csv(
        diagnostics_dir / "all_adjusted_points_preserved.csv", index=False, encoding="utf-8-sig"
    )
    events_table(all_events).to_csv(
        diagnostics_dir / "uturn_events.csv", index=False, encoding="utf-8-sig"
    )
    clusters_table(clusters).to_csv(
        diagnostics_dir / "uturn_clusters.csv", index=False, encoding="utf-8-sig"
    )
    accepted_structure_validation = clusters_table(clusters).copy()
    if not accepted_structure_validation.empty:
        accepted_structure_validation["validation_action"] = "accepted"
    rejected_structure_validation = pd.DataFrame(rejected_uturn_structures)
    if not rejected_structure_validation.empty:
        rejected_structure_validation["validation_action"] = "rejected"
    structure_validation = pd.concat(
        [accepted_structure_validation, rejected_structure_validation],
        ignore_index=True, sort=False,
    )
    structure_validation.to_csv(
        diagnostics_dir / "uturn_structure_validation.csv", index=False, encoding="utf-8-sig"
    )
    tile_summary.to_csv(
        diagnostics_dir / "gravity_tile_summary.csv", index=False, encoding="utf-8-sig"
    )

    if gravity_cfg.write_gpkg:
        base.write_gpkg(
            original, final, output_dir / "gravity_adjusted_tracks.gpkg", gravity_cfg.projected_crs
        )
        write_uturn_gpkg(
            clusters,
            output_dir / "uturn_consensus.gpkg",
            gravity_cfg.projected_crs,
            uturn_cfg.protection_buffer_m,
        )

    removed_edges = edge_diagnostics.loc[~edge_diagnostics["reliable"]]

    if len(final) != len(original):
        raise AssertionError(f"点保留失败：输入 {len(original)}，输出 {len(final)}")
    if final["clean_track_id"].nunique() != original["clean_track_id"].nunique():
        raise AssertionError("轨迹段保留失败：clean_track_id 数量发生变化")

    report = {
        "input": str(input_path),
        "output": str(output_csv),
        "input_rows": int(len(original)),
        "input_tracks": int(original.clean_track_id.nunique()),
        "independent_trajectories": int(original._contribution_id.nunique()),
        "road_eligible_rows": int(original.get("gravity_query_eligible", pd.Series(True, index=original.index)).fillna(False).sum()),
        "road_eligible_ratio": float(original.get("gravity_query_eligible", pd.Series(True, index=original.index)).fillna(False).mean()),
        "resolved_projected_crs": str(gravity_cfg.projected_crs),
        "activity_input_guard": activity_guard,
        "uturn_event_candidates": int(len(all_events)),
        "uturn_clusters": int(len(clusters)),
        "uturn_structure_rejected": int(len(rejected_uturn_structures)),
        "uturn_cluster_track_support": [int(cluster["track_support"]) for cluster in clusters],
        "uturn_anchored_points": int(adjusted["gravity_uturn_cluster_id"].gt(0).sum()),
        "uturn_edges_preserved_by_consensus": int(override_count),
        "mean_total_move_m": float(adjusted.gravity_displacement_m.mean()),
        "median_total_move_m": float(adjusted.gravity_displacement_m.median()),
        "p95_total_move_m": float(adjusted.gravity_displacement_m.quantile(0.95)),
        "max_total_move_m": float(adjusted.gravity_displacement_m.max()),
        "all_edges": int(len(edge_diagnostics)),
        "removed_edges": 0,
        "removed_edge_length_m": 0.0,
        "diagnostic_low_support_edges": int(len(removed_edges)),
        "diagnostic_low_support_edge_length_m": float(removed_edges.edge_length_m.sum()),
        "single_track_preserved_points": int(adjusted["gravity_single_track_preserved"].sum()),
        "output_rows": int(len(final)),
        "output_tracks": int(final.clean_track_id.nunique()),
        "row_retention_ratio": float(len(final) / max(len(original), 1)),
        "track_retention_ratio": float(final.clean_track_id.nunique() / max(original.clean_track_id.nunique(), 1)),
        "supported_subset_rows_diagnostic": int(len(supported_subset)) if supported_subset is not None else 0,
        "gravity_config": asdict(gravity_cfg),
        "gravity_tile_config": tile_info,
        "gravity_tile_count": int(len(tile_summary)),
        "uturn_config": config_dict(uturn_cfg),
    }
    (output_dir / "gravity_uturn_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] output_rows={len(final):,}/{len(original):,}, "
        f"tracks={final.clean_track_id.nunique():,}/{original.clean_track_id.nunique():,}, "
        f"uturn_clusters={len(clusters)}, deleted_edges=0, "
        f"diagnostic_low_support_edges={len(removed_edges):,}, "
        f"gravity_tiles={len(tile_summary):,}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="保留全部单轨迹路段的 U 形弯共识约束引力集中（560 m 分块查询版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--projected-crs", default="auto", help="默认继承输入 metric_crs；显式指定不一致时直接报错")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--search-radius-m", type=float, default=15.0)
    parser.add_argument("--min-support-tracks", type=int, default=2)
    parser.add_argument("--min-road-weight", type=float, default=0.35,
                        help="复核 S3 road_candidate_points.csv 的最低道路权重；不满足时直接报错，避免活动区轨迹重新进入引力")
    parser.add_argument("--edge-support-radius-m", type=float, default=8.0)
    parser.add_argument("--edge-min-supported-fraction", type=float, default=0.45)
    parser.add_argument("--uturn-min-cluster-tracks", type=int, default=3)
    parser.add_argument("--uturn-min-distinct-original-tracks", type=int, default=3)
    parser.add_argument("--uturn-max-leg-axis-difference-deg", type=float, default=25.0)
    parser.add_argument("--uturn-min-opposite-direction-score", type=float, default=0.70)
    parser.add_argument("--uturn-min-apex-turn-concentration", type=float, default=0.50)
    parser.add_argument("--uturn-min-leg-separation-m", type=float, default=4.0)
    parser.add_argument("--uturn-max-leg-separation-m", type=float, default=35.0)
    parser.add_argument("--uturn-min-leg-straightness", type=float, default=0.86)
    parser.add_argument("--uturn-min-leg-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--uturn-max-secondary-turn-peak-ratio", type=float, default=0.60)
    parser.add_argument("--uturn-activity-min-tracks", type=int, default=4)
    parser.add_argument("--uturn-activity-min-apex-concentration", type=float, default=0.68)
    parser.add_argument("--uturn-activity-max-secondary-peak-ratio", type=float, default=0.50)
    parser.add_argument("--uturn-activity-min-leg-straightness", type=float, default=0.90)
    parser.add_argument("--uturn-activity-min-leg-overlap-ratio", type=float, default=0.55)
    parser.add_argument("--uturn-anchor-max-move-m", type=float, default=18.0)
    parser.add_argument("--uturn-protection-buffer-m", type=float, default=8.0)
    parser.add_argument("--gravity-tile-size-m", type=float, default=DEFAULT_TILE_SIZE_M,
                        help="引力查询物理瓦片边长，建议与 KDE 的 --tile-size-m 保持一致")
    parser.add_argument("--gravity-tile-origin-x", type=float, default=None,
                        help="可选：瓦片左下角原点 X；若需严格对应 KDE，可填 KDE report 中 bounds[0]")
    parser.add_argument("--gravity-tile-origin-y", type=float, default=None,
                        help="可选：瓦片左下角原点 Y；若需严格对应 KDE，可填 KDE report 中 bounds[1]")
    parser.add_argument("--gravity-query-batch-size", type=int, default=50_000,
                        help="单个瓦片内仍过密时的最大查询点数批大小")
    parser.add_argument("--no-gpkg", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_crs = resolve_projected_crs(args.input, args.projected_crs)
    gravity_cfg = base.GravityConfig(
        projected_crs=resolved_crs,
        iterations=args.iterations,
        search_radius_m=args.search_radius_m,
        min_support_tracks=args.min_support_tracks,
        edge_support_radius_m=args.edge_support_radius_m,
        edge_min_supported_fraction=args.edge_min_supported_fraction,
        write_gpkg=not args.no_gpkg,
    )
    uturn_cfg = UTurnConfig(
        min_cluster_tracks=args.uturn_min_cluster_tracks,
        min_distinct_original_tracks=args.uturn_min_distinct_original_tracks,
        max_leg_axis_difference_deg=args.uturn_max_leg_axis_difference_deg,
        min_opposite_direction_score=args.uturn_min_opposite_direction_score,
        min_apex_turn_concentration=args.uturn_min_apex_turn_concentration,
        min_leg_separation_m=args.uturn_min_leg_separation_m,
        max_leg_separation_m=args.uturn_max_leg_separation_m,
        min_leg_straightness=args.uturn_min_leg_straightness,
        min_leg_longitudinal_overlap_ratio=args.uturn_min_leg_overlap_ratio,
        max_secondary_turn_peak_ratio=args.uturn_max_secondary_turn_peak_ratio,
        activity_min_distinct_original_tracks=args.uturn_activity_min_tracks,
        activity_min_apex_turn_concentration=args.uturn_activity_min_apex_concentration,
        activity_max_secondary_turn_peak_ratio=args.uturn_activity_max_secondary_peak_ratio,
        activity_min_leg_straightness=args.uturn_activity_min_leg_straightness,
        activity_min_leg_overlap_ratio=args.uturn_activity_min_leg_overlap_ratio,
        anchor_max_move_m=args.uturn_anchor_max_move_m,
        protection_buffer_m=args.uturn_protection_buffer_m,
    )
    try:
        run(
            args.input,
            args.output,
            gravity_cfg,
            uturn_cfg,
            gravity_tile_size_m=args.gravity_tile_size_m,
            gravity_tile_origin_x=args.gravity_tile_origin_x,
            gravity_tile_origin_y=args.gravity_tile_origin_y,
            gravity_query_batch_size=args.gravity_query_batch_size,
            min_road_weight=args.min_road_weight,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
