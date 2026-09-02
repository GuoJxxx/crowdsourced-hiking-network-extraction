#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7 路网清理与平滑：保持已验证拓扑，最终执行整条道路链连续平滑。"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from pipeline_modules.network_cleanup.cleaner import (
    clean_network, compress_collinear_paths, constrained_smooth_and_simplify, smooth_final_road_chains,
    collapse_directional_digital_loops, remove_digital_artifact_loops,
)
from pipeline_modules.network_cleanup.conservative import (
    audit_approved_directional_collapses, audit_approved_micro_gaps, audit_geometry_against_primary_raster,
    baseline_union, conservative_prune_network,
)
from pipeline_modules.network_cleanup.dem_constraints import attach_dem_attributes
from pipeline_modules.network_cleanup.gap_repair import repair_micro_gaps
from pipeline_modules.network_cleanup.s7_force_refine import (
    audit_s7_force_gaps,
    repair_s4_track_gaps,
    repair_strong_direction_gaps,
    resolve_polygon_cycles,
)
from pipeline_modules.network_cleanup.junction_cleanup import (
    audit_junction_repairs,
    rollback_failed_junction_repairs,
    prune_short_isolated_components,
    prune_short_leaf_chains,
    repair_endpoint_clusters,
    repair_endpoint_to_edge_junctions,
)
from pipeline_modules.network_cleanup.connectivity_repair import (
    audit_component_stitching, connectivity_audit, repair_fragmented_components, unresolved_endpoints,
)
from pipeline_modules.network_cleanup.graph_builder import build_network_from_rasters
from pipeline_modules.network_cleanup.hmm_matcher import match_tracks_hmm
from pipeline_modules.network_cleanup.models import NetworkData
from pipeline_modules.network_cleanup.optimizer import merge_degree_two
from pipeline_modules.network_cleanup.outputs import network_metrics, write_diagnostic_gpkg, write_network_bundle
from pipeline_modules.network_cleanup.tile_io import read_tile_config
from pipeline_modules.network_cleanup.track_evidence import attach_track_evidence, match_track_evidence
from pipeline_modules.network_cleanup.raw_track_topology import (
    RawTrackWitnessIndex, audit_raw_track_gaps, prune_parallel_false_bridges,
    prune_raw_negative_connectors, repair_raw_track_gaps,
    repair_high_support_uncovered_corridors, rollback_failed_raw_track_gap_repairs,
)
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles
from pipeline_modules.network_cleanup.uturn_topology import prune_uturn_shortcut_connectors
from pipeline_modules.network_cleanup.topology_utils import (
    clear_geometry_numeric_anomalies, geometry_numeric_anomalies,
)


def existing(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    return path if path.exists() else None


def stage(output: Path, number: int, name: str) -> Path:
    path = output / f"{number:02d}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(records, path: Path) -> None:
    rows = [{k: v for k, v in r.items() if k not in ("geometry", "_pixels")} for r in records]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_stage_network(network: NetworkData, directory: Path, basename: str, args) -> None:
    """Avoid repeated multi-gigabyte GPKG/SQLite writes unless explicitly requested."""
    metrics = network_metrics(network)
    (directory / f"{basename}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_intermediate_bundles:
        write_network_bundle(network, directory, basename)
    else:
        print("    [output] intermediate full bundle skipped (use --write-intermediate-bundles to enable)", flush=True)


def match(network: NetworkData, args, output_points: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    points = existing(args.gravity_points)
    if args.track_matcher == "none":
        return match_track_evidence(network, None)
    if args.track_matcher == "nearest":
        return match_track_evidence(
            network, points, args.match_max_distance_m, args.match_step,
            write_points_path=output_points,
        )
    return match_tracks_hmm(
        network, points,
        max_distance_m=args.match_max_distance_m,
        sample_step=args.match_step,
        max_candidates=args.hmm_max_candidates,
        emission_sigma_m=args.hmm_emission_sigma_m,
        heading_sigma_deg=args.hmm_heading_sigma_deg,
        heading_weight=args.hmm_heading_weight,
        transition_beta_m=args.hmm_transition_beta_m,
        transition_jump_penalty=args.hmm_transition_jump_penalty,
        transition_mode=args.hmm_transition_mode,
        transition_exact_radius_m=args.hmm_transition_exact_radius_m,
        min_road_weight=args.min_road_weight,
        write_points_path=output_points,
        progress_every_tracks=args.match_progress_every_tracks,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="S7 路网清理与平滑：最终连通性救援、残余伪环收尾与整条道路链连续平滑。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--auxiliary", type=Path)
    p.add_argument("--rescued", type=Path)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--kde", type=Path, required=True)
    p.add_argument("--distance", type=Path)
    p.add_argument("--gravity-points", type=Path)
    p.add_argument("--raw-tracks", type=Path,
                   help="S1 observed_tracks目录；原始轨迹仅作为拓扑连续性见证，不直接重建整网")
    p.add_argument("--uturn-gpkg", type=Path)
    p.add_argument("--activity-zones", type=Path)
    p.add_argument("--activity-layer", default="activity_suppression_zones")
    p.add_argument("--dem", type=Path)
    p.add_argument("--tile-report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile-size-m", type=float, default=560.0)
    p.add_argument("--keep-output", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="保留输出目录并复用第1阶段像元/证据检查点")
    p.add_argument("--stage1-progress-every-tiles", type=int, default=100,
                   help="第1阶段每读取多少个560m瓦片输出一次进度")
    p.add_argument("--stage1-chunk-pixels", type=int, default=1000000,
                   help="第1阶段矢量化邻接搜索的像元分块大小")
    p.add_argument("--stage1-progress-every-edges", type=int, default=10000,
                   help="第1阶段骨架链追踪每处理多少个出口输出一次进度")
    p.add_argument("--stage1-max-node-blob-pixels", type=int, default=64,
                   help="可折叠为单个拓扑节点的最大连续节点像元数；超出时拆分，避免射线")
    p.add_argument("--stage1-max-node-blob-span-cells", type=int, default=8,
                   help="可折叠节点团在行列方向的最大跨度；超出时拆分，避免长距离质心连接")
    p.add_argument("--geometry-source", choices=["primary", "primary_rescued", "union"], default="primary",
                   help="最终路网几何来源；primary 最严格，辅助/救援层仅作为证据")
    p.add_argument("--uturn-core-buffer-m", type=float, default=4.0,
                   help="U 型弯共识中心线核心保护半径；核心内不删除数字环边，且采用更保守平滑")
    p.add_argument("--uturn-context-buffer-m", type=float, default=10.0,
                   help="U 型弯上下文保护半径；用于剪枝保护，但不再阻止非核心数字环清理")
    p.add_argument("--no-thin-geometry", action="store_true",
                   help="关闭像元薄化；不建议，可能保留2x2矩形环")
    p.add_argument("--thin-halo-pixels", type=int, default=2,
                   help="分块薄化的边界重叠像元数")
    p.add_argument("--artifact-loop-max-m", type=float, default=50.0,
                   help="强制打断的栅格数字小环最大周长；仅删除现有边，不创建弦线")
    p.add_argument("--artifact-loop-max-span-m", type=float, default=18.0,
                   help="强制打断的栅格数字小环最大外包跨度")
    p.add_argument("--artifact-loop-rounds", type=int, default=8,
                   help="数字小环批量清理轮数")
    p.add_argument("--artifact-fallback-loop-max-m", type=float, default=28.0,
                   help="无正向重复证据的几何回退删环最大周长")
    p.add_argument("--artifact-fallback-loop-max-span-m", type=float, default=8.0,
                   help="无正向重复证据的几何回退删环最大跨度；仅处理很小栅格伪环")
    p.add_argument("--no-directional-loop-collapse", action="store_true",
                   help="关闭方向对角塌缩，仅使用弱边删环回退；默认启用方向塌缩")
    p.add_argument("--directional-loop-max-angle-deg", type=float, default=30.0,
                   help="数字环两端外部道路方向与中心线弦的最大夹角")
    p.add_argument("--directional-loop-evidence-distance-m", type=float, default=6.0,
                   help="方向对角候选距原始轨迹不超过该值视为距离证据")
    p.add_argument("--directional-loop-min-evidence-fraction", type=float, default=0.60,
                   help="方向对角候选沿线最少有效轨迹/KDE证据比例")
    p.add_argument("--disable-polygon-loop-resolver", action="store_true",
                   help="兼容旧参数：关闭任意多边形锁环解析")
    p.add_argument("--allow-polygon-loop-resolver", action="store_true",
                   help="显式启用任意多边形删环；默认关闭以保留真实道路环路")
    p.add_argument("--polygon-loop-max-perimeter-m", type=float, default=180.0,
                   help="任意形状锁环进入解析的最大周长")
    p.add_argument("--polygon-loop-max-span-m", type=float, default=60.0,
                   help="任意形状锁环进入解析的最大外包跨度")
    p.add_argument("--polygon-loop-max-area-m2", type=float, default=1800.0,
                   help="任意形状锁环进入解析的最大平面面积")
    p.add_argument("--polygon-loop-force-perimeter-m", type=float, default=100.0,
                   help="在无U形弯核心/批准补断时可强制处理的锁环最大周长")
    p.add_argument("--polygon-loop-force-span-m", type=float, default=35.0,
                   help="强制处理锁环的最大外包跨度")
    p.add_argument("--polygon-loop-force-area-m2", type=float, default=650.0,
                   help="强制处理锁环的最大面积")
    p.add_argument("--polygon-loop-thin-p90-width-m", type=float, default=14.0,
                   help="双门点环两分支P90距离不超过该值时视为重复双线")
    p.add_argument("--polygon-loop-thin-max-width-m", type=float, default=20.0,
                   help="双门点环两分支最大距离不超过该值时视为重复双线")
    p.add_argument("--polygon-loop-weak-branch-ratio", type=float, default=0.82,
                   help="非强制范围内，弱分支证据/强分支证据低于该比例时删除弱分支")
    p.add_argument("--polygon-loop-rounds", type=int, default=12,
                   help="任意多边形锁环迭代解析轮数")
    p.add_argument("--disable-polygon-loop-force", action="store_true",
                   help="关闭合理尺度内的强制去锁环，只保留证据/细长重复双线判断")
    p.add_argument("--disable-junction-repair", action="store_true",
                   help="关闭交叉口拓扑修复")
    p.add_argument("--junction-edge-snap-max-m", type=float, default=8.0,
                   help="断开的支路端点吸附到邻近道路边的最大距离")
    p.add_argument("--junction-edge-node-merge-m", type=float, default=2.5,
                   help="投影点距目标道路既有节点小于该值时直接共用节点，避免微小边")
    p.add_argument("--junction-edge-max-angle-deg", type=float, default=65.0,
                   help="端点原道路外向方向与交叉口连接方向最大夹角")
    p.add_argument("--junction-min-evidence-fraction", type=float, default=0.25,
                   help="交叉口补连沿线最低KDE/support/distance证据覆盖比例")
    p.add_argument("--junction-evidence-distance-m", type=float, default=10.0,
                   help="交叉口补连中距原始轨迹小于该值视为距离证据")
    p.add_argument("--junction-edge-rounds", type=int, default=4,
                   help="端点到道路边交叉口修复轮数")
    p.add_argument("--junction-edge-max-repairs", type=int, default=5000,
                   help="端点到道路边交叉口修复最大数量")

    p.add_argument("--junction-cluster-radius-m", type=float, default=8.0,
                   help="多个断端共同形成X/Y交叉口时的最大端点聚类半径")
    p.add_argument("--junction-cluster-center-search-m", type=float, default=4.0,
                   help="交叉口中心在KDE/support证据上的局部搜索半径")
    p.add_argument("--junction-cluster-max-angle-deg", type=float, default=75.0,
                   help="聚类交叉口每条支路指向共同中心的最大方向差")
    p.add_argument("--junction-cluster-min-evidence-fraction", type=float, default=0.20,
                   help="聚类交叉口每条补连臂最低证据比例")
    p.add_argument("--junction-cluster-min-size", type=int, default=3,
                   help="创建共享hub所需的最少有效断端数；2个断端交给micro-gap/S4轨迹补断处理")
    p.add_argument("--junction-cluster-max-size", type=int, default=6,
                   help="一次共同交叉口最多合并的断端数量")
    p.add_argument("--junction-cluster-max-repairs", type=int, default=2000,
                   help="聚类交叉口最多新增的补连臂数量")

    p.add_argument("--disable-cycle-closure", action="store_true",
                   help="关闭同一连通分量内的证据约束闭环补连；默认允许真实路网闭环")
    p.add_argument("--cycle-closure-min-detour-ratio", type=float, default=1.8,
                   help="同分量补连要求既有绕行长度/新连接长度的最小比例")
    p.add_argument("--cycle-closure-min-detour-excess-m", type=float, default=8.0,
                   help="同分量补连要求既有绕行比新连接至少多出的长度")
    p.add_argument("--disable-component-stitching", action="store_true",
                   help="关闭局部破碎分量拼接")
    p.add_argument("--component-stitch-max-m", type=float, default=28.0,
                   help="局部破碎分量端点之间允许自动拼接的最大距离")
    p.add_argument("--component-stitch-max-angle-deg", type=float, default=35.0,
                   help="短距离分量拼接两端最大方向偏差")
    p.add_argument("--component-stitch-min-evidence-fraction", type=float, default=0.35,
                   help="短距离分量拼接最低KDE/support/轨迹距离证据覆盖比例")
    p.add_argument("--component-stitch-long-threshold-m", type=float, default=15.0,
                   help="超过该距离后启用更严格的长缺口拼接门限")
    p.add_argument("--component-stitch-long-max-angle-deg", type=float, default=22.5,
                   help="长距离分量拼接最大方向偏差")
    p.add_argument("--component-stitch-long-min-evidence-fraction", type=float, default=0.65,
                   help="长距离分量拼接最低连续证据覆盖比例")
    p.add_argument("--component-stitch-evidence-distance-m", type=float, default=10.0,
                   help="分量拼接的轨迹距离证据阈值")
    p.add_argument("--component-stitch-max-repairs", type=int, default=10000,
                   help="分量拼接最多新增连接数量")

    p.add_argument("--isolated-component-hard-max-m", type=float, default=25.0,
                   help="完全孤立且总长度不超过该值的分量直接删除，不再因primary来源保留")
    p.add_argument("--isolated-component-weak-max-m", type=float, default=60.0,
                   help="完全孤立、无轨迹匹配且低置信度时允许删除的最大分量长度")
    p.add_argument("--isolated-component-weak-max-tracks", type=int, default=0,
                   help="弱孤立分量删除允许的最大匹配独立轨迹数")
    p.add_argument("--isolated-component-weak-max-confidence", type=float, default=0.42,
                   help="弱孤立分量删除的最大边置信度")

    p.add_argument("--junction-spur-hard-max-m", type=float, default=6.0,
                   help="交叉口处端点到交叉节点的极短毛刺链直接删除")
    p.add_argument("--junction-spur-weak-max-m", type=float, default=22.0,
                   help="交叉口处无轨迹支持且低置信度的短毛刺链最大删除长度")
    p.add_argument("--junction-spur-weak-max-tracks", type=int, default=0,
                   help="弱交叉口毛刺删除允许的最大匹配轨迹数")
    p.add_argument("--junction-spur-weak-max-confidence", type=float, default=0.40,
                   help="弱交叉口毛刺删除的最大边置信度")
    p.add_argument("--junction-spur-rounds", type=int, default=4,
                   help="交叉口短毛刺迭代删除轮数")
    p.add_argument("--disable-raw-track-gap-repair", action="store_true",
                   help="关闭原始轨迹拓扑见证补断")
    p.add_argument("--raw-track-max-step-m", type=float, default=100.0,
                   help="原始轨迹见证允许的最大相邻步长；仅用于剔除明显跳点")
    p.add_argument("--raw-track-max-speed-mps", type=float, default=8.0,
                   help="原始轨迹见证的最大合理速度，超过则断开见证链")
    p.add_argument("--raw-track-max-time-gap-s", type=float, default=1800.0,
                   help="原始轨迹见证的最大连续时间间隔")
    p.add_argument("--raw-track-isolated-spike-bridge-m", type=float, default=80.0,
                   help="允许跳过单个明显GPS尖峰后恢复原轨迹连续性的最大A-C距离")
    p.add_argument("--raw-track-gap-max-endpoint-distance-m", type=float, default=40.0,
                   help="原始同轨迹见证的断端-断端最大直线距离")
    p.add_argument("--raw-track-gap-max-path-m", type=float, default=65.0,
                   help="原始轨迹见证子路径最大实际长度")
    p.add_argument("--raw-track-gap-snap-m", type=float, default=7.0,
                   help="网络断端吸附到原始轨迹见证的最大距离")
    p.add_argument("--raw-track-gap-max-angle-deg", type=float, default=65.0,
                   help="原始轨迹见证补断与两侧道路方向的最大夹角")
    p.add_argument("--raw-track-gap-min-witness-tracks", type=int, default=1,
                   help="补断至少需要多少条独立原始轨迹连续见证")
    p.add_argument("--cycle-route-family-min-tracks", type=int, default=2,
                   help="0/1/多门点环保留为真实环路时，完整绕环原始轨迹族的最小独立贡献数")
    p.add_argument("--raw-track-gap-max-path-ratio", type=float, default=3.0,
                   help="原始见证子路径长度/断口直线距离最大比例")
    p.add_argument("--raw-track-gap-corridor-m", type=float, default=10.0,
                   help="原始见证子路径相对候选连接的P90最大偏离")
    p.add_argument("--raw-track-gap-max-repairs", type=int, default=5000,
                   help="原始轨迹见证补断最大数量")
    p.add_argument("--raw-coverage-max-endpoint-distance-m", type=float, default=90.0,
                   help="最终轨迹覆盖完整性审计允许恢复的断端最大距离")
    p.add_argument("--raw-coverage-max-path-m", type=float, default=140.0,
                   help="高支持缺失走廊允许的原始轨迹子路径最大长度")
    p.add_argument("--raw-coverage-min-witness-tracks", type=int, default=2,
                   help="覆盖完整性恢复至少需要的独立原始轨迹贡献数")
    p.add_argument("--raw-coverage-max-path-ratio", type=float, default=3.5,
                   help="覆盖恢复原始子路径长度/端点直距最大比例")
    p.add_argument("--raw-coverage-max-angle-deg", type=float, default=70.0,
                   help="覆盖恢复与两端既有道路方向最大夹角")
    p.add_argument("--raw-coverage-max-repairs", type=int, default=2000,
                   help="最终覆盖完整性恢复最大数量")
    p.add_argument("--thin-duplicate-min-independent-tracks", type=int, default=3,
                   help="细长重复双线保留为真实双路线时每条路线至少需要的独立原始轨迹数")
    p.add_argument("--raw-witness-relaxed-max-angle-deg", type=float, default=75.0,
                   help="原始轨迹强见证用于endpoint-to-edge时允许的最大接近角")
    p.add_argument("--parallel-bridge-max-m", type=float, default=35.0,
                   help="平行道路横向误桥接反审计的最大连接长度")
    p.add_argument("--parallel-bridge-axis-max-angle-deg", type=float, default=15.0,
                   help="两侧道路小于该轴向夹角时视为局部平行")
    p.add_argument("--parallel-bridge-min-cross-angle-deg", type=float, default=55.0,
                   help="连接杆与两侧平行道路均大于该夹角时视为横向桥")
    p.add_argument("--parallel-bridge-min-side-tracks", type=int, default=1,
                   help="仅当原始轨迹分别占据两侧平行道路时，缺少跨越见证才作为负证据")
    p.add_argument("--disable-micro-gap-repair", action="store_true",
                   help="关闭证据约束的10m以内短断裂修复")
    p.add_argument("--micro-gap-max-missing-pixels", type=int, default=0,
                   help="最大连续缺失像元数；0=按10m物理上限和实际分辨率自动推导")
    p.add_argument("--micro-gap-max-endpoint-distance-cells", type=float, default=0.0,
                   help="兼容像元距离上限；0=禁用，默认优先使用物理距离10m")
    p.add_argument("--micro-gap-max-distance-m", type=float, default=10.0,
                   help="断点中心物理距离上限；默认10m，跨研究区分辨率更稳定")
    p.add_argument("--micro-gap-long-threshold-m", type=float, default=6.0,
                   help="超过该长度的断裂启用更严格的连续证据和方向判据")
    p.add_argument("--micro-gap-long-max-angle-deg", type=float, default=30.0,
                   help="较长短断裂两端方向与连接线的最大夹角")
    p.add_argument("--micro-gap-long-min-evidence-fraction", type=float, default=0.50,
                   help="较长短断裂缺失区沿线最少有效证据比例")
    p.add_argument("--allow-long-gap-in-uturn-context", action="store_true",
                   help="允许>5m修复发生在U型弯上下文；默认关闭以防切弯")
    p.add_argument("--micro-gap-max-angle-deg", type=float, default=45.0,
                   help="两端道路外向方向与连接方向的最大夹角")
    p.add_argument("--micro-gap-strong-angle-deg", type=float, default=20.0,
                   help="无额外栅格证据时允许的一像素强方向约束")
    p.add_argument("--micro-gap-distance-evidence-m", type=float, default=10.0,
                   help="缺失像元到原始轨迹距离小于该值时视为有效证据")
    p.add_argument("--micro-gap-max-repairs", type=int, default=20000)
    p.add_argument("--micro-gap-progress-every", type=int, default=1000)
    p.add_argument("--allow-unevidenced-two-pixel-gap", action="store_true",
                   help="允许非常笔直但无辅助栅格证据的两像素断裂修复")
    p.add_argument("--disable-s4-track-gap-repair", action="store_true",
                   help="关闭基于S4同一条调整后徒步轨迹连续性的较长断裂修复")
    p.add_argument("--s4-track-gap-max-endpoint-distance-m", type=float, default=35.0,
                   help="同一S4轨迹连续穿过时，两断点最大直线距离")
    p.add_argument("--s4-track-gap-max-path-m", type=float, default=50.0,
                   help="同一S4轨迹用于补断的最大实际子路径长度")
    p.add_argument("--s4-track-gap-snap-m", type=float, default=6.0,
                   help="网络端点吸附到S4调整轨迹的最大距离")
    p.add_argument("--s4-track-gap-max-angle-deg", type=float, default=50.0,
                   help="S4同轨迹补断在两端与原道路方向最大夹角")
    p.add_argument("--s4-track-gap-max-repairs", type=int, default=5000,
                   help="S4同轨迹连续性补断最大数量")

    p.add_argument("--disable-force-direction-gap", action="store_true",
                   help="关闭最后一级强方向强制补断")
    p.add_argument("--force-direction-gap-max-m", type=float, default=15.0,
                   help="无同轨迹证据时，强方向+证据约束的最大强制补断距离")
    p.add_argument("--force-direction-gap-max-angle-deg", type=float, default=20.0,
                   help="强制补断两端与连接线的最大方向差")
    p.add_argument("--force-direction-gap-min-evidence-fraction", type=float, default=0.40,
                   help="强制补断沿线最低轨迹/KDE/support证据覆盖比例")
    p.add_argument("--force-direction-gap-evidence-distance-m", type=float, default=10.0,
                   help="强制补断沿线距原轨迹小于该值视为距离证据")
    p.add_argument("--force-direction-gap-max-repairs", type=int, default=5000,
                   help="强方向强制补断最大数量")

    p.add_argument("--track-matcher", choices=["hmm", "nearest", "none"], default="hmm")
    p.add_argument("--match-step", type=int, default=5)
    p.add_argument("--final-match-step", type=int, default=8)
    p.add_argument("--match-max-distance-m", type=float, default=45.0)
    p.add_argument("--hmm-max-candidates", type=int, default=5)
    p.add_argument("--hmm-emission-sigma-m", type=float, default=18.0)
    p.add_argument("--hmm-heading-sigma-deg", type=float, default=35.0)
    p.add_argument("--hmm-heading-weight", type=float, default=0.35)
    p.add_argument("--hmm-transition-beta-m", type=float, default=25.0)
    p.add_argument("--hmm-transition-jump-penalty", type=float, default=5.0)
    p.add_argument("--hmm-transition-mode", choices=["hybrid", "exact", "euclidean"], default="hybrid")
    p.add_argument("--hmm-transition-exact-radius-m", type=float, default=180.0)
    p.add_argument("--min-road-weight", type=float, default=0.30)
    p.add_argument("--write-matched-points", action="store_true")
    p.add_argument("--match-progress-every-tracks", type=int, default=100,
                   help="HMM 每处理多少条轨迹输出一次进度")

    p.add_argument("--dangle-aux-max-m", type=float, default=32.0)
    p.add_argument("--dangle-primary-max-m", type=float, default=12.0)
    p.add_argument("--dangle-rescued-max-m", type=float, default=8.0)
    p.add_argument("--prune-confidence", type=float, default=0.38)
    p.add_argument("--weak-component-max-m", type=float, default=45.0)
    p.add_argument("--weak-loop-max-m", type=float, default=85.0)

    p.add_argument("--parallel-min-distance-m", type=float, default=1.0)
    p.add_argument("--parallel-max-distance-m", type=float, default=6.0)
    p.add_argument("--parallel-max-angle-deg", type=float, default=12.0)
    p.add_argument("--parallel-min-length-m", type=float, default=20.0)
    p.add_argument("--parallel-min-overlap-ratio", type=float, default=0.68)
    p.add_argument("--parallel-score-ratio", type=float, default=0.72)
    p.add_argument("--loop-max-rounds", type=int, default=4,
                   help="弱环批量清理最大轮数；大区域建议 3-4")
    p.add_argument("--parallel-progress-every", type=int, default=1000,
                   help="平行线检测每处理多少条边打印一次进度")
    p.add_argument("--skip-parallel-cleanup", action="store_true",
                   help="跳过最耗时的重复平行线检测；用于快速诊断")
    p.add_argument("--aggressive-primary-pruning", action="store_true",
                   help="启用旧版基于证据的 primary 删除；默认关闭以避免骨架线路丢失")

    p.add_argument("--preserve-turn-deg", type=float, default=42.0)
    p.add_argument("--geometry-chain-turn-deg", type=float, default=110.0,
                   help="仅用于矢量平滑前的度2链合并。2m栅格台阶常产生45/90度伪转角；默认110度可把这些像元转角并入同一路段，同时U型弯核心仍受保护")
    p.add_argument("--topology-progress-every", type=int, default=50000,
                   help="度2链合并每处理多少条边输出一次进度")
    p.add_argument("--simplify-progress-every", type=int, default=10000,
                   help="几何简化每处理多少条边输出一次进度")
    p.add_argument("--dem-progress-every", type=int, default=10000,
                   help="DEM 属性每处理多少条边输出一次进度")
    p.add_argument("--write-intermediate-bundles", action="store_true",
                   help="输出每个阶段的 GPKG/SQLite；默认仅输出轻量诊断和最终路网")
    p.add_argument("--write-diagnostic-gpkg", action="store_true",
                   help="输出可能较大的诊断 GPKG；默认仅写 CSV")
    p.add_argument("--smooth-iterations", type=int, default=2,
                   help="受约束 Chaikin 平滑迭代次数；2m栅格默认2轮以消除连续像元台阶；0关闭")
    p.add_argument("--smooth-ratio", type=float, default=0.25,
                   help="普通道路 Chaikin 切角比例；2m栅格默认0.25")
    p.add_argument("--smooth-corridor-m", type=float, default=1.75,
                   help="普通道路相对原 primary 几何的最大 Hausdorff 偏移；2m栅格默认1.75m")
    p.add_argument("--uturn-smooth-ratio", type=float, default=0.08,
                   help="U 型弯核心区域平滑比例，较普通道路更保守")
    p.add_argument("--uturn-smooth-corridor-m", type=float, default=0.8,
                   help="U 型弯核心区域最大 Hausdorff 偏移")
    p.add_argument("--simplify-tolerance-m", type=float, default=0.0,
                   help="默认0：不对顺滑曲线执行 Douglas-Peucker 化简；仅显式设置>0时启用")
    p.add_argument("--simplify-corridor-m", type=float, default=1.75,
                   help="兼容字段；建议与 smooth-corridor-m 保持一致")
    p.add_argument("--uturn-simplify-tolerance-m", type=float, default=0.0,
                   help="默认0：U型弯不进行平滑后折线化简")
    p.add_argument("--final-smooth-sample-m", type=float, default=0.5,
                   help="最终整链平滑的输出采样间距；越小视觉越顺滑")
    p.add_argument("--final-smooth-sigma-m", type=float, default=3.0,
                   help="普通道路整链高斯平滑尺度；默认3m")
    p.add_argument("--final-smooth-corridor-m", type=float, default=2.5,
                   help="普通道路相对已验证路网允许的最大几何偏移")
    p.add_argument("--final-uturn-smooth-sigma-m", type=float, default=1.0,
                   help="U型弯整链平滑尺度；小于普通道路以保持 hairpin 形态")
    p.add_argument("--final-uturn-smooth-corridor-m", type=float, default=0.8,
                   help="U型弯最终平滑最大偏移")
    p.add_argument("--final-smooth-min-sigma-m", type=float, default=0.30,
                   help="走廊验证失败时自适应减弱平滑的最小尺度")
    p.add_argument("--disable-final-chain-smoothing", action="store_true",
                   help="关闭最终整条道路链平滑，仅保留前序几何")
    p.add_argument("--baseline-corridor-m", type=float, default=1.75,
                   help="普通道路相对 primary 栅格骨架的批准平滑走廊；审计会自动叠加半个像元对角线，不再误把像元足迹当偏移")
    p.add_argument("--dem-sample-spacing-m", type=float, default=5.0)
    return p


def run(args) -> Dict[str, Any]:
    clear_geometry_numeric_anomalies()
    output = args.output.resolve()
    if output.exists() and not (args.keep_output or args.resume):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    tile_cfg = read_tile_config(existing(args.tile_report), args.tile_size_m)

    report: Dict[str, Any] = {
        "version": "S7_network_cleanup_v0_5_smooth_network",
        "policy": {
            "new_route_creation": "raw_track_topology_witness_then_microgap_S4_directional_junction_component_stitching",
            "junction_shift": "local_endpoint_to_edge_or_shared_hub_only",
            "energy_shift": False,
            "parallel_midpoint_creation": False,
            "geometry_subset_of_s5": False,
            "geometry_subset_of_primary_plus_approved_microgaps": False,
            "geometry_within_primary_corridor_plus_approved_microgaps": True,
            "preserve_primary_except_hard_short_isolated_components": True,
            "geometry_source_default": "primary_only",
            "morphological_thinning": True,
            "digital_microloop_breaking": True,
            "directional_digital_loop_collapse": True,
            "shape_agnostic_polygon_cycle_resolver": bool(args.allow_polygon_loop_resolver and not args.disable_polygon_loop_resolver),
            "reasonable_range_forced_cycle_removal": False,
            "raw_track_topology_witness": True,
            "raw_track_is_topology_not_global_geometry": True,
            "parallel_false_bridge_guard": True,
            "raw_negative_topology_veto": True,
            "same_S4_track_gap_repair": True,
            "same_component_cycle_closure": not bool(args.disable_cycle_closure),
            "component_stitching": not bool(args.disable_component_stitching),
            "preserve_real_road_cycles": True,
            "raw_track_semantic_cycle_classification": True,
            "single_corridor_false_cycle_collapse": True,
            "uturn_shortcut_guard": True,
            "strong_direction_forced_gap_repair": True,
            "junction_endpoint_to_edge_snap": True,
            "junction_endpoint_cluster_hub": True,
            "short_isolated_component_hard_prune": True,
            "short_junction_spur_prune": True,
            "uturn_two_level_protection": True,
            "constrained_vector_smoothing": True,
            "repeated_matching": True,
            "pixel_footprint_corridor_audit": True,
        },
        "inputs": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stages": {},
    }

    print("[1/8] primary骨架薄化、严格伪环清理、分层补断与局部连通恢复...", flush=True)
    stage1_cache = output / "00_stage1_cache"
    network, tile_rows = build_network_from_rasters(
        args.primary, existing(args.auxiliary), existing(args.rescued),
        args.support, args.kde, existing(args.distance), tile_cfg,
        existing(args.uturn_gpkg), existing(args.activity_zones), args.activity_layer,
        uturn_core_buffer_m=max(0.0, float(args.uturn_core_buffer_m)),
        uturn_context_buffer_m=max(0.0, float(args.uturn_context_buffer_m)),
        work_dir=stage1_cache,
        resume=args.resume,
        progress_every_tiles=args.stage1_progress_every_tiles,
        chunk_pixels=max(100000, int(args.stage1_chunk_pixels)),
        progress_every_edges=max(1000, int(args.stage1_progress_every_edges)),
        max_node_blob_pixels=max(1, int(args.stage1_max_node_blob_pixels)),
        max_node_blob_span_cells=max(1, int(args.stage1_max_node_blob_span_cells)),
        geometry_source=args.geometry_source,
        thin_geometry=not args.no_thin_geometry,
        thin_halo_pixels=max(1, int(args.thin_halo_pixels)),
    )
    connectivity_before = connectivity_audit(network)
    allow_cycle_closure = not bool(args.disable_cycle_closure)

    raw_witness_index, raw_witness_report = RawTrackWitnessIndex.from_observed_tracks(
        args.raw_tracks, network.crs,
        max_step_m=max(1.0, float(args.raw_track_max_step_m)),
        max_speed_mps=max(0.1, float(args.raw_track_max_speed_mps)),
        max_time_gap_s=max(1.0, float(args.raw_track_max_time_gap_s)),
        isolated_spike_bridge_m=max(0.0, float(args.raw_track_isolated_spike_bridge_m)),
    )
    if raw_witness_index.enabled:
        print(
            "    [raw-track-witness] "
            f"files={raw_witness_report.get('files_seen', 0):,}, "
            f"chunks={raw_witness_report.get('motion_sane_chunks', 0):,}, "
            f"spikes_removed={raw_witness_report.get('isolated_spikes_removed', 0):,}",
            flush=True,
        )
    else:
        print(f"    [raw-track-witness] disabled: {raw_witness_report.get('reason', 'unavailable')}", flush=True)
    report["stages"]["raw_track_witness"] = raw_witness_report

    # v0.5: clean only strict raster/digitisation artefacts before adding missing roads.
    # The old shape-agnostic polygon resolver is opt-in because real road meshes
    # legitimately contain cycles and must not be opened merely due to size.
    if not bool(args.allow_polygon_loop_resolver) or bool(args.disable_polygon_loop_resolver):
        polygon_loop_rows = []
        polygon_loop_remaining = []
        print("    [polygon-cycle] 默认关闭：保留真实道路闭环", flush=True)
    else:
        network, polygon_loop_rows, polygon_loop_remaining = resolve_polygon_cycles(
            network,
            max_perimeter_m=max(0.0, float(args.polygon_loop_max_perimeter_m)),
            max_span_m=max(0.0, float(args.polygon_loop_max_span_m)),
            max_area_m2=max(0.0, float(args.polygon_loop_max_area_m2)),
            force_perimeter_m=max(0.0, float(args.polygon_loop_force_perimeter_m)),
            force_span_m=max(0.0, float(args.polygon_loop_force_span_m)),
            force_area_m2=max(0.0, float(args.polygon_loop_force_area_m2)),
            thin_p90_width_m=max(0.0, float(args.polygon_loop_thin_p90_width_m)),
            thin_max_width_m=max(0.0, float(args.polygon_loop_thin_max_width_m)),
            weak_branch_ratio=min(1.0, max(0.0, float(args.polygon_loop_weak_branch_ratio))),
            max_rounds=max(1, int(args.polygon_loop_rounds)),
            force_enabled=False,  # v0.5 never hard-forces cycle destruction.
        )

    if not args.no_directional_loop_collapse:
        network, diagonal_loop_rows = collapse_directional_digital_loops(
            network,
            primary_path=args.primary, support_path=args.support, kde_path=args.kde, distance_path=existing(args.distance),
            max_perimeter_m=max(0.0, float(args.artifact_loop_max_m)),
            max_span_m=max(0.0, float(args.artifact_loop_max_span_m)),
            max_rounds=max(1, int(args.artifact_loop_rounds)),
            max_gate_angle_deg=max(1.0, float(args.directional_loop_max_angle_deg)),
            evidence_distance_m=max(0.0, float(args.directional_loop_evidence_distance_m)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.directional_loop_min_evidence_fraction))),
        )
    else:
        diagonal_loop_rows = []
    network, microloop_rows = remove_digital_artifact_loops(
        network,
        max_perimeter_m=max(0.0, float(args.artifact_fallback_loop_max_m)),
        max_span_m=max(0.0, float(args.artifact_fallback_loop_max_span_m)),
        max_rounds=max(1, int(args.artifact_loop_rounds)),
    )

    # Tier 0 (v0.5 strongest topology cue): original S1 trajectory continuity.
    # Cleaning/gravity may remove or split points, but a motion-sane original track
    # that continuously traverses both sides remains valid topology evidence.
    if args.disable_raw_track_gap_repair or not raw_witness_index.enabled:
        raw_track_gap_rows = []
        if args.disable_raw_track_gap_repair:
            print("    [raw-track-gap] disabled", flush=True)
    else:
        network, raw_track_gap_rows = repair_raw_track_gaps(
            network, raw_witness_index,
            max_endpoint_distance_m=max(0.0, float(args.raw_track_gap_max_endpoint_distance_m)),
            max_track_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            max_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
            min_witness_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
            max_repairs=max(0, int(args.raw_track_gap_max_repairs)),
        )

    # Tier 1: very short gaps. Same-component closures are allowed when the
    # existing detour proves that this is a missing loop edge rather than a duplicate.
    if args.disable_micro_gap_repair:
        micro_gap_rows = []
        print("    [micro-gap] disabled", flush=True)
    else:
        network, micro_gap_rows = repair_micro_gaps(
            network,
            args.primary, existing(args.auxiliary), existing(args.rescued),
            args.support, args.kde, existing(args.distance),
            max_missing_pixels=int(args.micro_gap_max_missing_pixels),
            max_endpoint_distance_cells=max(0.0, float(args.micro_gap_max_endpoint_distance_cells)),
            max_endpoint_distance_m=max(0.0, float(args.micro_gap_max_distance_m)),
            max_angle_deg=max(1.0, float(args.micro_gap_max_angle_deg)),
            strong_angle_deg=max(1.0, float(args.micro_gap_strong_angle_deg)),
            distance_evidence_max_m=max(0.0, float(args.micro_gap_distance_evidence_m)),
            require_evidence_for_two_pixels=not args.allow_unevidenced_two_pixel_gap,
            long_gap_threshold_m=max(0.0, float(args.micro_gap_long_threshold_m)),
            long_gap_max_angle_deg=max(1.0, float(args.micro_gap_long_max_angle_deg)),
            long_gap_min_evidence_fraction=min(1.0, max(0.0, float(args.micro_gap_long_min_evidence_fraction))),
            allow_long_gap_in_uturn_context=bool(args.allow_long_gap_in_uturn_context),
            max_repairs=max(0, int(args.micro_gap_max_repairs)),
            progress_every=max(1, int(args.micro_gap_progress_every)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
        )

    # Tier 2: strongest continuity cue -- the same adjusted S4 trajectory crosses
    # the missing segment. v0.5 gives this mode a longer range for visible breaks.
    if args.disable_s4_track_gap_repair:
        s4_track_gap_rows = []
        print("    [track-gap] disabled", flush=True)
    else:
        network, s4_track_gap_rows = repair_s4_track_gaps(
            network,
            existing(args.gravity_points),
            max_endpoint_distance_m=max(0.0, float(args.s4_track_gap_max_endpoint_distance_m)),
            max_track_path_m=max(0.0, float(args.s4_track_gap_max_path_m)),
            endpoint_snap_m=max(0.0, float(args.s4_track_gap_snap_m)),
            max_angle_deg=max(1.0, float(args.s4_track_gap_max_angle_deg)),
            max_repairs=max(0, int(args.s4_track_gap_max_repairs)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
        )

    # Tier 3: direction + continuous raster/trajectory evidence.
    if args.disable_force_direction_gap:
        force_gap_rows = []
        print("    [force-gap] disabled", flush=True)
    else:
        network, force_gap_rows = repair_strong_direction_gaps(
            network,
            support_path=args.support,
            kde_path=args.kde,
            distance_path=existing(args.distance),
            max_distance_m=max(0.0, float(args.force_direction_gap_max_m)),
            max_angle_deg=max(1.0, float(args.force_direction_gap_max_angle_deg)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.force_direction_gap_min_evidence_fraction))),
            evidence_distance_m=max(0.0, float(args.force_direction_gap_evidence_distance_m)),
            max_repairs=max(0, int(args.force_direction_gap_max_repairs)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
        )

    # Tier 4: reconstruct T/Y/X junction topology after the simple gaps are closed.
    # Endpoint-to-edge repair splits the target edge at a real graph node; it never
    # creates a geometric crossing without a topological junction.
    if args.disable_junction_repair:
        junction_edge_rows = []
        junction_cluster_rows = []
        print("    [junction] disabled", flush=True)
    else:
        network, junction_edge_rows = repair_endpoint_to_edge_junctions(
            network,
            support_path=args.support, kde_path=args.kde, distance_path=existing(args.distance),
            max_snap_m=max(0.0, float(args.junction_edge_snap_max_m)),
            node_merge_m=max(0.0, float(args.junction_edge_node_merge_m)),
            max_approach_angle_deg=max(1.0, float(args.junction_edge_max_angle_deg)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.junction_min_evidence_fraction))),
            evidence_distance_m=max(0.0, float(args.junction_evidence_distance_m)),
            max_rounds=max(1, int(args.junction_edge_rounds)),
            max_repairs=max(0, int(args.junction_edge_max_repairs)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
            raw_witness_index=raw_witness_index,
            raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            raw_witness_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            raw_witness_max_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            raw_witness_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            raw_witness_corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
            raw_witness_relaxed_max_angle_deg=max(1.0, float(args.raw_witness_relaxed_max_angle_deg)),
            raw_parallel_guard_max_axis_angle_deg=max(0.0, float(args.parallel_bridge_axis_max_angle_deg)),
            raw_parallel_guard_min_cross_angle_deg=max(0.0, float(args.parallel_bridge_min_cross_angle_deg)),
            raw_parallel_guard_min_side_tracks=max(1, int(args.parallel_bridge_min_side_tracks)),
        )
        network, junction_cluster_rows = repair_endpoint_clusters(
            network,
            support_path=args.support, kde_path=args.kde, distance_path=existing(args.distance),
            cluster_radius_m=max(0.0, float(args.junction_cluster_radius_m)),
            center_search_m=max(0.0, float(args.junction_cluster_center_search_m)),
            max_approach_angle_deg=max(1.0, float(args.junction_cluster_max_angle_deg)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.junction_cluster_min_evidence_fraction))),
            evidence_distance_m=max(0.0, float(args.junction_evidence_distance_m)),
            min_cluster_size=max(3, int(args.junction_cluster_min_size)),
            max_cluster_size=max(3, int(args.junction_cluster_max_size)),
            max_repairs=max(0, int(args.junction_cluster_max_repairs)),
            allow_cycle_closure=allow_cycle_closure,
            cycle_min_detour_ratio=max(1.0, float(args.cycle_closure_min_detour_ratio)),
            cycle_min_detour_excess_m=max(0.0, float(args.cycle_closure_min_detour_excess_m)),
            raw_witness_index=raw_witness_index,
            raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            raw_witness_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            raw_witness_max_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            raw_witness_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            raw_witness_corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
            raw_witness_relaxed_max_angle_deg=max(1.0, float(args.raw_witness_relaxed_max_angle_deg)),
        )

    # Tier 5: restore a fragmented local road mesh. First, a strict long-range
    # endpoint-to-edge pass attaches a broken branch to the nearest through-road and
    # splits that road at a real graph node. This is the topology-safe answer to a
    # connector that would otherwise cross an intervening road.
    if args.disable_component_stitching:
        component_edge_stitch_rows = []
        component_stitch_rows = []
        print("    [component-stitch] disabled", flush=True)
    else:
        network, component_edge_stitch_rows = repair_endpoint_to_edge_junctions(
            network,
            support_path=args.support, kde_path=args.kde, distance_path=existing(args.distance),
            max_snap_m=max(0.0, float(args.component_stitch_max_m)),
            node_merge_m=max(0.0, float(args.junction_edge_node_merge_m)),
            max_approach_angle_deg=max(1.0, float(args.component_stitch_long_max_angle_deg)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.component_stitch_long_min_evidence_fraction))),
            evidence_distance_m=max(0.0, float(args.component_stitch_evidence_distance_m)),
            max_rounds=2,
            max_repairs=max(0, int(args.component_stitch_max_repairs)),
            allow_cycle_closure=False,
            raw_witness_index=raw_witness_index,
            raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            raw_witness_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            raw_witness_max_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            raw_witness_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            raw_witness_corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
            raw_witness_relaxed_max_angle_deg=max(1.0, float(args.raw_witness_relaxed_max_angle_deg)),
            raw_parallel_guard_max_axis_angle_deg=max(0.0, float(args.parallel_bridge_axis_max_angle_deg)),
            raw_parallel_guard_min_cross_angle_deg=max(0.0, float(args.parallel_bridge_min_cross_angle_deg)),
            raw_parallel_guard_min_side_tracks=max(1, int(args.parallel_bridge_min_side_tracks)),
        )
        for row in component_edge_stitch_rows:
            row["mode"] = "component_endpoint_to_edge_stitch"
        # Then join remaining mutually-facing component endpoints. Union-find is
        # updated only after all geometry checks pass, so every accepted edge really
        # reduces fragmentation and rejected crossings do not poison later choices.
        network, component_stitch_rows = repair_fragmented_components(
            network,
            support_path=args.support, kde_path=args.kde, distance_path=existing(args.distance),
            max_distance_m=max(0.0, float(args.component_stitch_max_m)),
            max_angle_deg=max(1.0, float(args.component_stitch_max_angle_deg)),
            min_evidence_fraction=min(1.0, max(0.0, float(args.component_stitch_min_evidence_fraction))),
            evidence_distance_m=max(0.0, float(args.component_stitch_evidence_distance_m)),
            long_gap_threshold_m=max(0.0, float(args.component_stitch_long_threshold_m)),
            long_gap_max_angle_deg=max(1.0, float(args.component_stitch_long_max_angle_deg)),
            long_gap_min_evidence_fraction=min(1.0, max(0.0, float(args.component_stitch_long_min_evidence_fraction))),
            max_repairs=max(0, int(args.component_stitch_max_repairs)),
        )

    # v0.5 topology veto: after every repair mode, remove ANY synthetic connector
    # contradicted by local raw trajectories. Unlike the older parallel-road guard,
    # this catches false T/diagonal/oblique connections as well.
    network, raw_negative_connector_rows = prune_raw_negative_connectors(
        network, raw_witness_index,
        max_connector_m=max(0.0, max(float(args.component_stitch_max_m), float(args.raw_track_gap_max_endpoint_distance_m))),
        raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
        raw_side_min_tracks=max(1, int(args.parallel_bridge_min_side_tracks)),
        raw_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
        raw_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
        raw_corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
    )

    network, parallel_false_bridge_rows = prune_parallel_false_bridges(
        network, raw_witness_index,
        max_bridge_m=max(0.0, float(args.parallel_bridge_max_m)),
        parallel_axis_max_angle_deg=max(0.0, float(args.parallel_bridge_axis_max_angle_deg)),
        transverse_min_angle_deg=max(0.0, float(args.parallel_bridge_min_cross_angle_deg)),
        raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
        raw_parallel_min_side_tracks=max(1, int(args.parallel_bridge_min_side_tracks)),
        raw_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
        raw_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
        raw_corridor_m=max(0.0, float(args.raw_track_gap_corridor_m)),
    )

    # v0.5 U-turn guard: a long raw trajectory around a hairpin is not evidence for
    # a short arm-to-arm cross-bar.  Only redundant synthetic cycle edges are eligible.
    network, uturn_shortcut_rows = prune_uturn_shortcut_connectors(
        network, raw_witness_index,
        max_connector_m=max(0.0, float(args.parallel_bridge_max_m)),
        parallel_axis_max_angle_deg=max(0.0, min(25.0, float(args.parallel_bridge_axis_max_angle_deg) + 10.0)),
        transverse_min_angle_deg=max(0.0, min(45.0, float(args.parallel_bridge_min_cross_angle_deg))),
        strict_endpoint_snap_m=max(1.0, min(4.0, float(args.raw_track_gap_snap_m))),
        strict_corridor_m=max(1.0, min(3.0, float(args.raw_track_gap_corridor_m))),
        strict_max_path_ratio=max(1.1, min(1.8, float(args.raw_track_gap_max_path_ratio))),
        min_witness_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
    )

    # v0.5 semantic cycle pass runs AFTER every repair mode, so a pair of individually
    # approved connectors cannot survive as a false local loop merely because both were
    # admissible in isolation.  Independent raw route families preserve true loops.
    network, semantic_cycle_rows = resolve_single_corridor_cycles(
        network, raw_witness_index,
        raw_corridor_m=max(1.0, min(4.0, float(args.raw_track_gap_corridor_m))),
        min_route_coverage=0.60,
        min_independent_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
        min_full_ring_tracks=max(1, int(args.cycle_route_family_min_tracks)),
        max_perimeter_m=max(0.0, max(220.0, float(args.polygon_loop_max_perimeter_m))),
        max_span_m=max(0.0, max(80.0, float(args.polygon_loop_max_span_m))),
        max_area_m2=max(0.0, max(3000.0, float(args.polygon_loop_max_area_m2))),
        max_rounds=max(1, int(args.polygon_loop_rounds)),
    )

    connectivity_after_repair = connectivity_audit(network)
    stage1_unresolved = unresolved_endpoints(network)
    print(
        "    [connectivity] components "
        f"{connectivity_before['component_count']:,} -> {connectivity_after_repair['component_count']:,}; "
        f"endpoints {connectivity_before['endpoint_count']:,} -> {connectivity_after_repair['endpoint_count']:,}; "
        f"cycles {connectivity_before['cyclomatic_number']:,} -> {connectivity_after_repair['cyclomatic_number']:,}",
        flush=True,
    )

    baseline = baseline_union(network)
    d1 = stage(output, 1, "s6_topology_and_microgap_graph")
    pd.DataFrame(tile_rows).to_csv(d1 / "tile_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(junction_edge_rows, d1 / "junction_endpoint_to_edge_repairs.csv")
    write_csv(junction_cluster_rows, d1 / "junction_endpoint_cluster_repairs.csv")
    write_csv(junction_edge_rows + junction_cluster_rows, d1 / "junction_candidate_ranking.csv")
    write_csv(polygon_loop_rows, d1 / "polygon_cycle_resolution.csv")
    write_csv(polygon_loop_remaining, d1 / "polygon_cycles_remaining.csv")
    write_csv(diagonal_loop_rows, d1 / "digital_loops_directional_collapsed.csv")
    write_csv(microloop_rows, d1 / "digital_microloops_removed.csv")
    write_csv(raw_track_gap_rows, d1 / "approved_raw_track_gap_repairs.csv")
    write_csv(
        [r for r in raw_track_gap_rows if str(r.get("raw_continuity_class", "")) == "direct_transition"],
        d1 / "direct_raw_continuity_repairs.csv",
    )
    write_csv(micro_gap_rows, d1 / "approved_micro_gap_repairs.csv")
    write_csv(s4_track_gap_rows, d1 / "approved_s4_track_gap_repairs.csv")
    write_csv(force_gap_rows, d1 / "approved_force_direction_gap_repairs.csv")
    write_csv(component_edge_stitch_rows, d1 / "component_endpoint_to_edge_stitch_repairs.csv")
    write_csv(component_stitch_rows, d1 / "component_stitch_repairs.csv")
    write_csv(raw_negative_connector_rows, d1 / "raw_negative_false_connectors_removed.csv")
    write_csv(parallel_false_bridge_rows, d1 / "parallel_false_bridges_removed.csv")
    write_csv(uturn_shortcut_rows, d1 / "uturn_shortcuts_removed.csv")
    write_csv(semantic_cycle_rows, d1 / "single_corridor_cycles_resolved.csv")
    write_csv(stage1_unresolved, d1 / "unresolved_endpoints_after_repair.csv")
    write_csv(geometry_numeric_anomalies(), d1 / "geometry_numeric_anomalies.csv")
    (d1 / "raw_track_witness_report.json").write_text(
        json.dumps(raw_witness_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (d1 / "connectivity_before_after.json").write_text(
        json.dumps({"before": connectivity_before, "after_repair": connectivity_after_repair}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.write_diagnostic_gpkg and raw_track_gap_rows:
        write_diagnostic_gpkg(
            d1 / "approved_raw_track_gap_repairs.gpkg",
            {"raw_track_gap_repairs": (raw_track_gap_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and raw_negative_connector_rows:
        write_diagnostic_gpkg(
            d1 / "raw_negative_false_connectors_removed.gpkg",
            {"raw_negative_false_connectors_removed": (raw_negative_connector_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and parallel_false_bridge_rows:
        write_diagnostic_gpkg(
            d1 / "parallel_false_bridges_removed.gpkg",
            {"parallel_false_bridges_removed": (parallel_false_bridge_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and uturn_shortcut_rows:
        write_diagnostic_gpkg(
            d1 / "uturn_shortcuts_removed.gpkg",
            {"uturn_shortcuts_removed": (uturn_shortcut_rows, network.crs)},
        )
    if micro_gap_rows:
        write_diagnostic_gpkg(
            d1 / "approved_micro_gap_repairs.gpkg",
            {"micro_gap_repairs": (micro_gap_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and s4_track_gap_rows:
        write_diagnostic_gpkg(
            d1 / "approved_s4_track_gap_repairs.gpkg",
            {"s4_track_gap_repairs": (s4_track_gap_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and force_gap_rows:
        write_diagnostic_gpkg(
            d1 / "approved_force_direction_gap_repairs.gpkg",
            {"force_direction_gap_repairs": (force_gap_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and component_edge_stitch_rows:
        write_diagnostic_gpkg(
            d1 / "component_endpoint_to_edge_stitch_repairs.gpkg",
            {"component_endpoint_to_edge_stitch": (component_edge_stitch_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and component_stitch_rows:
        write_diagnostic_gpkg(
            d1 / "component_stitch_repairs.gpkg",
            {"component_stitch_repairs": (component_stitch_rows, network.crs)},
        )
    if args.write_diagnostic_gpkg and stage1_unresolved:
        write_diagnostic_gpkg(
            d1 / "unresolved_endpoints_after_repair.gpkg",
            {"unresolved_endpoints": (stage1_unresolved, network.crs)},
        )
    write_stage_network(network, d1, "s6_topology_and_microgap_graph", args)
    report["stages"]["baseline"] = {
        **network_metrics(network),
        "geometry_source": args.geometry_source,
        "junction_endpoint_to_edge_repairs": len(junction_edge_rows),
        "junction_endpoint_cluster_repairs": len(junction_cluster_rows),
        "polygon_cycles_processed": len(polygon_loop_rows),
        "polygon_cycles_remaining": len(polygon_loop_remaining),
        "directional_loops_collapsed": len(diagonal_loop_rows),
        "digital_microloops_removed": len(microloop_rows),
        "approved_raw_track_gap_repairs": len(raw_track_gap_rows),
        "approved_micro_gap_repairs": len(micro_gap_rows),
        "approved_s4_track_gap_repairs": len(s4_track_gap_rows),
        "approved_force_direction_gap_repairs": len(force_gap_rows),
        "component_endpoint_to_edge_stitch_repairs": len(component_edge_stitch_rows),
        "component_stitch_repairs": len(component_stitch_rows),
        "raw_negative_false_connectors_removed": len(raw_negative_connector_rows),
        "parallel_false_bridges_removed": len(parallel_false_bridge_rows),
        "uturn_shortcuts_removed": len(uturn_shortcut_rows),
        "semantic_cycles_examined": len(semantic_cycle_rows),
        "single_corridor_false_cycles_collapsed": sum(r.get("action") == "collapse_single_corridor_false_cycle" for r in semantic_cycle_rows),
        "multigate_false_cycles_opened": sum(str(r.get("action", "")).startswith("open_multigate_false_cycle") for r in semantic_cycle_rows),
        "independent_dual_route_cycles_preserved": sum(r.get("action") == "preserve_independent_dual_route_cycle" for r in semantic_cycle_rows),
        "cycle_closures_added": sum(bool(r.get("cycle_closure")) for rows in (raw_track_gap_rows, micro_gap_rows, s4_track_gap_rows, force_gap_rows, junction_edge_rows, junction_cluster_rows) for r in rows),
        "connectivity_before": connectivity_before,
        "connectivity_after_repair": connectivity_after_repair,
        "unresolved_endpoint_count": len(stage1_unresolved),
    }

    print("[2/8] 第一次 HMM 匹配，建立路段证据...", flush=True)
    d2 = stage(output, 2, "first_track_matching")
    points_path = d2 / "matched_points.csv" if args.write_matched_points else None
    summary1, match1 = match(network, args, points_path)
    attach_track_evidence(network, summary1)
    summary1.to_csv(d2 / "edge_track_evidence.csv", index=False, encoding="utf-8-sig")
    report["stages"]["first_match"] = match1

    print("[3/8] 保留主体 primary；允许删除明确的短孤立分量与交叉口毛刺...", flush=True)
    if args.aggressive_primary_pruning:
        cleaned, diag = clean_network(
            network,
            args.dangle_aux_max_m, args.dangle_primary_max_m, args.dangle_rescued_max_m,
            args.prune_confidence, args.weak_component_max_m, args.weak_loop_max_m,
            args.parallel_min_distance_m, args.parallel_max_distance_m,
            args.parallel_max_angle_deg, args.parallel_min_length_m,
            args.parallel_min_overlap_ratio, args.parallel_score_ratio,
            args.loop_max_rounds, args.parallel_progress_every,
            args.skip_parallel_cleanup,
        )
    else:
        cleaned, conservative_removed = conservative_prune_network(
            network,
            terminal_length_m=args.dangle_aux_max_m,
            confidence_threshold=args.prune_confidence,
            min_component_length_m=args.weak_component_max_m,
        )
        diag = {
            "dangles": conservative_removed,
            "components": [], "loops": [], "parallel_dropped": [],
            "parallel_decisions": [],
        }
        print(
            f"    [cleanup] preserve-primary mode, removed_auxiliary={len(conservative_removed):,}, "
            f"primary_removed_by_conservative=0（后续短孤立规则可删除primary）", flush=True,
        )

    cleaned, isolated_removed = prune_short_isolated_components(
        cleaned,
        hard_max_length_m=max(0.0, float(args.isolated_component_hard_max_m)),
        weak_max_length_m=max(0.0, float(args.isolated_component_weak_max_m)),
        weak_max_tracks=max(0, int(args.isolated_component_weak_max_tracks)),
        weak_max_confidence=min(1.0, max(0.0, float(args.isolated_component_weak_max_confidence))),
    )
    cleaned, junction_spur_removed = prune_short_leaf_chains(
        cleaned,
        hard_max_length_m=max(0.0, float(args.junction_spur_hard_max_m)),
        weak_max_length_m=max(0.0, float(args.junction_spur_weak_max_m)),
        weak_max_tracks=max(0, int(args.junction_spur_weak_max_tracks)),
        weak_max_confidence=min(1.0, max(0.0, float(args.junction_spur_weak_max_confidence))),
        max_rounds=max(1, int(args.junction_spur_rounds)),
    )
    diag["isolated_components"] = isolated_removed
    diag["junction_spurs"] = junction_spur_removed
    print(
        f"    [cleanup] isolated_removed={len(isolated_removed):,}, "
        f"junction_spur_edges_removed={len(junction_spur_removed):,}",
        flush=True,
    )

    d3 = stage(output, 3, "evidence_cleanup")
    write_stage_network(cleaned, d3, "evidence_cleaned_graph", args)
    for name, rows in diag.items():
        write_csv(rows, d3 / f"{name}.csv")
    removed_all = (
        diag.get("dangles", [])
        + diag.get("components", [])
        + diag.get("loops", [])
        + diag.get("parallel_dropped", [])
        + diag.get("isolated_components", [])
        + diag.get("junction_spurs", [])
        + diag.get("conservative_removed", [])
    )
    if args.write_diagnostic_gpkg:
        write_diagnostic_gpkg(d3 / "removed_edges.gpkg", {"removed_edges": (removed_all, network.crs)})
    report["stages"]["cleanup"] = {**network_metrics(cleaned), "removed_edges": len(removed_all)}

    print("[4/8] 合并安全的度 2 节点，保留高曲率和 U 型弯节点...", flush=True)
    topology, merged = merge_degree_two(
        cleaned, preserve_turn_deg=args.preserve_turn_deg,
        progress_every=args.topology_progress_every,
    )
    d4 = stage(output, 4, "topology_simplification")
    write_stage_network(topology, d4, "topology_simplified_graph", args)
    report["stages"]["topology"] = {**network_metrics(topology), "degree2_merged": merged}

    print("[5/8] 先合并栅格伪转角形成连续道路链，再进行受约束平滑...", flush=True)
    # v0.5: topology 阶段仍保留较明显转角，避免影响网络语义；
    # 但几何平滑前再做一次“仅几何链”合并。2 m 栅格的45°/90°台阶
    # 不应被当成真实拓扑节点，否则每一小段只有2个顶点，平滑算法无法生效。
    geometry_chain_turn_deg = max(float(args.preserve_turn_deg), float(args.geometry_chain_turn_deg))
    geometry_topology, geometry_chain_merged = merge_degree_two(
        topology,
        preserve_turn_deg=geometry_chain_turn_deg,
        progress_every=args.topology_progress_every,
    )
    exact_net, collinear_rows = compress_collinear_paths(geometry_topology)
    ordinary_corridor = max(0.0, float(args.smooth_corridor_m))
    if float(args.simplify_corridor_m) > 0:
        ordinary_corridor = min(ordinary_corridor, float(args.simplify_corridor_m))
    simplified, smooth_rows = constrained_smooth_and_simplify(
        exact_net,
        smooth_iterations=max(0, int(args.smooth_iterations)),
        smooth_ratio=max(0.0, float(args.smooth_ratio)),
        smooth_corridor_m=ordinary_corridor,
        simplify_tolerance_m=max(0.0, float(args.simplify_tolerance_m)),
        uturn_smooth_ratio=max(0.0, float(args.uturn_smooth_ratio)),
        uturn_smooth_corridor_m=max(0.0, float(args.uturn_smooth_corridor_m)),
        uturn_simplify_tolerance_m=max(0.0, float(args.uturn_simplify_tolerance_m)),
        progress_every=max(1, int(args.simplify_progress_every)),
    )
    d5 = stage(output, 5, "evidence_constrained_smoothing")
    write_stage_network(simplified, d5, "smoothed_primary_corridor_graph", args)
    write_csv(collinear_rows, d5 / "collinear_compression_diagnostics.csv")
    write_csv(smooth_rows, d5 / "smoothing_diagnostics.csv")
    shifts = [float(r.get("hausdorff_shift_m", 0.0) or 0.0) for r in smooth_rows]
    report["stages"]["simplification"] = {
        **network_metrics(simplified),
        "max_geometry_shift_m": max(shifts) if shifts else 0.0,
        "smoothed_edges": sum(bool(r.get("smoothed")) for r in smooth_rows),
        "method": "degree2_geometry_chain_plus_endpoint_fixed_chaikin_plus_corridor_validated_dp",
        "geometry_chain_turn_deg": geometry_chain_turn_deg,
        "geometry_chain_merged": geometry_chain_merged,
    }

    print("[6/8] 对拓扑网络重新匹配，继续保持 primary 不删除...", flush=True)
    original_step = args.match_step
    args.match_step = args.final_match_step
    d6 = stage(output, 6, "rematching_and_final_prune")
    summary2, match2 = match(simplified, args, None)
    args.match_step = original_step
    attach_track_evidence(simplified, summary2)
    summary2.to_csv(d6 / "final_edge_track_evidence.csv", index=False, encoding="utf-8-sig")
    if args.aggressive_primary_pruning:
        final_net, final_diag = clean_network(
            simplified,
            args.dangle_aux_max_m, args.dangle_primary_max_m, args.dangle_rescued_max_m,
            args.prune_confidence + 0.02, args.weak_component_max_m, args.weak_loop_max_m,
            args.parallel_min_distance_m, args.parallel_max_distance_m,
            args.parallel_max_angle_deg, args.parallel_min_length_m,
            args.parallel_min_overlap_ratio, args.parallel_score_ratio,
            args.loop_max_rounds, args.parallel_progress_every,
            args.skip_parallel_cleanup,
        )
    else:
        final_net, final_removed = conservative_prune_network(
            simplified,
            terminal_length_m=args.dangle_aux_max_m,
            confidence_threshold=args.prune_confidence,
            min_component_length_m=args.weak_component_max_m,
        )
        final_diag = {"conservative_removed": final_removed}
        print(
            f"    [final-prune] preserve-primary mode, removed_auxiliary={len(final_removed):,}, "
            f"primary_removed_by_conservative=0（后续短孤立规则可删除primary）", flush=True,
        )
    final_net, final_isolated_removed = prune_short_isolated_components(
        final_net,
        hard_max_length_m=max(0.0, float(args.isolated_component_hard_max_m)),
        weak_max_length_m=max(0.0, float(args.isolated_component_weak_max_m)),
        weak_max_tracks=max(0, int(args.isolated_component_weak_max_tracks)),
        weak_max_confidence=min(1.0, max(0.0, float(args.isolated_component_weak_max_confidence))),
    )
    final_net, final_junction_spurs = prune_short_leaf_chains(
        final_net,
        hard_max_length_m=max(0.0, float(args.junction_spur_hard_max_m)),
        weak_max_length_m=max(0.0, float(args.junction_spur_weak_max_m)),
        weak_max_tracks=max(0, int(args.junction_spur_weak_max_tracks)),
        weak_max_confidence=min(1.0, max(0.0, float(args.junction_spur_weak_max_confidence))),
        max_rounds=max(1, int(args.junction_spur_rounds)),
    )
    final_diag["v0_5_isolated_components"] = final_isolated_removed
    final_diag["v0_5_junction_spurs"] = final_junction_spurs
    write_csv(final_isolated_removed, d6 / "final_short_isolated_components_removed.csv")
    write_csv(final_junction_spurs, d6 / "final_short_junction_spurs_removed.csv")
    print(
        f"    [final-prune] isolated_removed={len(final_isolated_removed):,}, "
        f"junction_spur_edges_removed={len(final_junction_spurs):,}",
        flush=True,
    )

    final_net, final_merged = merge_degree_two(
        final_net, preserve_turn_deg=args.preserve_turn_deg,
        progress_every=args.topology_progress_every,
    )

    # v0.5: pruning can remove a short bridge that was present during the stage-1
    # repair pass. Run one final raw-track-only rescue against the *actual* final
    # topology. This pass is deliberately no-cycle: it may only merge different
    # connected components. Curved missing road pieces follow the motion-sane raw
    # subpath rather than being forced onto the straight endpoint chord.
    final_rescue_endpoint_rows = []
    final_rescue_edge_rows = []
    if not bool(args.disable_raw_track_gap_repair) and raw_witness_index is not None and raw_witness_index.enabled:
        final_net, final_rescue_endpoint_rows = repair_raw_track_gaps(
            final_net, raw_witness_index,
            max_endpoint_distance_m=max(0.0, float(args.raw_track_gap_max_endpoint_distance_m)),
            max_track_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            max_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
            min_witness_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            corridor_m=max(0.1, float(args.raw_track_gap_corridor_m)),
            allow_cycle_closure=False,
            max_repairs=max(0, int(args.raw_track_gap_max_repairs)),
            use_raw_subpath_geometry=True,
            repair_mode="final_connectivity_rescue_endpoint_endpoint",
        )
        final_net, final_rescue_edge_rows = repair_endpoint_to_edge_junctions(
            final_net,
            support_path=None, kde_path=None, distance_path=None,
            max_snap_m=max(0.0, float(args.raw_track_gap_max_endpoint_distance_m)),
            node_merge_m=max(0.0, float(args.junction_edge_node_merge_m)),
            max_approach_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
            min_evidence_fraction=1.0,
            evidence_distance_m=max(0.0, float(args.junction_evidence_distance_m)),
            allow_cycle_closure=False,
            max_rounds=max(1, int(args.junction_edge_rounds)),
            max_repairs=max(0, int(args.raw_track_gap_max_repairs)),
            raw_witness_index=raw_witness_index,
            raw_witness_min_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
            raw_witness_endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
            raw_witness_max_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
            raw_witness_max_path_ratio=max(1.0, float(args.raw_track_gap_max_path_ratio)),
            raw_witness_corridor_m=max(0.1, float(args.raw_track_gap_corridor_m)),
            raw_witness_relaxed_max_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
            raw_parallel_guard_max_axis_angle_deg=max(1.0, float(args.parallel_bridge_axis_max_angle_deg)),
            raw_parallel_guard_min_cross_angle_deg=max(1.0, float(args.parallel_bridge_min_cross_angle_deg)),
            raw_parallel_guard_min_side_tracks=max(1, int(args.parallel_bridge_min_side_tracks)),
            use_raw_subpath_geometry=True,
            require_raw_witness=True,
            raw_target_max_axis_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
            repair_label="final_connectivity_rescue_endpoint_edge",
            include_protected_endpoints=True,
        )
    final_connectivity_rescue_rows = list(final_rescue_endpoint_rows) + list(final_rescue_edge_rows)
    write_csv(final_connectivity_rescue_rows, d6 / "final_connectivity_rescue.csv")
    print(
        f"    [final-connectivity] endpoint-endpoint={len(final_rescue_endpoint_rows):,}, "
        f"endpoint-edge={len(final_rescue_edge_rows):,}", flush=True,
    )

    # v0.5 coverage completeness audit: ordinary endpoint repair is intentionally
    # short-range.  A longer high-support raw corridor may still be absent after
    # pruning/activity filtering, so recover it only when >=2 independent raw
    # contributions support the curved subpath.  This pass never closes a cycle.
    final_net, raw_coverage_rows = repair_high_support_uncovered_corridors(
        final_net, raw_witness_index,
        max_endpoint_distance_m=max(0.0, float(args.raw_coverage_max_endpoint_distance_m)),
        max_track_path_m=max(0.0, float(args.raw_coverage_max_path_m)),
        endpoint_snap_m=max(0.0, float(args.raw_track_gap_snap_m)),
        max_angle_deg=max(1.0, float(args.raw_coverage_max_angle_deg)),
        min_witness_tracks=max(2, int(args.raw_coverage_min_witness_tracks)),
        max_path_ratio=max(1.0, float(args.raw_coverage_max_path_ratio)),
        max_repairs=max(0, int(args.raw_coverage_max_repairs)),
    ) if (raw_witness_index is not None and raw_witness_index.enabled and not bool(args.disable_raw_track_gap_repair)) else (final_net, [])
    write_csv(raw_coverage_rows, d6 / "raw_track_coverage_audit.csv")
    print(
        f"    [raw-coverage] restored={len(raw_coverage_rows):,}, no_cycle_closure=True",
        flush=True,
    )

    # Run semantic cycle resolution once more on the post-prune/post-rescue graph.
    # The expanded v0.5 thin-corridor envelope catches long bubble loops that were
    # intentionally outside the legacy 220 m polygon-cycle search window.
    final_net, final_cycle_rows = resolve_single_corridor_cycles(
        final_net, raw_witness_index,
        raw_corridor_m=max(1.0, min(4.0, float(args.raw_track_gap_corridor_m))),
        min_route_coverage=0.60,
        min_independent_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
        min_full_ring_tracks=max(1, int(args.cycle_route_family_min_tracks)),
        max_perimeter_m=max(0.0, max(220.0, float(args.polygon_loop_max_perimeter_m))),
        max_span_m=max(0.0, max(80.0, float(args.polygon_loop_max_span_m))),
        max_area_m2=max(0.0, max(3000.0, float(args.polygon_loop_max_area_m2))),
        max_rounds=max(1, int(args.polygon_loop_rounds)),
        thin_duplicate_max_perimeter_m=800.0,
        thin_duplicate_max_span_m=300.0,
        thin_duplicate_max_area_m2=8000.0,
        thin_duplicate_min_independent_tracks=max(1, int(args.thin_duplicate_min_independent_tracks)),
    )
    write_csv(final_cycle_rows, d6 / "final_cycle_resolution.csv")
    print(
        f"    [final-cycle] examined={len(final_cycle_rows):,}, "
        f"collapsed={sum(str(r.get('action', '')).startswith(('collapse_', 'open_')) for r in final_cycle_rows):,}",
        flush=True,
    )

    write_stage_network(final_net, d6, "rematched_final_pruned_graph", args)
    report["stages"]["rematch"] = match2
    report["stages"]["final_prune"] = {
        **network_metrics(final_net),
        "degree2_merged": final_merged,
        "short_isolated_edges_removed": len(final_isolated_removed),
        "short_junction_spur_edges_removed": len(final_junction_spurs),
        "final_connectivity_rescue_endpoint_endpoint": len(final_rescue_endpoint_rows),
        "final_connectivity_rescue_endpoint_edge": len(final_rescue_edge_rows),
        "raw_track_coverage_restored": len(raw_coverage_rows),
        "final_cycle_decisions": len(final_cycle_rows),
    }

    print("[7/8] 验证普通路线、交叉口修复、方向塌缩与分级补断...", flush=True)
    d7 = stage(output, 7, "monotonicity_audit")

    # v0.5 replays each junction connector's creation-time admissibility envelope
    # before the general geometry audits.  If a connector is genuinely invalid,
    # remove only that synthetic edge and continue; a few rejected repairs must not
    # abort the entire S7 run.
    junction_pre_audit = audit_junction_repairs(
        final_net,
        max_connector_m=max(
            float(args.junction_edge_snap_max_m) + float(args.junction_edge_node_merge_m),
            float(args.junction_cluster_radius_m) * 1.35,
            (float(args.component_stitch_max_m) + float(args.junction_edge_node_merge_m))
            if not bool(args.disable_component_stitching) else 0.0,
        ),
        max_angle_deg=max(float(args.junction_edge_max_angle_deg), float(args.junction_cluster_max_angle_deg)),
        min_evidence_fraction=min(
            float(args.junction_min_evidence_fraction),
            float(args.junction_cluster_min_evidence_fraction),
        ),
    )
    junction_pre_violations = [
        r for r in junction_pre_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    junction_rejected_rows = []
    if junction_pre_violations:
        final_net, junction_rejected_rows = rollback_failed_junction_repairs(
            final_net, junction_pre_violations
        )
        print(
            f"    [audit:junction:rollback] rejected={len(junction_rejected_rows):,}; "
            "继续审计剩余路网",
            flush=True,
        )
    write_csv(junction_rejected_rows, d7 / "junction_final_rejected.csv")
    report["stages"]["final_prune"]["junction_final_rejected"] = len(junction_rejected_rows)

    # v0.5 partitioned raw-gap audit: every connector is checked against the exact
    # admissibility envelope persisted at creation time.  In particular, long
    # high-support coverage rescues are not re-audited with the ordinary 40/65 m
    # short-gap limits.  A genuinely invalid connector is rolled back locally.
    raw_gap_pre_audit = audit_raw_track_gaps(
        final_net,
        max_endpoint_distance_m=max(0.0, float(args.raw_track_gap_max_endpoint_distance_m)),
        max_track_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
        max_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
        min_witness_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
    )
    raw_gap_pre_violations = [
        r for r in raw_gap_pre_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    raw_gap_rejected_rows = []
    if raw_gap_pre_violations:
        final_net, raw_gap_rejected_rows = rollback_failed_raw_track_gap_repairs(
            final_net, raw_gap_pre_violations
        )
        print(
            f"    [audit:raw-track-gap:rollback] rejected={len(raw_gap_rejected_rows):,}; "
            "继续审计剩余路网",
            flush=True,
        )
    write_csv(raw_gap_rejected_rows, d7 / "raw_track_gap_final_rejected.csv")
    report["stages"]["final_prune"]["raw_track_gap_final_rejected"] = len(raw_gap_rejected_rows)

    non_gap_net = final_net.copy()
    non_gap_net.edges = [
        edge for edge in final_net.edges
        if not bool(edge.get("gap_repair"))
        and not bool(edge.get("approved_gap_repair"))
        and not bool(edge.get("approved_directional_collapse"))
    ]
    audit = audit_geometry_against_primary_raster(
        non_gap_net, args.primary,
        corridor_m=max(0.0, float(args.baseline_corridor_m)),
        uturn_corridor_m=max(0.0, min(float(args.baseline_corridor_m), float(args.uturn_smooth_corridor_m))),
        progress_every=args.simplify_progress_every,
    )
    raw_gap_audit = audit_raw_track_gaps(
        final_net,
        max_endpoint_distance_m=max(0.0, float(args.raw_track_gap_max_endpoint_distance_m)),
        max_track_path_m=max(0.0, float(args.raw_track_gap_max_path_m)),
        max_angle_deg=max(1.0, float(args.raw_track_gap_max_angle_deg)),
        min_witness_tracks=max(1, int(args.raw_track_gap_min_witness_tracks)),
    )
    raw_gap_viol_preview = [r for r in raw_gap_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    print(
        f"    [audit:raw-track-gap] repairs={len(raw_gap_audit):,}, violations={len(raw_gap_viol_preview):,}",
        flush=True,
    )
    gap_audit = audit_approved_micro_gaps(
        final_net, args.primary,
        max_missing_pixels=int(args.micro_gap_max_missing_pixels),
        max_length_m=max(0.0, float(args.micro_gap_max_distance_m)),
        max_angle_deg=max(1.0, float(args.micro_gap_max_angle_deg)),
        long_gap_threshold_m=max(0.0, float(args.micro_gap_long_threshold_m)),
        long_gap_max_angle_deg=max(1.0, float(args.micro_gap_long_max_angle_deg)),
        long_gap_min_evidence_fraction=min(1.0, max(0.0, float(args.micro_gap_long_min_evidence_fraction))),
    )
    gap_violation_preview = [
        r for r in gap_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    print(
        f"    [audit:micro-gap] repairs={len(gap_audit):,}, "
        f"violations={len(gap_violation_preview):,}",
        flush=True,
    )
    if gap_violation_preview:
        from collections import Counter
        reason_counts = Counter(str(r.get("audit_reason", "unknown")) for r in gap_violation_preview)
        print(
            "    [audit:micro-gap:reasons] " +
            "; ".join(f"{reason}={count}" for reason, count in reason_counts.most_common()),
            flush=True,
        )
    junction_audit = audit_junction_repairs(
        final_net,
        max_connector_m=max(
            float(args.junction_edge_snap_max_m) + float(args.junction_edge_node_merge_m),
            float(args.junction_cluster_radius_m) * 1.35,
            (float(args.component_stitch_max_m) + float(args.junction_edge_node_merge_m))
            if not bool(args.disable_component_stitching) else 0.0,
        ),
        max_angle_deg=max(float(args.junction_edge_max_angle_deg), float(args.junction_cluster_max_angle_deg)),
        min_evidence_fraction=min(
            float(args.junction_min_evidence_fraction),
            float(args.junction_cluster_min_evidence_fraction),
        ),
    )
    junction_viol_preview = [
        r for r in junction_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    print(
        f"    [audit:junction] repairs={len(junction_audit):,}, "
        f"violations={len(junction_viol_preview):,}",
        flush=True,
    )
    if junction_viol_preview:
        from collections import Counter
        junction_reason_counts = Counter(
            str(r.get("failed_checks", "unknown")) for r in junction_viol_preview
        )
        print(
            "    [audit:junction:reasons] " +
            "; ".join(f"{reason}={count}" for reason, count in junction_reason_counts.most_common()),
            flush=True,
        )

    s7_force_gap_audit = audit_s7_force_gaps(
        final_net,
        same_track_max_endpoint_m=max(0.0, float(args.s4_track_gap_max_endpoint_distance_m)),
        same_track_max_path_m=max(0.0, float(args.s4_track_gap_max_path_m)),
        same_track_max_angle_deg=max(1.0, float(args.s4_track_gap_max_angle_deg)),
        forced_max_distance_m=max(0.0, float(args.force_direction_gap_max_m)),
        forced_max_angle_deg=max(1.0, float(args.force_direction_gap_max_angle_deg)),
        forced_min_evidence_fraction=min(1.0, max(0.0, float(args.force_direction_gap_min_evidence_fraction))),
    )
    s7_force_gap_viol_preview = [
        r for r in s7_force_gap_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    print(
        f"    [audit:s7-force-gap] repairs={len(s7_force_gap_audit):,}, "
        f"violations={len(s7_force_gap_viol_preview):,}",
        flush=True,
    )

    component_stitch_audit = audit_component_stitching(
        final_net,
        max_distance_m=max(0.0, float(args.component_stitch_max_m)),
        max_angle_deg=max(1.0, float(args.component_stitch_max_angle_deg)),
        min_evidence_fraction=min(1.0, max(0.0, float(args.component_stitch_min_evidence_fraction))),
        long_gap_threshold_m=max(0.0, float(args.component_stitch_long_threshold_m)),
        long_gap_max_angle_deg=max(1.0, float(args.component_stitch_long_max_angle_deg)),
        long_gap_min_evidence_fraction=min(1.0, max(0.0, float(args.component_stitch_long_min_evidence_fraction))),
    )
    component_stitch_viol_preview = [
        r for r in component_stitch_audit if float(r["outside_baseline_fraction"]) > 1e-6
    ]
    print(
        f"    [audit:component-stitch] repairs={len(component_stitch_audit):,}, "
        f"violations={len(component_stitch_viol_preview):,}",
        flush=True,
    )

    diagonal_audit = audit_approved_directional_collapses(
        final_net,
        min_evidence_fraction=min(1.0, max(0.0, float(args.directional_loop_min_evidence_fraction))),
        max_gate_angle_deg=max(1.0, float(args.directional_loop_max_angle_deg)),
    )
    print(
        f"    [audit:directional-loop] replacements={len(diagonal_audit):,}, "
        f"violations={sum(float(r['outside_baseline_fraction']) > 1e-6 for r in diagonal_audit):,}",
        flush=True,
    )
    audit.extend(raw_gap_audit)
    audit.extend(gap_audit)
    audit.extend(junction_audit)
    audit.extend(s7_force_gap_audit)
    audit.extend(component_stitch_audit)
    audit.extend(diagonal_audit)
    write_csv(audit, d7 / "geometry_audit.csv")
    write_csv(geometry_numeric_anomalies(), d7 / "geometry_numeric_anomalies.csv")
    violations = [r for r in audit if float(r["outside_baseline_fraction"]) > 1e-6]
    regular_violations = [r for r in violations if not bool(r.get("gap_repair")) and not bool(r.get("junction_repair")) and not bool(r.get("approved_directional_collapse"))]
    raw_gap_violations = [r for r in raw_gap_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    gap_violations = [r for r in gap_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    junction_violations = [r for r in junction_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    s7_force_gap_violations = [r for r in s7_force_gap_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    component_stitch_violations = [r for r in component_stitch_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    directional_violations = [r for r in diagonal_audit if float(r["outside_baseline_fraction"]) > 1e-6]
    if args.write_diagnostic_gpkg:
        write_diagnostic_gpkg(d7 / "geometry_audit.gpkg", {"outside_baseline": (violations, final_net.crs)})
    before = report["stages"]["baseline"]
    after = network_metrics(final_net)
    final_rescue_length_m = sum(
        float(e.get("geometry").length)
        for e in final_net.edges
        if bool(e.get("final_connectivity_rescue")) and e.get("geometry") is not None
    )
    # endpoint-to-edge rescue may split one target edge (+1 edge) and add one
    # connector (+1 edge); endpoint-to-endpoint adds one connector. These are
    # explicit approved v0.5 topology restorations, so monotonicity allows exactly
    # their structural/length budget and nothing more.
    final_rescue_edge_allowance = (
        len(final_rescue_endpoint_rows)
        + 2 * len(final_rescue_edge_rows)
        + len(raw_coverage_rows)
    )
    monotonic = {
        "edge_count_not_increased": after["edges"] <= before["edges"] + final_rescue_edge_allowance,
        "length_not_increased": after["length_m"] <= before["length_m"] + final_rescue_length_m + 1e-6,
        "final_rescue_edge_allowance": int(final_rescue_edge_allowance),
        "final_rescue_length_allowance_m": float(final_rescue_length_m),
        "regular_geometry_violations": len(regular_violations),
        "raw_track_gap_violations": len(raw_gap_violations),
        "micro_gap_violations": len(gap_violations),
        "junction_repair_violations": len(junction_violations),
        "s7_force_gap_violations": len(s7_force_gap_violations),
        "component_stitch_violations": len(component_stitch_violations),
        "directional_collapse_violations": len(directional_violations),
        "outside_baseline_edges": len(violations),
        "audit_model": "v0_5_smooth_network_mode_partitioned_audit",
        "ordinary_smoothing_corridor_m": max(0.0, float(args.baseline_corridor_m)),
        "uturn_core_smoothing_corridor_m": max(0.0, min(float(args.baseline_corridor_m), float(args.uturn_smooth_corridor_m))),
    }
    if not all([
        monotonic["edge_count_not_increased"],
        monotonic["length_not_increased"],
        len(regular_violations) == 0,
        len(raw_gap_violations) == 0,
        len(gap_violations) == 0,
        len(junction_violations) == 0,
        len(s7_force_gap_violations) == 0,
        len(component_stitch_violations) == 0,
        len(directional_violations) == 0,
    ]):
        raise RuntimeError(f"最终路网违反原始轨迹拓扑/几何约束：{monotonic}")
    report["stages"]["audit"] = monotonic
    report["stages"]["geometry_numeric_safety"] = {"anomalies_rejected": len(geometry_numeric_anomalies())}

    print("[8/8] 最终整链平滑、添加 DEM 属性并输出路网...", flush=True)
    d8 = stage(output, 8, "final_clean_network")
    final_smooth_rows = []
    if not bool(args.disable_final_chain_smoothing):
        topology_before_smoothing = connectivity_audit(final_net)
        final_net, final_smooth_rows = smooth_final_road_chains(
            final_net,
            sample_spacing_m=max(0.20, float(args.final_smooth_sample_m)),
            sigma_m=max(0.0, float(args.final_smooth_sigma_m)),
            corridor_m=max(0.0, float(args.final_smooth_corridor_m)),
            uturn_sigma_m=max(0.0, float(args.final_uturn_smooth_sigma_m)),
            uturn_corridor_m=max(0.0, float(args.final_uturn_smooth_corridor_m)),
            min_sigma_m=max(0.05, float(args.final_smooth_min_sigma_m)),
            progress_every=max(1, int(args.simplify_progress_every)),
        )
        topology_after_smoothing = connectivity_audit(final_net)
        invariant_keys = ("node_count", "edge_count", "component_count", "cyclomatic_number")
        topology_changed = {
            key: (topology_before_smoothing.get(key), topology_after_smoothing.get(key))
            for key in invariant_keys
            if topology_before_smoothing.get(key) != topology_after_smoothing.get(key)
        }
        if topology_changed:
            raise RuntimeError(f"最终整链平滑意外改变拓扑：{topology_changed}")
        write_csv(final_smooth_rows, d8 / "final_chain_smoothing.csv")
        report["stages"]["final_chain_smoothing"] = {
            "chains": len(final_smooth_rows),
            "smoothed_chains": sum(bool(r.get("smoothed")) for r in final_smooth_rows),
            "uturn_chains": sum(bool(r.get("uturn_core")) for r in final_smooth_rows),
            "gap_repair_chains": sum(bool(r.get("contains_gap_repair")) for r in final_smooth_rows),
            "max_hausdorff_shift_m": max(
                [float(r.get("hausdorff_shift_m", 0.0) or 0.0) for r in final_smooth_rows] or [0.0]
            ),
            "sample_spacing_m": max(0.20, float(args.final_smooth_sample_m)),
            "ordinary_sigma_m": max(0.0, float(args.final_smooth_sigma_m)),
            "ordinary_corridor_m": max(0.0, float(args.final_smooth_corridor_m)),
            "uturn_sigma_m": max(0.0, float(args.final_uturn_smooth_sigma_m)),
            "uturn_corridor_m": max(0.0, float(args.final_uturn_smooth_corridor_m)),
            "topology_invariant": True,
        }
    else:
        write_csv([], d8 / "final_chain_smoothing.csv")
        report["stages"]["final_chain_smoothing"] = {"disabled": True}

    dem_report = attach_dem_attributes(
        final_net, existing(args.dem), args.dem_sample_spacing_m,
        progress_every=args.dem_progress_every,
    )
    final_connectivity = connectivity_audit(final_net)
    final_unresolved = unresolved_endpoints(final_net)
    write_csv(final_unresolved, d8 / "unresolved_endpoints_final.csv")
    (d8 / "connectivity_final.json").write_text(
        json.dumps(final_connectivity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_diagnostic_gpkg and final_unresolved:
        write_diagnostic_gpkg(
            d8 / "unresolved_endpoints_final.gpkg",
            {"unresolved_endpoints": (final_unresolved, final_net.crs)},
        )
    paths = write_network_bundle(final_net, d8, "road_network_final_clean")
    report["stages"]["dem"] = dem_report
    report["final"] = {
        **network_metrics(final_net),
        "connectivity": final_connectivity,
        "unresolved_endpoint_count": len(final_unresolved),
        "paths": paths,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["final"], ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(parser().parse_args())
