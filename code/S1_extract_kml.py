#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅从 KML 中提取实测轨迹（gx:Track），并按“一条独立轨迹段一个 CSV”保存。

本脚本不会提取普通 KML LineString，因此不会把无逐点时间的计划路线混入
实测 GPS 轨迹数据。

支持：
1. 单个 gx:Track；
2. gx:MultiTrack 中的多个 gx:Track；
3. MultiGeometry 或多个 Placemark 中的多个 gx:Track；
4. 单个 KML 文件或批量文件夹；
5. 递归搜索子文件夹；
6. 每条轨迹单独 CSV、全量汇总、轨迹清单、文件级摘要；
7. 严格检查坐标—时间数量一致性，并提示重复或倒序时间。

输出结构：
output/
├─ observed_tracks/
│  ├─ 文件名__seg001__gxtrack.csv
│  └─ ...
├─ all_observed_points.csv
├─ track_manifest.csv
└─ extraction_summary.csv

每个点包含：经度、纬度、高程、UTC 时间、北京时间及来源信息。
仅使用 Python 标准库，无需安装 pandas、fastkml 等第三方包。

直接运行：
    python S1_extract_kml.py

临时覆盖路径：
    python S1_extract_kml.py input_folder -o output
    python S1_extract_kml.py input_folder -o output --no-combined
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

# 注意：KML_NS 和 GX_NS 是 XML 命名空间，不能用来保存磁盘路径。
KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
NS = {"kml": KML_NS, "gx": GX_NS}
CHINA_TZ = timezone(timedelta(hours=8))

# GitHub 发布版使用仓库相对路径；也可通过命令行参数显式覆盖。
DEFAULT_INPUT_PATH = Path("data/raw_kml")
DEFAULT_OUTPUT_DIR = Path("outputs/S1_TracksFromKML")

POINT_FIELDS = [
    "track_id",
    "source_file",
    "document_name",
    "placemark_name",
    "segment_id",
    "point_index",
    "raw_point_index",
    "longitude",
    "latitude",
    "elevation_m",
    "time_raw",
    "time_utc",
    "time_china",
    "time_source",
]

MANIFEST_FIELDS = [
    "track_id",
    "source_file",
    "document_name",
    "placemark_name",
    "segment_id",
    "output_file",
    "point_count",
    "start_time_utc",
    "end_time_utc",
    "duration_s",
    "time_count",
    "time_completeness",
    "elevation_count",
    "elevation_completeness",
    "min_elevation_m",
    "max_elevation_m",
    "mean_elevation_m",
    "start_longitude",
    "start_latitude",
    "end_longitude",
    "end_latitude",
    "min_longitude",
    "min_latitude",
    "max_longitude",
    "max_latitude",
    "warnings",
]

SUMMARY_FIELDS = [
    "source_file",
    "document_name",
    "gx_track_found",
    "gx_track_written",
    "observed_point_count",
    "ignored_linestring_count",
    "invalid_coordinate_count",
    "missing_time_count",
    "invalid_time_count",
    "missing_elevation_count",
    "warnings",
]


@dataclass
class Segment:
    source_file: str
    document_name: str
    placemark_name: str
    segment_id: int
    track_id: str
    points: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    invalid_coordinate_count: int = 0
    invalid_time_count: int = 0


def local_name(tag: str) -> str:
    """
    返回 XML 标签的本地名。

    兼容：
    1. {namespace}Placemark
    2. gx:Track
    3. Track
    """
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def direct_children_by_local(element: ET.Element, name: str) -> List[ET.Element]:
    """查找直接子节点，按本地名匹配，避免命名空间差异导致漏读。"""
    return [child for child in list(element) if local_name(child.tag) == name]


def descendants_by_local(element: ET.Element, name: str) -> List[ET.Element]:
    """查找所有后代节点，按本地名匹配。"""
    return [
        node
        for node in element.iter()
        if node is not element and local_name(node.tag) == name
    ]


def first_child_by_local(element: ET.Element, name: str) -> Optional[ET.Element]:
    """查找第一个直接子节点，按本地名匹配。"""
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def clean_text(value: Optional[str]) -> str:
    return (value or "").strip()


def finite_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_lon_lat(lon: Optional[float], lat: Optional[float]) -> bool:
    return (
        lon is not None
        and lat is not None
        and -180.0 <= lon <= 180.0
        and -90.0 <= lat <= 90.0
    )


def child_text(element: ET.Element, path: str) -> str:
    """
    读取子节点文本。

    优先按标准命名空间路径读取；
    失败后按本地名读取，例如 <name> 或 <kml:name> 都能兼容。
    """
    node = element.find(path, NS)
    if node is None:
        target_name = path.rsplit("/", 1)[-1].split(":", 1)[-1]
        node = first_child_by_local(element, target_name)
    return clean_text(node.text if node is not None else None)


def get_document(root: ET.Element) -> ET.Element:
    """获取 KML Document；若不存在，则返回 root。"""
    if local_name(root.tag) == "Document":
        return root

    document = root.find("kml:Document", NS)
    if document is None:
        matches = descendants_by_local(root, "Document")
        document = matches[0] if matches else None

    return document if document is not None else root


def parse_gx_coord(
    value: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """解析 gx:coord。KML gx:coord 顺序为 lon lat alt，使用空白分隔。"""
    parts = clean_text(value).split()
    if len(parts) < 2:
        return None, None, None

    lon = finite_float(parts[0])
    lat = finite_float(parts[1])
    alt = finite_float(parts[2]) if len(parts) >= 3 else None
    return lon, lat, alt


def parse_time(value: Optional[str]) -> Optional[datetime]:
    """解析 ISO 8601 时间，返回 UTC datetime；无法解析返回 None。"""
    text = clean_text(value)
    if not text:
        return None

    try:
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if dt.tzinfo is None:
        # KML 的 when 理论上应包含时区。
        # 缺失时区时按 UTC 处理，避免受机器本地时区影响。
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def format_utc(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def format_china(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.astimezone(CHINA_TZ).isoformat(timespec="milliseconds")


def safe_name(text: str, max_length: int = 100) -> str:
    """生成可用于 Windows/Linux 的文件名，保留中文。"""
    text = clean_text(text) or "unnamed"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")
    return (text or "unnamed")[:max_length]


def collect_kml_files(input_path: Path, recursive: bool) -> List[Path]:
    """
    收集 KML 文件。

    修正点：
    1. 文件夹模式下大小写不敏感；
    2. Linux 上不会漏掉 .KML、.KmL 等后缀。
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".kml":
            raise ValueError(f"输入文件不是 KML：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() == ".kml"
        ),
        key=lambda path: str(path).lower(),
    )


def build_source_labels(kml_files: Sequence[Path], input_root: Path) -> Dict[Path, str]:
    """
    为每个 KML 生成稳定且唯一的安全标签。

    不仅检查原始文件名是否重复，也检查文件名经过 safe_name() 清洗后是否
    发生碰撞，避免不同 KML 覆盖同一个输出 CSV。
    """
    base_labels = {path: safe_name(path.stem) for path in kml_files}
    label_counts = Counter(base_labels.values())

    labels: Dict[Path, str] = {}
    for path in kml_files:
        label = base_labels[path]

        if label_counts[label] > 1:
            try:
                relative = str(path.relative_to(input_root))
            except ValueError:
                relative = str(path)

            digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
            label = f"{label}__{digest}"

        labels[path] = label

    return labels


def write_csv(
    rows: Iterable[Dict[str, object]],
    output_path: Path,
    fields: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_dict_writer(
    output_path: Path,
    fields: Sequence[str],
) -> Tuple[TextIO, csv.DictWriter]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file = output_path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return file, writer


def numeric_values(points: Sequence[Dict[str, object]], field: str) -> List[float]:
    output: List[float] = []

    for point in points:
        value = point.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            output.append(float(value))

    return output


def build_manifest_row(segment: Segment, relative_output: str) -> Dict[str, object]:
    points = segment.points
    elevations = numeric_values(points, "elevation_m")
    lons = numeric_values(points, "longitude")
    lats = numeric_values(points, "latitude")

    times = [
        parse_time(str(point.get("time_utc") or ""))
        for point in points
        if point.get("time_utc")
    ]
    valid_times = [dt for dt in times if dt is not None]

    point_count = len(points)
    time_count = len(valid_times)
    elevation_count = len(elevations)

    start_time = min(valid_times) if valid_times else None
    end_time = max(valid_times) if valid_times else None

    duration_s: object = ""
    if start_time is not None and end_time is not None:
        duration_s = round((end_time - start_time).total_seconds(), 3)

    return {
        "track_id": segment.track_id,
        "source_file": segment.source_file,
        "document_name": segment.document_name,
        "placemark_name": segment.placemark_name,
        "segment_id": segment.segment_id,
        "output_file": relative_output,
        "point_count": point_count,
        "start_time_utc": format_utc(start_time),
        "end_time_utc": format_utc(end_time),
        "duration_s": duration_s,
        "time_count": time_count,
        "time_completeness": round(time_count / point_count, 6) if point_count else 0,
        "elevation_count": elevation_count,
        "elevation_completeness": (
            round(elevation_count / point_count, 6) if point_count else 0
        ),
        "min_elevation_m": min(elevations) if elevations else "",
        "max_elevation_m": max(elevations) if elevations else "",
        "mean_elevation_m": round(fmean(elevations), 3) if elevations else "",
        "start_longitude": points[0]["longitude"] if points else "",
        "start_latitude": points[0]["latitude"] if points else "",
        "end_longitude": points[-1]["longitude"] if points else "",
        "end_latitude": points[-1]["latitude"] if points else "",
        "min_longitude": min(lons) if lons else "",
        "min_latitude": min(lats) if lats else "",
        "max_longitude": max(lons) if lons else "",
        "max_latitude": max(lats) if lats else "",
        "warnings": " | ".join(segment.warnings),
    }


def extract_gx_tracks(
    kml_path: Path,
    source_label: str,
    min_points: int,
    source_ref: Optional[str] = None,
) -> Tuple[str, List[Segment], int, int, List[str]]:
    """
    从单个 KML 文件中提取 gx:Track。

    返回：
    document_name,
    有效轨迹段列表,
    发现的 gx:Track 数,
    LineString 数,
    文件级警告列表
    """
    try:
        tree = ET.parse(kml_path)
    except ET.ParseError as exc:
        raise ValueError(f"XML 解析失败：{exc}") from exc

    root = tree.getroot()
    document = get_document(root)
    source_file_ref = source_ref or kml_path.name

    document_name = child_text(document, "kml:name") or kml_path.stem

    # 只统计 LineString，不提取。
    ignored_linestring_count = len(descendants_by_local(document, "LineString"))

    placemarks = document.findall(".//kml:Placemark", NS)
    if not placemarks:
        placemarks = descendants_by_local(document, "Placemark")

    segments: List[Segment] = []
    file_warnings: List[str] = []
    gx_track_found = 0
    segment_id = 0

    for pm_index, placemark in enumerate(placemarks, start=1):
        placemark_name = child_text(placemark, "kml:name") or f"Placemark_{pm_index}"

        tracks = placemark.findall(".//gx:Track", NS)
        if not tracks:
            tracks = descendants_by_local(placemark, "Track")

        for track in tracks:
            gx_track_found += 1
            segment_id += 1
            track_id = f"{source_label}__seg{segment_id:03d}"

            segment = Segment(
                source_file=source_file_ref,
                document_name=document_name,
                placemark_name=placemark_name,
                segment_id=segment_id,
                track_id=track_id,
            )

            # gx:coord 和 when 在 gx:Track 中通常为直接子节点。
            # 用本地名匹配可兼容命名空间声明不规范的 KML。
            coord_nodes = direct_children_by_local(track, "coord")
            time_nodes = direct_children_by_local(track, "when")

            if len(coord_nodes) != len(time_nodes):
                # gx:Track 中缺少任意一个中间 when 时，仅靠剩余节点无法判断
                # 缺失发生在哪个坐标位置。继续按序配对会使后续所有时间错位。
                # 因此严格跳过该轨迹，而不是生成表面完整但时间错误的数据。
                file_warnings.append(
                    f"{track_id} 坐标数={len(coord_nodes)}、时间数={len(time_nodes)}，"
                    "无法保证逐点对应，已跳过"
                )
                continue

            output_index = 0

            for raw_index, coord_node in enumerate(coord_nodes, start=1):
                lon, lat, alt = parse_gx_coord(coord_node.text)

                if not valid_lon_lat(lon, lat):
                    segment.invalid_coordinate_count += 1
                    continue

                raw_time = (
                    clean_text(time_nodes[raw_index - 1].text)
                    if raw_index - 1 < len(time_nodes)
                    else ""
                )

                dt = parse_time(raw_time)
                if raw_time and dt is None:
                    segment.invalid_time_count += 1

                time_utc = format_utc(dt)

                output_index += 1
                segment.points.append(
                    {
                        "track_id": track_id,
                        "source_file": source_file_ref,
                        "document_name": document_name,
                        "placemark_name": placemark_name,
                        "segment_id": segment_id,
                        "point_index": output_index,
                        "raw_point_index": raw_index,
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "elevation_m": alt,
                        "time_raw": raw_time,
                        "time_utc": time_utc,
                        "time_china": format_china(dt),
                        "time_source": (
                            "gx:when" if dt is not None else "missing_or_invalid"
                        ),
                    }
                )

            if segment.invalid_coordinate_count:
                segment.warnings.append(
                    f"跳过无效坐标点 {segment.invalid_coordinate_count} 个"
                )

            if segment.invalid_time_count:
                segment.warnings.append(
                    f"无法解析时间 {segment.invalid_time_count} 个，已留空"
                )

            parsed_times = [
                parse_time(str(point.get("time_utc") or ""))
                for point in segment.points
            ]

            duplicate_time_count = 0
            reverse_time_count = 0

            for previous, current in zip(parsed_times, parsed_times[1:]):
                if previous is None or current is None:
                    continue

                delta_s = (current - previous).total_seconds()

                if delta_s == 0:
                    duplicate_time_count += 1
                elif delta_s < 0:
                    reverse_time_count += 1

            if duplicate_time_count:
                segment.warnings.append(
                    f"存在相邻重复时间戳 {duplicate_time_count} 处；保留原始点，"
                    "后续计算速度前需处理"
                )

            if reverse_time_count:
                segment.warnings.append(
                    f"存在时间倒序 {reverse_time_count} 处；保留原始顺序，"
                    "后续处理前需核查"
                )

            missing_elevation_count = sum(
                1
                for point in segment.points
                if point.get("elevation_m") in (None, "")
            )

            if missing_elevation_count:
                segment.warnings.append(f"高程缺失 {missing_elevation_count} 个")

            if len(segment.points) < min_points:
                file_warnings.append(
                    f"{track_id} 仅有 {len(segment.points)} 个有效点，"
                    f"少于 min_points={min_points}，未输出"
                )
                continue

            segments.append(segment)

    if gx_track_found == 0:
        file_warnings.append("未发现 gx:Track；普通 LineString 已按要求忽略")

    return (
        document_name,
        segments,
        gx_track_found,
        ignored_linestring_count,
        file_warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅提取 KML 中的实测 gx:Track，并按独立轨迹段输出 CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="单个 KML 文件或包含 KML 的文件夹",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出文件夹",
    )

    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否递归搜索子文件夹中的 KML",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="一条 gx:Track 至少包含多少个有效坐标点才输出",
    )

    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="不生成 all_observed_points.csv，仅输出单轨迹 CSV 和清单",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.min_points < 1:
        print("[ERROR] --min-points 必须大于或等于 1", file=sys.stderr)
        return 2

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    observed_dir = output_dir / "observed_tracks"
    observed_dir.mkdir(parents=True, exist_ok=True)

    try:
        kml_files = collect_kml_files(input_path, args.recursive)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not kml_files:
        print(f"[ERROR] 未找到 KML：{input_path}", file=sys.stderr)
        return 2

    input_root = input_path if input_path.is_dir() else input_path.parent
    source_labels = build_source_labels(kml_files, input_root)

    manifest_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    failed = 0
    total_tracks = 0
    total_points = 0

    combined_file: Optional[TextIO] = None
    combined_writer: Optional[csv.DictWriter] = None

    if not args.no_combined:
        combined_file, combined_writer = open_dict_writer(
            output_dir / "all_observed_points.csv", POINT_FIELDS
        )

    try:
        for kml_path in kml_files:
            source_ref = kml_path.name

            try:
                try:
                    source_ref = kml_path.relative_to(input_root).as_posix()
                except ValueError:
                    source_ref = kml_path.name

                (
                    document_name,
                    segments,
                    gx_track_found,
                    ignored_linestring_count,
                    file_warnings,
                ) = extract_gx_tracks(
                    kml_path=kml_path,
                    source_label=source_labels[kml_path],
                    min_points=args.min_points,
                    source_ref=source_ref,
                )

                file_point_count = 0
                file_invalid_coordinate_count = 0
                file_invalid_time_count = 0
                file_missing_time_count = 0
                file_missing_elevation_count = 0

                for segment in segments:
                    filename = (
                        f"{source_labels[kml_path]}__seg{segment.segment_id:03d}"
                        "__gxtrack.csv"
                    )

                    output_path = observed_dir / filename
                    write_csv(segment.points, output_path, POINT_FIELDS)

                    relative_output = output_path.relative_to(output_dir).as_posix()
                    manifest_rows.append(build_manifest_row(segment, relative_output))

                    if combined_writer is not None:
                        combined_writer.writerows(segment.points)

                    point_count = len(segment.points)
                    file_point_count += point_count
                    file_invalid_coordinate_count += segment.invalid_coordinate_count
                    file_invalid_time_count += segment.invalid_time_count

                    file_missing_time_count += sum(
                        1 for point in segment.points if not point.get("time_utc")
                    )

                    file_missing_elevation_count += sum(
                        1
                        for point in segment.points
                        if point.get("elevation_m") in (None, "")
                    )

                    total_tracks += 1
                    total_points += point_count

                    print(
                        f"[TRACK] {segment.track_id}: {point_count} 点 -> "
                        f"{relative_output}"
                    )

                all_warnings = list(file_warnings)

                for segment in segments:
                    all_warnings.extend(
                        f"segment {segment.segment_id}: {warning}"
                        for warning in segment.warnings
                    )

                summary_rows.append(
                    {
                        "source_file": source_ref,
                        "document_name": document_name,
                        "gx_track_found": gx_track_found,
                        "gx_track_written": len(segments),
                        "observed_point_count": file_point_count,
                        "ignored_linestring_count": ignored_linestring_count,
                        "invalid_coordinate_count": file_invalid_coordinate_count,
                        "missing_time_count": file_missing_time_count,
                        "invalid_time_count": file_invalid_time_count,
                        "missing_elevation_count": file_missing_elevation_count,
                        "warnings": " | ".join(all_warnings),
                    }
                )

                print(
                    f"[OK] {kml_path.name}: gx:Track发现={gx_track_found}，"
                    f"输出={len(segments)}，点数={file_point_count}，"
                    f"忽略LineString={ignored_linestring_count}"
                )

            except Exception as exc:
                failed += 1
                print(f"[FAILED] {kml_path}: {exc}", file=sys.stderr)

                summary_rows.append(
                    {
                        "source_file": source_ref,
                        "document_name": "",
                        "gx_track_found": 0,
                        "gx_track_written": 0,
                        "observed_point_count": 0,
                        "ignored_linestring_count": 0,
                        "invalid_coordinate_count": 0,
                        "missing_time_count": 0,
                        "invalid_time_count": 0,
                        "missing_elevation_count": 0,
                        "warnings": f"FAILED: {exc}",
                    }
                )

    finally:
        if combined_file is not None:
            combined_file.close()

    write_csv(manifest_rows, output_dir / "track_manifest.csv", MANIFEST_FIELDS)
    write_csv(summary_rows, output_dir / "extraction_summary.csv", SUMMARY_FIELDS)

    print("\n处理完成")
    print(f"KML 成功：{len(kml_files) - failed}，失败：{failed}")
    print(f"实测轨迹段：{total_tracks}")
    print(f"实测轨迹点：{total_points}")
    print(f"输出目录：{output_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())