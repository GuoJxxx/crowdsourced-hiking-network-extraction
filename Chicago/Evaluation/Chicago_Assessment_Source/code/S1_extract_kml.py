#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract measured trajectories (gx:Track) from KML only, saving each independent trajectory segment to a separate CSV.

This script does not extract ordinary KML LineString elements, so planned routes without per-point times
are excluded from the measured GPS trajectory data.

Supported inputs and outputs:
1. A single gx:Track;
2. Multiple gx:Track elements in gx:MultiTrack;
3. Multiple gx:Track elements in MultiGeometry or multiple Placemark elements;
4. A single KML file or a folder of files;
5. Recursive subfolder searches;
6. Per-trajectory CSVs, a complete aggregate, a trajectory inventory, and file summaries;
7. Strict coordinate/time count checks, with warnings for duplicate or reversed times.

Output structure:
output/
├─ observed_tracks/
|  +-- filename__seg001__gxtrack.csv
│  └─ ...
├─ all_observed_points.csv
├─ track_manifest.csv
└─ extraction_summary.csv

Each point includes longitude, latitude, elevation, UTC time, China Standard Time, and source information.
Uses only the Python standard library; no third-party packages such as pandas or fastkml are required.

Run directly:
    python S1_extract_kml.py

Override paths for this run:
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

# KML_NS and GX_NS are XML namespaces and must not store filesystem paths.
KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
NS = {"kml": KML_NS, "gx": GX_NS}
CHINA_TZ = timezone(timedelta(hours=8))

# Default input and output paths. Explicit command-line arguments are recommended on servers.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PACKAGE_ROOT / "data" / "raw_kml"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "S1_Extract"

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
    Return the local name of an XML tag.

    Supports:
    1. {namespace}Placemark
    2. gx:Track
    3. Track
    """
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def direct_children_by_local(element: ET.Element, name: str) -> List[ET.Element]:
    """Find direct children by local name to avoid omissions caused by namespace differences."""
    return [child for child in list(element) if local_name(child.tag) == name]


def descendants_by_local(element: ET.Element, name: str) -> List[ET.Element]:
    """Find all descendant nodes by local name."""
    return [
        node
        for node in element.iter()
        if node is not element and local_name(node.tag) == name
    ]


def first_child_by_local(element: ET.Element, name: str) -> Optional[ET.Element]:
    """Find the first direct child by local name."""
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
    Read child-node text.

    First try the standard namespace path;
    fall back to the local name, supporting both <name> and <kml:name>.
    """
    node = element.find(path, NS)
    if node is None:
        target_name = path.rsplit("/", 1)[-1].split(":", 1)[-1]
        node = first_child_by_local(element, target_name)
    return clean_text(node.text if node is not None else None)


def get_document(root: ET.Element) -> ET.Element:
    """Get the KML Document, or return root if none exists."""
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
    """Parse gx:coord: KML uses whitespace-separated lon lat alt values."""
    parts = clean_text(value).split()
    if len(parts) < 2:
        return None, None, None

    lon = finite_float(parts[0])
    lat = finite_float(parts[1])
    alt = finite_float(parts[2]) if len(parts) >= 3 else None
    return lon, lat, alt


def parse_time(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 time into a UTC datetime; return None if parsing fails."""
    text = clean_text(value)
    if not text:
        return None

    try:
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if dt.tzinfo is None:
        # KML when values should include a time zone.
        # Treat missing time zones as UTC to avoid dependence on the machine's local time zone.
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
    """Create a Windows/Linux-compatible filename while preserving Chinese characters."""
    text = clean_text(text) or "unnamed"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")
    return (text or "unnamed")[:max_length]


def collect_kml_files(input_path: Path, recursive: bool) -> List[Path]:
    """
    Collect KML files.

    Corrections:
    1. Folder searches are case-insensitive;
    2. Suffixes such as .KML and .KmL are also found on Linux.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".kml":
            raise ValueError(f"Input file is not KML: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

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
    Generate a stable, unique, safe label for each KML file.

    Check both duplicate original filenames and collisions after safe_name() sanitization,
    so different KML files cannot overwrite the same output CSV.
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
    Extract gx:Track elements from a single KML file.

    Returns:
    document_name,
    A list of valid trajectory segments,
    The number of gx:Track elements found,
    The number of LineString elements,
    A list of file-level warnings
    """
    try:
        tree = ET.parse(kml_path)
    except ET.ParseError as exc:
        raise ValueError(f"XML parsing failed: {exc}") from exc

    root = tree.getroot()
    document = get_document(root)
    source_file_ref = source_ref or kml_path.name

    document_name = child_text(document, "kml:name") or kml_path.stem

    # Count LineString elements without extracting them.
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

            # gx:coord and when are usually direct children of gx:Track.
            # Local-name matching supports KML with nonstandard namespace declarations.
            coord_nodes = direct_children_by_local(track, "coord")
            time_nodes = direct_children_by_local(track, "when")

            if len(coord_nodes) != len(time_nodes):
                # If any intermediate when is missing from gx:Track, the remaining nodes cannot identify
                # which coordinate lacks a time. Sequential pairing would shift all subsequent times.
                # Therefore skip the entire track rather than produce apparently complete data with incorrect times.
                file_warnings.append(
                    f"{track_id} coordinates={len(coord_nodes)}, timestamps={len(time_nodes)}; "
                    "point-by-point alignment cannot be guaranteed; skipped"
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
                    f"Skipped {segment.invalid_coordinate_count} invalid coordinate points"
                )

            if segment.invalid_time_count:
                segment.warnings.append(
                    f"Unparseable timestamps: {segment.invalid_time_count}; left blank"
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
                    f"Adjacent duplicate timestamps: {duplicate_time_count}; original points retained; "
                    "resolve before calculating speed"
                )

            if reverse_time_count:
                segment.warnings.append(
                    f"Out-of-order timestamps: {reverse_time_count}; original order retained; "
                    "review before further processing"
                )

            missing_elevation_count = sum(
                1
                for point in segment.points
                if point.get("elevation_m") in (None, "")
            )

            if missing_elevation_count:
                segment.warnings.append(f"Missing elevation at {missing_elevation_count} points")

            if len(segment.points) < min_points:
                file_warnings.append(
                    f"{track_id} has only {len(segment.points)} valid points, "
                    f"below min_points={min_points}; no output written"
                )
                continue

            segments.append(segment)

    if gx_track_found == 0:
        file_warnings.append("No gx:Track found; ordinary LineString features were ignored as specified")

    return (
        document_name,
        segments,
        gx_track_found,
        ignored_linestring_count,
        file_warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract observed gx:Track trajectories from KML into one CSV per independent segment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Single KML file or directory containing KML files",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search subdirectories recursively for KML files",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum valid coordinate points required to output a gx:Track",
    )

    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip all_observed_points.csv; write only individual track CSVs and the manifest",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.min_points < 1:
        print("[ERROR] --min-points must be at least 1", file=sys.stderr)
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
        print(f"[ERROR] No KML files found: {input_path}", file=sys.stderr)
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
                        f"[TRACK] {segment.track_id}: {point_count} points -> "
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
                    f"[OK] {kml_path.name}: gx:Track found={gx_track_found}, "
                    f"output={len(segments)}, points={file_point_count}, "
                    f"LineString ignored={ignored_linestring_count}"
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

    print("\nProcessing complete")
    print(f"KML files succeeded: {len(kml_files) - failed}, failed: {failed}")
    print(f"Observed track segments: {total_tracks}")
    print(f"Observed track points: {total_points}")
    print(f"Output directory: {output_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
