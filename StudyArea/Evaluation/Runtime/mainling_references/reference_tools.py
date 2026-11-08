from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


def prepare_compat_sources(
    original_root: Path | str,
    output_root: Path | str,
    edelkamp_source=None, biagioni_source=None,
) -> dict[str, object]:
    """Copy the two public implementations and mechanically translate Python 2."""
    from workflow.conversion import make_refactoring_tool

    original_root = Path(original_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Compatibility output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    method_dirs = {
        "edelkamp2003": Path(edelkamp_source) if edelkamp_source else original_root / "edelkamp2003",
        "biagoni2012": Path(biagioni_source) if biagioni_source else original_root / "biagoni2012",
    }
    for method, source_dir in method_dirs.items():
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Original source directory not found: {source_dir}")
        shutil.copytree(source_dir, output_root / method)

    refactor = make_refactoring_tool()
    counts: dict[str, int] = {}
    for method in method_dirs:
        python_files = sorted((output_root / method).rglob("*.py"))
        for source_path in python_files:
            source = source_path.read_text(encoding="utf-8").expandtabs(8)
            try:
                converted = str(
                    refactor.refactor_string(source + "\n", str(source_path))
                )
            except Exception as error:
                raise RuntimeError(f"Python 2 conversion failed for {source_path}") from error
            converted = converted.replace(
                "from rtree import Rtree",
                "from rtree.index import Index as Rtree",
            )
            source_path.write_text(converted, encoding="utf-8", newline="\n")
        counts[f"{method.split('20')[0].rstrip('i')}_python_files"] = len(
            python_files
        )

    compatibility_dir = next(p for p in Path(__file__).resolve().parents if (p/'settings.json').is_file())/'StudyArea/Methods/Biagioni/StudyArea_Biagioni_Compatibility'
    biagioni_dir = output_root / "biagoni2012"
    shutil.copy2(compatibility_dir / "biagioni_kde.py", biagioni_dir / "kde.py")
    shutil.copy2(
        compatibility_dir / "biagioni_skeleton.py",
        biagioni_dir / "skeleton.py",
    )
    shutil.copy2(
        compatibility_dir / "biagioni_subiterations.py",
        biagioni_dir / "subiterations.py",
    )
    shutil.copy2(
        compatibility_dir / "biagioni_spatialfunclib_accel.py",
        biagioni_dir / "pylibs" / "spatialfunclib_accel.py",
    )

    graph_extract = biagioni_dir / "graph_extract.py"
    graph_source = graph_extract.read_text(encoding="utf-8")
    graph_source = graph_source.replace(
        "from scipy.ndimage import imread\nfrom scipy.misc import imsave\n",
        "import cv2\n",
    )
    graph_source = graph_source.replace(
        "skeleton = imread(skeleton_filename)",
        "skeleton = cv2.imread(skeleton_filename, cv2.IMREAD_UNCHANGED)",
    )
    graph_source = graph_source.replace("np.bool", "np.bool_").replace(
        "np.int", "int"
    )
    graph_extract.write_text(graph_source, encoding="utf-8", newline="\n")

    streetmap = biagioni_dir / "streetmap.py"
    streetmap_source = streetmap.read_text(encoding="utf-8").replace(
        "import pyximport; pyximport.install()\n", ""
    )
    streetmap.write_text(streetmap_source, encoding="utf-8", newline="\n")

    viterbi = biagioni_dir / "viterbi.py"
    viterbi_source = viterbi.read_text(encoding="utf-8").replace(
        "(max_prob,max_from) = max(from_probs)",
        "(max_prob,max_from) = max(from_probs, key=lambda item: item[0])",
    )
    viterbi.write_text(viterbi_source, encoding="utf-8", newline="\n")

    refine_topology = biagioni_dir / "refine_topology.py"
    refine_source = refine_topology.read_text(encoding="utf-8")
    refine_helper = '''def collapse_would_create_parallel_edge(
    segment_head_node,
    segment_tail_node,
    head_edge_neighbors,
    tail_edge_neighbors,
):
    """Protect StreetMap's simple directed-edge lookup during a collapse."""
    prospective_keys = []
    for endpoint, neighbors in (
        (segment_head_node, head_edge_neighbors),
        (segment_tail_node, tail_edge_neighbors),
    ):
        for neighbor in neighbors:
            if endpoint in neighbor.out_nodes:
                prospective_keys.append(("incoming", neighbor.id))
            if endpoint in neighbor.in_nodes:
                prospective_keys.append(("outgoing", neighbor.id))
    return len(prospective_keys) != len(set(prospective_keys))


'''
    refine_source = refine_source.replace(
        "import math\n\nclass RefineTopology:",
        "import math\n\n" + refine_helper + "class RefineTopology:",
    )
    refine_guard_anchor = '''                if (segment.tail_edge.in_node in tail_edge_neighbors):
                    tail_edge_neighbors.remove(segment.tail_edge.in_node)
                
                for neighbor in head_edge_neighbors:
'''
    refine_guard = '''                if (segment.tail_edge.in_node in tail_edge_neighbors):
                    tail_edge_neighbors.remove(segment.tail_edge.in_node)

                if collapse_would_create_parallel_edge(
                    segment_head_node,
                    segment_tail_node,
                    head_edge_neighbors,
                    tail_edge_neighbors,
                ):
                    print("skipping collapse: parallel edge would be created")
                    del self.graphdb.nodes[new_intersection.id]
                    node_id -= 1
                    closed_segments_list.append(segment)
                    continue
                
                for neighbor in head_edge_neighbors:
'''
    if refine_guard_anchor not in refine_source:
        raise RuntimeError("Could not install Biagioni topology safety guard")
    refine_source = refine_source.replace(refine_guard_anchor, refine_guard, 1)
    refine_topology.write_text(refine_source, encoding="utf-8", newline="\n")

    process_matches = biagioni_dir / "process_map_matches.py"
    process_source = process_matches.read_text(encoding="utf-8")
    coalesce_helper = '''def coalesce_segment_pair(graphdb, head_segment, tail_segment):
    """Merge once and safely ignore a duplicate scheduled merge."""
    if (
        head_segment.id not in graphdb.segments
        or tail_segment.id not in graphdb.segments
        or head_segment.id == tail_segment.id
    ):
        return False
    head_segment.edges.extend(tail_segment.edges)
    for edge in tail_segment.edges:
        edge.segment = head_segment
    del graphdb.segments[tail_segment.id]
    return True


'''
    process_source = process_source.replace(
        "import math\n\nclass ProcessMapMatches:",
        "import math\n\n" + coalesce_helper + "class ProcessMapMatches:",
    )
    old_coalesce = '''            for head_segment, tail_segment in merge_segments:
                head_segment.edges.extend(tail_segment.edges)
                
                for edge in tail_segment.edges:
                    edge.segment = head_segment #self.graphdb.segment_lookup_table[edge.id] = head_segment
                
                del self.graphdb.segments[tail_segment.id]
'''
    new_coalesce = '''            for head_segment, tail_segment in merge_segments:
                coalesce_segment_pair(self.graphdb, head_segment, tail_segment)
'''
    if old_coalesce not in process_source:
        raise RuntimeError("Could not install Biagioni segment coalescing guard")
    process_source = process_source.replace(old_coalesce, new_coalesce, 1)
    process_matches.write_text(process_source, encoding="utf-8", newline="\n")

    return {
        "source_root": str(original_root.resolve()),
        "output_root": str(output_root.resolve()),
        "edelkamp_python_files": len(
            list((output_root / "edelkamp2003").rglob("*.py"))
        ),
        "biagioni_python_files": len(
            list((output_root / "biagoni2012").rglob("*.py"))
        ),
    }


def _run_logged_stage(
    name: str,
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = time.perf_counter() - started
    log_path = cwd / f"{name}.log"
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    (cwd / "runtime_seconds.txt").write_text(
        f"{elapsed:.6f}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr)[-6000:]
        raise RuntimeError(f"Stage {name} failed:\n{tail}")
    return {
        "name": name,
        "status": "completed",
        "command": command,
        "cwd": str(cwd.resolve()),
        "elapsed_seconds": elapsed,
        "log": str(log_path.resolve()),
    }


def run_edelkamp_pipeline(
    source_dir: Path | str,
    trips_dir: Path | str,
    output_dir: Path | str,
    *,
    timeout_seconds: int = 86400,
    seed_interval_m: float = 50.0,
    bearing_limit_deg: float = 45.0,
    intra_cluster_distance_m: float = 20.0,
    edge_bounding_box_m: float = 80.0,
    movement_threshold_m: float = 0.01,
    trip_limit: int | None = None,
) -> dict[str, object]:
    """Run the mechanically ported Edelkamp implementation reproducibly."""
    source_dir = Path(source_dir).resolve()
    trips_dir = Path(trips_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir() or not trips_dir.is_dir():
        raise FileNotFoundError("Edelkamp source or trip directory is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_trips = output_dir / "trips"
    if trips_dir != work_trips.resolve():
        if work_trips.exists():
            raise FileExistsError(f"Edelkamp work trips already exist: {work_trips}")
        shutil.copytree(trips_dir, work_trips)
    database = output_dir / "edelkamp_graph.db"
    if database.exists():
        raise FileExistsError(f"Edelkamp database already exists: {database}")

    command = [
        sys.executable,
        str(source_dir / "edelkamp2003.py"),
        "-i",
        str(seed_interval_m),
        "-b",
        str(bearing_limit_deg),
        "-d",
        str(intra_cluster_distance_m),
        "-s",
        str(edge_bounding_box_m),
        "-t",
        str(movement_threshold_m),
    ]
    if trip_limit is not None:
        command.extend(["-n", str(trip_limit)])
    report = _run_logged_stage("edelkamp", command, output_dir, timeout_seconds)
    result = {
        "method": "Edelkamp",
        "source_dir": str(source_dir),
        "trips_dir": str(work_trips.resolve()),
        "parameters": {
            "seed_interval_m": seed_interval_m,
            "bearing_limit_deg": bearing_limit_deg,
            "intra_cluster_distance_m": intra_cluster_distance_m,
            "edge_bounding_box_m": edge_bounding_box_m,
            "movement_threshold_m": movement_threshold_m,
            "trip_limit": trip_limit,
        },
        "runtime_total_seconds": report["elapsed_seconds"],
        "stage": report,
        "final_database": str(database.resolve()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_biagioni_pipeline(
    source_dir: Path | str,
    trips_dir: Path | str,
    output_dir: Path | str,
    *,
    timeout_seconds: int = 86400,
    cell_size_m: int = 1,
    gaussian_blur: int = 17,
    start_stage: int = 1,
) -> dict[str, object]:
    source_dir = Path(source_dir).resolve()
    trips_dir = Path(trips_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir() or not trips_dir.is_dir():
        raise FileNotFoundError("Biagioni source or trip directory is missing")
    if not 1 <= start_stage <= 8:
        raise ValueError("Biagioni start_stage must be between 1 and 8")

    stage_names = [
        "01_kde",
        "02_skeleton",
        "03_graph_extract",
        "04_matched_1",
        "05_prune_1",
        "06_refine",
        "07_matched_2",
        "08_final",
    ]
    stages = {name: output_dir / name for name in stage_names}
    for stage_dir in stages.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        (
            "01_kde",
            [
                sys.executable,
                str(source_dir / "kde.py"),
                "-c",
                str(cell_size_m),
                "-b",
                str(gaussian_blur),
                "-p",
                str(trips_dir),
            ],
        ),
        (
            "02_skeleton",
            [
                sys.executable,
                str(source_dir / "skeleton.py"),
                str(stages["01_kde"] / "kde.png"),
                "skeleton.png",
            ],
        ),
        (
            "03_graph_extract",
            [
                sys.executable,
                str(source_dir / "graph_extract.py"),
                str(stages["02_skeleton"] / "skeleton.png"),
                str(stages["01_kde"] / "bounding_box.txt"),
                "skeleton_map.db",
            ],
        ),
        (
            "04_matched_1",
            [
                sys.executable,
                str(source_dir / "graphdb_matcher_run.py"),
                "-d",
                str(stages["03_graph_extract"] / "skeleton_map.db"),
                "-t",
                str(trips_dir),
                "-o",
                str(stages["04_matched_1"]),
            ],
        ),
        (
            "05_prune_1",
            [
                sys.executable,
                str(source_dir / "process_map_matches.py"),
                "-d",
                str(stages["03_graph_extract"] / "skeleton_map.db"),
                "-t",
                str(stages["04_matched_1"]),
                "-o",
                "skeleton_map_mm1.db",
            ],
        ),
        (
            "06_refine",
            [
                sys.executable,
                str(source_dir / "refine_topology.py"),
                "-d",
                str(stages["05_prune_1"] / "skeleton_map_mm1.db"),
                "-t",
                str(stages["05_prune_1"] / "skeleton_map_mm1_traces.txt"),
                "-o",
                "skeleton_map_mm1_tr.db",
            ],
        ),
        (
            "07_matched_2",
            [
                sys.executable,
                str(source_dir / "graphdb_matcher_run.py"),
                "-d",
                str(stages["06_refine"] / "skeleton_map_mm1_tr.db"),
                "-t",
                str(trips_dir),
                "-o",
                str(stages["07_matched_2"]),
            ],
        ),
        (
            "08_final",
            [
                sys.executable,
                str(source_dir / "process_map_matches.py"),
                "-d",
                str(stages["06_refine"] / "skeleton_map_mm1_tr.db"),
                "-t",
                str(stages["07_matched_2"]),
                "-o",
                "skeleton_map_mm2.db",
            ],
        ),
    ]

    reports = []
    for stage_number, (name, command) in enumerate(commands, start=1):
        if stage_number < start_stage:
            runtime_path = stages[name] / "runtime_seconds.txt"
            log_path = stages[name] / f"{name}.log"
            if not runtime_path.is_file() or not log_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: completed stage evidence missing for {name}"
                )
            reports.append(
                {
                    "name": name,
                    "status": "reused",
                    "command": command,
                    "cwd": str(stages[name].resolve()),
                    "elapsed_seconds": float(
                        runtime_path.read_text(encoding="utf-8").strip()
                    ),
                    "log": str(log_path.resolve()),
                }
            )
        else:
            reports.append(
                _run_logged_stage(name, command, stages[name], timeout_seconds)
            )
    final_database = stages["08_final"] / "skeleton_map_mm2.db"
    result = {
        "method": "Biagioni",
        "source_dir": str(source_dir),
        "trips_dir": str(trips_dir),
        "cell_size_m": cell_size_m,
        "gaussian_blur": gaussian_blur,
        "resumed_from_stage": start_stage,
        "stages": reports,
        "runtime_total_seconds": sum(
            float(report["elapsed_seconds"]) for report in reports
        ),
        "final_database": str(final_database.resolve()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _parse_timestamp(value: str, fallback: float) -> float:
    text = (value or "").strip()
    if not text:
        return fallback
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _read_track(
    path: Path,
    *,
    require_projected: bool = False,
    expected_metric_crs: str = "EPSG:32647",
) -> list[dict[str, float | bool | str]]:
    points: list[dict[str, float | bool | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader):
            try:
                longitude = float(row.get("longitude", ""))
                latitude = float(row.get("latitude", ""))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(longitude) and math.isfinite(latitude)):
                continue
            time_text = (row.get("time_utc", "") or "").strip()
            timestamp = _parse_timestamp(time_text, fallback=float(row_number))
            timestamp_fallback = not time_text
            x = y = math.nan
            if require_projected:
                metric_crs = (row.get("metric_crs", "") or "").strip()
                if metric_crs != expected_metric_crs:
                    raise ValueError(
                        f"Unexpected metric_crs in {path}:{row_number + 2}: "
                        f"{metric_crs!r}; expected {expected_metric_crs}"
                    )
                try:
                    x_text = (row.get("x_m", "") or "").strip()
                    y_text = (row.get("y_m", "") or "").strip()
                    x = float(x_text)
                    y = float(y_text)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Missing numeric x_m/y_m in {path}:{row_number + 2}"
                    ) from error
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(
                        f"Non-finite x_m/y_m in {path}:{row_number + 2}"
                    )
            points.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": timestamp,
                    "timestamp_fallback": timestamp_fallback,
                    "x": x,
                    "y": y,
                    "x_text": x_text if require_projected else "",
                    "y_text": y_text if require_projected else "",
                }
            )
    return points


def _read_karagiorgou_track(
    path: Path,
    *,
    expected_metric_crs: str = "EPSG:32647",
) -> list[dict[str, float | bool | str]]:
    """Read every cleaned CSV data row using its authoritative metric columns."""
    points: list[dict[str, float | bool | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader):
            csv_row = row_number + 2
            metric_crs = (row.get("metric_crs", "") or "").strip()
            if metric_crs != expected_metric_crs:
                raise ValueError(
                    f"Unexpected metric_crs in {path}:{csv_row}: "
                    f"{metric_crs!r}; expected {expected_metric_crs}"
                )
            try:
                x_text = (row.get("x_m", "") or "").strip()
                y_text = (row.get("y_m", "") or "").strip()
                x = float(x_text)
                y = float(y_text)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Missing numeric x_m/y_m in {path}:{csv_row}"
                ) from error
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Non-finite x_m/y_m in {path}:{csv_row}")
            time_text = (row.get("time_utc", "") or "").strip()
            points.append(
                {
                    "timestamp": _parse_timestamp(
                        time_text, fallback=float(row_number)
                    ),
                    "timestamp_fallback": not time_text,
                    "x": x,
                    "y": y,
                    "x_text": x_text,
                    "y_text": y_text,
                }
            )
    return points


def _write_edelkamp_trip(
    path: Path, points: list[dict[str, float | bool | str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, point in enumerate(points):
            latitude = float(point["latitude"])
            longitude = float(point["longitude"])
            timestamp = float(point["timestamp"])
            previous_id = "None" if index == 0 else str(index - 1)
            next_id = "None" if index == len(points) - 1 else str(index + 1)
            handle.write(
                f"{index},{latitude:.10f},{longitude:.10f},{timestamp:.3f},"
                f"{previous_id},{next_id}\n"
            )


def _write_biagioni_trip(
    path: Path, points: list[dict[str, float | bool | str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, point in enumerate(points):
            latitude = float(point["latitude"])
            longitude = float(point["longitude"])
            timestamp = float(point["timestamp"])
            handle.write(
                f"{index},{latitude:.10f},{longitude:.10f},{timestamp:.3f}\n"
            )


def _write_karagiorgou_trip(
    path: Path,
    points: list[dict[str, float | bool | str]],
) -> None:
    """Write cleaned ``x_m y_m timestamp`` values without reprojection."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for point in points:
            timestamp = float(point["timestamp"])
            handle.write(
                f"{point['x_text']} {point['y_text']} {timestamp:.3f}\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_track_order(path: Path) -> tuple[str, str]:
    """Keep original S2 filename order when Windows requires a compact basename."""
    stem = path.stem
    if stem.startswith('track_') and len(stem) == 30 and all(c in '0123456789abcdef' for c in stem[6:]):
        with path.open('r', newline='', encoding='utf-8-sig') as handle:
            track_id = next(csv.DictReader(handle), {}).get('clean_track_id')
        if track_id:
            text = track_id.strip()
            for char in '<>:"/\\|?*':
                text = text.replace(char, '_')
            text = text.strip(' ._') or 'track'
            if len(text) > 170:
                text = text[:158] + '__' + hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
            return text + '.csv', path.name
    return path.name, path.name


def convert_clean_tracks(
    source_dir: Path | str,
    edelkamp_dir: Path | str,
    biagioni_dir: Path | str,
    *,
    karagiorgou_dir: Path | str | None = None,
    projected_crs: str = "EPSG:32647",
    limit: int | None = None,
) -> dict[str, object]:
    source_dir = Path(source_dir)
    edelkamp_dir = Path(edelkamp_dir)
    biagioni_dir = Path(biagioni_dir)
    karagiorgou_path = Path(karagiorgou_dir) if karagiorgou_dir is not None else None
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Clean-track directory not found: {source_dir}")
    edelkamp_dir.mkdir(parents=True, exist_ok=True)
    biagioni_dir.mkdir(parents=True, exist_ok=True)
    if karagiorgou_path is not None:
        karagiorgou_path.mkdir(parents=True, exist_ok=True)
    if list(edelkamp_dir.glob("trip_*.txt")) or list(
        biagioni_dir.glob("trip_*.txt")
    ):
        raise FileExistsError("Target trip directory already contains generated trips")
    if karagiorgou_path is not None and list(karagiorgou_path.glob("trip_*.txt")):
        raise FileExistsError("Target Karagiorgou directory already contains generated trips")

    source_files = sorted(source_dir.glob("*.csv"), key=_source_track_order)
    if limit is not None:
        source_files = source_files[:limit]

    rows: list[dict[str, object]] = []
    tracks_skipped_short = 0
    points_written = 0
    legacy_points_written = 0
    time_utc_fallback_tracks = 0
    time_utc_fallback_points = 0
    zero_dt_tracks = 0
    zero_dt_steps = 0
    consecutive_duplicate_xy_steps_retained = 0
    for source_path in source_files:
        legacy_points = _read_track(source_path)
        karagiorgou_points = (
            _read_karagiorgou_track(
                source_path, expected_metric_crs=projected_crs
            )
            if karagiorgou_path is not None
            else legacy_points
        )
        if len(karagiorgou_points) < 2:
            tracks_skipped_short += 1
            continue
        if len(legacy_points) < 2:
            raise ValueError(
                f"Track has {len(karagiorgou_points)} metric rows but fewer than two "
                f"valid longitude/latitude rows required by legacy methods: {source_path}"
            )
        trip_id = len(rows)
        edelkamp_path = edelkamp_dir / f"trip_{trip_id}.txt"
        biagioni_path = biagioni_dir / f"trip_{trip_id}.txt"
        _write_edelkamp_trip(edelkamp_path, legacy_points)
        _write_biagioni_trip(biagioni_path, legacy_points)
        karagiorgou_path_item = None
        if karagiorgou_path is not None:
            karagiorgou_path_item = karagiorgou_path / f"trip_{trip_id}.txt"
            _write_karagiorgou_trip(karagiorgou_path_item, karagiorgou_points)
        fallback_count = sum(
            bool(point["timestamp_fallback"]) for point in karagiorgou_points
        )
        if fallback_count:
            time_utc_fallback_tracks += 1
            time_utc_fallback_points += fallback_count
        track_zero_dt = sum(
            float(current["timestamp"]) == float(previous["timestamp"])
            for previous, current in zip(
                karagiorgou_points, karagiorgou_points[1:]
            )
        )
        if track_zero_dt:
            zero_dt_tracks += 1
            zero_dt_steps += track_zero_dt
        consecutive_duplicate_xy_steps_retained += sum(
            float(current["x"]) == float(previous["x"])
            and float(current["y"]) == float(previous["y"])
            for previous, current in zip(
                karagiorgou_points, karagiorgou_points[1:]
            )
        ) if karagiorgou_path is not None else 0
        points_written += len(karagiorgou_points)
        legacy_points_written += len(legacy_points)
        row = {
                "trip_id": trip_id,
                "source_file": source_path.name,
                "points": len(karagiorgou_points),
                "source_rows": len(karagiorgou_points),
                "karagiorgou_points": len(karagiorgou_points),
                "legacy_points": len(legacy_points),
                "source_sha256": _sha256(source_path),
                "edelkamp_sha256": _sha256(edelkamp_path),
                "biagioni_sha256": _sha256(biagioni_path),
            }
        if karagiorgou_path_item is not None:
            row["karagiorgou_sha256"] = _sha256(karagiorgou_path_item)
        rows.append(row)

    manifest = {
        "source_dir": str(source_dir.resolve()),
        "source_files_seen": len(source_files),
        "tracks_written": len(rows),
        "tracks_skipped_short": tracks_skipped_short,
        "points_written": points_written,
        "legacy_points_written": legacy_points_written,
        "time_utc_fallback_tracks": time_utc_fallback_tracks,
        "time_utc_fallback_points": time_utc_fallback_points,
        "zero_dt_tracks": zero_dt_tracks,
        "zero_dt_steps": zero_dt_steps,
        "consecutive_duplicate_xy_steps_retained": consecutive_duplicate_xy_steps_retained,
        "edelkamp_dir": str(edelkamp_dir.resolve()),
        "biagioni_dir": str(biagioni_dir.resolve()),
        "trips": rows,
    }
    if karagiorgou_path is not None:
        manifest["karagiorgou_dir"] = str(karagiorgou_path.resolve())
        manifest["karagiorgou_crs"] = projected_crs
        manifest["karagiorgou_coordinate_source"] = "cleaned_csv_x_m_y_m"
        manifest["karagiorgou_point_policy"] = "preserve_all_cleaned_points"
    return manifest


def _load_sqlite_graph(database: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(database) as connection:
        nodes = pd.read_sql_query(
            "SELECT id AS source_id, longitude, latitude FROM nodes",
            connection,
        )
        edges = pd.read_sql_query(
            "SELECT id AS source_edge_id, in_node AS u_source, "
            "out_node AS v_source FROM edges",
            connection,
        )
    return nodes, edges


def _graph_metrics(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> dict[str, object]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(int))
    graph.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))
    active_nodes = {node for edge in graph.edges for node in edge}
    active_graph = graph.subgraph(active_nodes).copy()
    components = list(nx.connected_components(active_graph)) if active_nodes else []
    degrees = dict(active_graph.degree())
    return {
        "nodes": int(len(nodes)),
        "active_nodes": int(len(active_nodes)),
        "edges": int(len(edges)),
        "length_m": float(edges.geometry.length.sum()) if len(edges) else 0.0,
        "connected_components": len(components),
        "largest_component_nodes": max(map(len, components), default=0),
        "endpoints": sum(degree == 1 for degree in degrees.values()),
        "intersections_degree_ge_3": sum(degree >= 3 for degree in degrees.values()),
        "cycle_rank": int(len(edges) - len(active_nodes) + len(components)),
    }


def normalize_sqlite_graph(
    database: Path | str,
    output_dir: Path | str,
    *,
    method: str,
    target_crs: str = "EPSG:32647",
) -> dict[str, object]:
    database = Path(database)
    output_dir = Path(output_dir)
    source_nodes, source_edges = _load_sqlite_graph(database)
    source_nodes = source_nodes.drop_duplicates("source_id", keep="first")
    source_nodes["source_id"] = source_nodes["source_id"].astype(int)

    geographic_nodes = gpd.GeoDataFrame(
        source_nodes,
        geometry=[
            Point(longitude, latitude)
            for longitude, latitude in source_nodes[
                ["longitude", "latitude"]
            ].itertuples(index=False)
        ],
        crs="EPSG:4326",
    ).to_crs(target_crs)
    source_to_node_id = {
        source_id: node_id
        for node_id, source_id in enumerate(sorted(geographic_nodes["source_id"]))
    }
    source_to_xy = {
        int(row.source_id): (float(row.geometry.x), float(row.geometry.y))
        for row in geographic_nodes.itertuples(index=False)
    }
    nodes = gpd.GeoDataFrame(
        [
            {
                "node_id": source_to_node_id[source_id],
                "source_id": source_id,
                "x": source_to_xy[source_id][0],
                "y": source_to_xy[source_id][1],
                "geometry": Point(source_to_xy[source_id]),
            }
            for source_id in sorted(source_to_node_id)
        ],
        crs=target_crs,
    )

    seen: set[tuple[int, int]] = set()
    normalized_edges: list[dict[str, object]] = []
    invalid_references = 0
    self_loops = 0
    duplicate_directions = 0
    for source_edge_id, u_source, v_source in source_edges.itertuples(index=False):
        u_source = int(u_source)
        v_source = int(v_source)
        if u_source not in source_to_node_id or v_source not in source_to_node_id:
            invalid_references += 1
            continue
        u = source_to_node_id[u_source]
        v = source_to_node_id[v_source]
        if u == v:
            self_loops += 1
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            duplicate_directions += 1
            continue
        seen.add(key)
        geometry = LineString([source_to_xy[u_source], source_to_xy[v_source]])
        if geometry.length == 0:
            self_loops += 1
            continue
        normalized_edges.append(
            {
                "edge_id": len(normalized_edges),
                "source_edge_id": int(source_edge_id),
                "u": u,
                "v": v,
                "length_m": float(geometry.length),
                "geometry": geometry,
            }
        )
    edges = gpd.GeoDataFrame(normalized_edges, crs=target_crs)

    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / "road_network.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    edges.to_file(gpkg, layer="edges", driver="GPKG")
    nodes.drop(columns="geometry").to_csv(output_dir / "nodes.csv", index=False)
    edge_table = edges.drop(columns="geometry").copy()
    edge_table["wkt"] = edges.geometry.to_wkt()
    edge_table.to_csv(output_dir / "edges.csv", index=False)

    with (output_dir / "benchmark_vertices.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for node_id, x, y in nodes[["node_id", "x", "y"]].itertuples(index=False):
            handle.write(f"{int(node_id)},{float(x):.6f},{float(y):.6f}\n")
    with (output_dir / "benchmark_edges.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for edge_id, u, v in edges[["edge_id", "u", "v"]].itertuples(index=False):
            handle.write(f"{int(edge_id)},{int(u)},{int(v)},1\n")

    result = {
        "method": method,
        "source_crs": "EPSG:4326",
        "normalization": {
            "target_crs": target_crs,
            "source_vertices": int(len(source_nodes)),
            "source_edges": int(len(source_edges)),
            "invalid_edge_references_removed": invalid_references,
            "self_loops_removed": self_loops,
            "reverse_or_duplicate_edges_collapsed": duplicate_directions,
        },
        "graph": {"crs": target_crs, **_graph_metrics(nodes, edges)},
        "outputs": {
            "gpkg": str(gpkg.resolve()),
            "nodes_csv": str((output_dir / "nodes.csv").resolve()),
            "edges_csv": str((output_dir / "edges.csv").resolve()),
            "benchmark_vertices": str(
                (output_dir / "benchmark_vertices.txt").resolve()
            ),
            "benchmark_edges": str((output_dir / "benchmark_edges.txt").resolve()),
        },
    }
    (output_dir / "graph_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def normalize_text_graph(
    vertices_path: Path | str,
    edges_path: Path | str,
    output_dir: Path | str,
    *,
    method: str,
    source_crs: str = "EPSG:32647",
    target_crs: str = "EPSG:32647",
) -> dict[str, object]:
    """Normalize the original TraceBundle vertex/edge text files."""
    vertices_path = Path(vertices_path)
    edges_path = Path(edges_path)
    output_dir = Path(output_dir)

    source_nodes = pd.read_csv(
        vertices_path,
        header=None,
        names=["source_id", "x", "y"],
    )
    source_edges = pd.read_csv(
        edges_path,
        header=None,
        names=["source_edge_id", "u_source", "v_source", "direction"],
    )
    source_nodes = source_nodes.drop_duplicates("source_id", keep="first")
    source_nodes["source_id"] = source_nodes["source_id"].astype(int)
    projected_nodes = gpd.GeoDataFrame(
        source_nodes,
        geometry=[
            Point(x, y)
            for x, y in source_nodes[["x", "y"]].itertuples(index=False)
        ],
        crs=source_crs,
    ).to_crs(target_crs)
    source_to_node_id = {
        source_id: node_id
        for node_id, source_id in enumerate(sorted(projected_nodes["source_id"]))
    }
    source_to_xy = {
        int(row.source_id): (float(row.geometry.x), float(row.geometry.y))
        for row in projected_nodes.itertuples(index=False)
    }
    nodes = gpd.GeoDataFrame(
        [
            {
                "node_id": source_to_node_id[source_id],
                "source_id": source_id,
                "x": source_to_xy[source_id][0],
                "y": source_to_xy[source_id][1],
                "geometry": Point(source_to_xy[source_id]),
            }
            for source_id in sorted(source_to_node_id)
        ],
        crs=target_crs,
    )

    seen: set[tuple[int, int]] = set()
    normalized_edges: list[dict[str, object]] = []
    invalid_references = 0
    self_loops = 0
    duplicate_directions = 0
    for source_edge_id, u_source, v_source, direction in source_edges.itertuples(
        index=False
    ):
        u_source = int(u_source)
        v_source = int(v_source)
        if u_source not in source_to_node_id or v_source not in source_to_node_id:
            invalid_references += 1
            continue
        u = source_to_node_id[u_source]
        v = source_to_node_id[v_source]
        if u == v:
            self_loops += 1
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            duplicate_directions += 1
            continue
        seen.add(key)
        geometry = LineString([source_to_xy[u_source], source_to_xy[v_source]])
        if geometry.length == 0:
            self_loops += 1
            continue
        normalized_edges.append(
            {
                "edge_id": len(normalized_edges),
                "source_edge_id": int(source_edge_id),
                "source_direction": int(direction),
                "u": u,
                "v": v,
                "length_m": float(geometry.length),
                "geometry": geometry,
            }
        )
    edges = gpd.GeoDataFrame(normalized_edges, crs=target_crs)

    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / "road_network.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    edges.to_file(gpkg, layer="edges", driver="GPKG")
    nodes.drop(columns="geometry").to_csv(output_dir / "nodes.csv", index=False)
    edge_table = edges.drop(columns="geometry").copy()
    edge_table["wkt"] = edges.geometry.to_wkt()
    edge_table.to_csv(output_dir / "edges.csv", index=False)
    with (output_dir / "benchmark_vertices.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for node_id, x, y in nodes[["node_id", "x", "y"]].itertuples(index=False):
            handle.write(f"{int(node_id)},{float(x):.6f},{float(y):.6f}\n")
    with (output_dir / "benchmark_edges.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for edge_id, u, v in edges[["edge_id", "u", "v"]].itertuples(index=False):
            handle.write(f"{int(edge_id)},{int(u)},{int(v)},1\n")

    result = {
        "method": method,
        "source_crs": source_crs,
        "normalization": {
            "target_crs": target_crs,
            "source_vertices": int(len(source_nodes)),
            "source_edges": int(len(source_edges)),
            "invalid_edge_references_removed": invalid_references,
            "self_loops_removed": self_loops,
            "reverse_or_duplicate_edges_collapsed": duplicate_directions,
        },
        "graph": {"crs": target_crs, **_graph_metrics(nodes, edges)},
        "outputs": {
            "gpkg": str(gpkg.resolve()),
            "nodes_csv": str((output_dir / "nodes.csv").resolve()),
            "edges_csv": str((output_dir / "edges.csv").resolve()),
            "benchmark_vertices": str(
                (output_dir / "benchmark_vertices.txt").resolve()
            ),
            "benchmark_edges": str((output_dir / "benchmark_edges.txt").resolve()),
        },
    }
    (output_dir / "graph_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
