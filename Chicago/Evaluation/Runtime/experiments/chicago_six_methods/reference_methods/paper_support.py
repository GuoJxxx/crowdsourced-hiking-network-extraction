"""Rebuild the manuscript's Chicago trajectory-support raster from source trips.

The assessment coverage raster follows the S2--S5 vehicle-compatible
trajectory branch, with the 40 m S2 continuity gate and the recorded S3--S5
settings. Source observations are the public Chicago trips supplied by
``prepare_inputs``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import deque


from workflow.common import source
PACKAGE = source('assessment')
CONFIG = PACKAGE / "configs/chicago_v0_5_strict_mainroad.json"
SCHEMA = PACKAGE / "configs/parameter_schema.json"
FIELDS = (
    "track_id", "source_file", "document_name", "placemark_name",
    "segment_id", "point_index", "raw_point_index", "longitude",
    "latitude", "elevation_m", "time_raw", "time_utc", "time_china",
    "time_source", "clean_track_id", "clean_point_index", "x_m", "y_m",
    "metric_crs", "dt_s", "step_m", "speed_mps", "cumulative_distance_m",
)
CHINA_TIME = timezone(timedelta(hours=8))


def _parts(projected: list[tuple[str, float, float]]) -> list[list[int]]:
    pieces: list[list[int]] = []
    current = [0]
    for index in range(1, len(projected)):
        previous, point = projected[index - 1], projected[index]
        if math.hypot(point[1] - previous[1], point[2] - previous[2]) > 40.0:
            pieces.append(current)
            current = []
        current.append(index)
    pieces.append(current)
    return [indices for indices in pieces if len(indices) >= 3 and sum(
        math.hypot(projected[right][1] - projected[left][1],
                   projected[right][2] - projected[left][2])
        for left, right in zip(indices, indices[1:])
    ) >= 8.0]


def write_split_s2(parsed: list[tuple], destination: Path) -> dict[str, int]:
    """Write the S2 points retained by the historical 40 m continuity gate."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    points = parts = 0
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(FIELDS)
        for member, projected, _geographic in parsed:
            source = Path(member).name
            track_id = f"{Path(source).stem}__seg001"
            for part_number, indices in enumerate(_parts(projected), 1):
                parts += 1
                for clean_index, index in enumerate(indices, 1):
                    _source_row, x, y = projected[index]
                    timestamp = float(_source_row.split()[2])
                    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    utc = moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                    china = moment.astimezone(CHINA_TIME).isoformat(timespec="milliseconds")
                    writer.writerow((
                        track_id, source, "Chicago_trips", Path(source).stem,
                        1, index + 1, index + 1, x, y, "", str(timestamp),
                        utc, china, "unix_timestamp", f"{track_id}__road{part_number:03d}",
                        clean_index, x, y, "EPSG:26916", "", "", "", "",
                    ))
                    points += 1
    return {"points": points, "parts": parts}


def load_projected_trips(manifest_path: Path) -> list[tuple]:
    """Read prepared archive derivatives using their original trip identities."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["trips"]
    if len(entries) != 889:
        raise ValueError(f"Paper support requires 889 prepared trips, found {len(entries)}")
    # Prepared filenames are sequential archive-order indices; the S1 source
    # filenames use original trip numbers.  The manifest preserves that map.
    entries = sorted(entries, key=lambda row: (
        f"{Path(row['member']).stem}__seg001__gxtrack.csv"))
    tracks_dir = (manifest_path.parent / "tracks_projected").resolve()
    parsed = []
    for entry in entries:
        path = (manifest_path.parent / entry["projected_path"]).resolve()
        if tracks_dir not in path.parents or not path.is_file():
            raise ValueError(f"Invalid prepared trip path: {entry['projected_path']}")
        projected = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path.name}:{line_number}: expected x y timestamp")
            projected.append((line, float(fields[0]), float(fields[1])))
        parsed.append((entry["member"], projected, None))
    return parsed


def _stage_args(stage: str, config: dict, schema: dict) -> list[str]:
    values = config["stages"][stage]
    arguments: list[str] = []
    for parameter in schema["stages"][stage]["parameters"]:
        value = values.get(parameter["dest"])
        kind = parameter["kind"]
        if kind == "bool_optional":
            arguments.append(parameter["positive_option"] if value else parameter["negative_option"])
        elif kind == "bool_flag":
            if bool(value) == bool(parameter["flag_value"]):
                arguments.append(parameter["option"])
        elif value is not None and str(value).strip():
            arguments.extend((parameter["option"], str(value)))
    return arguments


def _run(stage: str, source: Path, output: Path, config: dict, schema: dict,
         gravity_report: Path | None = None) -> None:
    command = [
        sys.executable, "-B", str(PACKAGE / "code" / f"{stage}.py"),
        "--input", str(source), "--output", str(output),
    ]
    if gravity_report is not None:
        command.extend(("--gravity-report", str(gravity_report)))
    command.extend(_stage_args(stage.split("_")[0], config, schema))
    descriptions = {
        "S3": "activity weighting for the assessment coverage raster",
        "S4": "trajectory adjustment for the assessment coverage raster",
        "S5": "rasterization of the shared assessment coverage",
    }
    label = stage.split("_")[0]
    log_path = output.parent / "logs" / (stage + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ASSESSMENT INPUTS] {label}: {descriptions[label]}", flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    tail = deque(maxlen=20)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        with subprocess.Popen(command, cwd=PACKAGE / "code", env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as child:
            for line in child.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                if line.lstrip().startswith("["):
                    print("[ASSESSMENT DETAIL] " + line.rstrip(), flush=True)
            code = child.wait()
        log.write(f"\nexit_code: {code}\n")
    if code:
        print(f"[ASSESSMENT INPUTS] {label} failed; log: {log_path}", flush=True)
        for line in tail:
            print("[ASSESSMENT DETAIL] " + line.rstrip(), flush=True)
        raise subprocess.CalledProcessError(code, command, output="".join(tail))
    print(f"[ASSESSMENT INPUTS] {label} complete; log: {log_path}", flush=True)


def build_paper_support(parsed: list[tuple], output_root: Path) -> tuple[Path, dict[str, int]]:
    """Generate S2--S5 under ``output_root`` and return the S5 raw union."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    config["stages"]["S3"].update(
        low_speed_fraction_min=1.0, mainroad_min_weight=0.35,
        max_output_link_m=0.0,
    )
    config["stages"]["S4"].update(min_support_tracks=2, min_road_weight=0.35)
    config["stages"]["S5"].update(
        max_link_m=0.0, min_road_weight=0.35,
        require_mainroad_candidate=False,
    )
    s2, s3, s4, s5 = (output_root / name for name in (
        "S2_Prepare", "S3_Activity", "S4_Gravity", "S5_KDE"))
    counts = write_split_s2(parsed, s2 / "all_cleaned_points.csv")
    _run("S3_activity_zone_filter", s2 / "all_cleaned_points.csv", s3, config, schema)
    _run("S4_uturn_consensus_gravity", s3 / "road_candidate_points.csv", s4, config, schema)
    _run("S5_directional_kde", s4 / "all_cleaned_points.csv", s5, config, schema,
         s4 / "gravity_uturn_report.json")
    raster = s5 / "raw_trajectory_union.tif"
    if not raster.is_file():
        raise FileNotFoundError(f"S5 did not create its raw trajectory union: {raster}")
    return raster, counts
