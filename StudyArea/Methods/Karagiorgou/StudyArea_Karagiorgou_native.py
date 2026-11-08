"""Faithful non-interactive runner for the public TraceBundle MATLAB source."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time


EXECUTION_CLASS = "official MATLAB source"
PARAMETERS = {"cluster_distance_m": 40, "angle_tolerance_deg": 12.5}
_PICKER = "[fnames,path]=uigetfile('*.txt','...select trajectory files!','MultiSelect','on');"
INTERMEDIATES = (
    "slk.mat",
    "pts_ln.mat",
    "vehicles.mat",
    "centre_pts.mat",
    "pts_to_inter.mat",
    "clusterseeds.mat",
    "inter_to_cluster.mat",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_filename_key(filename: str) -> tuple[tuple[int, str | int], ...]:
    """Match the digit-aware ordering used by the Windows file picker."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", Path(filename).name)
    )


def _manifest_entries(input_manifest: Path) -> list[dict[str, object]]:
    document = json.loads(input_manifest.read_text(encoding="utf-8"))
    if isinstance(document, dict) and "input_conversion" in document:
        document = document["input_conversion"]
    entries = document.get("trips") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Preparation manifest requires trajectory identity entries")
    return entries


def _track_files(
    directory: Path,
    input_manifest: Path | None = None,
) -> tuple[list[Path], list[dict[str, object]]]:
    files = sorted(
        directory.glob("trip_*.txt"),
        key=lambda path: _natural_filename_key(path.name),
    )
    if not files or any(not re.fullmatch(r"trip_\d+", path.stem) for path in files):
        raise ValueError("Projected trip_*.txt tracks are required")
    identifiers = [int(path.stem[5:]) for path in files]
    if len(set(identifiers)) != len(files):
        raise ValueError("Duplicate projected track identifiers")
    order: list[dict[str, object]] = [
        {"prepared_file": path.name, "trip_id": int(path.stem[5:])}
        for path in files
    ]
    if input_manifest is not None:
        selected = {path.resolve() for path in files}
        mapped: list[tuple[str, Path, int]] = []
        for entry in _manifest_entries(input_manifest):
            if not isinstance(entry, dict):
                raise ValueError("Invalid prepared trajectory identity")
            trip_id = int(entry.get("trip_id", -1))
            path = (directory / f"trip_{trip_id}.txt").resolve()
            source_file = str(entry.get("source_file", ""))
            if path not in selected or not source_file:
                raise ValueError("Invalid prepared trajectory path or original identity")
            mapped.append((source_file, path, trip_id))
        if len({item[1] for item in mapped}) != len(mapped) or {
            item[1] for item in mapped
        } != selected:
            raise ValueError("Preparation manifest must map each track exactly once")
        mapped.sort(key=lambda item: _natural_filename_key(item[0]))
        files = [item[1] for item in mapped]
        order = [
            {
                "prepared_file": path.name,
                "trip_id": trip_id,
                "source_file": source_file,
            }
            for source_file, path, trip_id in mapped
        ]
    for path in files:
        rows = path.read_text(encoding="utf-8").splitlines()
        if len(rows) < 2:
            raise ValueError(f"Track needs at least two rows: {path}")
        for row_number, row in enumerate(rows, 1):
            fields = row.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Track requires three numeric columns: {path}:{row_number}"
                )
            try:
                values = tuple(float(value) for value in fields)
            except ValueError as error:
                raise ValueError(
                    f"Track requires three numeric columns: {path}:{row_number}"
                ) from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Track requires finite numeric columns: {path}:{row_number}")
            if not (100000 <= values[0] <= 900000 and 0 <= values[1] <= 10000000):
                raise ValueError(
                    f"Track requires projected UTM metre coordinates: {path}:{row_number}"
                )
    return files, order


def stage_native_source(
    source_dir: Path | str,
    tracks: Path | str,
    native: Path | str,
    *,
    input_manifest: Path | None = None,
) -> Path:
    """Copy public code and replace only its interactive trajectory picker."""
    source_dir = Path(source_dir).resolve()
    tracks = Path(tracks).resolve()
    native = Path(native).resolve()
    extraction = source_dir / "source" / "intersection_nodes_extraction.m"
    bundling = source_dir / "source" / "tracebundle.m"
    libraries = source_dir / "libraries"
    for required in (extraction, bundling, libraries):
        if not required.exists():
            raise FileNotFoundError(f"Public TraceBundle dependency is missing: {required}")
    files, track_order = _track_files(tracks, input_manifest)
    original = extraction.read_bytes()
    picker = _PICKER.encode("utf-8")
    if original.count(picker) != 1:
        raise ValueError("Unexpected public file-picker source; refusing to patch")
    if native.exists():
        raise FileExistsError(f"Native stage already exists: {native}")
    native.mkdir(parents=True)
    (native / extraction.name).write_bytes(
        original.replace(picker, b"[fnames,path]=tracebundle_select_tracks();")
    )
    shutil.copy2(bundling, native / bundling.name)
    shutil.copytree(libraries, native / "libraries")
    provenance = {
        "execution_class": EXECUTION_CLASS,
        "parameters": PARAMETERS,
        "source_root": str(source_dir),
        "source_sha256": {
            path.relative_to(source_dir).as_posix(): _sha256(path)
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        },
        "track_sha256": {path.name: _sha256(path) for path in files},
        "track_order": track_order,
        "expected_intermediates": list(INTERMEDIATES),
        "file_picker_original": _PICKER,
        "file_picker_shim": "[fnames,path]=tracebundle_select_tracks();",
    }
    (native / "source_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (native / "track_manifest.txt").write_text(
        "\n".join(path.name for path in files) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (native / "track_directory.txt").write_text(
        str(tracks) + "\n", encoding="utf-8", newline="\n"
    )
    (native / "tracebundle_select_tracks.m").write_text(
        "function [fnames,path] = tracebundle_select_tracks()\n"
        "root = fileparts(mfilename('fullpath'));\n"
        "fid = fopen(fullfile(root, 'track_directory.txt'), 'r');\n"
        "if fid < 0, error('Missing track directory'); end\n"
        "path = fgetl(fid); fclose(fid);\n"
        "if ~ischar(path), error('Empty track directory'); end\n"
        "path = [path filesep];\n"
        "fid = fopen(fullfile(root, 'track_manifest.txt'), 'r');\n"
        "if fid < 0, error('Missing track manifest'); end\n"
        "fnames = {};\n"
        "while ~feof(fid)\n"
        "    name = fgetl(fid);\n"
        "    if ischar(name) && ~isempty(name), fnames{end+1} = name; end\n"
        "end\n"
        "fclose(fid);\n"
        "if isempty(fnames), error('Empty track manifest'); end\n"
        "end\n",
        encoding="utf-8",
        newline="\n",
    )
    return native


def validate_native_outputs(vertices: Path, edges: Path) -> tuple[int, int]:
    """Validate original TraceBundle text records before normalization."""
    for path in (vertices, edges):
        if not path.is_file():
            raise RuntimeError(f"TraceBundle output is missing: {path}")
    vertex_rows = vertices.read_text(encoding="utf-8").splitlines()
    edge_rows = edges.read_text(encoding="utf-8").splitlines()
    if not vertex_rows or not edge_rows:
        raise ValueError("TraceBundle output is empty")
    vertex_ids: set[int] = set()
    for row_number, row in enumerate(vertex_rows, 1):
        fields = row.split(",")
        if len(fields) != 3:
            raise ValueError(f"Invalid vertex columns at row {row_number}")
        identifier = int(fields[0])
        x, y = map(float, fields[1:])
        if identifier <= 0 or identifier in vertex_ids or not all(
            math.isfinite(value) for value in (x, y)
        ):
            raise ValueError(f"Invalid vertex at row {row_number}")
        if not (100000 <= x <= 900000 and 0 <= y <= 10000000):
            raise ValueError(f"Vertex is outside projected UTM metre range at row {row_number}")
        vertex_ids.add(identifier)
    edge_ids: set[int] = set()
    for row_number, row in enumerate(edge_rows, 1):
        fields = row.split(",")
        if len(fields) != 4:
            raise ValueError(f"Invalid edge columns at row {row_number}")
        identifier, start, end, direction = map(int, fields)
        if identifier <= 0 or identifier in edge_ids or start == end:
            raise ValueError(f"Invalid edge at row {row_number}")
        if start not in vertex_ids or end not in vertex_ids:
            raise ValueError(f"Edge references missing vertex at row {row_number}")
        if direction not in (0, 1):
            raise ValueError(f"Invalid edge direction at row {row_number}")
        edge_ids.add(identifier)
    return len(vertex_rows), len(edge_rows)


def run_native(
    source_dir: Path | str,
    tracks: Path | str,
    native: Path | str,
    *,
    matlab_executable: str | None = None,
    input_manifest: Path | None = None,
) -> dict[str, object]:
    """Run the two public scripts in their original order."""
    executable = matlab_executable or shutil.which("matlab")
    if not executable:
        raise RuntimeError(
            "MATLAB executable is required for authoritative TraceBundle execution"
        )
    native = stage_native_source(
        source_dir, tracks, native, input_manifest=input_manifest
    )
    intermediate_checks = " ".join(
        f"assert(exist('{name}','file') == 2, 'Missing TraceBundle intermediate: {name}');"
        for name in INTERMEDIATES
    )
    command = (
        "cd('"
        + str(native).replace("'", "''")
        + "'); addpath(genpath('libraries')); "
        "assert(exist('pdist','file') ~= 0 && exist('squareform','file') ~= 0, "
        "'Statistics and Machine Learning Toolbox is required'); "
        "run('intersection_nodes_extraction.m'); "
        + intermediate_checks
        + " run('tracebundle.m');"
    )
    started = time.perf_counter()
    result = subprocess.run(
        [executable, "-batch", command],
        cwd=native,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log = native / "matlab.log"
    log.write_text(
        f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
        newline="\n",
    )
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(
            f"MATLAB TraceBundle failed with exit code {result.returncode}: "
            f"{diagnostic}; see {log}"
        )
    vertices = native / "tracebundle_vertices.txt"
    edges = native / "tracebundle_edges.txt"
    vertex_count, edge_count = validate_native_outputs(vertices, edges)
    report = {
        "method": "Karagiorgou",
        "execution_class": EXECUTION_CLASS,
        "parameters": PARAMETERS,
        "runtime_total_seconds": elapsed,
        "vertices": vertex_count,
        "edges": edge_count,
        "source_crs": "EPSG:32647",
        "stage_order": ["intersection_nodes_extraction.m", "tracebundle.m"],
        "source_manifest": str((native / "source_manifest.json").resolve()),
        "matlab_log": str(log.resolve()),
        "final_vertices": str(vertices.resolve()),
        "final_edges": str(edges.resolve()),
    }
    (native.parent / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
