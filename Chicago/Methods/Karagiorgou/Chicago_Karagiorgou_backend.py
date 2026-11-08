"""Run the preserved, complete TraceBundle MATLAB source in an owned directory."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from collections import deque

from experiments.chicago_six_methods.reference_methods.common import MethodRunResult, sha256_file
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext
from .Chicago_Karagiorgou_progress import HELPER, instrument_matlab_source
from workflow.runtime_language import english_text, matlab_environment


PACKAGE = Path(__file__).resolve().parent
SOURCE_DIR = Path(__file__).resolve().parent / "Chicago_Karagiorgou_Source/tracebundle/matlab"
EXTRACTION = SOURCE_DIR / "source/intersection_nodes_extraction.m"
BUNDLING = SOURCE_DIR / "source/tracebundle.m"
SOURCE_PATHS = (SOURCE_DIR, HELPER, Path(__file__).with_name('Chicago_Karagiorgou_progress.py'))
EXECUTION_CLASS = "official MATLAB source"
PARAMETERS = {"cluster_distance_m": 40, "angle_tolerance_deg": 12.5}
_PICKER = "[fnames,path]=uigetfile('*.txt','...select trajectory files!','MultiSelect','on');"
INTERMEDIATES = ("slk.mat", "pts_ln.mat", "vehicles.mat", "centre_pts.mat",
                 "pts_to_inter.mat", "clusterseeds.mat", "inter_to_cluster.mat")


def _natural_filename_key(filename: str) -> tuple[tuple[int, str | int], ...]:
    """Match Windows' digit-aware ordering for the public file picker."""
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", Path(filename).name))


def _track_files(directory: Path, input_manifest: Path | None = None) -> tuple[list[Path], list[dict]]:
    files = sorted(directory.glob("trip_*.txt"), key=lambda path: _natural_filename_key(path.name))
    if not files or any(not re.fullmatch(r"trip_\d+", path.stem) for path in files):
        raise ValueError("Projected trip_*.txt tracks are required")
    if len({int(path.stem[5:]) for path in files}) != len(files):
        raise ValueError("Duplicate projected track identifiers")
    order = [{"prepared_file": path.name, "original_trip_id": int(path.stem[5:])}
             for path in files]
    if input_manifest is not None:
        document = json.loads(input_manifest.read_text(encoding="utf-8"))
        entries = document.get("trips") if isinstance(document, dict) else None
        selected_paths = {path.resolve() for path in files}
        if isinstance(entries, list) and entries:
            mapped = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Invalid prepared trajectory identity")
                identity = re.fullmatch(r"chicago/trips/trip_(\d+)\.txt", str(entry.get("member", "")))
                path = (input_manifest.parent / str(entry.get("projected_path", ""))).resolve()
                if identity is None or path.parent != directory.resolve() or path not in selected_paths:
                    raise ValueError("Invalid prepared trajectory path or original identity")
                mapped.append((int(identity[1]), path, entry["member"]))
            if len({item[0] for item in mapped}) != len(mapped):
                raise ValueError("Duplicate original numeric trajectory identifiers")
            if len({item[1] for item in mapped}) != len(mapped) or {item[1] for item in mapped} != selected_paths:
                raise ValueError("Prepared manifest must map each selected trajectory exactly once")
            mapped.sort(key=lambda item: _natural_filename_key(item[2]))
            files = [item[1] for item in mapped]
            order = [{"prepared_file": path.name, "original_trip_id": identifier, "original_member": member}
                     for identifier, path, member in mapped]
        else:
            recorded = document.get("track_order") if isinstance(document, dict) else None
            recorded_hashes = document.get("track_sha256") if isinstance(document, dict) else None
            if not isinstance(recorded, list) or not recorded or not isinstance(recorded_hashes, dict):
                raise ValueError("Prepared manifest requires trajectory identity entries")
            by_name = {path.name: path.resolve() for path in files}
            recorded_files = []
            recorded_order = []
            for entry in recorded:
                if not isinstance(entry, dict):
                    raise ValueError("Invalid recorded trajectory identity")
                name = str(entry.get("prepared_file", ""))
                path = by_name.get(name)
                if path is None:
                    raise ValueError("Recorded trajectory is absent from the selected inputs")
                expected_hash = str(recorded_hashes.get(name, "")).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                    raise ValueError("Recorded trajectory hash is missing or invalid")
                if sha256_file(path).lower() != expected_hash:
                    raise ValueError(f"Recorded trajectory hash mismatch: {name}")
                identifier = entry.get("trip_id", entry.get("original_trip_id"))
                if not isinstance(identifier, int):
                    raise ValueError("Recorded trajectory identifier must be an integer")
                provenance = {
                    "prepared_file": name,
                    "original_trip_id": identifier,
                }
                if entry.get("source_file") is not None:
                    provenance["source_file"] = str(entry["source_file"])
                recorded_files.append(path)
                recorded_order.append(provenance)
            if len(set(recorded_files)) != len(recorded_files) or set(recorded_files) != selected_paths:
                raise ValueError("Recorded manifest must map each selected trajectory exactly once")
            if set(recorded_hashes) != set(by_name):
                raise ValueError("Recorded trajectory hashes must cover each selected trajectory exactly once")
            files = recorded_files
            order = recorded_order
    for path in files:
        rows = path.read_text(encoding="utf-8").splitlines()
        if len(rows) < 2:
            raise ValueError(f"Track needs at least two rows: {path}")
        for index, row in enumerate(rows, 1):
            fields = row.split()
            if len(fields) != 3:
                raise ValueError(f"Track requires three numeric columns: {path}:{index}")
            try:
                values = tuple(float(value) for value in fields)
            except ValueError as exc:
                raise ValueError(f"Track requires three numeric columns: {path}:{index}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Track requires finite numeric columns: {path}:{index}")
            if not (100000 <= values[0] <= 900000 and 0 <= values[1] <= 10000000):
                raise ValueError(f"Track requires projected UTM metre coordinates: {path}:{index}")
    return files, order


def stage_native_source(tracks: Path, native: Path, *, input_manifest: Path | None = None) -> Path:
    """Prepare the trajectory picker and loop progress in staged source copies."""
    files, track_order = _track_files(Path(tracks), input_manifest)
    original = EXTRACTION.read_bytes()
    picker = _PICKER.encode("utf-8")
    if original.count(picker) != 1:
        raise ValueError("Unexpected public file-picker source; refusing to patch")
    if native.exists():
        raise FileExistsError(f"Native stage already exists: {native}")
    native.mkdir(parents=True)
    staged = native
    (staged / EXTRACTION.name).write_bytes(instrument_matlab_source(
        original.replace(picker, b"[fnames,path]=tracebundle_select_tracks();"), EXTRACTION.name))
    (staged / BUNDLING.name).write_bytes(instrument_matlab_source(BUNDLING.read_bytes(), BUNDLING.name))
    shutil.copy2(HELPER, staged / HELPER.name)
    shutil.copytree(SOURCE_DIR / "libraries", native / "libraries")
    provenance = {
        "source_sha256": {path.relative_to(SOURCE_DIR).as_posix(): sha256_file(path)
                          for path in sorted(SOURCE_DIR.rglob("*")) if path.is_file()},
        "track_sha256": {path.name: sha256_file(path) for path in files},
        "track_order": track_order,
        "expected_intermediates": list(INTERMEDIATES),
        "file_picker_original": _PICKER,
        "file_picker_shim": "[fnames,path]=tracebundle_select_tracks();",
        "progress_reporting": {
            "helper_sha256": sha256_file(staged / HELPER.name),
            "staged_script_sha256": {name: sha256_file(staged / name)
                                      for name in (EXTRACTION.name, BUNDLING.name)},
        },
    }
    (native / "source_manifest.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    (native / "track_manifest.txt").write_text(
        "\n".join(path.name for path in files) + "\n", encoding="utf-8", newline="\n")
    (native / "track_directory.txt").write_text(
        str(Path(tracks).resolve()) + "\n", encoding="utf-8", newline="\n")
    (staged / "tracebundle_select_tracks.m").write_text(
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
        encoding="utf-8", newline="\n")
    return native


def validate_native_outputs(vertices: Path, edges: Path) -> tuple[int, int]:
    """Validate the original CSV records before downstream normalization."""
    vertex_ids: set[int] = set()
    for path in (vertices, edges):
        if not path.is_file():
            raise RuntimeError(f"TraceBundle output is missing: {path}")
    vertex_rows = vertices.read_text(encoding="utf-8").splitlines()
    edge_rows = edges.read_text(encoding="utf-8").splitlines()
    if not vertex_rows or not edge_rows:
        raise ValueError("TraceBundle output is empty")
    for index, row in enumerate(vertex_rows, 1):
        fields = row.split(",")
        if len(fields) != 3:
            raise ValueError(f"Invalid vertex columns at row {index}")
        identifier = int(fields[0])
        x, y = map(float, fields[1:])
        if identifier <= 0 or identifier in vertex_ids or not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Invalid vertex at row {index}")
        # Chicago tracks are projected EPSG:26916 metres; reject obvious degree coordinates.
        if not (100000 <= x <= 900000 and 0 <= y <= 10000000):
            raise ValueError(f"Vertex is outside projected UTM metre range at row {index}")
        vertex_ids.add(identifier)
    edge_ids: set[int] = set()
    for index, row in enumerate(edge_rows, 1):
        fields = row.split(",")
        if len(fields) != 4:
            raise ValueError(f"Invalid edge columns at row {index}")
        identifier, start, end, direction = map(int, fields)
        if identifier <= 0 or identifier in edge_ids or start == end:
            raise ValueError(f"Invalid edge at row {index}")
        if start not in vertex_ids or end not in vertex_ids:
            raise ValueError(f"Edge references missing vertex at row {index}")
        if direction not in (0, 1):
            raise ValueError(f"Invalid edge direction at row {index}")
        edge_ids.add(identifier)
    return len(vertex_rows), len(edge_rows)


def _stream_matlab(arguments: list[str], native: Path, *, env=None) -> tuple[int, str]:
    """Display each MATLAB stage as it runs and retain the same lines on disk."""
    tail = deque(maxlen=30)
    with (native / "matlab.log").open("w", encoding="utf-8", newline="\n") as log:
        with subprocess.Popen(arguments, cwd=native, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as child:
            for raw in child.stdout:
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw.decode("mbcs" if os.name == "nt" else "utf-8", errors="replace")
                line = english_text(line.replace("\r\n", "\n"))
                log.write(line)
                log.flush()
                try:
                    print(line, end="", flush=True)
                except UnicodeEncodeError:
                    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                    printable = line.encode(encoding, errors="backslashreplace").decode(encoding)
                    print(printable, end="", flush=True)
                tail.append(line)
            code = child.wait()
        log.write(f"\nexit_code: {code}\n")
    return code, "".join(tail)[-2000:]


def run_native(tracks: Path, native: Path, matlab_executable: str | None = None,
               *, input_manifest: Path | None = None) -> tuple[Path, Path, float]:
    executable = matlab_executable or shutil.which("matlab")
    if not executable:
        raise RuntimeError("MATLAB executable is required for authoritative TraceBundle execution")
    stage_native_source(tracks, native, input_manifest=input_manifest)
    intermediate_checks = " ".join(
        f"assert(exist('{name}','file') == 2, 'Missing TraceBundle intermediate: {name}');"
        for name in INTERMEDIATES)
    # Initialize R2021b's settings service before the reference scripts clear variables.
    command = ("settings; cd('" + str(native.resolve()).replace("'", "''") + "'); "
               "if any(double(getString(message('MATLAB:singularMatrix')))>127),"
               "fprintf(2,'ERROR: English MATLAB diagnostics are required. Check the MATLAB language configuration.\\n');exit(2);end; "
               "addpath(genpath('libraries')); set(groot,'defaultFigureVisible','off'); "
               "assert(exist('pdist','file') ~= 0 && exist('squareform','file') ~= 0, "
               "'Statistics and Machine Learning Toolbox is required'); "
               "run('intersection_nodes_extraction.m'); "
               + intermediate_checks + " "
               +
               "run('tracebundle.m'); close all;")
    started = time.perf_counter()
    print("[RUNTIME] Karagiorgou follows the reference source code. "
          "Processing can take several hours or longer. Please wait.",
          flush=True)
    # Reference scripts call hold/gca and require JVM-enabled graphics.
    environment = matlab_environment(executable, native/'matlab_preferences')
    code, diagnostic = _stream_matlab([executable, "-batch", command], native, env=environment)
    elapsed = time.perf_counter() - started
    if code:
        raise RuntimeError(f"MATLAB TraceBundle failed with exit code {code}: "
                           f"{diagnostic}; see {native / 'matlab.log'}")
    vertices = native / "tracebundle_vertices.txt"
    edges = native / "tracebundle_edges.txt"
    validate_native_outputs(vertices, edges)
    return vertices, edges, elapsed


def run(context: RunContext, inputs: PreparedInputs) -> MethodRunResult:
    directory = context.output_root.resolve() / "karagiorgou"
    native = directory / "native"
    vertices, edges, elapsed = run_native(inputs.tracks_projected, native, input_manifest=inputs.manifest)
    vertex_count, edge_count = validate_native_outputs(vertices, edges)
    report = native / "report.json"
    report.write_text(json.dumps({
        "execution_class": EXECUTION_CLASS,
        "provenance": json.loads((native / "source_manifest.json").read_text(encoding="utf-8")),
        "track_manifest_sha256": sha256_file(native / "track_manifest.txt"),
        "file_picker_shim": "[fnames,path]=tracebundle_select_tracks();",
        "vertices": vertex_count, "edges": edge_count,
        "source_crs": "EPSG:26916",
        "stage_order": ["intersection_nodes_extraction.m", "tracebundle.m"],
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    stage = directory / "stage_manifest.json"
    stage.write_text(json.dumps({
        "execution_class": EXECUTION_CLASS,
        "graph": ["native/tracebundle_vertices.txt", "native/tracebundle_edges.txt"],
        "report": "native/report.json",
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return MethodRunResult("karagiorgou", EXECUTION_CLASS,
                           {"vertices": vertices, "edges": edges, "report": report}, elapsed, stage)
