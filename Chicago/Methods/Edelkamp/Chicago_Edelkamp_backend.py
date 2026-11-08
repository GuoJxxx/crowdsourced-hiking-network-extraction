"""Run the preserved Edelkamp implementation through a generated Python 3 copy."""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from experiments.chicago_six_methods.reference_methods.common import MethodRunResult, sha256_file
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext


SOURCE_DIR = Path(__file__).resolve().parent / "Chicago_Edelkamp_Source"
SOURCE_PATHS = (SOURCE_DIR,)
PARAMETERS = {
    "seed_interval_m": 50.0,
    "bearing_limit_deg": 45.0,
    "intra_cluster_distance_m": 20.0,
    "edge_bounding_box_m": 80.0,
    "movement_threshold_m": 0.01,
}


def _generate_compatibility_copy(destination: Path) -> dict[str, str]:
    """Apply the confirmed Mainling mechanical conversion to owned output."""
    from workflow.conversion import make_refactoring_tool

    if not (SOURCE_DIR / "edelkamp2003.py").is_file():
        raise FileNotFoundError(f"Preserved Edelkamp source is missing: {SOURCE_DIR}")
    shutil.copytree(SOURCE_DIR, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    refactor = make_refactoring_tool()
    source_hashes: dict[str, str] = {}
    for generated in sorted(destination.rglob("*.py")):
        relative = generated.relative_to(destination)
        source_hashes[relative.as_posix()] = sha256_file(SOURCE_DIR / relative)
        original = generated.read_text(encoding="utf-8").expandtabs(8)
        try:
            ast.parse(original)
            converted = original
        except SyntaxError:
            converted = str(refactor.refactor_string(original + "\n", str(generated)))
        converted = converted.replace("from rtree import Rtree", "from rtree.index import Index as Rtree")
        if relative.as_posix() == "location.py":
            # Seeding follows the numeric trajectory order of the paper run.
            converted = converted.replace(
                "trip_filenames = os.listdir(trips_path)",
                "trip_filenames = sorted((name for name in os.listdir(trips_path) "
                "if name.startswith('trip_') and name.endswith('.txt')), "
                "key=lambda name: int(name[5:-4]))",
            )
        generated.write_text(converted, encoding="utf-8", newline="\n")
    if not source_hashes or "from rtree.index import Index as Rtree" not in (
        destination / "edelkamp2003.py"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("Edelkamp compatibility conversion did not update the rtree import")
    return source_hashes


def _assert_graph_nonempty(database: Path) -> None:
    if not database.is_file() or database.stat().st_size == 0:
        raise RuntimeError("Edelkamp did not produce a SQLite graph")
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        for table in ("nodes", "edges"):
            if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] <= 0:
                raise RuntimeError(f"Edelkamp produced no {table}")


def run(context: RunContext, inputs: PreparedInputs) -> MethodRunResult:
    """Execute the original algorithm on the shared Edelkamp WGS84 trips."""
    started = time.perf_counter()
    directory = context.output_root.resolve() / "edelkamp"
    native = directory / "native"
    compatibility = directory / "compat" / "edelkamp2003"
    native.mkdir(parents=True)
    source_hashes = _generate_compatibility_copy(compatibility)
    shared_trips = (inputs.trips_wgs84 / "edelkamp").resolve()
    if not shared_trips.is_dir() or not list(shared_trips.glob("trip_*.txt")):
        raise FileNotFoundError(f"Prepared Edelkamp trips are missing: {shared_trips}")
    shutil.copytree(shared_trips, native / "trips")
    track_order = native / "track_order.txt"
    ordered = sorted(shared_trips.glob("trip_*.txt"), key=lambda item: int(item.stem[5:]))
    track_order.write_text("\n".join(item.name for item in ordered) + "\n",
                           encoding="utf-8", newline="\n")
    command = [
        sys.executable, "-B", str((compatibility / "edelkamp2003.py").resolve()),
        "-i", str(PARAMETERS["seed_interval_m"]),
        "-b", str(PARAMETERS["bearing_limit_deg"]),
        "-d", str(PARAMETERS["intra_cluster_distance_m"]),
        "-s", str(PARAMETERS["edge_bounding_box_m"]),
        "-t", str(PARAMETERS["movement_threshold_m"]),
    ]
    log_path = native / "run.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("command: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=native, stdout=log, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        log.write(f"\nexit_code: {completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    graph = native / "edelkamp_graph.db"
    _assert_graph_nonempty(graph)
    parameters = {
        **PARAMETERS,
        "input_path": str(shared_trips),
        "source_path": str(SOURCE_DIR.resolve()),
        "source_sha256": source_hashes,
        "command": command,
        "trajectory_order": [item.name for item in ordered],
         "compatibility": ["lib2to3 syntax conversion", "rtree.index.Index import", "numeric trajectory order"],
    }
    parameter_path = native / "parameters.json"
    parameter_path.write_text(json.dumps(parameters, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    stage = directory / "stage_manifest.json"
    stage.write_text(
        json.dumps({"stage": "edelkamp_clustering", "execution_class": "Python 3/current-library compatibility",
                    "generated_source": "compat/edelkamp2003", "graph": "native/edelkamp_graph.db",
                    "parameters": "native/parameters.json", "run_log": "native/run.log",
                    "track_order": "native/track_order.txt"},
                   sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return MethodRunResult(
        "edelkamp", "Python 3/current-library compatibility",
        {"graph": graph, "run_log": log_path, "parameters": parameter_path, "track_order": track_order},
        time.perf_counter() - started, stage,
    )
