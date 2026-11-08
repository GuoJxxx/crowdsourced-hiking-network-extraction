"""Compile and execute the preserved Ahmed map-construction Java sources."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
import re

from experiments.chicago_six_methods.reference_methods.common import MethodRunResult
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext
from workflow.runtime_language import java_arguments


SOURCE_DIR = Path(__file__).resolve().parent / "Chicago_Ahmed_Source"
SOURCE_PATHS = (SOURCE_DIR,)
PARAMETERS = {"epsilon": 150.0, "has_altitude": False, "alt_epsilon": 4.0}


def _command_text(arguments: list[str]) -> str:
    return subprocess.list2cmdline(arguments) if os.name == "nt" else " ".join(arguments)


def _execute(arguments: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.write_text(
        f"command: {_command_text(arguments)}\nexit_code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n",
        encoding="utf-8", newline="\n",
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, arguments, result.stdout, result.stderr)
    return result


def _version(arguments: list[str]) -> str:
    arguments = java_arguments(arguments)
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, arguments, result.stdout, result.stderr)
    return (result.stdout + result.stderr).strip()


def _ordered_tracks(directory: Path) -> list[Path]:
    """Restore trip_0, trip_1, ... from the recorded manuscript execution."""
    indexed = []
    for path in directory.glob("trip_*.txt"):
        match = re.fullmatch(r"trip_(\d+)\.txt", path.name)
        if match is None:
            raise ValueError(f"Invalid trajectory filename: {path.name}")
        indexed.append((int(match[1]), path))
    if not indexed or len({identifier for identifier, _ in indexed}) != len(indexed):
        raise ValueError("Ahmed requires unique numbered trajectory files")
    return [path for _, path in sorted(indexed)]


def run(context: RunContext, inputs: PreparedInputs) -> MethodRunResult:
    """Run the official Java generator against prepared projected trajectories."""
    started = time.perf_counter()
    java_version = _version(["java", "-version"])
    javac_version = _version(["javac", "-version"])
    directory = context.output_root.resolve() / "ahmed"
    native = directory / "native"
    classes = directory / "classes"
    native.mkdir()
    classes.mkdir()
    sources = sorted((SOURCE_DIR / "src/mapconstruction2").glob("*.java"))
    if not sources:
        raise FileNotFoundError(f"Ahmed Java sources are missing: {SOURCE_DIR}")
    compile_command = java_arguments(["javac", "--release", "17", "-d", str(classes.resolve()), *[str(path.resolve()) for path in sources]])
    compile_log = native / "compile.log"
    _execute(compile_command, compile_log)

    input_path = inputs.tracks_projected.resolve()
    # Ahmed updates the graph after each trajectory. Supply an explicit order
    # instead of using the filesystem's directory enumeration.
    trajectories = _ordered_tracks(input_path)
    track_order = native / "track_order.txt"
    track_order.write_text("\n".join(path.name for path in trajectories) + "\n",
                           encoding="utf-8", newline="\n")
    # Java concatenates this argument with "vertices.txt"/"edges.txt". A
    # trailing Windows backslash in a quoted subprocess argument is parsed as
    # an escaped quote by this JDK, so use Java's supported forward slashes.
    output_prefix = native.resolve().as_posix() + "/"
    run_command = java_arguments([
        "java", "-cp", str(classes.resolve()), "mapconstruction2.Chicago_Ahmed_Runner",
        str(input_path), output_prefix, "150.0", "false", "4.0", str(track_order.resolve()),
    ])
    run_log = native / "run.log"
    _execute(run_command, run_log)
    parameters = {
        **PARAMETERS,
        "input_path": str(input_path),
        "output_prefix": output_prefix,
        "java_version": java_version,
        "javac_version": javac_version,
        "compile_command": compile_command,
        "run_command": run_command,
        "source_path": str(SOURCE_DIR.resolve()),
        "trajectory_order": [path.name for path in trajectories],
    }
    parameter_path = native / "parameters.json"
    parameter_path.write_text(json.dumps(parameters, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    stage = directory / "stage_manifest.json"
    stage.write_text(
        json.dumps({"stage": "official_java_execution", "execution_class": "official Java",
                    "parameters": "native/parameters.json", "compile_log": "native/compile.log",
                    "run_log": "native/run.log", "track_order": "native/track_order.txt"}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return MethodRunResult(
        "ahmed", "official Java",
        {"vertices": native / "vertices.txt", "edges": native / "edges.txt",
         "compile_log": compile_log, "run_log": run_log, "parameters": parameter_path,
         "track_order": track_order},
        time.perf_counter() - started, stage,
    )
