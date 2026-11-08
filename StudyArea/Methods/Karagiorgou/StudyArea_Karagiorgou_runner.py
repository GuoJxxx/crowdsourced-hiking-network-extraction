"""Serial MATLAB runner for the deterministically accelerated TraceBundle source."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from pyproj import CRS

from experiments.chicago_six_methods.reference_methods.common import sha256_file
from experiments.chicago_six_methods.reference_methods.methods import (
    karagiorgou_native as strict_native,
)

from .StudyArea_Karagiorgou_source import EXECUTION_CLASS, stage_accelerated_source


INTERMEDIATES = strict_native.INTERMEDIATES
INTERMEDIATE_CAPTURE_DIR = "tracebundle_intermediates"


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run_accelerated_native(
    source_dir: Path,
    tracks: Path,
    native: Path,
    *,
    source_crs: str,
    matlab_executable: str | None = None,
    input_manifest: Path | None = None,
) -> dict[str, object]:
    """Stage and execute the two public scripts in their original serial order."""
    crs = CRS.from_user_input(source_crs)
    if not crs.is_projected:
        raise ValueError(f"TraceBundle source CRS must be projected: {source_crs}")
    normalized_crs = crs.to_string()
    executable = matlab_executable or shutil.which("matlab")
    if not executable:
        raise RuntimeError("MATLAB executable is required for accelerated TraceBundle")

    native = stage_accelerated_source(
        source_dir,
        tracks,
        native,
        input_manifest=input_manifest,
        cell_size_m=200.0,
    )
    checks = " ".join(
        f"assert(exist('{name}','file') == 2, 'Missing TraceBundle intermediate: {name}');"
        for name in INTERMEDIATES
    )
    captures = (
        f"if exist('{INTERMEDIATE_CAPTURE_DIR}','dir') ~= 7, "
        f"mkdir('{INTERMEDIATE_CAPTURE_DIR}'); end; "
        + " ".join(
            "copyfile('"
            + name
            + "', fullfile('"
            + INTERMEDIATE_CAPTURE_DIR
            + "','"
            + name
            + "'), 'f');"
            for name in INTERMEDIATES
        )
    )
    command = (
        "cd('"
        + str(native.resolve()).replace("'", "''")
        + "'); addpath(genpath('libraries')); "
        "assert(exist('pdist','file') ~= 0 && exist('squareform','file') ~= 0, "
        "'Statistics and Machine Learning Toolbox is required'); "
        "run('intersection_nodes_extraction.m'); "
        + checks
        + " "
        + captures
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

    captured_intermediates = native / INTERMEDIATE_CAPTURE_DIR
    missing = [
        name
        for name in INTERMEDIATES
        if not (captured_intermediates / name).is_file()
        or (captured_intermediates / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing or empty TraceBundle intermediate: {', '.join(missing)}")
    vertices = native / "tracebundle_vertices.txt"
    edges = native / "tracebundle_edges.txt"
    vertex_count, edge_count = strict_native.validate_native_outputs(vertices, edges)

    report: dict[str, object] = {
        "schema_version": 1,
        "method": "Karagiorgou",
        "execution_class": EXECUTION_CLASS,
        "parameters": {"cell_size_m": 200.0},
        "runtime_total_seconds": elapsed,
        "vertices": vertex_count,
        "edges": edge_count,
        "source_crs": normalized_crs,
        "stage_order": ["intersection_nodes_extraction.m", "tracebundle.m"],
        "source_manifest": "native/source_manifest.json",
        "matlab_log": "native/matlab.log",
        "final_vertices": "native/tracebundle_vertices.txt",
        "final_edges": "native/tracebundle_edges.txt",
        "source_manifest_sha256": sha256_file(native / "source_manifest.json"),
        "track_manifest_sha256": sha256_file(native / "track_manifest.txt"),
        "intermediate_sha256": {
            name: sha256_file(captured_intermediates / name)
            for name in INTERMEDIATES
        },
        "native_output_sha256": {
            vertices.name: sha256_file(vertices),
            edges.name: sha256_file(edges),
        },
    }
    _atomic_json(native.parent / "run_manifest.json", report)
    return report
