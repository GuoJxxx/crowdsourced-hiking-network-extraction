"""Normalize, evaluate and aggregate computed Chicago reference runs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .common import RunnerError, read_json_object, sha256_file, write_json_atomic


PACKAGE = Path(__file__).resolve().parent
from workflow.common import source
TOOLS = source('assessment_tools')
NORMALIZE_TOOL = TOOLS / "normalize_graph.py"
EVALUATE_TOOL = TOOLS / "evaluate_final_network.py"
COMPARE_TOOL = TOOLS / "generate_comparison.py"
EVALUATION_TOLERANCES = (4, 6, 8, 10, 15, 20, 25, 30, 35, 40)
NORMALIZATION_OWNER = ".normalization_owner.json"
EVALUATION_OWNER = ".evaluation_owner.json"


METHOD_SPECS = {
    "ahmed": {
        "format": "text",
        "vertices": "native/vertices.txt",
        "edges": "native/edges.txt",
        "source_crs": "EPSG:26916",
    },
    "biagioni": {
        "format": "sqlite",
        "database": "stages/08_final/skeleton_map_mm2.db",
        "source_crs": "EPSG:4326",
    },
    "davies": {
        "format": "text",
        "vertices": "native/vertices.txt",
        "edges": "native/edges.txt",
        "source_crs": "EPSG:4326",
        "swap_xy": True,
    },
    "edelkamp": {
        "format": "sqlite",
        "database": "native/edelkamp_graph.db",
        "source_crs": "EPSG:4326",
    },
    "karagiorgou": {
        "format": "text",
        "vertices": "native/tracebundle_vertices.txt",
        "edges": "native/tracebundle_edges.txt",
        "source_crs": "EPSG:26916",
    },
}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerError("invalid_output", f"Invalid JSON output: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError("invalid_output", f"JSON output is not an object: {path}")
    return value


def _method_root(results_root: Path, method: str) -> Path:
    method = method.lower()
    if method not in METHOD_SPECS:
        raise RunnerError("unknown_method", f"Unknown method: {method}")
    root = Path(results_root).resolve()
    directory = root / method
    if not directory.is_dir():
        raise RunnerError("missing_output", f"Generated method output is missing: {directory}")
    if method == "karagiorgou":
        completion = read_json_object(directory / "completion.json")
        if completion is None or completion.get("execution_class") != "official MATLAB source":
            raise RunnerError("non_authoritative_output",
                              "Karagiorgou postprocessing requires a completed native MATLAB source run")
    return directory


def _native_paths(directory: Path, method: str) -> list[Path]:
    spec = METHOD_SPECS[method]
    names = [name for name in ("vertices", "edges", "database") if name in spec]
    paths = [directory / str(spec[name]) for name in names]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RunnerError("missing_output", f"Generated native output is missing or empty: {path}")
    return paths


def _signature(paths: Iterable[Path], tool: Path) -> dict[str, object]:
    return {
        "tool_sha256": sha256_file(tool),
        "inputs_sha256": {path.name: sha256_file(path) for path in paths},
    }


def _run_tool(arguments: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-B", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(
        f"exit_code: {completed.returncode}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            [sys.executable, "-B", *arguments],
            completed.stdout,
            completed.stderr,
        )


def _remove_owned_directory(path: Path, owner_name: str, owner: dict[str, object]) -> None:
    if read_json_object(path / owner_name) != owner:
        raise RunnerError("unowned_output", f"Generated stage directory is not owned: {path}")
    shutil.rmtree(path)


def _portable_graph_metrics(path: Path, results_root: Path) -> None:
    document = _json(path)
    outputs = document.get("outputs")
    if isinstance(outputs, dict):
        for key, value in list(outputs.items()):
            output = Path(str(value)).resolve()
            try:
                outputs[key] = output.relative_to(results_root.resolve()).as_posix()
            except ValueError as exc:
                raise RunnerError("unsafe_output", f"Normalizer declared output outside run root: {output}") from exc
    write_json_atomic(path, document)


def normalize_method(
    results_root: Path,
    method: str,
    *,
    force: bool,
    resume: bool,
) -> Path:
    """Normalize one newly generated native graph through the shared tool."""
    method = method.lower()
    directory = _method_root(results_root, method)
    native = _native_paths(directory, method)
    output = directory / "normalized"
    owner = {"format": "chicago-reference-normalization-v1", "method": method}
    signature = {**_signature(native, NORMALIZE_TOOL), "method": method}
    metrics = output / "graph_metrics.json"
    completion = output / "normalization_manifest.json"
    if output.exists():
        marker = read_json_object(completion)
        if resume and not force and marker is not None:
            declared = marker.get("outputs_sha256")
            products = [metrics, output / "road_network.gpkg", output / "nodes.csv", output / "edges.csv"]
            if marker.get("signature") == signature and all(path.is_file() for path in products):
                actual = {path.name: sha256_file(path) for path in products}
                if actual == declared:
                    return metrics
        if not (force or resume):
            raise RunnerError("output_exists", f"Normalized output already exists: {output}")
        _remove_owned_directory(output, NORMALIZATION_OWNER, owner)
    output.mkdir(parents=True)
    write_json_atomic(output / NORMALIZATION_OWNER, owner)

    spec = METHOD_SPECS[method]
    arguments = [
        str(NORMALIZE_TOOL),
        "--format",
        str(spec["format"]),
        "--method",
        method.title(),
        "--output",
        str(output),
    ]
    for key, option in (("vertices", "--vertices"), ("edges", "--edges"), ("database", "--database")):
        if key in spec:
            arguments.extend([option, str(directory / str(spec[key]))])
    arguments.extend(["--source-crs", str(spec["source_crs"])])
    if spec.get("swap_xy"):
        arguments.append("--swap-xy")
    _run_tool(arguments, output / "normalize.log")
    _portable_graph_metrics(metrics, Path(results_root))

    products = [metrics, output / "road_network.gpkg", output / "nodes.csv", output / "edges.csv"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in products):
        raise RunnerError("invalid_output", f"Normalization produced incomplete output for {method}")
    write_json_atomic(completion, {
        "method": method,
        "signature": signature,
        "outputs_sha256": {path.name: sha256_file(path) for path in products},
    })
    return metrics


def evaluate_method(
    results_root: Path,
    method: str,
    *,
    force: bool,
    resume: bool,
) -> Path:
    """Evaluate one normalized generated graph on the fixed tolerance grid."""
    method = method.lower()
    directory = _method_root(results_root, method)
    root = Path(results_root).resolve()
    network = directory / "normalized" / "road_network.gpkg"
    inputs = root / "inputs"
    input_manifest = _json(inputs / "input_manifest.json")
    raw_union = (root / "support/S5_KDE/raw_trajectory_union.tif"
                 if input_manifest.get("profile") == "paper"
                 else inputs / "raw_trajectory_union.tif")
    sources = [
        network,
        raw_union,
        inputs / "truth" / "chicago_vertices_osm.txt",
        inputs / "truth" / "chicago_edges_osm.txt",
    ]
    for path in sources:
        if not path.is_file() or path.stat().st_size == 0:
            raise RunnerError("missing_output", f"Evaluation input is missing or empty: {path}")
    signature = {
        **_signature(sources, EVALUATE_TOOL),
        "method": method,
        "tolerances_m": list(EVALUATION_TOLERANCES),
        "coverage_m": 15.0,
    }
    output = directory / "evaluation.json"
    manifest = directory / "evaluation_manifest.json"
    owner_path = directory / EVALUATION_OWNER
    owner = {"format": "chicago-reference-evaluation-v1", "method": method}
    existing = read_json_object(manifest)
    if output.exists() or manifest.exists() or owner_path.exists():
        if resume and not force and existing is not None and existing.get("signature") == signature:
            if output.is_file() and existing.get("output_sha256") == sha256_file(output):
                return output
        if not (force or resume):
            raise RunnerError("output_exists", f"Evaluation output already exists: {output}")
        if read_json_object(owner_path) != owner:
            raise RunnerError("unowned_output", f"Evaluation output is not owned: {directory}")
        for path in (output, manifest, directory / "evaluate.log", owner_path):
            path.unlink(missing_ok=True)
    write_json_atomic(owner_path, owner)
    arguments = [
        str(EVALUATE_TOOL),
        "--network",
        str(network),
        "--layer",
        "edges",
        "--raw-union",
        str(sources[1]),
        "--vertices",
        str(sources[2]),
        "--edges",
        str(sources[3]),
        "--coverage-m",
        "15",
        "--tolerances",
        ",".join(map(str, EVALUATION_TOLERANCES)),
        "--output",
        str(output),
    ]
    _run_tool(arguments, directory / "evaluate.log")
    document = _json(output)
    document["network"] = network.relative_to(root).as_posix()
    write_json_atomic(output, document)
    proximity = document.get("proximity")
    if not isinstance(proximity, dict) or sorted(map(int, proximity)) != list(EVALUATION_TOLERANCES):
        raise RunnerError("invalid_output", f"Evaluation tolerance grid is incomplete for {method}")
    write_json_atomic(manifest, {
        "method": method,
        "signature": signature,
        "output_sha256": sha256_file(output),
    })
    return output


def normalize_methods(results_root: Path, methods: Iterable[str], *, force: bool, resume: bool) -> list[Path]:
    return [normalize_method(results_root, method, force=force, resume=resume) for method in methods]


def evaluate_methods(results_root: Path, methods: Iterable[str], *, force: bool, resume: bool) -> list[Path]:
    return [evaluate_method(results_root, method, force=force, resume=resume) for method in methods]


def write_reference_comparison(results_root: Path, methods: Iterable[str]) -> Path:
    """Invoke the shared comparison generator for the five reference methods."""
    root = Path(results_root).resolve()
    methods = tuple(method.lower() for method in methods)
    if set(methods) != set(METHOD_SPECS):
        raise RunnerError("incomplete_comparison", "Reference comparison requires all five methods")
    comparison = root / "comparison"
    comparison.mkdir(exist_ok=True)
    _run_tool(
        [str(COMPARE_TOOL), "--results-root", str(root), "--reference-only"],
        comparison / "generate_comparison.log",
    )
    path = comparison / "comparison.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise RunnerError("invalid_output", "Comparison generator did not write comparison.json")
    return path




