"""Deterministic equivalence checks for strict and accelerated TraceBundle runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from experiments.chicago_six_methods.reference_methods.geometry_signature import (
    canonical_geometry_signature,
)
from experiments.chicago_six_methods.reference_methods.common import sha256_file

from .StudyArea_Karagiorgou_runner import INTERMEDIATES


def _json_scalar(value: Any) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
    if isinstance(value, complex):
        return {"real": _json_scalar(value.real), "imag": _json_scalar(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    if mask.ndim == 0:
        return ()
    return tuple(int(value) for value in np.argwhere(mask)[0])


def _difference(
    filename: str,
    variable: str,
    kind: str,
    index: tuple[int, ...] | None,
    strict_value: object,
    accelerated_value: object,
) -> dict[str, object]:
    return {
        "file": filename,
        "variable": variable,
        "kind": kind,
        "index": None if index is None else list(index),
        "strict_value": _json_scalar(strict_value),
        "accelerated_value": _json_scalar(accelerated_value),
    }


def _load_mat_variables(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TraceBundle intermediate: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Invalid MAT file (empty): {path}")
    try:
        # mat_dtype=True can silently discard imaginary values in scipy.
        document = loadmat(path, mat_dtype=False, struct_as_record=True, squeeze_me=False)
    except Exception as exc:
        raise ValueError(f"Invalid MAT file: {path}") from exc
    variables = {
        name: np.asarray(value)
        for name, value in document.items()
        if not name.startswith("__")
    }
    if not variables:
        raise ValueError(f"Invalid MAT file (no variables): {path}")
    return variables


def _compare_arrays(
    filename: str,
    variable: str,
    strict: np.ndarray,
    accelerated: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object] | None:
    if strict.shape != accelerated.shape:
        return _difference(
            filename, variable, "shape", None, list(strict.shape), list(accelerated.shape)
        )
    if strict.dtype != accelerated.dtype:
        return _difference(
            filename, variable, "dtype", None, str(strict.dtype), str(accelerated.dtype)
        )
    if strict.dtype.kind == "c":
        for component in ("real", "imag"):
            result = _compare_arrays(filename, variable + "." + component,
                                     getattr(strict, component), getattr(accelerated, component),
                                     rtol=rtol, atol=atol)
            if result is not None:
                return result
        return None
    if strict.dtype.kind == "f":
        strict_finite = np.isfinite(strict)
        accelerated_finite = np.isfinite(accelerated)
        finite_difference = strict_finite != accelerated_finite
        if finite_difference.any():
            index = _first_index(finite_difference)
            return _difference(
                filename,
                variable,
                "finite_mask",
                index,
                strict[index],
                accelerated[index],
            )
        nonfinite_difference = (
            np.isnan(strict) != np.isnan(accelerated)
        ) | (np.isposinf(strict) != np.isposinf(accelerated)) | (
            np.isneginf(strict) != np.isneginf(accelerated)
        )
        if nonfinite_difference.any():
            index = _first_index(nonfinite_difference)
            return _difference(
                filename,
                variable,
                "nonfinite_value",
                index,
                strict[index],
                accelerated[index],
            )
        close = np.ones(strict.shape, dtype=bool)
        close[strict_finite] = np.isclose(
            strict[strict_finite],
            accelerated[strict_finite],
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        if not close.all():
            index = _first_index(~close)
            return _difference(
                filename, variable, "value", index, strict[index], accelerated[index]
            )
        return None
    try:
        equal = np.equal(strict, accelerated)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported MAT variable for deterministic comparison: {filename}:{variable}"
        ) from exc
    if not bool(np.asarray(equal).all()):
        index = _first_index(~np.asarray(equal, dtype=bool))
        return _difference(
            filename, variable, "value", index, strict[index], accelerated[index]
        )
    return None


def compare_mat_directories(
    strict_dir: Path,
    accelerated_dir: Path,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Compare the seven declared MATLAB intermediates without changing them."""
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("MAT comparison tolerances must be finite and nonnegative")
    strict_dir = Path(strict_dir)
    accelerated_dir = Path(accelerated_dir)
    file_reports: list[dict[str, object]] = []
    first_difference: dict[str, object] | None = None
    for filename in INTERMEDIATES:
        strict_variables = _load_mat_variables(strict_dir / filename)
        accelerated_variables = _load_mat_variables(accelerated_dir / filename)
        names = sorted(set(strict_variables) | set(accelerated_variables))
        variable_reports: list[dict[str, object]] = []
        for variable in names:
            if variable not in strict_variables or variable not in accelerated_variables:
                difference = _difference(
                    filename,
                    variable,
                    "variable_names",
                    None,
                    variable in strict_variables,
                    variable in accelerated_variables,
                )
            else:
                difference = _compare_arrays(
                    filename,
                    variable,
                    strict_variables[variable],
                    accelerated_variables[variable],
                    rtol=rtol,
                    atol=atol,
                )
            variable_reports.append({
                "name": variable,
                "passed": difference is None,
                "strict_shape": (
                    list(strict_variables[variable].shape)
                    if variable in strict_variables else None
                ),
                "accelerated_shape": (
                    list(accelerated_variables[variable].shape)
                    if variable in accelerated_variables else None
                ),
                "strict_dtype": (
                    str(strict_variables[variable].dtype)
                    if variable in strict_variables else None
                ),
                "accelerated_dtype": (
                    str(accelerated_variables[variable].dtype)
                    if variable in accelerated_variables else None
                ),
            })
            if first_difference is None and difference is not None:
                first_difference = difference
        file_reports.append({
            "file": filename,
            "passed": all(bool(item["passed"]) for item in variable_reports),
            "variables": variable_reports,
        })
    return {
        "schema_version": 1,
        "passed": first_difference is None,
        "rtol": rtol,
        "atol": atol,
        "files": file_reports,
        "first_difference": first_difference,
    }


def _read_nonempty_lines(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TraceBundle {label}: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"TraceBundle {label} is empty or contains blank rows")
    return lines


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_native_graph_signature(
    vertices: Path,
    edges: Path,
    *,
    precision_m: float = 0.001,
) -> dict[str, object]:
    """Hash a native graph without depending on row order or edge orientation."""
    if not math.isfinite(precision_m) or precision_m <= 0:
        raise ValueError("Coordinate precision must be finite and positive")
    vertex_map: dict[int, tuple[int, int]] = {}
    for row_index, row in enumerate(_read_nonempty_lines(Path(vertices), "vertices"), 1):
        fields = row.split(",")
        if len(fields) != 3:
            raise ValueError(f"Invalid vertex columns at row {row_index}")
        try:
            identifier = int(fields[0])
            x, y = float(fields[1]), float(fields[2])
        except ValueError as exc:
            raise ValueError(f"Invalid vertex at row {row_index}") from exc
        if identifier <= 0 or identifier in vertex_map or not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Invalid vertex at row {row_index}")
        vertex_map[identifier] = (round(x / precision_m), round(y / precision_m))

    edge_ids: set[int] = set()
    connectivity: list[tuple[int, int]] = []
    geometry: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row_index, row in enumerate(_read_nonempty_lines(Path(edges), "edges"), 1):
        fields = row.split(",")
        if len(fields) != 4:
            raise ValueError(f"Invalid edge columns at row {row_index}")
        try:
            identifier, start, end, direction = map(int, fields)
        except ValueError as exc:
            raise ValueError(f"Invalid edge at row {row_index}") from exc
        if identifier <= 0 or identifier in edge_ids or start == end:
            raise ValueError(f"Invalid edge at row {row_index}")
        if direction not in (0, 1):
            raise ValueError(f"Invalid edge direction at row {row_index}")
        if start not in vertex_map or end not in vertex_map:
            raise ValueError(f"Edge references missing vertex at row {row_index}")
        edge_ids.add(identifier)
        connectivity.append(tuple(sorted((start, end))))
        geometry.append(tuple(sorted((vertex_map[start], vertex_map[end]))))

    vertices_payload = sorted(
        (identifier, coordinates[0], coordinates[1])
        for identifier, coordinates in vertex_map.items()
    )
    return {
        "schema_version": 1,
        "precision_m": precision_m,
        "vertex_count": len(vertex_map),
        "edge_count": len(connectivity),
        "vertex_hash": _hash_json(vertices_payload),
        "edge_connectivity_hash": _hash_json(sorted(connectivity)),
        "geometry_hash": _hash_json(sorted(geometry)),
    }


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Invalid {label}: {path}")
    return document


def _method_root(root: Path) -> Path:
    root = Path(root)
    nested = root / "karagiorgou"
    return nested if nested.is_dir() else root


def _input_manifest(root: Path, method: Path) -> Path:
    candidates = (
        Path(root) / "inputs" / "input_manifest.json",
        method.parent / "inputs" / "input_manifest.json",
        method.parent.parent / "inputs" / "input_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing Chicago input manifest below {root}")


def _public_source_hashes(manifest: dict[str, object]) -> dict[str, str]:
    for key in ("source_sha256", "public_source_sha256"):
        candidate = manifest.get(key)
        if isinstance(candidate, dict):
            return {str(name): str(value) for name, value in candidate.items()}
    raise ValueError("Source manifest lacks public source hashes")


def _compare_expected(
    failures: list[str], label: str, actual: object, expected: object, *, atol: float = 1e-12
) -> None:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        matches = math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=atol)
    else:
        matches = actual == expected
    if not matches:
        failures.append(f"{label}: expected {expected!r}, found {actual!r}")


def validate_chicago_equivalence(
    strict_root: Path,
    accelerated_root: Path,
    baseline: Path,
) -> dict[str, object]:
    """Validate two complete Chicago results against the fixed scientific baseline."""
    expected = _load_json(Path(baseline), "Chicago strict baseline")
    required = {
        "schema_version", "expected_epsg", "precision_m", "graph",
        "proximity_15m", "geometry_signature", "native_graph_signature",
        "strict_provenance",
    }
    missing = sorted(required - set(expected))
    if missing or expected.get("schema_version") != 1:
        raise ValueError(f"Invalid Chicago strict baseline schema; missing={missing}")

    strict_method = _method_root(Path(strict_root))
    accelerated_method = _method_root(Path(accelerated_root))
    precision = float(expected["precision_m"])
    epsg = int(expected["expected_epsg"])
    failures: list[str] = []

    native_reports: dict[str, dict[str, object]] = {}
    geometry_reports: dict[str, str] = {}
    metrics_reports: dict[str, dict[str, object]] = {}
    evaluation_reports: dict[str, dict[str, object]] = {}
    for label, method in (("strict", strict_method), ("accelerated", accelerated_method)):
        native = method / "native"
        native_reports[label] = canonical_native_graph_signature(
            native / "tracebundle_vertices.txt",
            native / "tracebundle_edges.txt",
            precision_m=precision,
        )
        geometry_reports[label] = canonical_geometry_signature(
            method / "normalized" / "road_network.gpkg",
            precision_m=precision,
            expected_epsg=epsg,
        )
        metrics = _load_json(method / "normalized" / "graph_metrics.json", "graph metrics")
        graph = metrics.get("graph")
        if not isinstance(graph, dict):
            raise ValueError(f"Invalid graph metrics: {method}")
        metrics_reports[label] = graph
        evaluation = _load_json(method / "evaluation.json", "Chicago evaluation")
        proximity = evaluation.get("proximity")
        at_15 = proximity.get("15") if isinstance(proximity, dict) else None
        if not isinstance(at_15, dict):
            raise ValueError(f"Invalid Chicago 15 m evaluation: {method}")
        evaluation_reports[label] = at_15

    for key, expected_value in expected["native_graph_signature"].items():
        for label in ("strict", "accelerated"):
            _compare_expected(
                failures,
                f"{label} native {key}",
                native_reports[label].get(key),
                expected_value,
            )
    for label in ("strict", "accelerated"):
        _compare_expected(
            failures,
            f"{label} geometry signature",
            geometry_reports[label],
            expected["geometry_signature"],
        )
        for key, expected_value in expected["graph"].items():
            _compare_expected(
                failures, f"{label} graph {key}", metrics_reports[label].get(key), expected_value
            )
        for key, expected_value in expected["proximity_15m"].items():
            _compare_expected(
                failures,
                f"{label} proximity 15 m {key}",
                evaluation_reports[label].get(key),
                expected_value,
            )

    provenance = expected["strict_provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("Invalid Chicago strict provenance")
    expected_sources = provenance.get("public_source_sha256")
    expected_inputs = provenance.get("input_sha256")
    expected_native = provenance.get("native_output_sha256")
    if not all(isinstance(value, dict) for value in (expected_sources, expected_inputs, expected_native)):
        raise ValueError("Invalid Chicago strict provenance hashes")
    for label, root, method in (
        ("strict", Path(strict_root), strict_method),
        ("accelerated", Path(accelerated_root), accelerated_method),
    ):
        source_manifest = _load_json(method / "native" / "source_manifest.json", "source manifest")
        actual_sources = _public_source_hashes(source_manifest)
        for name, expected_hash in expected_sources.items():
            _compare_expected(
                failures, f"{label} public source {name}", actual_sources.get(name), expected_hash
            )
        input_document = _load_json(_input_manifest(root, method), "Chicago input manifest")
        outputs = input_document.get("outputs")
        for name, expected_hash in expected_inputs.items():
            entry = outputs.get(name) if isinstance(outputs, dict) else None
            actual_hash = entry.get("sha256") if isinstance(entry, dict) else None
            _compare_expected(
                failures, f"{label} input {name}", actual_hash, expected_hash
            )
    for name, expected_hash in expected_native.items():
        _compare_expected(
            failures,
            f"strict native output {name}",
            sha256_file(strict_method / "native" / name),
            expected_hash,
        )

    return {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "baseline": str(Path(baseline).resolve()),
        "native_graph": native_reports,
        "geometry": geometry_reports,
        "graph": metrics_reports,
        "proximity_15m": evaluation_reports,
    }
