from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from reference_tools import (
    convert_clean_tracks,
    normalize_text_graph,
    normalize_sqlite_graph,
    prepare_compat_sources,
    run_biagioni_pipeline,
    run_edelkamp_pipeline,
)
from karagiorgou_native import (
    EXECUTION_CLASS as KARAGIORGOU_EXECUTION_CLASS,
    run_native as run_karagiorgou_native,
    validate_native_outputs,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ORIGINAL = (
    HERE.parent / "experiments/chicago_six_methods/vendor/gpx_to_road_map-master"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path) -> dict[str, Path]:
    return {
        "compat": root / "code" / "python3_compat",
        "edelkamp_raw": root / "Edelkamp" / "raw",
        "edelkamp_trips": root / "Edelkamp" / "raw" / "trips",
        "edelkamp_normalized": root / "Edelkamp" / "normalized",
        "biagioni_inputs": root / "Biagioni" / "inputs" / "trips",
        "biagioni_stages": root / "Biagioni" / "stages",
        "biagioni_normalized": root / "Biagioni" / "normalized",
        "karagiorgou_inputs": root / "Karagiorgou" / "inputs" / "trips",
        "karagiorgou_native": root / "Karagiorgou" / "native",
        "karagiorgou_normalized": root / "Karagiorgou" / "normalized",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Edelkamp, Biagioni, and Karagiorgou on cleaned focus-area tracks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=HERE)
    prepare.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    prepare.add_argument("--edelkamp-source", type=Path)
    prepare.add_argument("--biagioni-source", type=Path)
    prepare.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Directory containing the FULL workflow's S2 cleaned trajectory CSV files.",
    )
    prepare.add_argument("--limit", type=int)

    edelkamp = subparsers.add_parser("run-edelkamp")
    edelkamp.add_argument("--root", type=Path, default=HERE)
    edelkamp.add_argument("--timeout", type=int, default=172800)
    edelkamp.add_argument("--limit", type=int)

    biagioni = subparsers.add_parser("run-biagioni")
    biagioni.add_argument("--root", type=Path, default=HERE)
    biagioni.add_argument("--timeout", type=int, default=172800)
    biagioni.add_argument("--start-stage", type=int, choices=range(1, 9), default=1)

    karagiorgou = subparsers.add_parser("run-karagiorgou")
    karagiorgou.add_argument("--root", type=Path, default=HERE)
    karagiorgou.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    karagiorgou.add_argument(
        "--matlab-executable",
        help="MATLAB executable path; omit when matlab is already on PATH.",
    )

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--root", type=Path, default=HERE)
    normalize.add_argument(
        "--method",
        choices=("Edelkamp", "Biagioni", "Karagiorgou", "both", "all"),
        default="all",
    )
    normalize.add_argument("--target-crs", default="EPSG:32647")

    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, default=HERE)
    report.add_argument("--tracks-gpkg", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, default=HERE)
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    paths = _paths(root)
    sources = prepare_compat_sources(args.original_root, paths["compat"], args.edelkamp_source, args.biagioni_source)
    conversion = convert_clean_tracks(
        args.tracks,
        paths["edelkamp_trips"],
        paths["biagioni_inputs"],
        karagiorgou_dir=paths["karagiorgou_inputs"],
        limit=args.limit,
    )
    result = {"compatibility_sources": sources, "input_conversion": conversion}
    _write_json(root / "preparation_manifest.json", result)
    return result


def _run_edelkamp(args: argparse.Namespace) -> dict[str, object]:
    paths = _paths(args.root.resolve())
    return run_edelkamp_pipeline(
        paths["compat"] / "edelkamp2003",
        paths["edelkamp_trips"],
        paths["edelkamp_raw"],
        timeout_seconds=args.timeout,
        trip_limit=args.limit,
    )


def _run_biagioni(args: argparse.Namespace) -> dict[str, object]:
    paths = _paths(args.root.resolve())
    return run_biagioni_pipeline(
        paths["compat"] / "biagoni2012",
        paths["biagioni_inputs"],
        paths["biagioni_stages"],
        timeout_seconds=args.timeout,
        start_stage=args.start_stage,
    )


def _run_karagiorgou(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    paths = _paths(root)
    source_dir = (
        args.original_root.resolve()
        / "karagiorgou"
        / "tracebundle"
        / "matlab"
    )
    return run_karagiorgou_native(
        source_dir,
        paths["karagiorgou_inputs"],
        paths["karagiorgou_native"],
        matlab_executable=args.matlab_executable,
        input_manifest=root / "preparation_manifest.json",
    )


def _normalize(args: argparse.Namespace) -> dict[str, object]:
    paths = _paths(args.root.resolve())
    results: dict[str, object] = {}
    if args.method in ("Edelkamp", "both", "all"):
        results["Edelkamp"] = normalize_sqlite_graph(
            paths["edelkamp_raw"] / "edelkamp_graph.db",
            paths["edelkamp_normalized"],
            method="Edelkamp",
            target_crs=args.target_crs,
        )
    if args.method in ("Biagioni", "both", "all"):
        results["Biagioni"] = normalize_sqlite_graph(
            paths["biagioni_stages"] / "08_final" / "skeleton_map_mm2.db",
            paths["biagioni_normalized"],
            method="Biagioni",
            target_crs=args.target_crs,
        )
    if args.method in ("Karagiorgou", "all"):
        results["Karagiorgou"] = normalize_text_graph(
            paths["karagiorgou_native"] / "tracebundle_vertices.txt",
            paths["karagiorgou_native"] / "tracebundle_edges.txt",
            paths["karagiorgou_normalized"],
            method="Karagiorgou",
            source_crs="EPSG:32647",
            target_crs=args.target_crs,
        )
    return results


def _report(args: argparse.Namespace) -> dict[str, object]:
    import geopandas as gpd
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.root.resolve()
    comparison_dir = root / "comparison"
    figures_dir = comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    preparation = json.loads(
        (root / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    input_summary = preparation["input_conversion"]
    edelkamp_run = json.loads(
        (root / "Edelkamp" / "raw" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    biagioni_run = json.loads(
        (root / "Biagioni" / "stages" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    karagiorgou_run = json.loads(
        (root / "Karagiorgou" / "run_manifest.json").read_text(encoding="utf-8")
    )

    rows = []
    method_paths = {
        "Edelkamp": root / "Edelkamp" / "normalized",
        "Biagioni": root / "Biagioni" / "normalized",
        "Karagiorgou": root / "Karagiorgou" / "normalized",
    }
    runtimes = {
        "Edelkamp": float(edelkamp_run["runtime_total_seconds"]),
        "Biagioni": float(biagioni_run["runtime_total_seconds"]),
        "Karagiorgou": float(karagiorgou_run["runtime_total_seconds"]),
    }
    metrics_by_method: dict[str, dict[str, object]] = {}
    for method, normalized_dir in method_paths.items():
        method_metrics = json.loads(
            (normalized_dir / "graph_metrics.json").read_text(encoding="utf-8")
        )
        graph = method_metrics["graph"]
        rows.append(
            {
                "method": method,
                "input_tracks": int(input_summary["tracks_written"]),
                "input_points": int(input_summary["points_written"]),
                "runtime_seconds": runtimes[method],
                "runtime_minutes": runtimes[method] / 60.0,
                "nodes": int(graph["nodes"]),
                "edges": int(graph["edges"]),
                "length_km": float(graph["length_m"]) / 1000.0,
                "connected_components": int(graph["connected_components"]),
                "largest_component_nodes": int(graph["largest_component_nodes"]),
                "endpoints": int(graph["endpoints"]),
                "intersections_degree_ge_3": int(
                    graph["intersections_degree_ge_3"]
                ),
                "cycle_rank": int(graph["cycle_rank"]),
                "crs": graph["crs"],
            }
        )
        metrics_by_method[method] = method_metrics

    metrics_table = pd.DataFrame(rows)
    metrics_csv = comparison_dir / "reference_method_metrics.csv"
    metrics_json = comparison_dir / "reference_method_metrics.json"
    metrics_table.to_csv(metrics_csv, index=False)
    _write_json(metrics_json, rows)

    stage_rows = [
        {
            "method": "Edelkamp",
            "stage": "clustering_and_graph",
            "runtime_seconds": runtimes["Edelkamp"],
        }
    ]
    stage_rows.extend(
        {
            "method": "Biagioni",
            "stage": stage["name"],
            "runtime_seconds": float(stage["elapsed_seconds"]),
        }
        for stage in biagioni_run["stages"]
    )
    stage_rows.append(
        {
            "method": "Karagiorgou",
            "stage": "intersection_extraction_and_trace_bundling",
            "runtime_seconds": runtimes["Karagiorgou"],
        }
    )
    runtime_csv = comparison_dir / "runtime_by_stage.csv"
    pd.DataFrame(stage_rows).to_csv(runtime_csv, index=False)

    tracks = gpd.read_file(args.tracks_gpkg).to_crs("EPSG:32647")
    networks = {
        method: gpd.read_file(
            normalized_dir / "road_network.gpkg",
            layer="edges",
        ).to_crs("EPSG:32647")
        for method, normalized_dir in method_paths.items()
    }
    minx, miny, maxx, maxy = tracks.total_bounds
    padding = 150.0
    extent = (minx - padding, maxx + padding, miny - padding, maxy + padding)
    colors = {
        "Edelkamp": "#d62728",
        "Biagioni": "#0066cc",
        "Karagiorgou": "#2ca02c",
    }

    def draw_network(ax, method: str) -> None:
        tracks.plot(ax=ax, color="#b8b8b8", linewidth=0.18, alpha=0.28)
        networks[method].plot(
            ax=ax,
            color=colors[method],
            linewidth=0.55,
            alpha=0.92,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"{method}  |  {len(networks[method]):,} edges",
            fontsize=12,
        )

    figure_paths = []
    for method in method_paths:
        fig, ax = plt.subplots(figsize=(9, 7.2))
        draw_network(ax, method)
        fig.tight_layout()
        path = figures_dir / f"{method.lower()}_road_network.png"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figure_paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), sharex=True, sharey=True)
    for ax, method in zip(axes, method_paths):
        draw_network(ax, method)
    fig.suptitle(
        "Reference road-network construction on identical focus-area tracks",
        fontsize=14,
    )
    fig.tight_layout()
    side_by_side = figures_dir / "three_reference_methods_road_network.png"
    fig.savefig(side_by_side, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figure_paths.append(side_by_side)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    plot_specs = [
        ("runtime_minutes", "Runtime (min)"),
        ("length_km", "Network length (km)"),
        ("edges", "Normalized edges"),
        ("connected_components", "Connected components"),
    ]
    for ax, (column, title) in zip(axes.flat, plot_specs):
        ax.bar(
            metrics_table["method"],
            metrics_table[column],
            color=[colors[method] for method in metrics_table["method"]],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for index, value in enumerate(metrics_table[column]):
            ax.text(index, value, f"{value:,.1f}", ha="center", va="bottom")
    fig.tight_layout()
    structural_figure = figures_dir / "reference_method_structural_metrics.png"
    fig.savefig(structural_figure, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figure_paths.append(structural_figure)

    compatibility_notes = {
        "algorithm_parameters_changed": False,
        "input": {
            "tracks": int(input_summary["tracks_written"]),
            "points": int(input_summary["points_written"]),
            "source": input_summary["source_dir"],
            "coordinate_source": input_summary["karagiorgou_coordinate_source"],
            "metric_crs": input_summary["karagiorgou_crs"],
            "time_utc_fallback_tracks": int(
                input_summary["time_utc_fallback_tracks"]
            ),
            "time_utc_fallback_points": int(
                input_summary["time_utc_fallback_points"]
            ),
            "zero_dt_tracks": int(input_summary["zero_dt_tracks"]),
            "zero_dt_steps": int(input_summary["zero_dt_steps"]),
            "consecutive_duplicate_xy_steps_retained": int(
                input_summary["consecutive_duplicate_xy_steps_retained"]
            ),
        },
        "mechanical_compatibility": [
            "Python 2 syntax translated to Python 3.",
            "Deprecated SciPy image I/O replaced by OpenCV equivalents.",
            "Legacy rtree import updated to the current package API.",
            "Viterbi ties compare probability only under Python 3.",
            "Topology collapses that would violate StreetMap's simple-graph invariant are skipped (8 occurrences).",
            "Duplicate scheduled segment coalesces are treated idempotently.",
            "Karagiorgou executes the public MATLAB scripts unchanged except for replacing the interactive file picker with a deterministic manifest.",
        ],
        "preserved_original_limitations": [
            "Biagioni keeps the original fixed geographic distance constants.",
            "Biagioni keeps the original handling of the terminal matched segment of each trace.",
        ],
    }
    notes_path = comparison_dir / "compatibility_notes.json"
    _write_json(notes_path, compatibility_notes)

    table_lines = []
    for row in rows:
        table_lines.append(
            f"| {row['method']} | {row['runtime_minutes']:.2f} | "
            f"{row['nodes']:,} | {row['edges']:,} | {row['length_km']:.2f} | "
            f"{row['connected_components']} | {row['largest_component_nodes']:,} | "
            f"{row['endpoints']:,} | {row['intersections_degree_ge_3']:,} | "
            f"{row['cycle_rank']:,} |"
        )
    result_table = "\n".join(table_lines)
    readme = f"""# Study-area reference-method comparison

## Experiment inputs

- Data: `Full/S2_Clean/cleaned_hiking_tracks`
- Valid tracks: {input_summary['tracks_written']:,}
- Valid track points: {input_summary['points_written']:,}
- All three reference methods use the same cleaned tracks, with final outputs normalized to EPSG:32647.

## Methods and parameters

- Edelkamp: 50 m seed spacing, 45-degree direction difference, 20 m within-cluster distance, 80 m bounding box and 0.01 m convergence threshold.
- Biagioni: 1 m raster and 17-pixel Gaussian smoothing; execute all eight stages covering KDE, skeletonization, graph extraction, two matching/pruning rounds and topology refinement.
- Karagiorgou: execute the public TraceBundle MATLAB source with 40 m clustering distance and 12.5-degree turn tolerance, running `intersection_nodes_extraction.m` and then `tracebundle.m`.
- Public-method algorithm parameters remain unchanged; Karagiorgou only replaces interactive file selection with a fixed list in natural order of original cleaned-track filenames.

## Final results

| Method | Runtime (min) | Nodes | Edges | Total length (km) | Components | Largest-component nodes | Endpoints | Degree>=3 intersections | Cycle rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{result_table}

## Directories

- `Edelkamp/raw`: raw run database, seed/edge files, logs and parameter manifest.
- `Edelkamp/normalized`: standardized GPKG, node/edge CSV, evaluation text and graph metrics.
- `Biagioni/stages`: all eight stages of intermediate results, two rounds of matching files and logs.
- `Biagioni/normalized`: standardized GPKG, node/edge CSV, evaluation text and graph metrics.
- `Karagiorgou/inputs/trips`: MATLAB inputs generated directly from the same cleaned-track `x_m/y_m`.
- `Karagiorgou/native`: public MATLAB source runtime directory, native vertices/edges, logs and hash manifest.
- `Karagiorgou/normalized`: standardized GPKG, node/edge CSV, evaluation text and graph metrics.
- `comparison`: summary tables, timing tables, compatibility notes and figures.
- `code/python3_compat`: runnable compatibility copies of the two public Python implementations.
"""
    readme_path = root / 'README_Results.md'
    readme_path.write_text(readme, encoding="utf-8")

    result = {
        "metrics_csv": str(metrics_csv.resolve()),
        "metrics_json": str(metrics_json.resolve()),
        "runtime_csv": str(runtime_csv.resolve()),
        "compatibility_notes": str(notes_path.resolve()),
        "readme": str(readme_path.resolve()),
        "figures": [str(path.resolve()) for path in figure_paths],
    }
    _write_json(comparison_dir / "report_manifest.json", result)
    return result


def _verify(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    paths = _paths(root)
    metrics: dict[str, object] = {}
    required_outputs: list[str] = []
    for method, directory in (
        ("Edelkamp", paths["edelkamp_normalized"]),
        ("Biagioni", paths["biagioni_normalized"]),
        ("Karagiorgou", paths["karagiorgou_normalized"]),
    ):
        metrics_path = directory / "graph_metrics.json"
        method_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if method_metrics["graph"]["nodes"] <= 0:
            raise RuntimeError(f"{method} normalized graph has no nodes")
        if method_metrics["graph"]["edges"] <= 0:
            raise RuntimeError(f"{method} normalized graph has no edges")
        for output in method_metrics["outputs"].values():
            output_path = Path(output)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty output: {output_path}")
            required_outputs.append(str(output_path))
        metrics[method] = method_metrics

    preparation = json.loads(
        (root / "preparation_manifest.json").read_text(encoding="utf-8")
    )["input_conversion"]
    if preparation.get("karagiorgou_coordinate_source") != "cleaned_csv_x_m_y_m":
        raise RuntimeError("Karagiorgou must use cleaned CSV x_m/y_m coordinates")
    if preparation.get("karagiorgou_crs") != "EPSG:32647":
        raise RuntimeError("Karagiorgou preparation CRS must be EPSG:32647")
    run_manifest = json.loads(
        (root / "Karagiorgou" / "run_manifest.json").read_text(encoding="utf-8")
    )
    if run_manifest.get("execution_class") != KARAGIORGOU_EXECUTION_CLASS:
        raise RuntimeError("Karagiorgou verification requires official MATLAB source")
    native_dir = paths["karagiorgou_native"]
    vertices = native_dir / "tracebundle_vertices.txt"
    edges = native_dir / "tracebundle_edges.txt"
    native_vertex_count, native_edge_count = validate_native_outputs(vertices, edges)
    source_manifest_path = native_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_tracks = int(preparation["tracks_written"])
    if len(source_manifest.get("track_sha256", {})) != expected_tracks:
        raise RuntimeError("Karagiorgou native track manifest count does not match preparation")
    public_source_root = Path(source_manifest.get("source_root", ""))
    for relative_name, expected_hash in source_manifest.get(
        "source_sha256", {}
    ).items():
        source_path = public_source_root / relative_name
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise RuntimeError(f"Karagiorgou public source hash drift: {source_path}")
    cleaned_source_root = Path(preparation.get("source_dir", ""))
    generated_trip_root = Path(preparation.get("karagiorgou_dir", ""))
    native_track_hashes = source_manifest.get("track_sha256", {})
    source_rows_total = 0
    for trip in preparation.get("trips", []):
        source_path = cleaned_source_root / str(trip["source_file"])
        if (
            not source_path.is_file()
            or _sha256(source_path) != trip.get("source_sha256")
        ):
            raise RuntimeError(f"Karagiorgou cleaned source CSV hash drift: {source_path}")
        trip_name = f"trip_{int(trip['trip_id'])}.txt"
        generated_path = generated_trip_root / trip_name
        source_rows = int(trip.get("source_rows", -1))
        karagiorgou_points = int(trip.get("karagiorgou_points", -1))
        generated_points = (
            len(generated_path.read_text(encoding="utf-8").splitlines())
            if generated_path.is_file()
            else -1
        )
        if source_rows != karagiorgou_points or source_rows != generated_points:
            raise RuntimeError(
                f"Karagiorgou source row count differs from generated point count: "
                f"{generated_path}"
            )
        source_rows_total += source_rows
        generated_hash = _sha256(generated_path) if generated_path.is_file() else None
        if generated_hash != trip.get("karagiorgou_sha256"):
            raise RuntimeError(f"Karagiorgou generated trip hash drift: {generated_path}")
        if generated_hash != native_track_hashes.get(trip_name):
            raise RuntimeError(
                f"Karagiorgou staged track hash differs from preparation: {generated_path}"
            )
    if source_rows_total != int(preparation.get("points_written", -1)):
        raise RuntimeError("Karagiorgou total source row count differs from preparation")
    normalized_graph = metrics["Karagiorgou"]["normalization"]
    if int(normalized_graph["source_vertices"]) != native_vertex_count:
        raise RuntimeError("Karagiorgou normalized vertex count differs from native output")
    if int(normalized_graph["source_edges"]) != native_edge_count:
        raise RuntimeError("Karagiorgou normalized edge count differs from native output")
    required_outputs.extend([str(vertices), str(edges), str(source_manifest_path)])

    result = {
        "status": "verified",
        "root": str(root),
        "methods": metrics,
        "required_outputs": required_outputs,
        "karagiorgou_input_audit": {
            key: preparation[key]
            for key in (
                "tracks_written",
                "points_written",
                "time_utc_fallback_tracks",
                "time_utc_fallback_points",
                "zero_dt_tracks",
                "zero_dt_steps",
                "consecutive_duplicate_xy_steps_retained",
                "karagiorgou_coordinate_source",
                "karagiorgou_crs",
                "karagiorgou_point_policy",
            )
            if key in preparation
        },
    }
    _write_json(root / "final_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "prepare": _prepare,
        "run-edelkamp": _run_edelkamp,
        "run-biagioni": _run_biagioni,
        "run-karagiorgou": _run_karagiorgou,
        "normalize": _normalize,
        "report": _report,
        "verify": _verify,
    }
    result = actions[args.command](args)
    console_result = result
    if args.command == "prepare":
        conversion = result["input_conversion"]
        console_result = {
            "manifest": str((args.root.resolve() / "preparation_manifest.json")),
            "source_files_seen": conversion["source_files_seen"],
            "tracks_written": conversion["tracks_written"],
            "tracks_skipped_short": conversion["tracks_skipped_short"],
            "points_written": conversion["points_written"],
            "time_utc_fallback_tracks": conversion["time_utc_fallback_tracks"],
            "time_utc_fallback_points": conversion["time_utc_fallback_points"],
            "zero_dt_tracks": conversion["zero_dt_tracks"],
            "zero_dt_steps": conversion["zero_dt_steps"],
            "consecutive_duplicate_xy_steps_retained": conversion[
                "consecutive_duplicate_xy_steps_retained"
            ],
        }
    print(json.dumps(console_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
