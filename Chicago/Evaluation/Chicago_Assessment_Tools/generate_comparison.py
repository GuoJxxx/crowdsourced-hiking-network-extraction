#!/usr/bin/env python3
"""Build the common Chicago comparison tables, plot, and Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DISPLAY = {
    "ahmed": "Ahmed",
    "biagioni": "Biagioni",
    "davies": "Davies",
    "edelkamp": "Edelkamp",
    "karagiorgou": "Karagiorgou",
    "activity_aware": 'Proposed method',
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def optional_runtime(root: Path, method: str) -> float | None:
    candidates = {
        "ahmed": [root / method / "runtime.txt"],
        "biagioni": [root / method / "runtime_total_seconds.txt"],
        "davies": [root / method / "runtime.txt", root / method / "raw" / "runtime_seconds.txt"],
        "edelkamp": [root / method / "runtime.txt"],
        "karagiorgou": [root / method / "runtime.txt", root / method / "raw" / "runtime_seconds.txt"],
    }.get(method, [])
    for path in candidates:
        if path.exists():
            try:
                value = path.read_text(encoding="utf-8").strip()
                match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
                if match:
                    return float(match.group(0))
            except ValueError:
                pass
    report = root / method / "raw" / "report.json"
    if report.exists():
        data = read_json(report)
        if "elapsed_seconds" in data:
            return float(data["elapsed_seconds"])
    return None


def load_rows(root: Path, *, include_activity_aware: bool = True) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    order = ["ahmed", "biagioni", "davies", "edelkamp", "karagiorgou"]
    eval_paths = {m: root / m / "evaluation.json" for m in order}
    metric_paths = {m: root / m / "normalized" / "graph_metrics.json" for m in order}
    if include_activity_aware:
        eval_paths["activity_aware"] = root / "comparison" / "activity_aware" / "evaluation.json"
        metric_paths["activity_aware"] = root / "comparison" / "activity_aware" / "graph_metrics.json"

    missing = [str(p) for p in [*eval_paths.values(), *metric_paths.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required result files are missing:\n" + "\n".join(missing))

    tolerances: list[int] | None = None
    selected = [*order, "activity_aware"] if include_activity_aware else order
    for method in selected:
        evaluation = read_json(eval_paths[method])
        metrics = read_json(metric_paths[method])["graph"]
        current_tolerances = sorted(int(k) for k in evaluation["proximity"])
        if tolerances is None:
            tolerances = current_tolerances
        elif current_tolerances != tolerances:
            raise ValueError(f"Tolerance grid mismatch for {method}")
        row = {
            "method_key": method,
            "method": DISPLAY[method],
            "execution": "official MATLAB source" if method == "karagiorgou" else ('Complete v0.5 smoothed vehicle run' if method == "activity_aware" else 'Original method code / equivalent compatibility port'),
            "nodes": metrics["active_nodes"],
            "edges": metrics["edges"],
            "length_m": metrics["length_m"],
            "components": metrics["connected_components"],
            "endpoints": metrics["endpoints"],
            "intersections": metrics["intersections_degree_ge_3"],
            "cycle_rank": metrics["cycle_rank"],
            "runtime_seconds": optional_runtime(root, method),
        }
        for tolerance in current_tolerances:
            proximity = evaluation["proximity"][str(tolerance)]
            for metric in ("precision", "recall", "f1"):
                row[f"{metric}_{tolerance}m"] = proximity[metric]
        rows.append(row)
    assert tolerances is not None
    return rows, tolerances


def write_csv(path: Path, rows: list[dict], tolerances: list[int]) -> None:
    fields = [
        "method_key", "method", "execution", "nodes", "edges", "length_m",
        "components", "endpoints", "intersections", "cycle_rank", "runtime_seconds",
    ]
    fields += [f"{metric}_{t}m" for t in tolerances for metric in ("precision", "recall", "f1")]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict], tolerances: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, metric, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1")):
        for row in rows:
            linewidth = 3.0 if row["method_key"] == "activity_aware" else 1.7
            marker = "o" if row["method_key"] == "activity_aware" else None
            ax.plot(
                tolerances,
                [row[f"{metric}_{t}m"] for t in tolerances],
                label=row["method"],
                linewidth=linewidth,
                marker=marker,
                markersize=4,
            )
        ax.set_title(title)
        ax.set_xlabel('Distance tolerance (m)')
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('Score')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle('Chicago: unified geometric evaluation of five reference methods and the proposed method')
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda r: r["f1_15m"], reverse=True)
    ours = next((r for r in rows if r["method_key"] == "activity_aware"), None)
    edelkamp = next(r for r in rows if r["method_key"] == "edelkamp")
    karagiorgou = next(r for r in rows if r["method_key"] == "karagiorgou")
    davies = next(r for r in rows if r["method_key"] == "davies")
    ours_rank = (next(rank for rank, row in enumerate(ranked, 1)
                      if row["method_key"] == "activity_aware") if ours else None)
    highest_recall = max(rows, key=lambda r: r["recall_15m"])
    best_f1 = ranked[0]
    lines = [
        '# Chicago: five reference method reproductions and unified comparison',
        "",
        '## Evaluation protocol',
        "",
        '- Input: 889 official Chicago tracks (118,360 sample points); common evaluation CRS EPSG:26916.',
        '- Ground truth: official mapconstruction Chicago road map; Recall includes only observable ground-truth roads within 15 m of the original tracks.',
        '- Geometry metrics: bidirectional distance transforms between output and ground truth; Precision, Recall and F1 at tolerances of 4/6/8/10/15/20/25/30/35/40 m.',
        '- Graph metrics: computed after normalizing all outputs to undirected geometric graphs and collapsing exact reverse duplicate edges.',
        "",
        '## Main results at 15 m',
        "",
        '| Rank | Method | Precision | Recall | F1 | Length (km) | Components | Intersections | Cycle rank |',
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['precision_15m']:.4f} | "
            f"{row['recall_15m']:.4f} | {row['f1_15m']:.4f} | {row['length_m']/1000:.3f} | "
            f"{row['components']} | {row['intersections']} | {row['cycle_rank']} |"
        )
    lines += ["", '## Interpretation', ""]
    if ours is None:
        lines += [
            f"""- {best_f1['method']} has the highest 15 m F1 ({best_f1['f1_15m']:.4f}).""",
            f"""- {highest_recall['method']} has the highest Recall ({highest_recall['recall_15m']:.4f}).""",
            f"""- Native Karagiorgou MATLAB results at 15 m: P={karagiorgou['precision_15m']:.4f}, R={karagiorgou['recall_15m']:.4f}, F1={karagiorgou['f1_15m']:.4f}.""",
            '- Karagiorgou results come from native execution of the public TraceBundle MATLAB source, running intersection_nodes_extraction.m and then tracebundle.m. Independent ground truth is used only for unified evaluation after network generation.',
        ]
    else:
        lines += [
            f"""- Proposed method at 15 m: P={ours['precision_15m']:.4f}, R={ours['recall_15m']:.4f}, F1={ours['f1_15m']:.4f}.""",
            f"""- Proposed method F1 rank: {ours_rank}; gap to first-place {best_f1['method']}: {best_f1['f1_15m']-ours['f1_15m']:.4f}.""",
            f"""- {highest_recall['method']} has the highest Recall ({highest_recall['recall_15m']:.4f}), indicating the most complete coverage of observable official ground-truth roads.""",
            f"""- Relative to native Karagiorgou MATLAB results, proposed-method Recall is higher by {ours['recall_15m']-karagiorgou['recall_15m']:.4f}, Precision lower by {karagiorgou['precision_15m']-ours['precision_15m']:.4f}, and F1 higher by {ours['f1_15m']-karagiorgou['f1_15m']:.4f}.""",
            f"""- Davies has the highest Precision ({davies['precision_15m']:.4f}) but the lowest Recall ({davies['recall_15m']:.4f}), reflecting conservative network construction.""",
            f"""- Edelkamp F1 exceeds the proposed method by {edelkamp['f1_15m']-ours['f1_15m']:.4f}; its network length is {edelkamp['length_m']/1000:.3f} km, cycle rank {edelkamp['cycle_rank']}; the proposed method has {ours['length_m']/1000:.3f} km, cycle rank {ours['cycle_rank']}. Interpret these topology differences together with geometric F1.""",
            f"""- After unified normalization, the proposed method has {ours['nodes']} nodes and {ours['edges']} short segments; node/edge counts reflect 0.5 m geometry densification. Length, components, endpoints, intersections and cycle rank are {ours['length_m']/1000:.3f} km, {ours['components']}, {ours['endpoints']}, {ours['intersections']}, {ours['cycle_rank']}, respectively, for topology comparison.""",
            '- Karagiorgou results come from native execution of the public TraceBundle MATLAB source, running intersection_nodes_extraction.m and then tracebundle.m. Independent ground truth is used only for unified evaluation after network generation.',
        ]
    lines += [
        "",
        "Full per-tolerance data are in `comparison.csv`; curves are in `comparison_curves.png`. Each method's raw outputs, logs, normalized graph and `evaluation.json` are in the corresponding parent-level directory.",
    ]
    path_root = path.parent / "path_based"
    hausdorff_rows = []
    for row in rows:
        summary_path = path_root / row["method_key"] / "hausdorff" / "summary.json"
        if summary_path.exists():
            hausdorff_rows.append((row["method"], read_json(summary_path)))
    if hausdorff_rows:
        lines += [
            "",
            '## Supplementary Ahmed2014 path Hausdorff results',
            "",
            'Direction: constructed graph paths to the official ground-truth graph. This supplementary directed path-distance metric does not replace the symmetric raster P/R/F1 above.',
            "",
            '| Method | Paths | Mean (m) | Median (m) | P95 (m) | Max (m) | Mean path length (m) |',
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, summary in hausdorff_rows:
            lines.append(
                f"| {name} | {summary['path_count']} | {summary['distance_mean_m']:.3f} | "
                f"{summary['distance_median_m']:.3f} | {summary['distance_p95_m']:.3f} | "
                f"{summary['distance_max_m']:.3f} | {summary['mean_path_length_m']:.3f} |"
            )
        lines += [
            "",
            "LineSegment results depend on network segment length (short edges generally attain smaller distances), so mean path length is also reported. The original evaluator's Frechet three-chain path computation grows combinatorially with graph density and can take a long time. Unfinished supplementary runs have `INCOMPLETE.md` and must not be cited as numerical results.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_path_hausdorff_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method_key", "method", "path_count", "distance_mean_m",
        "distance_median_m", "distance_p95_m", "distance_max_m",
        "mean_path_length_m", "mean_path_complexity", "elapsed_seconds",
    ]
    output_rows = []
    path_root = path.parent / "path_based"
    for row in rows:
        summary_path = path_root / row["method_key"] / "hausdorff" / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        output_rows.append({
            "method_key": row["method_key"],
            "method": row["method"],
            **{field: summary[field] for field in fields[2:]},
        })
    if output_rows:
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Aggregate the five reference methods without requiring activity-aware output.",
    )
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows, tolerances = load_rows(root, include_activity_aware=not args.reference_only)
    write_csv(output / "comparison.csv", rows, tolerances)
    write_plot(output / "comparison_curves.png", rows, tolerances)
    write_markdown(output / "REPRODUCTION_REPORT.md", rows)
    write_path_hausdorff_csv(output / "path_hausdorff.csv", rows)
    (output / "comparison.json").write_text(
        json.dumps({"tolerances_m": tolerances, "methods": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"methods": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
