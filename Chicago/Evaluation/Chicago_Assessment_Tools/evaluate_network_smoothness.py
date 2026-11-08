from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    return []


def _sample_points(line: LineString, spacing_m: float) -> list[tuple[float, float]]:
    length = float(line.length)
    if not math.isfinite(length) or length <= 0:
        return []
    distances = list(np.arange(0.0, length, spacing_m))
    if not distances or not math.isclose(distances[-1], length, rel_tol=0.0, abs_tol=1e-9):
        distances.append(length)
    return [(float(point.x), float(point.y)) for point in (line.interpolate(distance) for distance in distances)]


def _signed_turns(points: list[tuple[float, float]]) -> list[float]:
    turns = []
    for previous, current, following in zip(points, points[1:], points[2:]):
        ax, ay = current[0] - previous[0], current[1] - previous[1]
        bx, by = following[0] - current[0], following[1] - current[1]
        norm_a = math.hypot(ax, ay)
        norm_b = math.hypot(bx, by)
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            continue
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        angle = math.degrees(math.atan2(cross, dot))
        turns.append(0.0 if abs(angle) < 1e-10 else angle)
    return turns


def summarize_lines(lines, spacing_m: float = 2.0) -> dict:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    all_turns: list[float] = []
    alternating_zigzags = 0
    network_length_m = 0.0
    for geometry in lines:
        for line in _line_parts(geometry):
            network_length_m += float(line.length)
            turns = _signed_turns(_sample_points(line, spacing_m))
            all_turns.extend(turns)
            significant = [turn for turn in turns if abs(turn) >= 5.0]
            alternating_zigzags += sum(
                1
                for first, second in zip(significant, significant[1:])
                if first * second < 0.0
            )

    absolute = np.abs(np.asarray(all_turns, dtype=float))
    sharp_count = int(np.count_nonzero(absolute >= 15.0))
    sample_count = int(absolute.size)
    length_km = network_length_m / 1000.0

    def percentile(value: float) -> float:
        return float(np.percentile(absolute, value)) if sample_count else 0.0

    return {
        "sample_spacing_m": float(spacing_m),
        "network_length_m": float(network_length_m),
        "turn_sample_count": sample_count,
        "mean_abs_turn_deg": float(np.mean(absolute)) if sample_count else 0.0,
        "median_abs_turn_deg": percentile(50),
        "p90_abs_turn_deg": percentile(90),
        "p95_abs_turn_deg": percentile(95),
        "p99_abs_turn_deg": percentile(99),
        "max_abs_turn_deg": float(np.max(absolute)) if sample_count else 0.0,
        "sharp_turn_count_ge_15": sharp_count,
        "sharp_turn_rate_ge_15": sharp_count / sample_count if sample_count else 0.0,
        "alternating_zigzag_count": int(alternating_zigzags),
        "alternating_zigzags_per_km": alternating_zigzags / length_km if length_km else 0.0,
    }


def evaluate_gpkg(path: Path, spacing_m: float) -> dict:
    edges = gpd.read_file(path, layer="edges")
    if edges.crs is None or not edges.crs.is_projected:
        raise ValueError(f"{path} must use a projected CRS")
    summary = summarize_lines(edges.geometry, spacing_m=spacing_m)
    summary.update(
        {
            "path": str(path.resolve()),
            "crs": str(edges.crs),
            "edge_count": int(len(edges)),
            "empty_geometry_count": int(edges.geometry.is_empty.sum()),
            "invalid_geometry_count": int((~edges.geometry.is_valid).sum()),
        }
    )
    return summary


def render_plot(before: dict, after: dict, output_path: Path) -> None:
    metrics = [
        ("p90_abs_turn_deg", 'P90 local turn angle (degrees)'),
        ("sharp_turn_rate_ge_15", 'Sharp turns >=15 degrees (%)'),
        ("alternating_zigzags_per_km", 'Alternating zigzags (/km)'),
    ]
    before_values = [before[key] * (100.0 if key == "sharp_turn_rate_ge_15" else 1.0) for key, _ in metrics]
    after_values = [after[key] * (100.0 if key == "sharp_turn_rate_ge_15" else 1.0) for key, _ in metrics]
    labels = [label for _, label in metrics]
    x = np.arange(len(labels))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.34
    before_bars = ax.bar(x - width / 2, before_values, width, color="#2f6f91", label='v0.4 unrestricted vehicle version')
    after_bars = ax.bar(x + width / 2, after_values, width, color="#c8472d", label='v0.5 smoothed vehicle version')
    ax.set_title(f"""Network smoothness at fixed {after['sample_spacing_m']:.0f} m resampling (lower values indicate smoother geometry)""")
    ax.set_xticks(x, labels)
    ax.set_ylabel('Metric value')
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    ax.bar_label(before_bars, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(after_bars, fmt="%.3f", padding=3, fontsize=9)
    fig.text(
        0.5,
        0.015,
        'Both versions are resampled at the same spacing in EPSG:26916; this chart measures local geometric oscillation, not road positional accuracy.',
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare road-network geometric smoothness using fixed-spacing resampling")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    before = evaluate_gpkg(args.before, args.spacing_m)
    after = evaluate_gpkg(args.after, args.spacing_m)
    result = {
        "method": "fixed_spacing_local_turn_angle",
        "interpretation": "Lower local-turn and alternating-zigzag values indicate smoother geometry; this is not a positional-accuracy metric.",
        "before_v04": before,
        "after_v05": after,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["version", *before.keys()]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"version": "v0.4", **before})
        writer.writerow({"version": "v0.5", **after})
    render_plot(before, after, args.output_png)


if __name__ == "__main__":
    main()
