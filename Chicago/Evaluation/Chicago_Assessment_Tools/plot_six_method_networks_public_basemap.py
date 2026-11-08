#!/usr/bin/env python3
"""Plot six reconstructed Chicago road networks over the official public map."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


def load_truth_segments(vertices_path: Path, edges_path: Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    vertices: dict[int, tuple[float, float]] = {}
    with vertices_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3:
                vertices[int(row[0])] = (float(row[1]), float(row[2]))

    segments = []
    with edges_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            u, v = int(row[1]), int(row[2])
            if u in vertices and v in vertices and u != v:
                segments.append((vertices[u], vertices[v]))
    if not segments:
        raise ValueError("Official Chicago map contains no valid segments")
    return segments


def load_method(spec: dict, *, base_dir: Path | None = None) -> dict:
    normalized = Path(spec["normalized_dir"])
    evaluation_path = Path(spec["evaluation"])
    if base_dir is not None:
        if not normalized.is_absolute():
            normalized = base_dir / normalized
        if not evaluation_path.is_absolute():
            evaluation_path = base_dir / evaluation_path
    normalized = normalized.resolve()
    evaluation_path = evaluation_path.resolve()
    nodes: dict[int, tuple[float, float]] = {}
    with (normalized / "nodes.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            nodes[int(row["node_id"])] = (float(row["x"]), float(row["y"]))

    segments = []
    with (normalized / "edges.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            u, v = int(row["u"]), int(row["v"])
            if u in nodes and v in nodes and u != v:
                segments.append((nodes[u], nodes[v]))
    if not segments:
        raise ValueError(f"{spec['label']} contains no valid road segments")

    graph = json.loads((normalized / "graph_metrics.json").read_text(encoding="utf-8"))["graph"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return {
        "label": spec["label"],
        "segments": segments,
        "nodes": nodes,
        "length_km": float(graph["length_m"]) / 1000.0,
        "components": int(graph["connected_components"]),
        "intersections": int(graph["intersections_degree_ge_3"]),
        "f1": float(evaluation["proximity"]["15"]["f1"]),
        "proposed": bool(spec.get("proposed", False)),
    }


def common_extent(truth_segments: list, methods: list[dict]) -> tuple[float, float, float, float]:
    points = [point for segment in truth_segments for point in segment]
    points.extend(point for method in methods for segment in method["segments"] for point in segment)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan = max(xmax - xmin, 1.0)
    yspan = max(ymax - ymin, 1.0)
    return xmin - 0.025 * xspan, xmax + 0.025 * xspan, ymin - 0.035 * yspan, ymax + 0.035 * yspan


def nice_scale_length(width_m: float) -> float:
    target = width_m * 0.18
    magnitude = 10 ** math.floor(math.log10(max(target, 1.0)))
    candidates = [1, 2, 5, 10]
    return max(value * magnitude for value in candidates if value * magnitude <= target)


def add_scale_bar(ax, extent, font: FontProperties) -> None:
    xmin, xmax, ymin, ymax = extent
    length = nice_scale_length(xmax - xmin)
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.055 * (ymax - ymin)
    x1 = x0 + length
    tick = 0.012 * (ymax - ymin)
    ax.plot([x0, x1], [y0, y0], color="#242424", linewidth=1.5, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#242424", linewidth=1.0, zorder=5)
    ax.plot([x1, x1], [y0 - tick, y0 + tick], color="#242424", linewidth=1.0, zorder=5)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text((x0 + x1) / 2, y0 + 1.7 * tick, label, ha="center", va="bottom", fontproperties=font, fontsize=8)


def add_north_arrow(ax, extent, font: FontProperties) -> None:
    xmin, xmax, ymin, ymax = extent
    x = xmax - 0.055 * (xmax - xmin)
    y = ymax - 0.08 * (ymax - ymin)
    dy = 0.045 * (ymax - ymin)
    ax.annotate("", xy=(x, y + dy), xytext=(x, y - dy * 0.35), arrowprops={"arrowstyle": "-|>", "color": "#242424", "lw": 1.1}, zorder=5)
    ax.text(x, y + dy * 1.14, "N", ha="center", va="bottom", fontproperties=font, fontsize=9)


def resolve_font(*, weight: str | None = None) -> FontProperties:
    """Use the Windows installation's CJK font when present, else defaults."""
    windows_dir = os.environ.get("WINDIR")
    if windows_dir:
        font_path = Path(windows_dir) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            return FontProperties(fname=str(font_path), weight=weight)
    return FontProperties(weight=weight)


def display_label(label: str, language: str) -> str:
    if language != "en":
        return label
    if "Karagiorgou" in label:
        return "(e) Karagiorgou et al. (2012)"
    if "Activity-Aware" in label:
        return "(f) Proposed method"
    return label


def render_static(truth_segments: list, methods: list[dict], extent, output_dir: Path, language: str = "zh") -> tuple[list[Path], dict]:
    font = resolve_font()
    title_font = resolve_font(weight="bold")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.8))
    fig.patch.set_facecolor("white")
    figure_title = fig.suptitle(
        "Chicago benchmark: reconstructed networks from six methods"
        if language == "en" else
        'Chicago public dataset: reconstructed networks from six methods',
        fontproperties=title_font, fontsize=21, y=0.985,
    )
    figure_subtitle = fig.text(
        0.5, 0.928,
        "Light gray: official reference network; blue: reference methods; red: proposed method (common extent, EPSG:26916)"
        if language == "en" else
        'Light gray: official public ground-truth network; blue: reference methods; red: Activity-Aware v0.5 strict main-road version (common extent, EPSG:26916)',
        ha="center", fontproperties=font, fontsize=10.5, color="#505050",
    )

    xmin, xmax, ymin, ymax = extent
    for ax, method in zip(axes.flat, methods):
        ax.add_collection(LineCollection(truth_segments, colors="#c5c8ca", linewidths=0.34, alpha=0.82, zorder=1))
        color = "#b4472f" if method["proposed"] else "#156082"
        width = 0.80 if method["proposed"] else 0.68
        ax.add_collection(LineCollection(method["segments"], colors=color, linewidths=width, alpha=0.94, capstyle="round", joinstyle="round", zorder=2))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fbfbfa")
        for spine in ax.spines.values():
            spine.set_color("#aeb2b4")
            spine.set_linewidth(0.75)
        ax.set_title(display_label(method["label"], language), fontproperties=title_font, fontsize=13, pad=22)
        metrics = (
            f"{method['length_km']:.1f} km  |  components {method['components']}  |  intersections {method['intersections']}  |  F1 {method['f1']:.3f}"
            if language == "en" else
            f"""{method['length_km']:.1f} km  |  components {method['components']}  |  intersections {method['intersections']}  |  F1 {method['f1']:.3f}"""
        )
        ax.text(0.5, 1.014, metrics, transform=ax.transAxes, ha="center", va="bottom", fontproperties=font, fontsize=8.5, color="#4f4f4f")
        add_scale_bar(ax, extent, font)
        add_north_arrow(ax, extent, font)

    legend = [
        Line2D([0], [0], color="#c5c8ca", lw=2.0, label="Official reference network" if language == "en" else 'Chicago official ground-truth basemap'),
        Line2D([0], [0], color="#156082", lw=2.0, label="Reference-method output" if language == "en" else 'Reference method output'),
        Line2D([0], [0], color="#b4472f", lw=2.2, label="Proposed-method output" if language == "en" else 'Activity-Aware v0.5 output'),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, prop=font, bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.065, top=0.885, wspace=0.065, hspace=0.16)
    fig.canvas.draw()
    title_subtitle_overlap = bool(figure_title.get_window_extent().overlaps(figure_subtitle.get_window_extent()))
    if title_subtitle_overlap:
        raise RuntimeError("Figure title overlaps the subtitle")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "road_network_comparison_six_methods.png",
        output_dir / "road_network_comparison_six_methods.pdf",
        output_dir / "road_network_comparison_six_methods.svg",
    ]
    fig.savefig(outputs[0], dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(outputs[1], facecolor="white", bbox_inches="tight")
    fig.savefig(outputs[2], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return outputs, {"title_subtitle_overlap": title_subtitle_overlap}


def svg_path(segments: list, extent, width: float = 1000.0, height: float = 660.0) -> str:
    xmin, xmax, ymin, ymax = extent
    pad = 18.0
    scale = min((width - 2 * pad) / (xmax - xmin), (height - 2 * pad) / (ymax - ymin))
    draw_w = (xmax - xmin) * scale
    draw_h = (ymax - ymin) * scale
    ox = (width - draw_w) / 2.0
    oy = (height - draw_h) / 2.0
    parts = []
    seen = set()
    for (x1, y1), (x2, y2) in segments:
        sx1, sy1 = ox + (x1 - xmin) * scale, height - (oy + (y1 - ymin) * scale)
        sx2, sy2 = ox + (x2 - xmin) * scale, height - (oy + (y2 - ymin) * scale)
        start = (round(sx1), round(sy1))
        end = (round(sx2), round(sy2))
        if start == end:
            continue
        key = tuple(sorted((start, end)))
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"M{start[0]},{start[1]}L{end[0]},{end[1]}")
    return "".join(parts)


def render_html(truth_segments: list, methods: list[dict], extent, html_output: Path) -> None:
    basemap_d = svg_path(truth_segments, extent)
    panels = []
    for method in methods:
        network_class = "inferred-network proposed" if method["proposed"] else "inferred-network"
        metrics = f"""{method['length_km']:.1f} km / components {method['components']} / intersections {method['intersections']} · F1 {method['f1']:.3f}"""
        panels.append(
            f"""<section class=\"network-panel\">
  <h3>{method["label"]}</h3>
  <div class=\"metric-line\">{metrics}</div>
  <svg viewBox=\"0 0 1000 660\" role=\"img\" aria-label=\"{method['label']} network compared with Chicago official ground-truth basemap\">
    <title>{method['label']} network</title>
    <desc>Light gray lines show official ground-truth roads from the Chicago public dataset; colored lines show method outputs.</desc>
    <rect class=\"map-frame\" x=\"1\" y=\"1\" width=\"998\" height=\"658\"></rect>
    <use href=\"#public-chicago-basemap\" class=\"official-basemap\"></use>
    <path class=\"{network_class}\" d=\"{svg_path(method['segments'], extent)}\"></path>
    <g class=\"north\" aria-hidden=\"true\"><text x=\"944\" y=\"48\">N</text><path d=\"M944,60 L934,87 L944,80 L954,87 Z\"></path></g>
  </svg>
</section>"""
        )

    fragment = f"""<div id=\"chicago-public-road-network-comparison\">
  <h2>Chicago public dataset: network comparison for six methods</h2>
  <div class=\"legend\"><span><i class=\"truth-swatch\"></i>Official ground-truth basemap</span><span><i class=\"reference-swatch\"></i>Reference methods</span><span><i class=\"proposed-swatch\"></i>Activity-Aware v0.5</span></div>
  <svg aria-hidden=\"true\" width=\"0\" height=\"0\" class=\"shared-defs\"><defs><path id=\"public-chicago-basemap\" d=\"{basemap_d}\"></path></defs></svg>
  <div class=\"network-grid\">{"".join(panels)}</div>
</div>
<style>
#chicago-public-road-network-comparison {{ width: 100%; color: var(--foreground); }}
#chicago-public-road-network-comparison h2 {{ margin: 0 0 5px; text-align: center; font-weight: 500; }}
#chicago-public-road-network-comparison .legend {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 14px; margin-bottom: 12px; color: var(--muted-foreground); }}
#chicago-public-road-network-comparison .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
#chicago-public-road-network-comparison .legend i {{ width: 22px; height: 2px; display: inline-block; }}
#chicago-public-road-network-comparison .truth-swatch {{ background: var(--border); }}
#chicago-public-road-network-comparison .reference-swatch {{ background: var(--viz-series-1); }}
#chicago-public-road-network-comparison .proposed-swatch {{ background: var(--viz-series-2); }}
#chicago-public-road-network-comparison .shared-defs {{ position: absolute; }}
#chicago-public-road-network-comparison .network-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 15px 13px; }}
#chicago-public-road-network-comparison .network-panel {{ min-width: 0; }}
#chicago-public-road-network-comparison h3 {{ margin: 0; text-align: center; font-weight: 500; }}
#chicago-public-road-network-comparison .metric-line {{ margin: 2px 0 5px; text-align: center; color: var(--muted-foreground); font-size: 12px; }}
#chicago-public-road-network-comparison .network-panel svg {{ display: block; width: 100%; height: auto; }}
#chicago-public-road-network-comparison .map-frame {{ fill: transparent; stroke: var(--border); vector-effect: non-scaling-stroke; }}
#chicago-public-road-network-comparison .official-basemap {{ fill: none; stroke: var(--border); stroke-width: .55; opacity: .78; vector-effect: non-scaling-stroke; }}
#chicago-public-road-network-comparison .inferred-network {{ fill: none; stroke: var(--viz-series-1); stroke-width: 1.15; opacity: .94; vector-effect: non-scaling-stroke; stroke-linecap: round; stroke-linejoin: round; }}
#chicago-public-road-network-comparison .inferred-network.proposed {{ stroke: var(--viz-series-2); stroke-width: 1.25; }}
#chicago-public-road-network-comparison .north path {{ fill: var(--foreground); }}
#chicago-public-road-network-comparison .north text {{ fill: var(--foreground); font-size: 24px; text-anchor: middle; }}
@media (max-width: 720px) {{ #chicago-public-road-network-comparison .network-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 440px) {{ #chicago-public-road-network-comparison .network-grid {{ grid-template-columns: 1fr; }} }}
</style>
"""
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-vertices", type=Path, required=True)
    parser.add_argument("--truth-edges", type=Path, required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    args = parser.parse_args()

    specs = json.loads(args.methods_manifest.read_text(encoding="utf-8"))
    if len(specs) != 6:
        raise ValueError(f"Expected exactly six methods, found {len(specs)}")
    truth_segments = load_truth_segments(args.truth_vertices, args.truth_edges)
    methods = [load_method(spec, base_dir=args.methods_manifest.parent) for spec in specs]
    extent = common_extent(truth_segments, methods)
    outputs, layout_audit = render_static(truth_segments, methods, extent, args.output_dir, args.language)
    render_html(truth_segments, methods, extent, args.html_output)

    summary = {
        "panel_count": len(methods),
        "official_basemap": {
            "vertices": str(args.truth_vertices.resolve()),
            "edges": str(args.truth_edges.resolve()),
        },
        "official_basemap_segments": len(truth_segments),
        "method_segments": [len(method["segments"]) for method in methods],
        "layout_audit": layout_audit,
        "extent_epsg_26916": list(extent),
        "outputs": [str(path.resolve()) for path in outputs] + [str(args.html_output.resolve())],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "road_network_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
