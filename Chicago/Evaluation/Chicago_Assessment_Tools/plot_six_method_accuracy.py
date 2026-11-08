from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


SHORT_NAMES_ZH = {
    "Ahmed": "Ahmed",
    "Biagioni": "Biagioni",
    "Davies": "Davies",
    "Edelkamp": "Edelkamp",
    "Karagiorgou": "Karagiorgou",
    "Activity-Aware": 'Proposed method',
}

SHORT_NAMES_EN = {
    "Ahmed": "Ahmed",
    "Biagioni": "Biagioni",
    "Davies": "Davies",
    "Edelkamp": "Edelkamp",
    "Karagiorgou": "Karagiorgou",
    "Activity-Aware": "Proposed method",
}


def short_name(label: str, language: str = "zh") -> str:
    names = SHORT_NAMES_EN if language == "en" else SHORT_NAMES_ZH
    for key, value in names.items():
        if key in label:
            return value
    return label


def load_methods(manifest: Path, language: str = "zh") -> list[dict]:
    methods = []
    for spec in json.loads(manifest.read_text(encoding="utf-8")):
        evaluation_path = Path(spec["evaluation"])
        if not evaluation_path.is_absolute():
            evaluation_path = manifest.parent / evaluation_path
        evaluation = json.loads(evaluation_path.resolve().read_text(encoding="utf-8"))
        p15 = evaluation["proximity"]["15"]
        methods.append({
            "label": spec["label"],
            "short": short_name(spec["label"], language),
            "proposed": bool(spec.get("proposed", False)),
            "precision": float(p15["precision"]),
            "recall": float(p15["recall"]),
            "f1": float(p15["f1"]),
            "proximity": evaluation["proximity"],
        })
    if len(methods) != 6:
        raise ValueError(f"Expected six methods, got {len(methods)}")
    return methods


def write_csvs(methods: list[dict], output_dir: Path) -> None:
    with (output_dir / "six_methods_accuracy_at_15m.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "precision", "recall", "f1"])
        writer.writeheader()
        for method in methods:
            writer.writerow({
                "method": method["label"],
                "precision": method["precision"],
                "recall": method["recall"],
                "f1": method["f1"],
            })

    tolerances = sorted({int(k) for method in methods for k in method["proximity"]})
    with (output_dir / "six_methods_f1_by_tolerance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tolerance_m", *[method["short"] for method in methods]])
        for tolerance in tolerances:
            writer.writerow([
                tolerance,
                *[method["proximity"].get(str(tolerance), {}).get("f1", "") for method in methods],
            ])


def render_grouped(methods: list[dict], output_dir: Path, font: FontProperties, title_font: FontProperties, language: str = "zh", layout: str = "standard") -> None:
    x = np.arange(len(methods))
    width = 0.23
    metrics = [
        ("Precision", "precision", "#2f6f9f"),
        ("Recall", "recall", "#d18a2e"),
        ("F1", "f1", "#b4472f"),
    ]
    manuscript = layout == "manuscript"
    fig, ax = plt.subplots(figsize=(6.3, 5.45) if manuscript else (15.5, 8.2))
    for offset, (label, key, color) in zip((-width, 0.0, width), metrics):
        values = [method[key] for method in methods]
        bars = ax.bar(x + offset, values, width=width, color=color, alpha=0.92, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=6.8 if manuscript else 9, fontproperties=font,
            )

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score (0–1)" if language == "en" else 'Evaluation score (0-1)', fontproperties=font, fontsize=9 if manuscript else 12)
    ax.set_xlabel("Map-construction method" if language == "en" else 'Network reconstruction method', fontproperties=font, fontsize=9 if manuscript else 12)
    tick_labels = [method["short"] for method in methods]
    if manuscript and language == "en":
        tick_labels = ["Proposed\nmethod" if value == "Proposed method" else value for value in tick_labels]
    ax.set_xticks(x, tick_labels, fontproperties=font, fontsize=7 if manuscript else 10)
    ax.grid(axis="y", color="#d5d8da", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if manuscript:
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
            frameon=False, prop=font, fontsize=7,
        )
    else:
        ax.legend(loc="upper center", ncol=3, frameon=False, prop=font)
    if not manuscript:
        ax.set_title(
            "Chicago benchmark: Precision, Recall, and F1 at 15 m"
            if language == "en" else
            'Chicago public dataset: Precision, Recall and F1 for six methods',
            fontproperties=title_font, fontsize=19, pad=34,
        )
        ax.text(
            0.5, 1.035,
            "Official reference network · trajectory coverage buffer 15 m · matching tolerance 15 m"
            if language == "en" else
            'Official public reference network / 15 m track coverage band / 15 m matching tolerance',
            transform=ax.transAxes, ha="center", fontproperties=font, fontsize=10, color="#505050",
        )
    for label, method in zip(ax.get_xticklabels(), methods):
        if method["proposed"]:
            label.set_fontweight("bold")
            label.set_color("#b4472f")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"six_methods_precision_recall_f1.{suffix}", dpi=240 if suffix == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_f1_curves(methods: list[dict], output_dir: Path, font: FontProperties, title_font: FontProperties, language: str = "zh", layout: str = "standard") -> None:
    colors = ["#156082", "#5a7f3e", "#8a62a8", "#d18a2e", "#6c757d", "#b4472f"]
    manuscript = layout == "manuscript"
    fig, ax = plt.subplots(figsize=(6.3, 5.15) if manuscript else (13.5, 8.0))
    for method, color in zip(methods, colors):
        points = sorted((int(k), float(v["f1"])) for k, v in method["proximity"].items())
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs, ys, marker="o", markersize=5.5 if method["proposed"] else 4.2,
            linewidth=2.8 if method["proposed"] else 1.6,
            color=color, label=method["short"],
        )
    ax.axvline(15, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        15.4, 0.08,
        "Reported tolerance: 15 m" if language == "en" else 'Final reporting tolerance: 15 m',
        fontproperties=font, fontsize=7 if manuscript else 9, color="#555555",
    )
    ax.set_xlim(3, 41)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Matching tolerance (m)" if language == "en" else 'Matching tolerance (m)', fontproperties=font, fontsize=9 if manuscript else 12)
    ax.set_ylabel("F1", fontproperties=font, fontsize=9 if manuscript else 12)
    ax.set_xticks([4, 6, 8, 10, 15, 20, 25, 30, 35, 40])
    ax.grid(color="#d5d8da", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", frameon=False, prop=font, ncol=2, fontsize=7 if manuscript else None)
    if not manuscript:
        ax.set_title(
            "F1 by matching tolerance for six methods"
            if language == "en" else
            'F1 versus matching tolerance for six methods',
            fontproperties=title_font, fontsize=19, pad=22,
        )
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"six_methods_f1_tolerance_curves.{suffix}", dpi=240 if suffix == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def resolve_font(*, weight: str | None = None) -> FontProperties:
    """Use the Windows installation's CJK font when present, else defaults."""
    windows_dir = os.environ.get("WINDIR")
    if windows_dir:
        font_path = Path(windows_dir) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            return FontProperties(fname=str(font_path), weight=weight)
    return FontProperties(weight=weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--layout", choices=("standard", "manuscript"), default="standard")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = resolve_font()
    title_font = resolve_font(weight="bold")
    methods = load_methods(args.manifest, args.language)
    write_csvs(methods, args.output_dir)
    render_grouped(methods, args.output_dir, font, title_font, args.language, args.layout)
    render_f1_curves(methods, args.output_dir, font, title_font, args.language, args.layout)
    print(json.dumps({
        "methods": len(methods),
        "proposed_f1_15m": next(method["f1"] for method in methods if method["proposed"]),
        "outputs": [str(path) for path in sorted(args.output_dir.glob("six_methods_*"))],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
