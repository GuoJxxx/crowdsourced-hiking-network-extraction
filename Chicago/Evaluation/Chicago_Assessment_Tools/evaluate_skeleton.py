#!/usr/bin/env python3
"""Evaluate an S5 skeleton against the official Chicago benchmark map.

Recall is measured only on ground-truth pixels that are within a configurable
distance of an input trajectory.  This avoids penalising the reconstruction
for streets that were never observed by the supplied 889 trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import LineString


def graph_length(mask: np.ndarray, cell: float) -> float:
    horizontal = np.count_nonzero(mask[:, :-1] & mask[:, 1:])
    vertical = np.count_nonzero(mask[:-1, :] & mask[1:, :])
    diagonal = (
        np.count_nonzero(mask[:-1, :-1] & mask[1:, 1:])
        + np.count_nonzero(mask[1:, :-1] & mask[:-1, 1:])
    )
    return float(cell * (horizontal + vertical + math.sqrt(2.0) * diagonal))


def load_ground_truth(vertices_path: Path, edges_path: Path):
    vertices = pd.read_csv(
        vertices_path, header=None, names=["id", "x", "y"]
    ).set_index("id")
    edges = pd.read_csv(
        edges_path, header=None, names=["id", "u", "v", "flag"]
    )
    lines = []
    for u, v in edges[["u", "v"]].itertuples(index=False):
        lines.append(
            LineString(
                [
                    (float(vertices.at[u, "x"]), float(vertices.at[u, "y"])),
                    (float(vertices.at[v, "x"]), float(vertices.at[v, "y"])),
                ]
            )
        )
    return lines


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--raw-union", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--vertices", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--coverage-m", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.skeleton) as src:
        prediction = src.read(1) > 0
        profile = {
            "shape": prediction.shape,
            "transform": src.transform,
            "crs": str(src.crs),
            "cell_x": abs(float(src.transform.a)),
            "cell_y": abs(float(src.transform.e)),
        }
    with rasterio.open(args.raw_union) as src:
        raw = src.read(1) > 0
        if src.shape != prediction.shape or src.transform != profile["transform"]:
            raise ValueError("raw-union grid is not aligned with skeleton")
    with rasterio.open(args.support) as src:
        support = src.read(1)
        if src.shape != prediction.shape or src.transform != profile["transform"]:
            raise ValueError("support grid is not aligned with skeleton")

    if not math.isclose(profile["cell_x"], profile["cell_y"], rel_tol=1e-9):
        raise ValueError("evaluation currently requires square pixels")
    cell = profile["cell_x"]

    lines = load_ground_truth(args.vertices, args.edges)
    truth = rasterize(
        ((line, 1) for line in lines),
        out_shape=prediction.shape,
        transform=profile["transform"],
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    truth_distance = ndi.distance_transform_edt(~truth) * cell
    prediction_distance = ndi.distance_transform_edt(~prediction) * cell
    raw_distance = ndi.distance_transform_edt(~raw) * cell
    covered_truth = truth & (raw_distance <= float(args.coverage_m))

    thresholds = [4.0, 6.0, 8.0, 10.0, 15.0]
    proximity = {}
    pred_count = int(prediction.sum())
    covered_truth_count = int(covered_truth.sum())
    for tolerance in thresholds:
        precision = safe_ratio(
            int(np.count_nonzero(prediction & (truth_distance <= tolerance))),
            pred_count,
        )
        recall = safe_ratio(
            int(np.count_nonzero(covered_truth & (prediction_distance <= tolerance))),
            covered_truth_count,
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        proximity[str(int(tolerance))] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    labels, component_count = ndi.label(
        prediction, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.ravel())[1:]
    neighbours = ndi.convolve(
        prediction.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - prediction.astype(np.uint8)

    pred_support = support[prediction]
    result = {
        "skeleton": str(args.skeleton.resolve()),
        "crs": profile["crs"],
        "cell_size_m": cell,
        "coverage_distance_m": float(args.coverage_m),
        "prediction_pixels": pred_count,
        "prediction_graph_length_m": graph_length(prediction, cell),
        "truth_pixels_in_grid": int(truth.sum()),
        "covered_truth_pixels": covered_truth_count,
        "covered_truth_ratio": safe_ratio(covered_truth_count, int(truth.sum())),
        "proximity": proximity,
        "components_8_connected": int(component_count),
        "components_ge_5_pixels": int(np.count_nonzero(sizes >= 5)),
        "components_ge_20_pixels": int(np.count_nonzero(sizes >= 20)),
        "largest_component_pixels": int(sizes.max()) if sizes.size else 0,
        "endpoints_pixel_degree_1": int(np.count_nonzero(prediction & (neighbours == 1))),
        "junction_pixels_degree_ge_3": int(np.count_nonzero(prediction & (neighbours >= 3))),
        "support_on_prediction": {
            "mean": float(pred_support.mean()) if pred_support.size else 0.0,
            "median": float(np.median(pred_support)) if pred_support.size else 0.0,
            "fraction_ge_1": safe_ratio(int(np.count_nonzero(pred_support >= 1)), pred_count),
            "fraction_ge_2": safe_ratio(int(np.count_nonzero(pred_support >= 2)), pred_count),
            "fraction_ge_3": safe_ratio(int(np.count_nonzero(pred_support >= 3)), pred_count),
        },
        "raw_trace_precision_at_8m": safe_ratio(
            int(np.count_nonzero(raw & (truth_distance <= 8.0))), int(raw.sum())
        ),
        "raw_trace_recall_on_covered_truth_at_8m": safe_ratio(
            int(np.count_nonzero(covered_truth & (raw_distance <= 8.0))),
            covered_truth_count,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
