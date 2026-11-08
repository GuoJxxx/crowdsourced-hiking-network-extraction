#!/usr/bin/env python3
"""Chicago raster-proximity assessment (manuscript Eqs. 11–13)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi

from evaluate_skeleton import graph_length, load_ground_truth, safe_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--layer", default="edges")
    parser.add_argument("--raw-union", type=Path, required=True)
    parser.add_argument("--vertices", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--coverage-m", type=float, default=15.0)
    parser.add_argument(
        "--tolerances",
        default="4,6,8,10,15,20,25,30,35,40",
        help="Comma-separated matching tolerances in metres.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.raw_union) as src:
        raw = src.read(1) > 0
        transform = src.transform
        shape = src.shape
        raster_crs = src.crs
        cell = abs(float(transform.a))

    network = gpd.read_file(args.network, layer=args.layer)
    if network.crs is None:
        raise ValueError("final network has no CRS")
    if network.crs != raster_crs:
        network = network.to_crs(raster_crs)
    geometries = [geometry for geometry in network.geometry if geometry is not None and not geometry.is_empty]
    prediction = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    truth_lines = load_ground_truth(args.vertices, args.edges)
    truth = rasterize(
        ((line, 1) for line in truth_lines),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    truth_distance = ndi.distance_transform_edt(~truth) * cell
    prediction_distance = ndi.distance_transform_edt(~prediction) * cell
    raw_distance = ndi.distance_transform_edt(~raw) * cell
    covered_truth = truth & (raw_distance <= float(args.coverage_m))

    pred_count = int(prediction.sum())
    truth_count = int(covered_truth.sum())
    proximity = {}
    tolerances = sorted(
        {
            float(value.strip())
            for value in str(args.tolerances).split(",")
            if value.strip()
        }
    )
    for tolerance in tolerances:
        # Eq. 12: reconstructed pixels supported by the independent reference.
        precision = safe_ratio(
            int(np.count_nonzero(prediction & (truth_distance <= tolerance))), pred_count
        )
        # Eq. 11: covered reference pixels reached by the reconstruction.
        recall = safe_ratio(
            int(np.count_nonzero(covered_truth & (prediction_distance <= tolerance))), truth_count
        )
        # Eq. 13: harmonic mean of precision and trajectory-covered recall.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        proximity[str(int(tolerance))] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    labels, component_count = ndi.label(
        prediction, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.ravel())[1:]
    result = {
        "network": str(args.network.resolve()),
        "layer": args.layer,
        "crs": str(network.crs),
        "feature_edges": int(len(network)),
        "feature_length_m": float(network.geometry.length.sum()),
        "rasterized_pixels": pred_count,
        "rasterized_graph_length_m": graph_length(prediction, cell),
        "covered_truth_pixels": truth_count,
        "proximity": proximity,
        "raster_components_8_connected": int(component_count),
        "largest_raster_component_pixels": int(sizes.max()) if sizes.size else 0,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
