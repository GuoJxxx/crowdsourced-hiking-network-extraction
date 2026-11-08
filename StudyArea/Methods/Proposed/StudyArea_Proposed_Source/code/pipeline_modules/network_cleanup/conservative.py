from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import LineString

from .models import NetworkData, reindex_network
from .optimizer import edge_confidence


PROTECTED_SOURCES = {"primary", "mixed", "support_rescued"}


def conservative_prune_network(
    network: NetworkData,
    terminal_length_m: float = 12.0,
    confidence_threshold: float = 0.22,
    min_component_length_m: float = 20.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Only remove unmistakable auxiliary noise.

    S5 primary, mixed, rescued and U-turn edges are immutable evidence and are never
    removed. This prevents S6 from degrading a good skeleton merely because HMM
    coverage is locally sparse.
    """
    net = network.copy()
    for edge in net.edges:
        edge["confidence"] = edge_confidence(edge)
        edge["retained_reason"] = "s5_baseline"

    dropped: List[Dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        degree = defaultdict(int)
        for edge in net.edges:
            degree[int(edge["u"])] += 1
            degree[int(edge["v"])] += 1

        kept: List[Dict[str, Any]] = []
        for edge in net.edges:
            source = str(edge.get("source_type", "primary"))
            protected = (bool(edge.get("uturn_protected")) or bool(edge.get("gap_repair"))
                         or bool(edge.get("approved_gap_repair")) or bool(edge.get("approved_directional_collapse")) or source in PROTECTED_SOURCES)
            terminal = degree[int(edge["u"])] == 1 or degree[int(edge["v"])] == 1
            tracks = int(edge.get("distinct_tracks", 0) or 0)
            support_max = float(edge.get("support_max", 0) or 0)
            support_mean = float(edge.get("support_mean", 0) or 0)
            kde_max = float(edge.get("kde_max", 0) or 0)
            weak = float(edge.get("confidence", 0) or 0) < float(confidence_threshold)

            delete = (
                source == "auxiliary"
                and terminal
                and float(edge.get("length_m", 0) or 0) < float(terminal_length_m)
                and weak
                and tracks == 0
                and support_max < 2.0
                and support_mean < 0.50
                and kde_max <= 0.0
                and not protected
            )
            if delete:
                row = dict(edge)
                row["drop_reason"] = "unmistakable_short_auxiliary_noise"
                dropped.append(row)
                changed = True
            else:
                if protected:
                    edge["retained_reason"] = (
                        "uturn_exact_preserved" if edge.get("uturn_protected")
                        else "s5_primary_or_rescued_preserved"
                    )
                elif tracks > 0:
                    edge["retained_reason"] = "track_supported"
                elif support_max >= 2 or support_mean >= 0.50:
                    edge["retained_reason"] = "raster_supported"
                else:
                    edge["retained_reason"] = "topology_continuity"
                kept.append(edge)
        net.edges = kept

    # Only remove a tiny isolated component when every edge is auxiliary and weak.
    adjacency = defaultdict(list)
    for idx, edge in enumerate(net.edges):
        adjacency[int(edge["u"])].append(idx)
        adjacency[int(edge["v"])].append(idx)
    remaining = set(range(len(net.edges)))
    remove_ids = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            idx = stack.pop()
            edge = net.edges[idx]
            for node_id in (int(edge["u"]), int(edge["v"])):
                for other in adjacency[node_id]:
                    if other in remaining:
                        remaining.remove(other)
                        component.add(other)
                        stack.append(other)
        length = sum(float(net.edges[idx].get("length_m", 0) or 0) for idx in component)
        if length < float(min_component_length_m) and all(
            str(net.edges[idx].get("source_type", "")) == "auxiliary"
            and not bool(net.edges[idx].get("uturn_protected"))
            and int(net.edges[idx].get("distinct_tracks", 0) or 0) == 0
            and float(net.edges[idx].get("support_max", 0) or 0) < 2.0
            and float(net.edges[idx].get("confidence", 0) or 0) < float(confidence_threshold)
            for idx in component
        ):
            remove_ids.update(component)

    if remove_ids:
        kept = []
        for idx, edge in enumerate(net.edges):
            if idx in remove_ids:
                row = dict(edge)
                row["drop_reason"] = "tiny_isolated_auxiliary_component"
                dropped.append(row)
            else:
                kept.append(edge)
        net.edges = kept

    return reindex_network(net), dropped


class BaselineAuditIndex:
    def __init__(self, network: NetworkData) -> None:
        self.edge_count = int(len(network.edges))
        self.length_m = float(sum(float(edge.get("length_m", 0) or 0) for edge in network.edges))


def baseline_union(network: NetworkData):
    """Return a lightweight provenance index instead of a huge unary union.

    S6 cleanup never creates geometry.  Every edge carries its maximum shift from the
    original S5 line, so a global line union and buffer are unnecessary and can consume
    many gigabytes on a full study area.
    """
    return BaselineAuditIndex(network)


def audit_geometry_against_baseline(
    network: NetworkData,
    baseline_geometry,
    corridor_m: float = 1.5,
) -> List[Dict[str, Any]]:
    """Fast monotonicity audit based on immutable S5 geometry provenance."""
    rows: List[Dict[str, Any]] = []
    for edge in network.edges:
        geom = edge["geometry"]
        shift = float(edge.get("max_geometry_shift_m", 0.0) or 0.0)
        origin = str(edge.get("geometry_origin", "s5_skeleton"))
        valid_origin = origin.startswith("s5_") or origin == "s5_skeleton"
        violation = (not valid_origin) or shift > float(corridor_m) + 1e-9
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "source_type": str(edge.get("source_type", "")),
            "uturn_protected": bool(edge.get("uturn_protected")),
            "length_m": float(geom.length),
            "geometry_origin": origin,
            "max_geometry_shift_m": shift,
            "outside_baseline_m": float(geom.length) if violation else 0.0,
            "outside_baseline_fraction": 1.0 if violation else 0.0,
            "geometry": geom,
        })
    return rows



def audit_geometry_against_primary_raster(
    network: NetworkData,
    primary_path,
    tolerance_cells: float = 0.0,
    corridor_m: float = 0.0,
    uturn_corridor_m: Optional[float] = None,
    progress_every: int = 10000,
) -> List[Dict[str, Any]]:
    """Audit ordinary final geometry against the *primary pixel footprint* plus
    the explicitly approved smoothing corridor.

    v0.5 compared every final sample directly with primary **pixel centres**.
    That is too strict for a 2 m raster: even an unshifted diagonal pixel path can
    be ~half a pixel diagonal away from the nearest pixel centre between vertices,
    and a valid 1.5 m Hausdorff smoothing can therefore be falsely reported as
    outside the baseline corridor.

    v0.5 treats every active primary cell as an area footprint first, then adds the
    configured vector-smoothing corridor.  In centre-distance form the acceptance
    radius is therefore::

        half_pixel_diagonal + approved_smoothing_corridor

    A second provenance guard remains active: ``max_geometry_shift_m`` must not
    exceed the approved corridor (ordinary or U-turn core).  Thus the spatial audit
    is relaxed only by the raster cell footprint; it does *not* silently allow a
    larger vector shift.

    ``tolerance_cells`` is retained as a compatibility floor for older callers.
    """
    import math
    import time
    import numpy as np
    import rasterio
    from .tile_io import RasterBlockSampler

    primary_path = str(primary_path)
    with rasterio.open(primary_path) as ds:
        if ds.crs != network.crs:
            raise ValueError(f"""primary CRS differs from network CRS: {ds.crs} != {network.crs}""")
        transform = ds.transform
        inv = ~transform
        # Works for ordinary north-up rasters and remains meaningful for a rotated grid.
        cell_x = float(math.hypot(transform.a, transform.d))
        cell_y = float(math.hypot(transform.b, transform.e))
        cell_min = max(min(cell_x, cell_y), 1e-9)
        cell_max = max(cell_x, cell_y, 1e-9)
        half_pixel_diagonal = 0.5 * float(math.hypot(cell_x, cell_y))
        width, height = int(ds.width), int(ds.height)

    ordinary_corridor = max(0.0, float(corridor_m))
    core_corridor = ordinary_corridor if uturn_corridor_m is None else max(0.0, float(uturn_corridor_m))
    compatibility_floor_m = max(0.0, float(tolerance_cells)) * cell_max
    spacing = max(cell_min * 0.5, 0.25)

    rows_out: List[Dict[str, Any]] = []
    started = time.perf_counter()
    allowed_origin_prefixes = (
        "s5_",
        "primary_corridor_",
    )

    with RasterBlockSampler({"primary": primary_path}, cache_blocks=256) as sampler:
        total = len(network.edges)
        for index, edge in enumerate(network.edges, 1):
            geom = edge["geometry"]
            length = float(geom.length)
            uturn_core = bool(edge.get("uturn_core_protected", False))
            approved_corridor = core_corridor if uturn_core else ordinary_corridor
            center_tolerance = max(
                compatibility_floor_m,
                half_pixel_diagonal + approved_corridor,
            )
            # Search enough neighbour cells to cover the full centre-distance radius.
            radius_r = max(1, int(math.ceil(center_tolerance / max(cell_y, 1e-9))) + 1)
            radius_c = max(1, int(math.ceil(center_tolerance / max(cell_x, 1e-9))) + 1)

            distances = np.arange(0.0, length, spacing, dtype=float)
            if len(distances) == 0 or distances[-1] < length:
                distances = np.r_[distances, length]
            xs = np.empty(len(distances), dtype=float)
            ys = np.empty(len(distances), dtype=float)
            for i, distance in enumerate(distances):
                point = geom.interpolate(float(distance))
                xs[i], ys[i] = float(point.x), float(point.y)

            cols_f, rows_f = inv * (xs, ys)
            base_rows = np.floor(np.asarray(rows_f, dtype=float)).astype(np.int64)
            base_cols = np.floor(np.asarray(cols_f, dtype=float)).astype(np.int64)
            min_dist = np.full(len(distances), np.inf, dtype=float)
            supported = np.zeros(len(distances), dtype=bool)

            for dr in range(-radius_r, radius_r + 1):
                for dc in range(-radius_c, radius_c + 1):
                    rr = base_rows + dr
                    cc = base_cols + dc
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    if not np.any(valid):
                        continue
                    values = np.zeros(len(rr), dtype=float)
                    values[valid] = sampler.values_rc(
                        "primary", rr[valid], cc[valid], default=0.0
                    )
                    active = valid & (values != 0)
                    if not np.any(active):
                        continue
                    center_x = (
                        transform.c
                        + (cc[active] + 0.5) * transform.a
                        + (rr[active] + 0.5) * transform.b
                    )
                    center_y = (
                        transform.f
                        + (cc[active] + 0.5) * transform.d
                        + (rr[active] + 0.5) * transform.e
                    )
                    distance_xy = np.hypot(xs[active] - center_x, ys[active] - center_y)
                    min_dist[active] = np.minimum(min_dist[active], distance_xy)
                    supported[active] |= distance_xy <= center_tolerance + 1e-9

            spatial_outside = ~supported
            spatial_outside_fraction = float(np.mean(spatial_outside)) if len(spatial_outside) else 0.0

            origin = str(edge.get("geometry_origin", ""))
            shift = float(edge.get("max_geometry_shift_m", 0.0) or 0.0)
            origin_valid = bool(origin.startswith(allowed_origin_prefixes) or origin in {"s5_skeleton", "s5_skeleton_subset"})
            provenance_valid = bool(origin_valid and shift <= approved_corridor + 1e-9)
            violation_fraction = 1.0 if not provenance_valid else spatial_outside_fraction

            rows_out.append({
                "edge_id": int(edge["edge_id"]),
                "source_type": str(edge.get("source_type", "")),
                "uturn_protected": bool(edge.get("uturn_protected")),
                "uturn_core_protected": uturn_core,
                "length_m": length,
                "sample_count": int(len(distances)),
                "max_distance_to_primary_pixel_center_m": (
                    float(np.max(min_dist[np.isfinite(min_dist)]))
                    if np.any(np.isfinite(min_dist)) else float("inf")
                ),
                "pixel_half_diagonal_m": float(half_pixel_diagonal),
                "approved_smoothing_corridor_m": float(approved_corridor),
                "effective_center_tolerance_m": float(center_tolerance),
                "spatial_outside_fraction": spatial_outside_fraction,
                "geometry_origin": origin,
                "max_geometry_shift_m": shift,
                "origin_valid": origin_valid,
                "provenance_valid": provenance_valid,
                "outside_baseline_m": float(length * violation_fraction),
                "outside_baseline_fraction": float(violation_fraction),
                "geometry": geom,
            })
            if progress_every > 0 and (index % int(progress_every) == 0 or index == total):
                print(
                    f"    [audit:pixel-footprint] edges={index:,}/{total:,}, "
                    f"violations={sum(r['outside_baseline_fraction'] > 1e-6 for r in rows_out):,}, "
                    f"half_pixel_diag={half_pixel_diagonal:.3f}m, elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    return rows_out

def audit_approved_micro_gaps(
    network: NetworkData,
    primary_path,
    max_missing_pixels: int = 0,
    max_step_cells: float = 1.45,
    max_length_m: float = 10.0,
    max_angle_deg: float = 45.0,
    long_gap_threshold_m: float = 6.0,
    long_gap_max_angle_deg: float = 30.0,
    long_gap_min_evidence_fraction: float = 0.50,
) -> List[Dict[str, Any]]:
    """Audit evidence-approved gap repairs independently of their final vector shape.

    v0.5 changed the *output geometry* of approved gap repairs from a Bresenham
    stair-step polyline to a straight directional connector.  Therefore the audit
    must not require every vector vertex to advance by one raster cell, nor require
    geometry_origin == ``micro_gap_bresenham``.

    Instead, v0.5 reconstructs the endpoint-to-endpoint Bresenham cell chain only
    for *audit/evidence* purposes and validates:
      1. both endpoints are on active primary pixels;
      2. the reconstructed raster path is 8-connected;
      3. missing primary cells do not exceed the approved physical 10 m rule;
      4. vector length is within max_length_m;
      5. endpoint direction angles satisfy the same short/long gap thresholds;
      6. long gaps satisfy the stored evidence-fraction threshold;
      7. geometry provenance is an explicitly approved gap-repair origin.

    Thus a valid straight connector may intentionally pass through primary=0 cells;
    those cells are exactly the gap being repaired and are not an audit violation.
    """
    import math
    import numpy as np
    import rasterio
    from .tile_io import RasterBlockSampler

    def _bresenham_rc(r0: int, c0: int, r1: int, c1: int):
        x0, y0, x1, y1 = int(c0), int(r0), int(c1), int(r1)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        rows, cols = [], []
        while True:
            rows.append(y0)
            cols.append(x0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    # Audit ONLY genuine micro-gap repairs.  v0.5 introduced several longer
    # evidence-approved connector classes (junction repair, S4/forced gaps and
    # component stitching) which intentionally also carry ``gap_repair=True`` so
    # downstream pruning protects them.  Re-auditing those connectors with the
    # <=10 m micro-gap envelope is a category error and caused valid component
    # stitches to be rejected at the final audit.
    gap_edges = [
        edge for edge in network.edges
        if (
            bool(edge.get("micro_gap_repair"))
            or str(edge.get("source_type", "")) == "micro_gap_repair"
            or str(edge.get("geometry_origin", "")).startswith("micro_gap_")
        )
        and (bool(edge.get("gap_repair")) or bool(edge.get("approved_gap_repair")))
        and not bool(edge.get("s7_force_gap_repair"))
        and not bool(edge.get("junction_repair"))
        and not bool(edge.get("approved_junction_repair"))
        and not bool(edge.get("component_stitching_repair"))
    ]
    if not gap_edges:
        return []

    with rasterio.open(primary_path) as ds:
        transform = ds.transform
        inv = ~transform
        if ds.crs != network.crs:
            raise ValueError(f"""primary CRS differs from network CRS: {ds.crs} != {network.crs}""")
        cell_x = float(math.hypot(transform.a, transform.d))
        cell_y = float(math.hypot(transform.b, transform.e))
        cell = max(cell_x, cell_y, 1e-9)

    effective_max_missing = int(max_missing_pixels)
    if effective_max_missing <= 0:
        effective_max_missing = max(1, int(math.ceil(float(max_length_m) / cell)) - 1)

    approved_origins = {
        "micro_gap_bresenham",
        "micro_gap_directional_straight_connector_v0_5",
        "micro_gap_directional_straight_connector_v0_5",
    }

    rows_out: List[Dict[str, Any]] = []
    with RasterBlockSampler({"primary": primary_path}, cache_blocks=128) as sampler:
        for edge in gap_edges:
            geom = edge["geometry"]
            coords = np.asarray(geom.coords, dtype=float)
            if len(coords) < 2:
                rows_out.append({
                    "edge_id": int(edge["edge_id"]),
                    "gap_repair": True,
                    "outside_baseline_m": float(getattr(geom, "length", 0.0)),
                    "outside_baseline_fraction": 1.0,
                    "reason": "invalid_gap_geometry",
                    "geometry": geom,
                })
                continue

            # Use only topology endpoints to reconstruct the raster path.
            endpoint_xy = np.asarray([coords[0], coords[-1]], dtype=float)
            cols_f, rows_f = inv * (endpoint_xy[:, 0], endpoint_xy[:, 1])
            end_rows = np.floor(np.asarray(rows_f, dtype=float)).astype(np.int64)
            end_cols = np.floor(np.asarray(cols_f, dtype=float)).astype(np.int64)
            rows, cols = _bresenham_rc(
                int(end_rows[0]), int(end_cols[0]),
                int(end_rows[1]), int(end_cols[1]),
            )

            values = sampler.values_rc("primary", rows, cols, default=0.0)
            missing = values == 0
            missing_count = (
                int(np.count_nonzero(missing[1:-1]))
                if len(missing) > 2 else int(np.count_nonzero(missing))
            )
            endpoint_ok = bool(
                len(values) >= 2 and values[0] != 0 and values[-1] != 0
            )

            # Reconstructed Bresenham chain must be 8-connected.
            dr = np.abs(np.diff(rows))
            dc = np.abs(np.diff(cols))
            raster_step_ok = bool(
                len(rows) >= 2
                and np.all(dr <= 1)
                and np.all(dc <= 1)
                and np.all((dr + dc) >= 1)
            )

            length_m = float(geom.length)
            raw_attr_missing = edge.get("micro_gap_missing_pixels", missing_count)
            try:
                attr_missing = (
                    int(raw_attr_missing)
                    if raw_attr_missing is not None and np.isfinite(float(raw_attr_missing))
                    else missing_count
                )
            except Exception:
                attr_missing = missing_count

            def _safe_float_attr(name: str, default: float) -> float:
                raw = edge.get(name, default)
                if raw is None:
                    return float(default)
                try:
                    value = float(raw)
                except Exception:
                    return float(default)
                return value if np.isfinite(value) else float(default)

            evidence_fraction = _safe_float_attr("micro_gap_evidence_fraction", 0.0)
            # IMPORTANT: 0.0 is a valid perfect-alignment angle.  Do not use
            # ``value or 180`` here; that was the v0.5 bug that converted 0° to 180°.
            angle_u = _safe_float_attr("micro_gap_angle_u_deg", 180.0)
            angle_v = _safe_float_attr("micro_gap_angle_v_deg", 180.0)
            angle_max = max(angle_u, angle_v)

            # Match the GENERATION logic exactly:
            # - the tighter 30° angle threshold is triggered by *physical distance*
            #   > long_gap_threshold_m;
            # - dense evidence is required for a physically long gap OR >2 missing
            #   primary pixels.
            # v0.5 incorrectly used missing_count>2 to tighten the angle as well.
            physically_long = bool(length_m > float(long_gap_threshold_m))
            requires_dense_evidence = bool(physically_long or missing_count > 2)
            angle_limit = (
                float(long_gap_max_angle_deg) if physically_long else float(max_angle_deg)
            )
            angle_ok = bool(angle_max <= angle_limit + 1e-9)
            evidence_ok = bool(
                (not requires_dense_evidence)
                or evidence_fraction >= float(long_gap_min_evidence_fraction) - 1e-12
            )

            origin = str(edge.get("geometry_origin", ""))
            origin_ok = bool(origin in approved_origins)
            approval_ok = bool(edge.get("approved_gap_repair", edge.get("gap_repair", False)))
            missing_ok = bool(
                missing_count <= int(effective_max_missing)
                and attr_missing <= int(effective_max_missing)
                and abs(attr_missing - missing_count) <= 1
            )
            length_ok = bool(length_m <= float(max_length_m) + 1e-6)

            valid = bool(
                approval_ok
                and endpoint_ok
                and raster_step_ok
                and missing_ok
                and length_ok
                and angle_ok
                and evidence_ok
                and origin_ok
            )

            rows_out.append({
                "edge_id": int(edge["edge_id"]),
                "source_type": str(edge.get("source_type", "")),
                "uturn_protected": bool(edge.get("uturn_protected")),
                "gap_repair": True,
                "approved_gap_repair": approval_ok,
                "length_m": length_m,
                "vector_vertex_count": int(len(coords)),
                "audit_raster_sample_count": int(len(rows)),
                "micro_gap_missing_pixels": missing_count,
                "attribute_missing_pixels": attr_missing,
                "effective_max_missing_pixels": int(effective_max_missing),
                "evidence_fraction": evidence_fraction,
                "physically_long_gap": physically_long,
                "requires_dense_evidence": requires_dense_evidence,
                "evidence_ok": evidence_ok,
                "angle_u_deg": angle_u,
                "angle_v_deg": angle_v,
                "angle_limit_deg": angle_limit,
                "angle_ok": angle_ok,
                "endpoint_primary_supported": endpoint_ok,
                "raster_path_8connected": raster_step_ok,
                "length_ok": length_ok,
                "origin_ok": origin_ok,
                "outside_baseline_m": 0.0 if valid else length_m,
                "outside_baseline_fraction": 0.0 if valid else 1.0,
                "geometry_origin": origin,
                "audit_reason": (
                    "approved_gap_repair_valid"
                    if valid else
                    "failed:" + ",".join([
                        name for name, ok in (
                            ("approval", approval_ok),
                            ("endpoint_primary", endpoint_ok),
                            ("raster_8connected", raster_step_ok),
                            ("missing_pixels", missing_ok),
                            ("length", length_ok),
                            ("angle", angle_ok),
                            ("evidence", evidence_ok),
                            ("origin", origin_ok),
                        ) if not ok
                    ])
                ),
                "geometry": geom,
            })
    return rows_out


def audit_approved_directional_collapses(
    network: NetworkData,
    min_evidence_fraction: float = 0.60,
    max_gate_angle_deg: float = 30.0,
) -> List[Dict[str, Any]]:
    """Audit v0.5 evidence-approved digital-loop centreline replacements.

    These edges intentionally need not follow the original primary pixel centres, so
    they are audited by their immutable approval evidence instead of the primary
    raster corridor used for ordinary edges.
    """
    nodes = {int(n["node_id"]): n for n in network.nodes}
    rows: List[Dict[str, Any]] = []
    for edge in network.edges:
        if not bool(edge.get("approved_directional_collapse")):
            continue
        geom = edge.get("geometry")
        ev_value = edge.get("directional_collapse_evidence_fraction")
        aa_value = edge.get("directional_collapse_gate_angle_a_deg")
        ab_value = edge.get("directional_collapse_gate_angle_b_deg")
        ev = float(0.0 if ev_value is None else ev_value)
        # 0° is a valid perfect gate alignment; preserve numeric zero.
        aa = float(180.0 if aa_value is None else aa_value)
        ab = float(180.0 if ab_value is None else ab_value)
        u, v = int(edge["u"]), int(edge["v"])
        endpoint_ok = False
        if isinstance(geom, LineString) and u in nodes and v in nodes and len(geom.coords) >= 2:
            a = geom.coords[0]; b = geom.coords[-1]
            nu = nodes[u]; nv = nodes[v]
            endpoint_ok = (
                ((a[0]-float(nu["x"]))**2 + (a[1]-float(nu["y"]))**2) ** 0.5 <= 1e-6
                and ((b[0]-float(nv["x"]))**2 + (b[1]-float(nv["y"]))**2) ** 0.5 <= 1e-6
            ) or (
                ((b[0]-float(nu["x"]))**2 + (b[1]-float(nu["y"]))**2) ** 0.5 <= 1e-6
                and ((a[0]-float(nv["x"]))**2 + (a[1]-float(nv["y"]))**2) ** 0.5 <= 1e-6
            )
        valid = bool(
            isinstance(geom, LineString) and geom.length > 0 and geom.is_simple
            and endpoint_ok
            and ev >= float(min_evidence_fraction)
            and max(aa, ab) <= float(max_gate_angle_deg) + 1e-9
            and str(edge.get("geometry_origin", "")) == "directional_loop_diagonal_v0_5"
        )
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "source_type": str(edge.get("source_type", "")),
            "approved_directional_collapse": True,
            "length_m": float(geom.length) if isinstance(geom, LineString) else 0.0,
            "evidence_fraction": ev,
            "gate_angle_a_deg": aa,
            "gate_angle_b_deg": ab,
            "endpoint_topology_preserved": endpoint_ok,
            "outside_baseline_m": 0.0 if valid else float(geom.length if isinstance(geom, LineString) else 0.0),
            "outside_baseline_fraction": 0.0 if valid else 1.0,
            "geometry_origin": str(edge.get("geometry_origin", "")),
            "geometry": geom,
        })
    return rows
