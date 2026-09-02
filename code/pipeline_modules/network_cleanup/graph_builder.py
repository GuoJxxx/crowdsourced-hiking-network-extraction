from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import thin
from shapely import intersects as shapely_intersects
from shapely.geometry import LineString, Point
from rasterio.windows import Window

from .models import NetworkData, compute_degrees, reindex_network
from .tile_io import RasterBlockSampler, TileConfig, tile_windows, validate_aligned_rasters


OFFSETS_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
# Half-neighbourhood is enough for undirected node-pixel clustering.
OFFSETS_HALF: Tuple[Tuple[int, int], ...] = ((0, 1), (1, -1), (1, 0), (1, 1))


def _path_signature(paths: Iterable[Optional[Path]], extra: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw).resolve()
        if not path.exists():
            continue
        stat = path.stat()
        items.append({
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = {"files": items, "extra": extra}
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return payload


def _cache_matches(path: Path, signature: Dict[str, Any]) -> bool:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        return previous.get("sha256") == signature.get("sha256")
    except Exception:
        return False


def _pixel_xy_from_flat(flat: np.ndarray, width: int, transform) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=np.int64)
    rows = flat // int(width)
    cols = flat - rows * int(width)
    # General affine transform, including the uncommon rotated-raster case.
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _deduplicate_consecutive(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 1:
        return coords
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep]


def _build_line(
    path_indices: Sequence[int],
    active_flat: np.ndarray,
    width: int,
    transform,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
) -> Optional[LineString]:
    idx = np.asarray(path_indices, dtype=np.int64)
    flats = np.asarray(active_flat[idx], dtype=np.int64)
    xs, ys = _pixel_xy_from_flat(flats, width, transform)
    coords = np.column_stack([xs, ys])
    coords[0] = np.asarray(start_xy, dtype=float)
    coords[-1] = np.asarray(end_xy, dtype=float)
    coords = _deduplicate_consecutive(coords)
    if len(coords) < 2:
        return None
    geom = LineString(coords)
    return geom if geom.length > 0 else None


def collect_skeleton_pixels_fast(
    primary_path: Path,
    auxiliary_path: Optional[Path],
    rescued_path: Optional[Path],
    tile_cfg: TileConfig,
    work_dir: Optional[Path] = None,
    resume: bool = False,
    progress_every_tiles: int = 100,
    geometry_source: str = "primary",
    thin_geometry: bool = True,
    thin_halo_pixels: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """Read S5 geometry as a strict, one-pixel-wide sparse mask.

    ``geometry_source='primary'`` is the safe default: auxiliary and rescued rasters
    are sampled only as evidence on primary pixels and cannot create extra roads.
    A halo-aware morphological thinning pass removes 2x2 pixel blocks that otherwise
    become rectangular graph cycles. The output geometry is always a subset of the
    selected S5 raster pixels.
    """
    geometry_source = str(geometry_source).lower().strip()
    if geometry_source not in {"primary", "primary_rescued", "union"}:
        raise ValueError(f"不支持的 geometry_source: {geometry_source}")
    paths = [Path(primary_path)]
    if auxiliary_path and Path(auxiliary_path).exists():
        paths.append(Path(auxiliary_path))
    if rescued_path and Path(rescued_path).exists():
        paths.append(Path(rescued_path))
    ref = validate_aligned_rasters(paths)

    signature = _path_signature(
        paths,
        {
            "tile_size_m": float(tile_cfg.tile_size_m),
            "origin_x": tile_cfg.origin_x,
            "origin_y": tile_cfg.origin_y,
            "shape": [int(ref["height"]), int(ref["width"])],
            "transform": tuple(ref["transform"]),
            "crs": str(ref["crs"]),
            "geometry_source": geometry_source,
            "thin_geometry": bool(thin_geometry),
            "thin_halo_pixels": int(thin_halo_pixels),
        },
    )
    cache_active = cache_flags = cache_tiles = cache_meta = None
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_active = work_dir / "active_flat.npy"
        cache_flags = work_dir / "source_flags.npy"
        cache_tiles = work_dir / "tile_summary.csv"
        cache_meta = work_dir / "skeleton_cache_signature.json"
        if (
            resume and cache_active.exists() and cache_flags.exists()
            and cache_tiles.exists() and cache_meta.exists()
            and _cache_matches(cache_meta, signature)
        ):
            print("    [stage1:read] reuse sparse-pixel checkpoint", flush=True)
            active = np.load(cache_active, mmap_mode="r")
            flags = np.load(cache_flags, mmap_mode="r")
            tile_rows = pd.read_csv(cache_tiles).to_dict("records")
            print(f"    [stage1:read] active_pixels={len(active):,}", flush=True)
            return active, flags, ref, tile_rows

    windows = list(tile_windows(int(ref["width"]), int(ref["height"]), ref["transform"], tile_cfg))
    total_tiles = len(windows)
    print(
        f"    [stage1:read] tiles={total_tiles:,}, raster={ref['width']:,}x{ref['height']:,}, "
        f"cell={ref['cell_size']:.3f}m",
        flush=True,
    )

    primary = rasterio.open(primary_path)
    auxiliary = rasterio.open(auxiliary_path) if auxiliary_path and Path(auxiliary_path).exists() else None
    rescued = rasterio.open(rescued_path) if rescued_path and Path(rescued_path).exists() else None
    flat_parts: List[np.ndarray] = []
    flag_parts: List[np.ndarray] = []
    tile_rows: List[Dict[str, Any]] = []
    width = int(ref["width"])
    started = time.perf_counter()
    try:
        with rasterio.Env(GDAL_CACHEMAX=512):
            for number, (tile_id, window, info) in enumerate(windows, 1):
                halo = max(0, int(thin_halo_pixels)) if thin_geometry else 0
                row0 = max(0, int(window.row_off) - halo)
                col0 = max(0, int(window.col_off) - halo)
                row1 = min(int(ref["height"]), int(window.row_off + window.height) + halo)
                col1 = min(int(ref["width"]), int(window.col_off + window.width) + halo)
                expanded = Window(col0, row0, col1-col0, row1-row0)

                p_exp = primary.read(1, window=expanded, masked=False) != 0
                a_exp = auxiliary.read(1, window=expanded, masked=False) != 0 if auxiliary else np.zeros_like(p_exp)
                r_exp = rescued.read(1, window=expanded, masked=False) != 0 if rescued else np.zeros_like(p_exp)
                if geometry_source == "primary":
                    geometry_exp = p_exp
                elif geometry_source == "primary_rescued":
                    geometry_exp = p_exp | r_exp
                else:
                    geometry_exp = p_exp | a_exp | r_exp
                if thin_geometry and np.any(geometry_exp):
                    geometry_exp = thin(geometry_exp)

                cr0 = int(window.row_off) - row0
                cc0 = int(window.col_off) - col0
                cr1 = cr0 + int(window.height)
                cc1 = cc0 + int(window.width)
                geometry = geometry_exp[cr0:cr1, cc0:cc1]
                p = p_exp[cr0:cr1, cc0:cc1]
                a = a_exp[cr0:cr1, cc0:cc1]
                r = r_exp[cr0:cr1, cc0:cc1]

                rr, cc = np.nonzero(geometry)
                if len(rr):
                    global_r = rr.astype(np.int64, copy=False) + int(window.row_off)
                    global_c = cc.astype(np.int64, copy=False) + int(window.col_off)
                    flat_parts.append(global_r * width + global_c)
                    flag_parts.append(
                        p[rr, cc].astype(np.uint8)
                        | (a[rr, cc].astype(np.uint8) << 1)
                        | (r[rr, cc].astype(np.uint8) << 2)
                    )
                tile_rows.append({
                    **info,
                    "primary_pixels": int(np.count_nonzero(p)),
                    "auxiliary_pixels": int(np.count_nonzero(a)),
                    "rescued_pixels": int(np.count_nonzero(r)),
                    "geometry_pixels_before_thin": int(np.count_nonzero(
                        p if geometry_source == "primary" else
                        (p | r) if geometry_source == "primary_rescued" else
                        (p | a | r)
                    )),
                    "active_pixels": int(len(rr)),
                    "geometry_source": geometry_source,
                })
                if number == 1 or number == total_tiles or number % max(1, progress_every_tiles) == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"    [stage1:read] {number:,}/{total_tiles:,} tiles, "
                        f"collected={sum(len(x) for x in flat_parts):,}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )
    finally:
        primary.close()
        if auxiliary is not None:
            auxiliary.close()
        if rescued is not None:
            rescued.close()

    if not flat_parts:
        raise ValueError("输入骨架没有非零像元")
    print("    [stage1:sort] concatenate and sort sparse pixels...", flush=True)
    active = np.concatenate(flat_parts).astype(np.int64, copy=False)
    flags = np.concatenate(flag_parts).astype(np.uint8, copy=False)
    del flat_parts, flag_parts
    order = np.argsort(active, kind="quicksort")
    active = active[order]
    flags = flags[order]
    del order
    # Tile windows are disjoint; duplicates should not occur. Keep a defensive path.
    duplicate = np.flatnonzero(np.diff(active) == 0)
    if len(duplicate):
        starts = np.r_[0, np.flatnonzero(np.diff(active) != 0) + 1]
        active_unique = active[starts]
        flags_unique = np.bitwise_or.reduceat(flags, starts)
        active, flags = active_unique, flags_unique
    print(
        f"    [stage1:sort] active_pixels={len(active):,}, elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )

    if cache_active is not None and cache_meta is not None and cache_tiles is not None and cache_flags is not None:
        np.save(cache_active, active, allow_pickle=False)
        np.save(cache_flags, flags, allow_pickle=False)
        pd.DataFrame(tile_rows).to_csv(cache_tiles, index=False, encoding="utf-8-sig")
        cache_meta.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
        print("    [stage1:cache] sparse-pixel checkpoint saved", flush=True)
    return active, flags, ref, tile_rows


def _search_hits(
    sorted_flat: np.ndarray,
    source_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    dr: int,
    dc: int,
    width: int,
    height: int,
    chunk_pixels: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    delta = int(dr) * int(width) + int(dc)
    for start in range(0, len(source_indices), chunk_pixels):
        src = source_indices[start:start + chunk_pixels]
        rr = rows[src]
        cc = cols[src]
        valid = (rr + dr >= 0) & (rr + dr < height) & (cc + dc >= 0) & (cc + dc < width)
        if not np.any(valid):
            continue
        src_valid = src[valid]
        targets = sorted_flat[src_valid] + delta
        pos = np.searchsorted(sorted_flat, targets)
        hit = pos < len(sorted_flat)
        if np.any(hit):
            safe = np.minimum(pos, len(sorted_flat) - 1)
            hit &= sorted_flat[safe] == targets
        if np.any(hit):
            src_parts.append(src_valid[hit].astype(np.int64, copy=False))
            dst_parts.append(pos[hit].astype(np.int64, copy=False))
    return src_parts, dst_parts



def _contains_sorted(sorted_flat: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return vectorized membership in a sorted int64 array."""
    targets = np.asarray(targets, dtype=np.int64)
    pos = np.searchsorted(sorted_flat, targets)
    hit = pos < len(sorted_flat)
    if np.any(hit):
        safe = np.minimum(pos, len(sorted_flat) - 1)
        hit &= sorted_flat[safe] == targets
    return hit


def _search_hits_no_corner_cut(
    sorted_flat: np.ndarray,
    source_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    dr: int,
    dc: int,
    width: int,
    height: int,
    chunk_pixels: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """8-neighbour hits with diagonal corner shortcuts removed.

    A diagonal link is retained only when neither orthogonal bridge pixel exists.
    This preserves a true one-pixel diagonal road, but prevents staircase pixels and
    2x2 junction blocks from becoming artificial high-degree blobs.
    """
    src_parts, dst_parts = _search_hits(
        sorted_flat, source_indices, rows, cols, dr, dc, width, height, chunk_pixels
    )
    if abs(int(dr)) != 1 or abs(int(dc)) != 1 or not src_parts:
        return src_parts, dst_parts
    out_src: List[np.ndarray] = []
    out_dst: List[np.ndarray] = []
    vertical_delta = int(dr) * int(width)
    horizontal_delta = int(dc)
    for src, dst in zip(src_parts, dst_parts):
        base = sorted_flat[src]
        bridge_vertical = _contains_sorted(sorted_flat, base + vertical_delta)
        bridge_horizontal = _contains_sorted(sorted_flat, base + horizontal_delta)
        keep = ~(bridge_vertical | bridge_horizontal)
        if np.any(keep):
            out_src.append(src[keep].astype(np.int64, copy=False))
            out_dst.append(dst[keep].astype(np.int64, copy=False))
    return out_src, out_dst


def build_sparse_neighbour_table(
    active_flat: np.ndarray,
    width: int,
    height: int,
    chunk_pixels: int = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized pixel graph using no-corner-cut 8-connectivity.

    Cardinal neighbours are always retained. A diagonal neighbour is retained only
    when neither orthogonal bridge pixel exists. The rule removes raster staircase
    shortcuts that previously converted long roads into giant junction components.
    """
    n = len(active_flat)
    rows = (active_flat // int(width)).astype(np.int32, copy=False)
    cols = (active_flat - rows.astype(np.int64) * int(width)).astype(np.int32, copy=False)
    degree = np.zeros(n, dtype=np.uint8)
    nbr1 = np.full(n, -1, dtype=np.int32)
    nbr2 = np.full(n, -1, dtype=np.int32)
    all_idx = np.arange(n, dtype=np.int64)
    started = time.perf_counter()
    for number, (dr, dc) in enumerate(OFFSETS_8, 1):
        src_parts, dst_parts = _search_hits_no_corner_cut(
            active_flat, all_idx, rows, cols, dr, dc, width, height, chunk_pixels
        )
        if src_parts:
            src = np.concatenate(src_parts)
            dst = np.concatenate(dst_parts).astype(np.int32, copy=False)
            slots = degree[src]
            first = slots == 0
            second = slots == 1
            nbr1[src[first]] = dst[first]
            nbr2[src[second]] = dst[second]
            degree[src] = slots + 1
            hits = len(src)
        else:
            hits = 0
        print(
            f"    [stage1:adjacency] direction={number}/8, retained_links={hits:,}, "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return rows, cols, degree, nbr1, nbr2



def cluster_node_pixels(
    active_flat: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    degree: np.ndarray,
    width: int,
    height: int,
    transform,
    chunk_pixels: int,
    max_blob_pixels: int = 64,
    max_blob_span_cells: int = 8,
) -> Tuple[
    List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Create pixel-faithful graph nodes without centroid chords.

    Small connected non-degree-2 pixel blobs are represented by a skeleton-pixel
    medoid. A parent tree inside each blob preserves the exact raster path between
    the medoid and every exit. Oversized blobs are split into singleton nodes rather
    than collapsed to one centroid, preventing star/ray artefacts.
    """
    node_active_idx = np.flatnonzero(degree != 2).astype(np.int64, copy=False)
    m = len(node_active_idx)
    node_id_by_active = np.full(len(active_flat), -1, dtype=np.int32)
    parent_active = np.full(len(active_flat), -1, dtype=np.int64)
    if m == 0:
        print("    [stage1:nodes] no non-degree2 pixels; all components are loops", flush=True)
        return (
            [], node_id_by_active, node_active_idx, parent_active,
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        )

    node_flats = active_flat[node_active_idx]
    local_idx = np.arange(m, dtype=np.int64)
    pair_a: List[np.ndarray] = []
    pair_b: List[np.ndarray] = []
    started = time.perf_counter()
    node_rows = rows[node_active_idx]
    node_cols = cols[node_active_idx]

    # Build adjacency between node-candidate pixels using exactly the same
    # no-corner-cut rule as the complete skeleton graph.
    for dr, dc in OFFSETS_HALF:
        delta = int(dr) * int(width) + int(dc)
        for start in range(0, m, chunk_pixels):
            local = local_idx[start:start + chunk_pixels]
            rr = node_rows[local]
            cc = node_cols[local]
            valid = (rr + dr >= 0) & (rr + dr < height) & (cc + dc >= 0) & (cc + dc < width)
            if not np.any(valid):
                continue
            src = local[valid]
            targets = node_flats[src] + delta
            pos = np.searchsorted(node_flats, targets)
            hit = pos < m
            if np.any(hit):
                safe = np.minimum(pos, m - 1)
                hit &= node_flats[safe] == targets
            if not np.any(hit):
                continue
            src_hit = src[hit]
            dst_hit = pos[hit].astype(np.int64, copy=False)
            if abs(int(dr)) == 1 and abs(int(dc)) == 1:
                active_src = node_active_idx[src_hit]
                base = active_flat[active_src]
                bridge_vertical = _contains_sorted(active_flat, base + int(dr) * int(width))
                bridge_horizontal = _contains_sorted(active_flat, base + int(dc))
                keep = ~(bridge_vertical | bridge_horizontal)
                src_hit = src_hit[keep]
                dst_hit = dst_hit[keep]
            if len(src_hit):
                pair_a.append(src_hit.astype(np.int32, copy=False))
                pair_b.append(dst_hit.astype(np.int32, copy=False))

    if pair_a:
        pa = np.concatenate(pair_a).astype(np.int32, copy=False)
        pb = np.concatenate(pair_b).astype(np.int32, copy=False)
        graph = coo_matrix((np.ones(len(pa), dtype=np.uint8), (pa, pb)), shape=(m, m)).tocsr()
        raw_count, raw_labels = connected_components(graph, directed=False, return_labels=True)
        del graph
    else:
        pa = np.empty(0, dtype=np.int32)
        pb = np.empty(0, dtype=np.int32)
        raw_count = m
        raw_labels = np.arange(m, dtype=np.int32)

    # Split oversized candidate components instead of collapsing them to one node.
    order = np.argsort(raw_labels, kind="stable")
    sorted_labels = raw_labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels) != 0) + 1]
    stops = np.r_[starts[1:], len(order)]
    final_labels = np.full(m, -1, dtype=np.int32)
    final_members: List[np.ndarray] = []
    split_components = 0
    for start, stop in zip(starts, stops):
        members = order[start:stop]
        rr = node_rows[members]
        cc = node_cols[members]
        row_span = int(rr.max() - rr.min() + 1)
        col_span = int(cc.max() - cc.min() + 1)
        compact = (
            len(members) <= max(1, int(max_blob_pixels))
            and row_span <= max(1, int(max_blob_span_cells))
            and col_span <= max(1, int(max_blob_span_cells))
        )
        if compact:
            nid = len(final_members)
            final_labels[members] = nid
            final_members.append(members.astype(np.int64, copy=False))
        else:
            split_components += 1
            for member in members:
                nid = len(final_members)
                final_labels[int(member)] = nid
                final_members.append(np.asarray([member], dtype=np.int64))

    cluster_count = len(final_members)
    node_id_by_active[node_active_idx] = final_labels
    xs, ys = _pixel_xy_from_flat(node_flats, width, transform)

    # Local adjacency of candidate pixels, used to build a skeleton-constrained
    # medoid-to-exit tree for each compact blob.
    adjacency: Dict[int, List[int]] = {}
    if len(pa):
        for a, b in zip(pa.tolist(), pb.tolist()):
            if final_labels[a] != final_labels[b]:
                continue
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))

    nodes: List[Dict[str, Any]] = []
    anchors = np.full(cluster_count, -1, dtype=np.int64)
    for nid, members in enumerate(final_members):
        mx = float(np.mean(xs[members]))
        my = float(np.mean(ys[members]))
        dist2 = (xs[members] - mx) ** 2 + (ys[members] - my) ** 2
        best_value = float(np.min(dist2))
        tied = members[np.flatnonzero(np.isclose(dist2, best_value))]
        if len(tied) > 1:
            tied_degree = degree[node_active_idx[tied]]
            anchor_local = int(tied[int(np.argmax(tied_degree))])
        else:
            anchor_local = int(tied[0])
        anchor_active = int(node_active_idx[anchor_local])
        anchors[nid] = anchor_active
        parent_active[anchor_active] = anchor_active

        # BFS tree entirely inside the node-candidate blob.
        queue = [anchor_local]
        seen = {anchor_local}
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for nxt in adjacency.get(current, ()):  # singleton blobs have no entry
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
                parent_active[int(node_active_idx[nxt])] = int(node_active_idx[current])
        # A compact component should be connected; defensive singleton fallback
        # keeps all paths on actual skeleton pixels even if malformed input occurs.
        for local in members:
            active_idx = int(node_active_idx[int(local)])
            if parent_active[active_idx] < 0:
                parent_active[active_idx] = active_idx

        x = float(xs[anchor_local])
        y = float(ys[anchor_local])
        rr = node_rows[members]
        cc = node_cols[members]
        nodes.append({
            "node_id": int(nid),
            "x": x,
            "y": y,
            "pixel_count": int(len(members)),
            "raw_max_pixel_degree": int(np.max(degree[node_active_idx[members]])),
            "blob_row_span_cells": int(rr.max() - rr.min() + 1),
            "blob_col_span_cells": int(cc.max() - cc.min() + 1),
            "anchor_flat": int(active_flat[anchor_active]),
            "geometry": Point(x, y),
            "uturn_protected": False,
            "activity_zone": False,
        })

    # Direct retained links between different node clusters. These are essential
    # when an oversized junction component has been split into singleton nodes.
    if len(pa):
        cross = final_labels[pa] != final_labels[pb]
        link_a = node_active_idx[pa[cross]].astype(np.int64, copy=False)
        link_b = node_active_idx[pb[cross]].astype(np.int64, copy=False)
    else:
        link_a = np.empty(0, dtype=np.int64)
        link_b = np.empty(0, dtype=np.int64)

    print(
        f"    [stage1:nodes] node_pixels={m:,}, raw_components={raw_count:,}, "
        f"clusters={cluster_count:,}, oversized_split={split_components:,}, "
        f"direct_node_links={len(link_a):,}, elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )
    return nodes, node_id_by_active, node_active_idx, parent_active, anchors, link_a, link_b



def find_node_exits(
    active_flat: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    degree: np.ndarray,
    node_id_by_active: np.ndarray,
    node_active_idx: np.ndarray,
    width: int,
    height: int,
    chunk_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster_parts: List[np.ndarray] = []
    prev_parts: List[np.ndarray] = []
    curr_parts: List[np.ndarray] = []
    for dr, dc in OFFSETS_8:
        src_parts, dst_parts = _search_hits_no_corner_cut(
            active_flat, node_active_idx, rows, cols, dr, dc, width, height, chunk_pixels
        )
        for src, dst in zip(src_parts, dst_parts):
            keep = degree[dst] == 2
            if np.any(keep):
                src_keep = src[keep]
                dst_keep = dst[keep]
                cluster_parts.append(node_id_by_active[src_keep].astype(np.int64, copy=False))
                prev_parts.append(src_keep.astype(np.int64, copy=False))
                curr_parts.append(dst_keep.astype(np.int64, copy=False))
    if not cluster_parts:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    clusters = np.concatenate(cluster_parts)
    prev = np.concatenate(prev_parts)
    curr = np.concatenate(curr_parts)
    order = np.lexsort((curr, clusters))
    clusters, prev, curr = clusters[order], prev[order], curr[order]
    keep = np.ones(len(curr), dtype=bool)
    keep[1:] = (clusters[1:] != clusters[:-1]) | (curr[1:] != curr[:-1])
    return clusters[keep].astype(np.int32), prev[keep], curr[keep]


def sample_active_rasters(
    active_flat: np.ndarray,
    width: int,
    paths: Dict[str, Optional[Path]],
    work_dir: Optional[Path],
    resume: bool,
    chunk_pixels: int,
) -> Dict[str, np.ndarray]:
    valid_paths = {k: Path(v) if v is not None and Path(v).exists() else None for k, v in paths.items()}
    signature = _path_signature(
        valid_paths.values(),
        {"active_pixels": int(len(active_flat)), "active_first": int(active_flat[0]), "active_last": int(active_flat[-1])},
    )
    cache_meta = None
    cache_files: Dict[str, Path] = {}
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_meta = work_dir / "evidence_cache_signature.json"
        cache_files = {name: work_dir / f"active_{name}.npy" for name in valid_paths}
        if (
            resume and cache_meta.exists() and _cache_matches(cache_meta, signature)
            and all(cache_files[name].exists() for name in valid_paths)
        ):
            print("    [stage1:evidence] reuse active-pixel evidence checkpoint", flush=True)
            return {name: np.load(path, mmap_mode="r") for name, path in cache_files.items()}

    result = {name: np.zeros(len(active_flat), dtype=np.float32) for name in valid_paths}
    rows_all = active_flat // int(width)
    cols_all = active_flat - rows_all * int(width)
    started = time.perf_counter()
    with RasterBlockSampler(valid_paths, cache_blocks=256) as sampler:
        total_chunks = max(1, math.ceil(len(active_flat) / chunk_pixels))
        for chunk_no, start in enumerate(range(0, len(active_flat), chunk_pixels), 1):
            stop = min(len(active_flat), start + chunk_pixels)
            rows = rows_all[start:stop]
            cols = cols_all[start:stop]
            for name in valid_paths:
                if valid_paths[name] is not None:
                    result[name][start:stop] = sampler.values_rc(name, rows, cols, default=0.0).astype(np.float32)
            print(
                f"    [stage1:evidence] chunk={chunk_no}/{total_chunks}, pixels={stop:,}/{len(active_flat):,}, "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    if cache_meta is not None:
        for name, path in cache_files.items():
            np.save(path, result[name], allow_pickle=False)
        cache_meta.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
        print("    [stage1:cache] active-pixel evidence checkpoint saved", flush=True)
    return result


def _edge_source_and_evidence(
    path_idx: np.ndarray,
    source_flags: np.ndarray,
    evidence: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    flags = np.asarray(source_flags[path_idx], dtype=np.uint8)
    primary_fraction = float(np.mean((flags & 1) > 0))
    auxiliary_fraction = float(np.mean((flags & 2) > 0))
    rescued_fraction = float(np.mean((flags & 4) > 0))
    if rescued_fraction >= max(primary_fraction, auxiliary_fraction) and rescued_fraction > 0:
        source_type = "support_rescued"
    elif primary_fraction >= auxiliary_fraction:
        source_type = "primary"
    else:
        source_type = "auxiliary"
    if sum(value > 0.15 for value in (primary_fraction, auxiliary_fraction, rescued_fraction)) > 1:
        source_type = "mixed"

    support = np.asarray(evidence.get("support", np.zeros(0))[path_idx], dtype=float) if "support" in evidence else np.zeros(len(path_idx))
    kde = np.asarray(evidence.get("kde", np.zeros(0))[path_idx], dtype=float) if "kde" in evidence else np.zeros(len(path_idx))
    distance = np.asarray(evidence.get("distance", np.zeros(0))[path_idx], dtype=float) if "distance" in evidence else np.zeros(len(path_idx))
    return {
        "source_type": source_type,
        "primary_fraction": primary_fraction,
        "auxiliary_fraction": auxiliary_fraction,
        "rescued_fraction": rescued_fraction,
        "support_mean": float(np.mean(support)) if len(support) else 0.0,
        "support_median": float(np.median(support)) if len(support) else 0.0,
        "support_max": float(np.max(support)) if len(support) else 0.0,
        "support_ge2_fraction": float(np.mean(support >= 2)) if len(support) else 0.0,
        "kde_mean": float(np.mean(kde)) if len(kde) else 0.0,
        "kde_median": float(np.median(kde)) if len(kde) else 0.0,
        "kde_max": float(np.max(kde)) if len(kde) else 0.0,
        "distance_mean_m": float(np.mean(distance)) if len(distance) else 0.0,
    }


def trace_graph_fast(
    active_flat: np.ndarray,
    source_flags: np.ndarray,
    evidence: Dict[str, np.ndarray],
    ref: Dict[str, Any],
    chunk_pixels: int = 1_000_000,
    progress_every_edges: int = 10_000,
    max_node_blob_pixels: int = 64,
    max_node_blob_span_cells: int = 8,
) -> NetworkData:
    width, height = int(ref["width"]), int(ref["height"])
    transform, crs = ref["transform"], ref["crs"]
    started = time.perf_counter()
    rows, cols, degree, nbr1, nbr2 = build_sparse_neighbour_table(
        active_flat, width, height, chunk_pixels=chunk_pixels
    )
    (
        nodes, node_id_by_active, node_active_idx, parent_active, anchors,
        direct_link_a, direct_link_b,
    ) = cluster_node_pixels(
        active_flat, rows, cols, degree, width, height, transform, chunk_pixels,
        max_blob_pixels=max_node_blob_pixels,
        max_blob_span_cells=max_node_blob_span_cells,
    )
    clusters, exit_prev, exit_curr = find_node_exits(
        active_flat, rows, cols, degree, node_id_by_active, node_active_idx,
        width, height, chunk_pixels,
    )
    print(
        f"    [stage1:trace] exits={len(exit_curr):,}, direct_node_links={len(direct_link_a):,}",
        flush=True,
    )

    visited_degree2 = np.zeros(len(active_flat), dtype=bool)
    edges: List[Dict[str, Any]] = []

    def anchor_to_member(member_active: int, node_id: int) -> List[int]:
        anchor = int(anchors[int(node_id)])
        member_active = int(member_active)
        if member_active == anchor:
            return [anchor]
        reverse_path = [member_active]
        current = member_active
        guard = 0
        while current != anchor:
            guard += 1
            if guard > max(2, int(nodes[int(node_id)].get("pixel_count", 1)) + 2):
                raise RuntimeError("节点内部骨架路径追踪异常")
            parent = int(parent_active[current])
            if parent < 0 or parent == current:
                raise RuntimeError(
                    f"节点 {node_id} 内部像元 {current} 无法沿骨架到达锚点；已拒绝生成直线弦"
                )
            reverse_path.append(parent)
            current = parent
        return list(reversed(reverse_path))

    def emit(path: Sequence[int], u: int, v: int, relation: str = "degree2_chain") -> None:
        if not path:
            return
        compact_path = [int(path[0])]
        for value in path[1:]:
            value = int(value)
            if value != compact_path[-1]:
                compact_path.append(value)
        idx = np.asarray(compact_path, dtype=np.int64)
        geom = _build_line(
            idx, active_flat, width, transform,
            (float(nodes[u]["x"]), float(nodes[u]["y"])),
            (float(nodes[v]["x"]), float(nodes[v]["y"])),
        )
        if geom is None:
            return
        # Hard invariant: every original stage-1 vertex must be an active skeleton
        # pixel and consecutive vertices must be 8-neighbour pixels. Any violation
        # would recreate the ray artefact, so fail loudly rather than write bad data.
        flats = active_flat[idx]
        rr = flats // width
        cc = flats - rr * width
        if len(idx) > 1:
            max_step = int(max(np.max(np.abs(np.diff(rr))), np.max(np.abs(np.diff(cc)))))
        else:
            max_step = 0
        if max_step > 1:
            raise RuntimeError(
                f"检测到脱离骨架的长弦：u={u}, v={v}, max_pixel_step={max_step}"
            )
        attrs = _edge_source_and_evidence(idx, source_flags, evidence)
        edges.append({
            "edge_id": len(edges),
            "u": int(u),
            "v": int(v),
            "length_m": float(geom.length),
            "pixel_count": int(len(idx)),
            **attrs,
            "geometry": geom,
            "uturn_protected": False,
            "activity_zone": False,
            "gap_repair": False,
            "geometry_origin": "s5_pixel_path_exact",
            "max_geometry_shift_m": 0.0,
            "baseline_length_m": float(geom.length),
            "stage1_relation": relation,
            "max_pixel_step": max_step,
        })

    # Retained direct adjacency between separate node clusters.
    for a_raw, b_raw in zip(direct_link_a, direct_link_b):
        a, b = int(a_raw), int(b_raw)
        u, v = int(node_id_by_active[a]), int(node_id_by_active[b])
        if u < 0 or v < 0 or u == v:
            continue
        left = anchor_to_member(a, u)
        right_anchor_to_member = anchor_to_member(b, v)
        path = left + [b] + list(reversed(right_anchor_to_member))[1:]
        emit(path, u, v, relation="direct_node_link")

    for exit_no, (u_raw, prev_raw, curr_raw) in enumerate(zip(clusters, exit_prev, exit_curr), 1):
        u = int(u_raw)
        prev, curr = int(prev_raw), int(curr_raw)
        if visited_degree2[curr]:
            continue
        degree2_path = [curr]
        visited_degree2[curr] = True
        target = -1
        target_member = -1
        valid_chain = True
        guard = 0
        while True:
            guard += 1
            if guard > len(active_flat) + 10:
                raise RuntimeError("骨架链追踪出现异常循环")
            a, b = int(nbr1[curr]), int(nbr2[curr])
            nxt = b if a == prev else a
            if nxt < 0:
                valid_chain = False
                break
            degree2_path.append(nxt)
            node_target = int(node_id_by_active[nxt])
            if node_target >= 0:
                target = node_target
                target_member = nxt
                break
            if visited_degree2[nxt]:
                valid_chain = False
                break
            visited_degree2[nxt] = True
            prev, curr = curr, nxt
        if valid_chain and target >= 0:
            start_internal = anchor_to_member(int(prev_raw), u)
            end_internal = anchor_to_member(target_member, target)
            full_path = start_internal + degree2_path + list(reversed(end_internal))[1:]
            emit(full_path, u, target, relation="degree2_chain")
        if exit_no == 1 or exit_no == len(exit_curr) or exit_no % max(1, progress_every_edges) == 0:
            print(
                f"    [stage1:trace] exits={exit_no:,}/{len(exit_curr):,}, edges={len(edges):,}, "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    remaining = np.flatnonzero((degree == 2) & (~visited_degree2))
    loop_count = 0
    for start_raw in remaining:
        start = int(start_raw)
        if visited_degree2[start]:
            continue
        xs, ys = _pixel_xy_from_flat(np.asarray([active_flat[start]], dtype=np.int64), width, transform)
        anchor = len(nodes)
        nodes.append({
            "node_id": anchor,
            "x": float(xs[0]),
            "y": float(ys[0]),
            "pixel_count": 1,
            "raw_max_pixel_degree": 2,
            "blob_row_span_cells": 1,
            "blob_col_span_cells": 1,
            "anchor_flat": int(active_flat[start]),
            "geometry": Point(float(xs[0]), float(ys[0])),
            "uturn_protected": False,
            "activity_zone": False,
        })
        path = [start]
        visited_degree2[start] = True
        prev = start
        curr = int(nbr1[start])
        closed = False
        while curr >= 0:
            if curr == start:
                path.append(start)
                closed = True
                break
            if int(node_id_by_active[curr]) >= 0 or visited_degree2[curr]:
                break
            path.append(curr)
            visited_degree2[curr] = True
            a, b = int(nbr1[curr]), int(nbr2[curr])
            nxt = b if a == prev else a
            prev, curr = curr, nxt
        if closed and len(path) >= 4:
            emit(path, anchor, anchor, relation="closed_loop")
            loop_count += 1

    # Orientation-independent exact duplicate removal.
    unique: set[Tuple[int, int, bytes]] = set()
    deduped: List[Dict[str, Any]] = []
    for edge in edges:
        coords = list(edge["geometry"].coords)
        forward = edge["geometry"].wkb
        reverse = LineString(list(reversed(coords))).wkb
        key = (
            min(int(edge["u"]), int(edge["v"])),
            max(int(edge["u"]), int(edge["v"])),
            min(forward, reverse),
        )
        if key in unique:
            continue
        unique.add(key)
        edge["edge_id"] = len(deduped)
        deduped.append(edge)

    network = NetworkData(nodes, deduped, crs, transform, width, height)
    compute_degrees(network.nodes, network.edges)
    result = reindex_network(network)
    ray_edges = sum(int(edge.get("max_pixel_step", 0)) > 1 for edge in result.edges)
    if ray_edges:
        raise RuntimeError(f"第1阶段像元忠实性检查失败：ray_edges={ray_edges}")
    print(
        f"    [stage1:trace] done nodes={len(result.nodes):,}, edges={len(result.edges):,}, "
        f"loops={loop_count:,}, ray_edges=0, elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )
    return result



def _read_union_geometry(path: Optional[Path], layers: Sequence[str], crs, buffer_if_line: float = 0.0):
    if path is None or not Path(path).exists():
        return None
    for layer in layers:
        try:
            zones = gpd.read_file(path, layer=layer)
            if not len(zones):
                continue
            if zones.crs != crs:
                zones = zones.to_crs(crs)
            if buffer_if_line > 0 and all(g in ("LineString", "MultiLineString") for g in zones.geom_type.unique()):
                zones = zones.copy()
                zones["geometry"] = zones.geometry.buffer(buffer_if_line)
            return zones.geometry.union_all()
        except Exception:
            continue
    return None


def _mark_intersections(records: List[Dict[str, Any]], zone, field: str, chunk_size: int = 25_000) -> None:
    if zone is None:
        for record in records:
            record[field] = False
        return
    for start in range(0, len(records), chunk_size):
        stop = min(len(records), start + chunk_size)
        geoms = np.asarray([record["geometry"] for record in records[start:stop]], dtype=object)
        try:
            mask = np.asarray(shapely_intersects(geoms, zone), dtype=bool)
        except Exception:
            mask = np.asarray([zone.intersects(geom) for geom in geoms], dtype=bool)
        for record, value in zip(records[start:stop], mask):
            record[field] = bool(value)


def apply_vector_protection_fast(
    network: NetworkData,
    uturn_gpkg: Optional[Path],
    activity_gpkg: Optional[Path],
    activity_layer: str = "activity_suppression_zones",
    uturn_core_buffer_m: float = 4.0,
    uturn_context_buffer_m: float = 10.0,
) -> None:
    """Apply two-level U-turn protection.

    ``uturn_core_protected`` is a narrow corridor around the consensus centerline and
    is the only U-turn flag that blocks digital-loop collapse and aggressive geometry
    simplification. ``uturn_context_protected`` is the wider context corridor used by
    pruning and evidence cleanup.  ``uturn_protected`` is retained as a backwards
    compatible alias of the context flag.
    """
    started = time.perf_counter()
    core = _read_union_geometry(
        uturn_gpkg,
        ("uturn_consensus_lines",),
        network.crs,
        buffer_if_line=max(0.0, float(uturn_core_buffer_m)),
    )
    context = _read_union_geometry(
        uturn_gpkg,
        ("uturn_consensus_lines", "uturn_protection_zones"),
        network.crs,
        buffer_if_line=max(0.0, float(uturn_context_buffer_m)),
    )
    # Defensive fallback for legacy GPKGs that only contain the old protection-zone layer.
    if core is None:
        core = _read_union_geometry(
            uturn_gpkg, ("uturn_protection_zones",), network.crs
        )
    if context is None:
        context = core

    activity = _read_union_geometry(activity_gpkg, (activity_layer,), network.crs)
    _mark_intersections(network.nodes, core, "uturn_core_protected")
    _mark_intersections(network.edges, core, "uturn_core_protected")
    _mark_intersections(network.nodes, context, "uturn_context_protected")
    _mark_intersections(network.edges, context, "uturn_context_protected")
    for record in network.nodes:
        record["uturn_protected"] = bool(record.get("uturn_context_protected"))
    for record in network.edges:
        record["uturn_protected"] = bool(record.get("uturn_context_protected"))
    _mark_intersections(network.nodes, activity, "activity_zone")
    _mark_intersections(network.edges, activity, "activity_zone")
    print(
        f"    [stage1:zones] U-turn core={float(uturn_core_buffer_m):g} m, "
        f"context={float(uturn_context_buffer_m):g} m; elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )


def build_network_from_rasters(
    primary_path: Path,
    auxiliary_path: Optional[Path],
    rescued_path: Optional[Path],
    support_path: Path,
    kde_path: Path,
    distance_path: Optional[Path],
    tile_cfg: TileConfig,
    uturn_gpkg: Optional[Path] = None,
    activity_gpkg: Optional[Path] = None,
    activity_layer: str = "activity_suppression_zones",
    uturn_core_buffer_m: float = 4.0,
    uturn_context_buffer_m: float = 10.0,
    work_dir: Optional[Path] = None,
    resume: bool = False,
    progress_every_tiles: int = 100,
    chunk_pixels: int = 1_000_000,
    progress_every_edges: int = 10_000,
    max_node_blob_pixels: int = 64,
    max_node_blob_span_cells: int = 8,
    geometry_source: str = "primary",
    thin_geometry: bool = True,
    thin_halo_pixels: int = 2,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    active, flags, ref, tile_rows = collect_skeleton_pixels_fast(
        primary_path, auxiliary_path, rescued_path, tile_cfg,
        work_dir=work_dir, resume=resume,
        progress_every_tiles=progress_every_tiles,
        geometry_source=geometry_source,
        thin_geometry=thin_geometry,
        thin_halo_pixels=thin_halo_pixels,
    )
    evidence = sample_active_rasters(
        active, int(ref["width"]),
        {"support": support_path, "kde": kde_path, "distance": distance_path},
        work_dir=work_dir, resume=resume, chunk_pixels=chunk_pixels,
    )
    network = trace_graph_fast(
        active, flags, evidence, ref,
        chunk_pixels=chunk_pixels,
        progress_every_edges=progress_every_edges,
        max_node_blob_pixels=max_node_blob_pixels,
        max_node_blob_span_cells=max_node_blob_span_cells,
    )
    apply_vector_protection_fast(
        network, uturn_gpkg, activity_gpkg, activity_layer,
        uturn_core_buffer_m=uturn_core_buffer_m,
        uturn_context_buffer_m=uturn_context_buffer_m,
    )
    compute_degrees(network.nodes, network.edges)
    network.metadata.update({
        "active_pixels": int(len(active)),
        "tile_size_m": float(tile_cfg.tile_size_m),
        "tile_origin_x": tile_cfg.origin_x,
        "tile_origin_y": tile_cfg.origin_y,
        "stage1_algorithm": "strict_primary_thinned_pixel_graph_v4_6",
        "geometry_source": str(geometry_source),
        "thin_geometry": bool(thin_geometry),
        "thin_halo_pixels": int(thin_halo_pixels),
        "max_node_blob_pixels": int(max_node_blob_pixels),
        "max_node_blob_span_cells": int(max_node_blob_span_cells),
    })
    return network, tile_rows
