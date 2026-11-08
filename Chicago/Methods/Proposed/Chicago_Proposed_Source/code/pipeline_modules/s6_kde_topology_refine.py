from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, convolve, label, maximum_filter
from skimage.morphology import thin


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
]


def _inside(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(pixel: Tuple[int, int], mask: np.ndarray) -> List[Tuple[int, int]]:
    r, c = pixel
    out = []
    for dr, dc in NEIGHBORS_8:
        rr, cc = r + dr, c + dc
        if not (_inside(mask.shape, rr, cc) and mask[rr, cc]):
            continue
        # Avoid counting a diagonal shortcut when an orthogonal skeleton step exists.
        # This keeps raster staircases as degree-2 chains instead of false junctions.
        if dr != 0 and dc != 0:
            if mask[r + dr, c] or mask[r, c + dc]:
                continue
        out.append((rr, cc))
    return out


def _edge_key(a: Tuple[int, int], b: Tuple[int, int]):
    return (a, b) if a <= b else (b, a)


def trace_degree2_chains(mask: np.ndarray) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """Return open degree-2 chains and pure cycles from an 8-connected 1-pixel skeleton.

    Junctions/endpoints are held fixed by the ridge refinement.  Pure cycles are
    deliberately not moved here; S7 is responsible for cycle removal.
    """
    pix: Set[Tuple[int, int]] = set(map(tuple, np.argwhere(mask.astype(bool))))
    degree = {p: len([q for q in _neighbors(p, mask) if q in pix]) for p in pix}
    nodes = {p for p, d in degree.items() if d != 2}
    visited = set()
    chains: List[List[Tuple[int, int]]] = []

    for start in sorted(nodes):
        for nb in sorted(_neighbors(start, mask)):
            ek = _edge_key(start, nb)
            if ek in visited:
                continue
            visited.add(ek)
            path = [start, nb]
            prev, cur = start, nb
            guard = 0
            while cur not in nodes and guard < len(pix) + 10:
                guard += 1
                nbs = [q for q in _neighbors(cur, mask) if q != prev]
                if not nbs:
                    break
                # In a clean skeleton a degree-2 pixel has exactly one forward option.
                nxt = nbs[0]
                ek = _edge_key(cur, nxt)
                if ek in visited:
                    break
                visited.add(ek)
                path.append(nxt)
                prev, cur = cur, nxt
            if len(path) >= 2:
                chains.append(path)

    cycles: List[List[Tuple[int, int]]] = []
    for p in sorted(pix):
        for nb in sorted(_neighbors(p, mask)):
            ek = _edge_key(p, nb)
            if ek in visited:
                continue
            visited.add(ek)
            path = [p, nb]
            prev, cur = p, nb
            guard = 0
            while guard < len(pix) + 10:
                guard += 1
                nbs = [q for q in _neighbors(cur, mask) if q != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                if nxt == path[0]:
                    path.append(nxt)
                    break
                ek = _edge_key(cur, nxt)
                if ek in visited:
                    break
                visited.add(ek)
                path.append(nxt)
                prev, cur = cur, nxt
            cycles.append(path)
    return chains, cycles


def _bresenham(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    r0, c0 = a; r1, c1 = b
    x0, y0, x1, y1 = c0, r0, c1, r1
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    out = []
    while True:
        out.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return out


def _chain_tangent(path: Sequence[Tuple[int, int]], i: int) -> np.ndarray:
    # x=column, y=row in pixel space; only direction matters.
    if i <= 0:
        a = np.asarray(path[0], float); b = np.asarray(path[min(2, len(path)-1)], float)
    elif i >= len(path)-1:
        a = np.asarray(path[max(0, len(path)-3)], float); b = np.asarray(path[-1], float)
    else:
        a = np.asarray(path[max(0, i-2)], float); b = np.asarray(path[min(len(path)-1, i+2)], float)
    v = b - a  # row, col
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.asarray([0.0, 1.0])


def refine_primary_to_kde_ridge(
    skeleton: np.ndarray,
    scaled_kde: np.ndarray,
    support: np.ndarray,
    distance: np.ndarray,
    cell_size_m: float,
    search_radius_m: float = 4.0,
    min_chain_m: float = 16.0,
    min_local_ridge_ratio: float = 0.20,
    support_weight: float = 0.25,
    distance_weight: float = 0.25,
    offset_penalty: float = 0.08,
    transition_penalty: float = 0.16,
    min_mean_kde_gain: float = 0.03,
    max_distance_m: float = 6.0,
    exclusion_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Move long degree-2 primary chains toward a *local* KDE ridge.

    The skeleton topology is used only as a scaffold. Junction/endpoints remain fixed;
    pure cycles are left untouched for S7. Candidate positions are searched along the
    local normal. A Viterbi-like first-order DP favors KDE ridge pixels while penalizing
    lateral jumps. No road is created outside the current chain corridor.
    """
    sk = thin(skeleton.astype(bool)).astype(bool)
    if not sk.any() or search_radius_m <= 0:
        return sk, np.zeros_like(sk), []

    cell = max(float(cell_size_m), 1e-9)
    radius_px = max(1, int(math.ceil(float(search_radius_m) / cell)))
    min_chain_px = max(4, int(math.ceil(float(min_chain_m) / cell)))
    kde = scaled_kde.astype(float, copy=False)
    local_max = maximum_filter(kde, size=2 * radius_px + 1, mode="nearest")
    ratio_map = np.divide(kde, np.maximum(local_max, 1.0), out=np.zeros_like(kde, float), where=local_max > 0)
    sup_scale = np.clip(support.astype(float) / 3.0, 0.0, 1.0)
    dist_score = np.exp(-np.maximum(distance.astype(float), 0.0) / 4.0)
    exclusion = np.zeros_like(sk) if exclusion_mask is None else exclusion_mask.astype(bool)

    chains, _ = trace_degree2_chains(sk)
    result = sk.copy()
    moved_mask = np.zeros_like(sk)
    rows: List[Dict[str, object]] = []

    for chain_id, path in enumerate(chains, start=1):
        if len(path) < min_chain_px:
            continue
        arr = np.asarray(path, dtype=int)
        if np.any(exclusion[arr[:, 0], arr[:, 1]]):
            continue

        offsets = np.arange(-radius_px, radius_px + 1, dtype=int)
        candidates: List[List[Tuple[int, int, int, float]]] = []
        original_kde = []
        for i, (r, c) in enumerate(path):
            original_kde.append(float(kde[r, c]))
            tangent = _chain_tangent(path, i)
            normal = np.asarray([-tangent[1], tangent[0]], float)  # row,col normal
            opts = []
            fixed = i < 2 or i >= len(path)-2
            for off in offsets:
                if fixed and off != 0:
                    continue
                rr = int(round(r + normal[0] * off))
                cc = int(round(c + normal[1] * off))
                if not _inside(sk.shape, rr, cc):
                    continue
                if exclusion[rr, cc] or not np.isfinite(distance[rr, cc]):
                    continue
                if distance[rr, cc] > float(max_distance_m) or kde[rr, cc] <= 0:
                    continue
                ridge = float(ratio_map[rr, cc])
                if ridge < float(min_local_ridge_ratio) and off != 0:
                    continue
                unary = (
                    1.00 * ridge
                    + float(support_weight) * float(sup_scale[rr, cc])
                    + float(distance_weight) * float(dist_score[rr, cc])
                    - float(offset_penalty) * abs(off) / max(radius_px, 1)
                )
                opts.append((rr, cc, int(off), float(unary)))
            if not opts:
                opts = [(int(r), int(c), 0, float(ratio_map[r, c]))]
            candidates.append(opts)

        # First-order DP. This is enough to prevent per-pixel zig-zagging while still
        # allowing a road to shift gradually from one side of the KDE band to the ridge.
        back: List[np.ndarray] = []
        dp = np.asarray([o[3] for o in candidates[0]], dtype=float)
        back.append(np.full(len(candidates[0]), -1, dtype=int))
        for i in range(1, len(candidates)):
            prev = candidates[i-1]
            cur = candidates[i]
            ndp = np.full(len(cur), -1e30, dtype=float)
            bidx = np.full(len(cur), -1, dtype=int)
            for j, opt in enumerate(cur):
                off = opt[2]
                vals = np.asarray([
                    dp[k] - float(transition_penalty) * abs(off - prev[k][2])
                    for k in range(len(prev))
                ])
                k = int(np.argmax(vals))
                ndp[j] = float(vals[k] + opt[3])
                bidx[j] = k
            dp = ndp
            back.append(bidx)

        idx = int(np.argmax(dp))
        chosen = [None] * len(path)
        for i in range(len(path)-1, -1, -1):
            chosen[i] = candidates[i][idx]
            if i > 0:
                idx = int(back[i][idx])
        chosen_pixels = [(int(o[0]), int(o[1])) for o in chosen]
        chosen_kde = np.asarray([kde[r, c] for r, c in chosen_pixels], float)
        old = np.asarray(original_kde, float)
        old_mean = float(np.mean(old)) if len(old) else 0.0
        new_mean = float(np.mean(chosen_kde)) if len(chosen_kde) else 0.0
        gain = (new_mean - old_mean) / max(old_mean, 1.0)
        max_off_m = max(abs(int(o[2])) for o in chosen) * cell
        if gain < float(min_mean_kde_gain) or max_off_m <= 1e-9:
            continue

        # Replace only this chain interior. Endpoints are fixed and therefore keep all
        # junction connectivity. Rasterization uses Bresenham between optimized points.
        new_chain = np.zeros_like(sk)
        for a, b in zip(chosen_pixels[:-1], chosen_pixels[1:]):
            for rr, cc in _bresenham(a, b):
                if _inside(sk.shape, rr, cc):
                    new_chain[rr, cc] = True
        result[arr[1:-1, 0], arr[1:-1, 1]] = False
        result |= new_chain
        moved_mask |= new_chain
        rows.append({
            "chain_id": chain_id,
            "pixels": len(path),
            "length_m_approx": float((len(path)-1) * cell),
            "old_mean_scaled_kde": old_mean,
            "new_mean_scaled_kde": new_mean,
            "relative_kde_gain": float(gain),
            "max_lateral_shift_m": float(max_off_m),
        })

    result = thin(result).astype(bool)
    return result, moved_mask, rows


def local_ridge_ratio(scaled_kde: np.ndarray, radius_px: int) -> np.ndarray:
    radius_px = max(1, int(radius_px))
    mx = maximum_filter(scaled_kde.astype(float), size=2*radius_px+1, mode="nearest")
    return np.divide(
        scaled_kde.astype(float), np.maximum(mx, 1.0),
        out=np.zeros_like(mx, dtype=float), where=mx > 0,
    )


def filter_rescue_by_ridge_and_attachments(
    rescue: np.ndarray,
    existing_skeleton: np.ndarray,
    scaled_kde: np.ndarray,
    support: np.ndarray,
    cell_size_m: float,
    min_ridge_ratio: float = 0.45,
    ridge_radius_m: float = 6.0,
    max_primary_attachments: int = 1,
    bridge_min_support: int = 3,
    bridge_min_support_fraction: float = 0.50,
    bridge_min_ridge_ratio: float = 0.65,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Evidence-first support rescue.

    v0.5 distinguishes two very different multi-attachment cases:
    (1) a rescue corridor joins *different* primary connected components -- this
        reduces fragmentation and is preferred when repeated support exists;
    (2) it touches the same already-connected component more than once -- this can
        close a false bubble and therefore still needs strong ridge/support evidence.
    """
    cell = max(float(cell_size_m), 1e-9)
    ridge_radius_px = max(1, int(math.ceil(float(ridge_radius_m) / cell)))
    ratio = local_ridge_ratio(scaled_kde, ridge_radius_px)
    existing = existing_skeleton.astype(bool)
    existing_halo = binary_dilation(existing, iterations=1)
    novel = rescue.astype(bool) & ~existing_halo
    novel &= (ratio >= float(min_ridge_ratio)) | (support >= max(4, int(bridge_min_support)+1))
    novel = thin(novel).astype(bool)

    labs, count = label(novel, structure=np.ones((3,3), dtype=np.uint8))
    base_labs, _ = label(existing, structure=np.ones((3,3), dtype=np.uint8))
    kept = np.zeros_like(novel)
    removed_multi = kept_bridge = isolated = single_attach = distinct_bridge = 0
    for cid in range(1, count+1):
        comp = labs == cid
        if not comp.any():
            continue
        ring = binary_dilation(comp, iterations=2) & existing
        touched_labels = np.unique(base_labs[ring])
        touched_labels = touched_labels[touched_labels > 0]
        # Contact groups are useful when all contacts land on one base component.
        _, contact_groups = label(ring, structure=np.ones((3,3), dtype=np.uint8))
        attachments = int(contact_groups)
        distinct_components = int(len(touched_labels))
        sup_fraction = float(np.mean(support[comp] >= int(bridge_min_support)))
        med_ridge = float(np.median(ratio[comp])) if comp.any() else 0.0

        if attachments == 0:
            kept |= comp; isolated += 1; continue
        if attachments <= int(max_primary_attachments):
            kept |= comp; single_attach += 1; continue
        # The key v0.5 rule: a repeated-support corridor that joins two separate
        # primary components is a continuity repair, not a cycle risk.
        if distinct_components >= 2 and (
            sup_fraction >= min(0.35, float(bridge_min_support_fraction))
            or med_ridge >= min(0.45, float(bridge_min_ridge_ratio))
        ):
            kept |= comp; distinct_bridge += 1; continue
        if sup_fraction >= float(bridge_min_support_fraction) and med_ridge >= float(bridge_min_ridge_ratio):
            kept |= comp; kept_bridge += 1
        else:
            removed_multi += 1

    return thin(kept).astype(bool), {
        "novel_rescue_pixels_before_filter": int((rescue.astype(bool) & ~existing_halo).sum()),
        "novel_rescue_pixels_after_filter": int(kept.sum()),
        "rescue_components": int(count),
        "rescue_isolated_components_kept": int(isolated),
        "rescue_single_attachment_components_kept": int(single_attach),
        "rescue_distinct_component_bridges_kept": int(distinct_bridge),
        "rescue_strong_same_component_multi_attachment_kept": int(kept_bridge),
        "rescue_multi_attachment_components_removed": int(removed_multi),
    }


def filter_additions_connecting_distinct_components(
    additions: np.ndarray,
    base_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Keep only additions that connect two *different* base components.

    Used for route-endpoint connectors. If both ends touch the same already-connected
    base component, the connector would close a cycle and is rejected at S6 instead of
    leaving the problem for S7.
    """
    add = additions.astype(bool) & ~base_mask.astype(bool)
    labs, count = label(add, structure=np.ones((3,3), dtype=np.uint8))
    base_labs, _ = label(base_mask.astype(bool), structure=np.ones((3,3), dtype=np.uint8))
    kept = np.zeros_like(add)
    accepted = rejected_same = rejected_other = 0
    for cid in range(1, count+1):
        comp = labs == cid
        ring = binary_dilation(comp, iterations=1) & (base_labs > 0)
        touched = np.unique(base_labs[ring])
        touched = touched[touched > 0]
        if len(touched) == 2:
            kept |= comp; accepted += 1
        elif len(touched) == 1:
            rejected_same += 1
        else:
            rejected_other += 1
    return thin(kept).astype(bool), {
        "connector_components": int(count),
        "connector_components_accepted": int(accepted),
        "connector_same_component_cycle_risk_rejected": int(rejected_same),
        "connector_ambiguous_rejected": int(rejected_other),
    }


def filter_auxiliary_leaf_only(
    auxiliary: np.ndarray,
    primary: np.ndarray,
    max_primary_attachments: int = 1,
    max_primary_contact_fraction: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Keep single-track auxiliary roads as leaves/isolated paths by default.

    This preserves genuine low-density hiking paths while preventing single-track
    geometry from becoming an alternate route between two primary locations.
    """
    aux = thin(auxiliary.astype(bool) & ~primary.astype(bool)).astype(bool)
    labs, count = label(aux, structure=np.ones((3,3), dtype=np.uint8))
    kept = np.zeros_like(aux)
    primary_labs, _ = label(primary.astype(bool), structure=np.ones((3,3), dtype=np.uint8))
    removed = 0; isolated = 0; leaves = 0; hugging_removed = 0
    for cid in range(1, count+1):
        comp = labs == cid
        ring = binary_dilation(comp, iterations=1) & (primary_labs > 0)
        contact_labels, contact_groups = label(ring, structure=np.ones((3,3), dtype=np.uint8))
        attachments = int(contact_groups)
        contact_fraction = float(ring.sum()) / max(float(comp.sum()), 1.0)
        # A single-track line that hugs primary for a substantial fraction is usually a
        # duplicate side of the same road, not a genuine leaf branch.
        if contact_fraction > float(max_primary_contact_fraction) and comp.sum() >= 6:
            hugging_removed += 1
            continue
        if attachments == 0:
            kept |= comp; isolated += 1
        elif attachments <= int(max_primary_attachments):
            kept |= comp; leaves += 1
        else:
            removed += 1
    return thin(kept).astype(bool), {
        "auxiliary_components": int(count),
        "auxiliary_isolated_kept": int(isolated),
        "auxiliary_leaf_components_kept": int(leaves),
        "auxiliary_multi_primary_components_removed": int(removed),
        "auxiliary_primary_hugging_components_removed": int(hugging_removed),
    }
