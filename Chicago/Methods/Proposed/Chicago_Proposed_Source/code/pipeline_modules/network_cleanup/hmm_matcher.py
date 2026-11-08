from __future__ import annotations

import csv
import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from .models import NetworkData, component_ids, make_nx_graph


def _axis_angle_difference_deg(a: float, b: float) -> float:
    """Undirected angle difference in [0, 90] degrees."""
    d = abs(float(a) - float(b)) % math.pi
    d = min(d, math.pi - d)
    return float(math.degrees(d))


def _edge_tangent_angle(geom, along: float, delta: float = 2.0) -> float:
    length = float(geom.length)
    if length <= 1e-9:
        return 0.0
    a = max(0.0, min(length, along - delta))
    b = max(0.0, min(length, along + delta))
    if b - a < 1e-6:
        a, b = 0.0, length
    p0 = geom.interpolate(a)
    p1 = geom.interpolate(b)
    return float(math.atan2(p1.y - p0.y, p1.x - p0.x) % math.pi)


def _query_indices(tree: STRtree, geoms: Sequence[Any], envelope) -> List[int]:
    result = tree.query(envelope)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(v) for v in result]
    lookup = {id(g): i for i, g in enumerate(geoms)}
    return [lookup[id(g)] for g in result if id(g) in lookup]


def _bool_values(series: pd.Series, default: bool = True) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).to_numpy(dtype=bool)
    values = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


@dataclass(frozen=True)
class Candidate:
    edge_index: int
    edge_id: int
    along_m: float
    distance_m: float
    projected_x: float
    projected_y: float
    tangent_angle_rad: float
    emission_cost: float


class TransitionModel:
    """Network-aware transition distance with bounded caching.

    Modes:
    - euclidean: projection-to-projection straight distance.
    - hybrid: exact for same/adjacent/local edges, Euclidean approximation farther away.
    - exact: A* shortest path between edge endpoints.
    """

    def __init__(
        self,
        network: NetworkData,
        mode: str = "hybrid",
        exact_radius_m: float = 180.0,
        cache_size: int = 200_000,
    ) -> None:
        self.network = network
        self.mode = str(mode)
        self.exact_radius_m = float(exact_radius_m)
        self.graph = make_nx_graph(network.nodes, network.edges)
        self.nodes = {int(n["node_id"]): (float(n["x"]), float(n["y"])) for n in network.nodes}
        self.edges = {int(e["edge_id"]): e for e in network.edges}
        self.components = component_ids(network.nodes, network.edges)
        self.cache_size = max(1_000, int(cache_size))
        self.node_cache: OrderedDict[Tuple[int, int], float] = OrderedDict()

    def _heuristic(self, a: int, b: int) -> float:
        ax, ay = self.nodes.get(int(a), (0.0, 0.0))
        bx, by = self.nodes.get(int(b), (0.0, 0.0))
        return float(math.hypot(ax - bx, ay - by))

    def _node_distance(self, u: int, v: int, allow_exact: bool) -> float:
        u, v = int(u), int(v)
        if u == v:
            return 0.0
        key = (min(u, v), max(u, v))
        if key in self.node_cache:
            value = self.node_cache.pop(key)
            self.node_cache[key] = value
            return value
        euclidean = self._heuristic(u, v)
        if self.components.get(u) != self.components.get(v):
            value = float("inf")
        elif not allow_exact:
            value = euclidean * 1.12
        else:
            try:
                value = float(nx.astar_path_length(
                    self.graph,
                    u,
                    v,
                    heuristic=lambda a, b: self._heuristic(int(a), int(b)),
                    weight="weight",
                ))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                value = float("inf")
        self.node_cache[key] = value
        while len(self.node_cache) > self.cache_size:
            self.node_cache.popitem(last=False)
        return value

    def distance(self, previous: Candidate, current: Candidate) -> float:
        if self.mode == "euclidean":
            return float(math.hypot(
                current.projected_x - previous.projected_x,
                current.projected_y - previous.projected_y,
            ))

        e0 = self.edges[previous.edge_id]
        e1 = self.edges[current.edge_id]
        l0 = float(e0["length_m"])
        l1 = float(e1["length_m"])

        if previous.edge_id == current.edge_id:
            return abs(float(current.along_m) - float(previous.along_m))

        endpoint0 = (
            (int(e0["u"]), float(previous.along_m)),
            (int(e0["v"]), max(0.0, l0 - float(previous.along_m))),
        )
        endpoint1 = (
            (int(e1["u"]), float(current.along_m)),
            (int(e1["v"]), max(0.0, l1 - float(current.along_m))),
        )

        best = float("inf")
        for n0, d0 in endpoint0:
            for n1, d1 in endpoint1:
                straight = self._heuristic(n0, n1)
                exact = self.mode == "exact" or straight <= self.exact_radius_m
                middle = self._node_distance(n0, n1, allow_exact=exact)
                best = min(best, d0 + middle + d1)
        return best


def _read_schema(points_csv: Path) -> Dict[str, Optional[str]]:
    columns = set(pd.read_csv(points_csv, nrows=0).columns)
    x = "gravity_x_m" if "gravity_x_m" in columns else "x_m"
    y = "gravity_y_m" if "gravity_y_m" in columns else "y_m"
    if x not in columns or y not in columns:
        raise ValueError('Track points are missing gravity_x_m/y_m or x_m/y_m')
    clean = "clean_track_id" if "clean_track_id" in columns else "track_id"
    raw = "track_id" if "track_id" in columns else clean
    order = "clean_point_index" if "clean_point_index" in columns else (
        "point_index" if "point_index" in columns else None
    )
    source = "source_file" if "source_file" in columns else None
    time = "time" if "time" in columns else ("timestamp" if "timestamp" in columns else None)
    return {
        "x": x,
        "y": y,
        "clean": clean,
        "raw": raw,
        "order": order,
        "source": source,
        "time": time,
        "weight": "road_extraction_weight" if "road_extraction_weight" in columns else None,
        "candidate": "mainroad_candidate" if "mainroad_candidate" in columns else None,
    }


def iter_tracks(
    points_csv: Path,
    chunksize: int = 250_000,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Stream contiguous tracks from CSV while preserving tracks split across chunks."""
    schema = _read_schema(points_csv)
    usecols = [v for v in schema.values() if v is not None]
    usecols = list(dict.fromkeys(usecols))
    carry: Optional[pd.DataFrame] = None
    closed_tracks: set[str] = set()

    for chunk in pd.read_csv(points_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        if carry is not None and not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True, sort=False)
            carry = None
        clean_col = str(schema["clean"])
        if chunk.empty:
            continue
        ids = chunk[clean_col].astype(str)
        last_id = str(ids.iloc[-1])
        last_mask = ids == last_id
        complete = chunk.loc[~last_mask]
        carry = chunk.loc[last_mask].copy()
        for track_id, frame in complete.groupby(clean_col, sort=False):
            tid = str(track_id)
            if tid in closed_tracks:
                raise ValueError(
                    'Input CSV contains noncontiguous rows for the same clean_track_id; sort by clean_track_id and clean_point_index first'
                )
            closed_tracks.add(tid)
            yield tid, frame.copy()

    if carry is not None and not carry.empty:
        clean_col = str(schema["clean"])
        for track_id, frame in carry.groupby(clean_col, sort=False):
            yield str(track_id), frame.copy()


def _prepare_track(
    frame: pd.DataFrame,
    schema: Dict[str, Optional[str]],
    sample_step: int,
    min_road_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(frame[str(schema["x"])], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[str(schema["y"])], errors="coerce").to_numpy(float)
    if schema["order"]:
        order = pd.to_numeric(frame[str(schema["order"])], errors="coerce").to_numpy(float)
    else:
        order = np.arange(len(frame), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(order)
    if schema["candidate"]:
        valid &= _bool_values(frame[str(schema["candidate"])], default=True)
    if schema["weight"]:
        weight = pd.to_numeric(frame[str(schema["weight"])], errors="coerce").fillna(0).to_numpy(float)
        valid &= weight >= float(min_road_weight)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=int)
    indices = indices[np.argsort(order[indices], kind="stable")]
    step = max(1, int(sample_step))
    sampled = indices[::step]
    if indices[-1] != sampled[-1]:
        sampled = np.r_[sampled, indices[-1]]
    xs, ys = x[sampled], y[sampled]
    headings = np.full(len(sampled), np.nan, dtype=float)
    if len(sampled) >= 2:
        dx, dy = np.gradient(xs), np.gradient(ys)
        moving = np.hypot(dx, dy) > 1e-6
        headings[moving] = np.mod(np.arctan2(dy[moving], dx[moving]), math.pi)
    return xs, ys, headings, sampled


def _candidates_for_observation(
    x: float,
    y: float,
    heading: float,
    network: NetworkData,
    geoms: Sequence[Any],
    tree: STRtree,
    max_distance_m: float,
    max_candidates: int,
    emission_sigma_m: float,
    heading_sigma_deg: float,
    heading_weight: float,
) -> List[Candidate]:
    point = Point(float(x), float(y))
    envelope = box(
        x - max_distance_m,
        y - max_distance_m,
        x + max_distance_m,
        y + max_distance_m,
    )
    indices = _query_indices(tree, geoms, envelope)
    candidates: List[Candidate] = []
    for edge_index in indices:
        geom = geoms[edge_index]
        distance = float(point.distance(geom))
        if distance > max_distance_m:
            continue
        along = float(geom.project(point))
        projected = geom.interpolate(along)
        tangent = _edge_tangent_angle(geom, along)
        angle = 0.0 if not np.isfinite(heading) else _axis_angle_difference_deg(heading, tangent)
        emission = 0.5 * (distance / max(emission_sigma_m, 1e-6)) ** 2
        emission += float(heading_weight) * 0.5 * (angle / max(heading_sigma_deg, 1e-6)) ** 2
        source = str(network.edges[edge_index].get("source_type", "primary"))
        source_penalty = {"primary": 0.0, "mixed": 0.0, "support_rescued": -0.08, "gap_repair": 0.05, "auxiliary": 0.18}.get(source, 0.08)
        emission += source_penalty
        candidates.append(Candidate(
            edge_index=int(edge_index),
            edge_id=int(network.edges[edge_index]["edge_id"]),
            along_m=along,
            distance_m=distance,
            projected_x=float(projected.x),
            projected_y=float(projected.y),
            tangent_angle_rad=tangent,
            emission_cost=float(emission),
        ))
    candidates.sort(key=lambda c: (c.emission_cost, c.distance_m, c.edge_id))
    return candidates[: max(1, int(max_candidates))]


def _viterbi_segment(
    observations: np.ndarray,
    candidates: Sequence[Sequence[Candidate]],
    transition_model: TransitionModel,
    transition_beta_m: float,
    transition_jump_penalty: float,
) -> List[Candidate]:
    if not candidates:
        return []
    costs = np.asarray([c.emission_cost for c in candidates[0]], dtype=float)
    back: List[np.ndarray] = []

    for t in range(1, len(candidates)):
        previous_candidates = candidates[t - 1]
        current_candidates = candidates[t]
        observed_distance = float(np.linalg.norm(observations[t] - observations[t - 1]))
        current_costs = np.full(len(current_candidates), float("inf"), dtype=float)
        pointers = np.full(len(current_candidates), -1, dtype=np.int64)
        for j, current in enumerate(current_candidates):
            for i, previous in enumerate(previous_candidates):
                network_distance = transition_model.distance(previous, current)
                if not np.isfinite(network_distance):
                    transition = transition_jump_penalty + observed_distance / max(transition_beta_m, 1e-6)
                else:
                    transition = abs(network_distance - observed_distance) / max(transition_beta_m, 1e-6)
                    if observed_distance > 1.0 and network_distance > max(100.0, observed_distance * 5.0):
                        transition += transition_jump_penalty
                value = costs[i] + transition + current.emission_cost
                if value < current_costs[j]:
                    current_costs[j] = value
                    pointers[j] = i
        if not np.isfinite(current_costs).any():
            current_costs = np.asarray([c.emission_cost for c in current_candidates], dtype=float)
            pointers[:] = int(np.argmin(costs)) if len(costs) else -1
        back.append(pointers)
        costs = current_costs

    state = int(np.argmin(costs))
    sequence = [candidates[-1][state]]
    for t in range(len(candidates) - 2, -1, -1):
        state = int(back[t][state])
        if state < 0:
            state = 0
        sequence.append(candidates[t][state])
    sequence.reverse()
    return sequence


class _PointWriter:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None
        self.rows: List[Dict[str, Any]] = []
        self.header_written = False
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()

    def append(self, row: Dict[str, Any]) -> None:
        if self.path is None:
            return
        self.rows.append(row)
        if len(self.rows) >= 50_000:
            self.flush()

    def flush(self) -> None:
        if self.path is None or not self.rows:
            return
        pd.DataFrame(self.rows).to_csv(
            self.path,
            mode="a",
            header=not self.header_written,
            index=False,
            encoding="utf-8-sig" if not self.header_written else "utf-8",
        )
        self.header_written = True
        self.rows.clear()

    def close(self) -> None:
        self.flush()


def match_tracks_hmm(
    network: NetworkData,
    points_csv: Optional[Path],
    max_distance_m: float = 50.0,
    sample_step: int = 5,
    max_candidates: int = 5,
    emission_sigma_m: float = 18.0,
    heading_sigma_deg: float = 35.0,
    heading_weight: float = 0.35,
    transition_beta_m: float = 25.0,
    transition_jump_penalty: float = 5.0,
    transition_mode: str = "hybrid",
    transition_exact_radius_m: float = 180.0,
    min_road_weight: float = 0.30,
    chunksize: int = 250_000,
    write_points_path: Optional[Path] = None,
    progress_every_tracks: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """HMM/Viterbi map matching and per-edge trajectory evidence aggregation."""
    if points_csv is None or not Path(points_csv).exists():
        empty = pd.DataFrame([{
            "edge_id": int(e["edge_id"]),
            "match_points": 0,
            "distinct_tracks": 0,
            "distinct_clean_tracks": 0,
            "mean_match_dist_m": np.nan,
            "p95_match_dist_m": np.nan,
            "coverage_ratio": 0.0,
            "hmm_sequence_count": 0,
        } for e in network.edges])
        return empty, {"enabled": False, "reason": "gravity points not provided"}

    points_csv = Path(points_csv)
    schema = _read_schema(points_csv)
    geoms = [e["geometry"] for e in network.edges]
    if not geoms:
        raise ValueError('Road network has no matchable edges')
    tree = STRtree(geoms)
    transition = TransitionModel(
        network,
        mode=transition_mode,
        exact_radius_m=transition_exact_radius_m,
    )

    accum: Dict[int, Dict[str, Any]] = {
        int(e["edge_id"]): {
            "count": 0,
            "sum_dist": 0.0,
            "distances": [],
            "tracks": set(),
            "clean_tracks": set(),
            "min_along": float("inf"),
            "max_along": float("-inf"),
            "sequences": 0,
        }
        for e in network.edges
    }
    writer = _PointWriter(write_points_path)
    total_tracks = total_observations = total_matched = zero_candidate = 0
    sequence_breaks = 0
    started = time.perf_counter()
    print(
        f"    [hmm] start edges={len(network.edges):,}, sample_step={sample_step}, "
        f"max_candidates={max_candidates}, transition={transition_mode}",
        flush=True,
    )

    for track_id, frame in iter_tracks(points_csv, chunksize=chunksize):
        total_tracks += 1
        xs, ys, headings, original_indices = _prepare_track(
            frame,
            schema,
            sample_step=sample_step,
            min_road_weight=min_road_weight,
        )
        if len(xs) == 0:
            continue
        total_observations += len(xs)
        observation_candidates: List[List[Candidate]] = []
        for x, y, heading in zip(xs, ys, headings):
            cand = _candidates_for_observation(
                float(x), float(y), float(heading), network, geoms, tree,
                max_distance_m=max_distance_m,
                max_candidates=max_candidates,
                emission_sigma_m=emission_sigma_m,
                heading_sigma_deg=heading_sigma_deg,
                heading_weight=heading_weight,
            )
            observation_candidates.append(cand)
            if not cand:
                zero_candidate += 1

        raw_col = str(schema["raw"])
        source_col = schema["source"]
        raw_value = str(frame.iloc[0][raw_col]) if len(frame) else track_id
        contribution = f"{frame.iloc[0][str(source_col)]}||{raw_value}" if source_col else raw_value

        start = 0
        while start < len(xs):
            while start < len(xs) and not observation_candidates[start]:
                start += 1
            if start >= len(xs):
                break
            end = start + 1
            while end < len(xs) and observation_candidates[end]:
                end += 1
            if start > 0 or end < len(xs):
                sequence_breaks += 1
            observations = np.c_[xs[start:end], ys[start:end]]
            sequence = _viterbi_segment(
                observations,
                observation_candidates[start:end],
                transition,
                transition_beta_m=transition_beta_m,
                transition_jump_penalty=transition_jump_penalty,
            )
            visited_edges = set()
            for local_idx, candidate in enumerate(sequence):
                global_idx = start + local_idx
                a = accum[candidate.edge_id]
                a["count"] += 1
                a["sum_dist"] += candidate.distance_m
                if len(a["distances"]) < 10_000:
                    a["distances"].append(candidate.distance_m)
                a["tracks"].add(contribution)
                a["clean_tracks"].add(track_id)
                a["min_along"] = min(a["min_along"], candidate.along_m)
                a["max_along"] = max(a["max_along"], candidate.along_m)
                visited_edges.add(candidate.edge_id)
                total_matched += 1
                writer.append({
                    "clean_track_id": track_id,
                    "track_id": raw_value,
                    "sample_index": int(global_idx),
                    "source_row_index": int(original_indices[global_idx]),
                    "x": float(xs[global_idx]),
                    "y": float(ys[global_idx]),
                    "matched_edge_id": candidate.edge_id,
                    "match_distance_m": candidate.distance_m,
                    "along_m": candidate.along_m,
                    "matched_x": candidate.projected_x,
                    "matched_y": candidate.projected_y,
                    "emission_cost": candidate.emission_cost,
                })
            for edge_id in visited_edges:
                accum[edge_id]["sequences"] += 1
            start = end
        if progress_every_tracks > 0 and total_tracks % int(progress_every_tracks) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"    [hmm] tracks={total_tracks:,}, observations={total_observations:,}, "
                f"matched={total_matched:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    writer.close()
    print(
        f"    [hmm] done tracks={total_tracks:,}, observations={total_observations:,}, "
        f"matched={total_matched:,}, elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    edge_lookup = {int(e["edge_id"]): e for e in network.edges}
    for edge_id in sorted(edge_lookup):
        edge = edge_lookup[edge_id]
        a = accum[edge_id]
        distances = np.asarray(a["distances"], dtype=float)
        coverage = 0.0
        if a["count"] >= 2 and np.isfinite(a["min_along"]) and np.isfinite(a["max_along"]):
            coverage = (a["max_along"] - a["min_along"]) / max(float(edge["length_m"]), 1e-9)
        rows.append({
            "edge_id": edge_id,
            "match_points": int(a["count"]),
            "distinct_tracks": int(len(a["tracks"])),
            "distinct_clean_tracks": int(len(a["clean_tracks"])),
            "mean_match_dist_m": float(a["sum_dist"] / a["count"]) if a["count"] else np.nan,
            "p95_match_dist_m": float(np.percentile(distances, 95)) if len(distances) else np.nan,
            "coverage_ratio": float(np.clip(coverage, 0.0, 1.0)),
            "hmm_sequence_count": int(a["sequences"]),
        })

    report = {
        "enabled": True,
        "method": "HMM_Viterbi",
        "transition_mode": transition_mode,
        "tracks": int(total_tracks),
        "sampled_observations": int(total_observations),
        "matched_observations": int(total_matched),
        "matched_ratio": float(total_matched / max(total_observations, 1)),
        "zero_candidate_observations": int(zero_candidate),
        "sequence_breaks": int(sequence_breaks),
        "sample_step": int(sample_step),
        "max_candidates": int(max_candidates),
        "max_distance_m": float(max_distance_m),
        "emission_sigma_m": float(emission_sigma_m),
        "transition_beta_m": float(transition_beta_m),
    }
    return pd.DataFrame(rows), report
