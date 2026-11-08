from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

from .models import NetworkData, reindex_network
from .topology_utils import cycle_closure_allowed, safe_intersection, weighted_graph
from .junction_cleanup import (
    _angle_deg,
    _continuation_vector,
    _crosses_unmodelled_existing_edge,
)


@dataclass
class RawTrackChunk:
    track_id: str
    contribution_id: str
    source_file: str
    geometry: LineString
    point_count: int


def _query_indices(tree: Optional[STRtree], geoms: Sequence[LineString], query_geom) -> List[int]:
    if tree is None:
        return []
    result = tree.query(query_geom)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in result]
    lookup = {id(g): i for i, g in enumerate(geoms)}
    return [lookup[id(g)] for g in result if id(g) in lookup]


def _safe_times(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def _transition_ok(
    a: np.ndarray,
    b: np.ndarray,
    ta,
    tb,
    max_step_m: float,
    max_speed_mps: float,
    max_time_gap_s: float,
) -> bool:
    distance = float(np.linalg.norm(b - a))
    if not np.isfinite(distance) or distance <= 0.0 or distance > float(max_step_m):
        return False
    if pd.notna(ta) and pd.notna(tb):
        dt = float((tb - ta).total_seconds())
        if not np.isfinite(dt) or dt <= 0.0 or dt > float(max_time_gap_s):
            return False
        if distance / max(dt, 1e-9) > float(max_speed_mps):
            return False
    return True


def _remove_isolated_spikes(
    xy: np.ndarray,
    times: Sequence,
    max_step_m: float,
    max_speed_mps: float,
    max_time_gap_s: float,
    bridge_m: float,
) -> Tuple[np.ndarray, List, int]:
    if len(xy) < 3:
        return xy, list(times), 0
    keep = np.ones(len(xy), dtype=bool)
    removed = 0
    for i in range(1, len(xy) - 1):
        if not keep[i - 1]:
            continue
        bad_left = not _transition_ok(
            xy[i - 1], xy[i], times[i - 1], times[i],
            max_step_m, max_speed_mps, max_time_gap_s,
        )
        bad_right = not _transition_ok(
            xy[i], xy[i + 1], times[i], times[i + 1],
            max_step_m, max_speed_mps, max_time_gap_s,
        )
        if not (bad_left and bad_right):
            continue
        bridge_dist = float(np.linalg.norm(xy[i + 1] - xy[i - 1]))
        if bridge_dist > float(bridge_m):
            continue
        if _transition_ok(
            xy[i - 1], xy[i + 1], times[i - 1], times[i + 1],
            max(max_step_m, bridge_m), max_speed_mps, max_time_gap_s,
        ):
            keep[i] = False
            removed += 1
    return xy[keep], [times[i] for i in range(len(times)) if keep[i]], removed


def _split_motion_sane_chunks(
    xy: np.ndarray,
    times: Sequence,
    max_step_m: float,
    max_speed_mps: float,
    max_time_gap_s: float,
) -> List[np.ndarray]:
    if len(xy) < 2:
        return []
    chunks: List[np.ndarray] = []
    start = 0
    for i in range(1, len(xy)):
        if not _transition_ok(
            xy[i - 1], xy[i], times[i - 1], times[i],
            max_step_m, max_speed_mps, max_time_gap_s,
        ):
            if i - start >= 2:
                chunks.append(xy[start:i].copy())
            start = i
    if len(xy) - start >= 2:
        chunks.append(xy[start:].copy())
    return chunks


def _snap_subline_endpoints(line: LineString, a: Tuple[float, float], b: Tuple[float, float]) -> Optional[LineString]:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    coords[0] = (float(a[0]), float(a[1]))
    coords[-1] = (float(b[0]), float(b[1]))
    clean = [coords[0]]
    for p in coords[1:]:
        q = (float(p[0]), float(p[1]))
        if q != clean[-1]:
            clean.append(q)
    if len(clean) < 2:
        return None
    geom = LineString(clean)
    return geom if geom.length > 0 and geom.is_simple else None


def _sample_p90_distance(source: LineString, target: LineString, n: int = 13) -> float:
    if source.length <= 0:
        return float("inf")
    ds = np.linspace(0.0, float(source.length), max(3, int(n)))
    distances = [float(source.interpolate(float(d)).distance(target)) for d in ds]
    return float(np.percentile(np.asarray(distances, dtype=float), 90))


class RawTrackWitnessIndex:
    """Spatial index of motion-sane S1 original trajectory chunks.

    Original GPS is used only as a *topology witness*.  It is intentionally not a
    replacement geometry source for the whole network.  A connector is witnessed only
    when the same original track chunk passes near both connector ends and the subpath
    between those projections stays close to the proposed connection corridor.
    """

    def __init__(self, chunks: Sequence[RawTrackChunk], crs: Any, report: Optional[Dict[str, Any]] = None):
        self.chunks = list(chunks)
        self.crs = crs
        self.geoms = [c.geometry for c in self.chunks]
        self.tree = STRtree(self.geoms) if self.geoms else None
        self.report = dict(report or {})

    @classmethod
    def disabled(cls, reason: str = "raw tracks unavailable") -> "RawTrackWitnessIndex":
        return cls([], None, {"enabled": False, "reason": reason, "chunks": 0})

    @property
    def enabled(self) -> bool:
        return bool(self.chunks)

    @classmethod
    def from_observed_tracks(
        cls,
        observed_tracks: Optional[Path],
        target_crs: Any,
        pattern: str = "*__gxtrack.csv",
        max_step_m: float = 100.0,
        max_speed_mps: float = 8.0,
        max_time_gap_s: float = 1800.0,
        isolated_spike_bridge_m: float = 80.0,
    ) -> Tuple["RawTrackWitnessIndex", Dict[str, Any]]:
        folder = Path(observed_tracks) if observed_tracks is not None else None
        if folder is None or not folder.is_dir():
            idx = cls.disabled("observed_tracks directory not found")
            return idx, idx.report
        files = sorted(folder.rglob(pattern))
        if not files:
            idx = cls.disabled(f"no raw track files matching {pattern}")
            return idx, idx.report

        try:
            dst = CRS.from_user_input(target_crs)
        except Exception as exc:
            idx = cls.disabled(f"invalid network CRS: {exc}")
            return idx, idx.report
        transformer = Transformer.from_crs("EPSG:4326", dst, always_xy=True)

        chunks: List[RawTrackChunk] = []
        points_seen = points_valid = 0
        invalid_motion_breaks = 0
        spikes_removed = 0
        failed_files = 0
        for file in files:
            try:
                df = pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
            except Exception:
                failed_files += 1
                continue
            points_seen += len(df)
            if not {"longitude", "latitude"}.issubset(df.columns):
                failed_files += 1
                continue
            order_col = "point_index" if "point_index" in df.columns else "raw_point_index" if "raw_point_index" in df.columns else None
            if order_col is not None:
                df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
                df = df.sort_values(order_col, kind="stable")
            lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(float)
            lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(float)
            valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
            if not np.any(valid):
                continue
            df = df.loc[valid].reset_index(drop=True)
            lon = lon[valid]; lat = lat[valid]
            x, y = transformer.transform(lon, lat)
            xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
            finite = np.isfinite(xy).all(axis=1)
            xy = xy[finite]
            df = df.loc[finite].reset_index(drop=True)
            if len(xy) < 2:
                continue
            points_valid += len(xy)
            times = _safe_times(df["time_utc"]) if "time_utc" in df.columns else pd.Series([pd.NaT] * len(df))
            time_list = list(times)
            xy, time_list, removed = _remove_isolated_spikes(
                xy, time_list,
                max_step_m=float(max_step_m),
                max_speed_mps=float(max_speed_mps),
                max_time_gap_s=float(max_time_gap_s),
                bridge_m=float(isolated_spike_bridge_m),
            )
            spikes_removed += int(removed)
            sane_chunks = _split_motion_sane_chunks(
                xy, time_list,
                max_step_m=float(max_step_m),
                max_speed_mps=float(max_speed_mps),
                max_time_gap_s=float(max_time_gap_s),
            )
            invalid_motion_breaks += max(0, len(sane_chunks) - 1)
            raw_track = str(df.iloc[0].get("track_id", file.stem)) if len(df) else file.stem
            source_file = str(df.iloc[0].get("source_file", file.name)) if len(df) else file.name
            contribution = f"{source_file}||{raw_track}"
            for arr in sane_chunks:
                # Remove duplicate consecutive coordinates before Linestring creation.
                clean = [tuple(map(float, arr[0]))]
                for row in arr[1:]:
                    p = tuple(map(float, row))
                    if p != clean[-1]:
                        clean.append(p)
                if len(clean) < 2:
                    continue
                geom = LineString(clean)
                if geom.length <= 0 or not geom.is_simple:
                    # A self-crossing raw GPS chunk is not a safe topology witness.
                    continue
                chunks.append(RawTrackChunk(raw_track, contribution, source_file, geom, len(clean)))

        report = {
            "enabled": bool(chunks),
            "observed_tracks": str(folder),
            "files_seen": int(len(files)),
            "failed_files": int(failed_files),
            "points_seen": int(points_seen),
            "points_valid": int(points_valid),
            "isolated_spikes_removed": int(spikes_removed),
            "motion_sane_chunks": int(len(chunks)),
            "invalid_motion_breaks": int(invalid_motion_breaks),
            "max_step_m": float(max_step_m),
            "max_speed_mps": float(max_speed_mps),
            "max_time_gap_s": float(max_time_gap_s),
            "isolated_spike_bridge_m": float(isolated_spike_bridge_m),
            "target_crs": dst.to_string(),
        }
        if not chunks:
            report["reason"] = "no motion-sane original trajectory chunks"
        return cls(chunks, dst, report), report

    def tracks_near_geometry(self, geometry: LineString, distance_m: float = 6.0) -> Set[str]:
        """Distinct original-track contributions observed near one road geometry."""
        if not self.enabled or not isinstance(geometry, LineString) or geometry.length <= 0:
            return set()
        query = geometry.buffer(max(0.0, float(distance_m)) + 1e-9)
        out: Set[str] = set()
        for i in _query_indices(self.tree, self.geoms, query):
            chunk = self.chunks[int(i)]
            if float(chunk.geometry.distance(geometry)) <= float(distance_m) + 1e-9:
                out.add(chunk.contribution_id)
        return out

    def count_tracks_near_geometry(self, geometry: LineString, distance_m: float = 6.0) -> int:
        return int(len(self.tracks_near_geometry(geometry, distance_m)))

    def route_support(
        self,
        geometry: LineString,
        corridor_m: float = 4.0,
        min_coverage: float = 0.60,
        sample_spacing_m: float = 2.0,
        endpoint_snap_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return original-track contributions that genuinely follow a route.

        Mere proximity is not enough: a contribution must pass near both route gates
        and remain inside the route corridor for a substantial fraction of sampled
        route positions.  This is used for semantic cycle classification, where two
        nearby branches must not both inherit support from the same raw trajectory.
        """
        empty = {
            "support_tracks": 0, "support_chunks": 0, "contribution_ids": set(),
            "contribution_coverage": {}, "contribution_p90_m": {},
            "best_coverage": 0.0, "best_p90_m": float("inf"),
        }
        if not self.enabled or not isinstance(geometry, LineString) or geometry.length <= 0:
            return empty
        corridor = max(0.1, float(corridor_m))
        endpoint_snap = float(endpoint_snap_m) if endpoint_snap_m is not None else max(corridor * 1.5, corridor + 1.0)
        n = max(7, int(math.ceil(float(geometry.length) / max(0.5, float(sample_spacing_m)))) + 1)
        samples = [geometry.interpolate(float(d)) for d in np.linspace(0.0, float(geometry.length), n)]
        a, b = Point(geometry.coords[0]), Point(geometry.coords[-1])
        query = geometry.buffer(endpoint_snap + corridor + 1e-9)
        accepted: List[Tuple[str, float, float]] = []
        for i in _query_indices(self.tree, self.geoms, query):
            chunk = self.chunks[int(i)]
            g = chunk.geometry
            if float(g.distance(a)) > endpoint_snap or float(g.distance(b)) > endpoint_snap:
                continue
            dist = np.asarray([float(g.distance(pt)) for pt in samples], dtype=float)
            coverage = float(np.mean(dist <= corridor + 1e-9))
            if coverage + 1e-12 < float(min_coverage):
                continue
            p90 = float(np.percentile(dist, 90))
            accepted.append((chunk.contribution_id, coverage, p90))
        if not accepted:
            return empty
        contribution_ids = {x[0] for x in accepted}
        coverage_by: Dict[str, float] = {}
        p90_by: Dict[str, float] = {}
        for cid, coverage, p90 in accepted:
            coverage_by[cid] = max(float(coverage), coverage_by.get(cid, 0.0))
            p90_by[cid] = min(float(p90), p90_by.get(cid, float("inf")))
        best = max(accepted, key=lambda x: (x[1], -x[2]))
        return {
            "support_tracks": int(len(contribution_ids)),
            "support_chunks": int(len(accepted)),
            "contribution_ids": contribution_ids,
            "contribution_coverage": coverage_by,
            "contribution_p90_m": p90_by,
            "best_coverage": float(best[1]),
            "best_p90_m": float(min(x[2] for x in accepted)),
        }

    def direct_transition_witness(
        self,
        connector: LineString,
        endpoint_snap_m: float = 4.0,
        max_path_m: float = 40.0,
        max_path_ratio: float = 1.8,
        corridor_m: float = 3.0,
    ) -> Dict[str, Any]:
        """Strict raw witness for a *direct* connector.

        In particular, a raw trajectory that travels around a long U-turn from one
        arm to the other is not evidence for a short cross-bar between those arms.
        The tighter path-ratio/corridor constraints enforce that distinction.
        """
        return self.query_connector(
            connector, endpoint_snap_m=float(endpoint_snap_m),
            max_path_m=float(max_path_m), max_path_ratio=float(max_path_ratio),
            corridor_m=float(corridor_m),
        )

    def query_connector(
        self,
        connector: LineString,
        endpoint_snap_m: float = 6.0,
        max_path_m: float = 60.0,
        max_path_ratio: float = 3.0,
        corridor_m: float = 10.0,
    ) -> Dict[str, Any]:
        empty = {
            "witness_tracks": 0,
            "witness_chunks": 0,
            "best_track_id": "",
            "best_contribution_id": "",
            "best_source_path_m": float("inf"),
            "best_snap_a_m": float("inf"),
            "best_snap_b_m": float("inf"),
            "best_corridor_p90_m": float("inf"),
            "contribution_ids": set(),
            "best_geometry": None,
        }
        if not self.enabled or not isinstance(connector, LineString) or connector.length <= 0:
            return empty
        a = Point(connector.coords[0]); b = Point(connector.coords[-1])
        search = connector.buffer(max(float(endpoint_snap_m), float(corridor_m)) + 1e-6)
        candidates = _query_indices(self.tree, self.geoms, search)
        accepted: List[Tuple[float, RawTrackChunk, Dict[str, Any]]] = []
        for i in candidates:
            chunk = self.chunks[int(i)]
            geom = chunk.geometry
            da, db = float(geom.distance(a)), float(geom.distance(b))
            if da > float(endpoint_snap_m) or db > float(endpoint_snap_m):
                continue
            ma, mb = float(geom.project(a)), float(geom.project(b))
            if abs(mb - ma) <= 1e-6:
                continue
            lo, hi = sorted((ma, mb))
            sub = substring(geom, lo, hi)
            if not isinstance(sub, LineString) or sub.length <= 0:
                continue
            if mb < ma:
                sub = LineString(list(sub.coords)[::-1])
            path_len = float(sub.length)
            euclid = float(a.distance(b))
            if path_len > float(max_path_m):
                continue
            if euclid > 1e-6 and path_len / euclid > float(max_path_ratio):
                continue
            p90 = _sample_p90_distance(sub, connector)
            if p90 > float(corridor_m):
                continue
            snapped = _snap_subline_endpoints(
                sub, (float(a.x), float(a.y)), (float(b.x), float(b.y))
            )
            if snapped is None:
                continue
            score = da + db + 0.5 * p90 + 0.02 * path_len
            accepted.append((score, chunk, {
                "source_path_m": path_len,
                "snap_a_m": da,
                "snap_b_m": db,
                "corridor_p90_m": p90,
                "geometry": snapped,
            }))
        if not accepted:
            return empty
        accepted.sort(key=lambda x: x[0])
        distinct = {c.contribution_id for _, c, _ in accepted}
        _, best_chunk, best = accepted[0]
        return {
            "witness_tracks": int(len(distinct)),
            "witness_chunks": int(len(accepted)),
            "best_track_id": best_chunk.track_id,
            "best_contribution_id": best_chunk.contribution_id,
            "best_source_path_m": float(best["source_path_m"]),
            "best_snap_a_m": float(best["snap_a_m"]),
            "best_snap_b_m": float(best["snap_b_m"]),
            "best_corridor_p90_m": float(best["corridor_p90_m"]),
            "contribution_ids": set(distinct),
            "best_geometry": best["geometry"],
        }

    def query_transition_path(
        self,
        connector: LineString,
        endpoint_snap_m: float = 6.0,
        max_path_m: float = 60.0,
        max_path_ratio: float = 3.0,
    ) -> Dict[str, Any]:
        """Return a motion-sane raw subpath between connector endpoints.

        v0.5 final connectivity rescue deliberately does *not* require the raw
        trajectory to stay close to the straight endpoint chord.  That older
        corridor test rejected real curved missing road segments.  Safety comes
        instead from endpoint snapping, source-path length/path-ratio limits and
        caller-side tangent/topology checks.
        """
        empty = {
            "witness_tracks": 0,
            "witness_chunks": 0,
            "best_track_id": "",
            "best_contribution_id": "",
            "best_source_path_m": float("inf"),
            "best_snap_a_m": float("inf"),
            "best_snap_b_m": float("inf"),
            "best_corridor_p90_m": float("inf"),
            "contribution_ids": set(),
            "best_geometry": None,
        }
        if not self.enabled or not isinstance(connector, LineString) or connector.length <= 0:
            return empty
        a = Point(connector.coords[0]); b = Point(connector.coords[-1])
        # Query by the chord envelope only to obtain candidates; no chord-distance
        # acceptance rule is applied below.  A chunk touching both endpoint
        # neighbourhoods necessarily intersects this envelope spatially.
        search = connector.buffer(max(float(endpoint_snap_m), 0.5) + 1e-6)
        candidates = _query_indices(self.tree, self.geoms, search)
        accepted: List[Tuple[float, RawTrackChunk, Dict[str, Any]]] = []
        for i in candidates:
            chunk = self.chunks[int(i)]
            geom = chunk.geometry
            da, db = float(geom.distance(a)), float(geom.distance(b))
            if da > float(endpoint_snap_m) or db > float(endpoint_snap_m):
                continue
            ma, mb = float(geom.project(a)), float(geom.project(b))
            if abs(mb - ma) <= 1e-6:
                continue
            lo, hi = sorted((ma, mb))
            sub = substring(geom, lo, hi)
            if not isinstance(sub, LineString) or sub.length <= 0:
                continue
            if mb < ma:
                sub = LineString(list(sub.coords)[::-1])
            path_len = float(sub.length)
            euclid = float(a.distance(b))
            if path_len > float(max_path_m):
                continue
            if euclid > 1e-6 and path_len / euclid > float(max_path_ratio):
                continue
            snapped = _snap_subline_endpoints(
                sub, (float(a.x), float(a.y)), (float(b.x), float(b.y))
            )
            if snapped is None:
                continue
            p90 = _sample_p90_distance(sub, connector)
            score = da + db + 0.02 * path_len + 0.01 * p90
            accepted.append((score, chunk, {
                "source_path_m": path_len,
                "snap_a_m": da,
                "snap_b_m": db,
                "corridor_p90_m": p90,
                "geometry": snapped,
            }))
        if not accepted:
            return empty
        accepted.sort(key=lambda x: x[0])
        distinct = {c.contribution_id for _, c, _ in accepted}
        _, best_chunk, best = accepted[0]
        return {
            "witness_tracks": int(len(distinct)),
            "witness_chunks": int(len(accepted)),
            "best_track_id": best_chunk.track_id,
            "best_contribution_id": best_chunk.contribution_id,
            "best_source_path_m": float(best["source_path_m"]),
            "best_snap_a_m": float(best["snap_a_m"]),
            "best_snap_b_m": float(best["snap_b_m"]),
            "best_corridor_p90_m": float(best["corridor_p90_m"]),
            "contribution_ids": set(distinct),
            "best_geometry": best["geometry"],
        }

    def rank_connector_evidence(
        self,
        connector: LineString,
        *,
        endpoint_snap_m: float = 6.0,
        max_path_m: float = 60.0,
        max_path_ratio: float = 3.0,
        corridor_m: float = 10.0,
        side_distance_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Explainable raw evidence used by S7 candidate competition."""
        direct = self.query_connector(
            connector,
            endpoint_snap_m=float(endpoint_snap_m),
            max_path_m=float(max_path_m),
            max_path_ratio=float(max_path_ratio),
            corridor_m=float(corridor_m),
        )
        side_d = float(side_distance_m) if side_distance_m is not None else float(endpoint_snap_m)
        a = Point(connector.coords[0]); b = Point(connector.coords[-1])
        side_a = self.tracks_near_geometry(LineString([(a.x, a.y), (a.x + 1e-6, a.y)]), side_d) if self.enabled else set()
        side_b = self.tracks_near_geometry(LineString([(b.x, b.y), (b.x + 1e-6, b.y)]), side_d) if self.enabled else set()
        return {
            **direct,
            "direct_witness_tracks": int(direct.get("witness_tracks", 0) or 0),
            "distinct_raw_tracks": int(len(set(direct.get("contribution_ids", set())))),
            "side_a_tracks": int(len(side_a)),
            "side_b_tracks": int(len(side_b)),
            "raw_density_support": int(len(side_a) + len(side_b)),
        }


def _endpoint_state(net: NetworkData):
    nodes = {int(n["node_id"]): n for n in net.nodes}
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    graph = nx.Graph()
    for edge in net.edges:
        u, v = int(edge["u"]), int(edge["v"])
        incident[u].append(edge); incident[v].append(edge)
        graph.add_edge(u, v)
    endpoints = [nid for nid, edges in incident.items() if len(edges) == 1]
    return nodes, incident, graph, endpoints


def repair_raw_track_gaps(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    max_endpoint_distance_m: float = 40.0,
    max_track_path_m: float = 65.0,
    endpoint_snap_m: float = 7.0,
    max_angle_deg: float = 65.0,
    min_witness_tracks: int = 1,
    max_path_ratio: float = 3.0,
    corridor_m: float = 10.0,
    allow_cycle_closure: bool = True,
    cycle_min_detour_ratio: float = 1.8,
    cycle_min_detour_excess_m: float = 8.0,
    max_repairs: int = 5000,
    use_raw_subpath_geometry: bool = False,
    repair_mode: str = "raw_track_witness",
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    idx = witness_index
    if idx is None or not idx.enabled or int(max_repairs) <= 0:
        return network.copy(), []
    net = reindex_network(network.copy())
    nodes, incident, graph, endpoints = _endpoint_state(net)
    if len(endpoints) < 2:
        return net, []
    weighted = weighted_graph(net)
    xy = np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in endpoints], dtype=float)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(float(max_endpoint_distance_m), output_type="set")
    edge_geoms = [e["geometry"] for e in net.edges]
    edge_ids = [int(e["edge_id"]) for e in net.edges]
    edge_tree = STRtree(edge_geoms) if edge_geoms else None
    candidates: List[Dict[str, Any]] = []
    for i, j in pairs:
        a, b = int(endpoints[i]), int(endpoints[j])
        pa, pb = xy[i], xy[j]
        euclid = float(np.linalg.norm(pb - pa))
        if euclid <= 0.35:
            continue
        straight = LineString([tuple(pa), tuple(pb)])
        if bool(use_raw_subpath_geometry):
            witness = idx.query_transition_path(
                straight,
                endpoint_snap_m=float(endpoint_snap_m),
                max_path_m=float(max_track_path_m),
                max_path_ratio=float(max_path_ratio),
            )
        else:
            witness = idx.query_connector(
                straight,
                endpoint_snap_m=float(endpoint_snap_m),
                max_path_m=float(max_track_path_m),
                max_path_ratio=float(max_path_ratio),
                corridor_m=float(corridor_m),
            )
        if int(witness["witness_tracks"]) < int(min_witness_tracks):
            continue
        geom = witness.get("best_geometry")
        if not isinstance(geom, LineString) or geom.length <= 0:
            continue
        va = _continuation_vector(incident[a][0], a)
        vb = _continuation_vector(incident[b][0], b)
        coords = np.asarray(geom.coords, dtype=float)
        if va is None or vb is None or len(coords) < 2:
            continue
        ta = coords[min(2, len(coords) - 1)] - coords[0]
        tb = coords[max(0, len(coords) - 3)] - coords[-1]
        aa = _angle_deg(va, ta); ab = _angle_deg(vb, tb)
        if max(aa, ab) > float(max_angle_deg):
            continue
        same_component = nx.has_path(weighted, a, b)
        detour = float("inf")
        if same_component:
            if not bool(allow_cycle_closure):
                continue
            ok, detour = cycle_closure_allowed(
                weighted, a, b, float(geom.length),
                min_detour_ratio=float(cycle_min_detour_ratio),
                min_detour_excess_m=float(cycle_min_detour_excess_m),
            )
            if not ok:
                continue
        ignore = {int(incident[a][0]["edge_id"]), int(incident[b][0]["edge_id"])}
        if _crosses_unmodelled_existing_edge(
            geom, edge_geoms, edge_ids, edge_tree, ignore,
            allowed_touch_points=[Point(tuple(pa)), Point(tuple(pb))],
        ):
            continue
        score = (
            max(aa, ab)
            + 0.15 * euclid
            + float(witness["best_snap_a_m"])
            + float(witness["best_snap_b_m"])
            - 4.0 * min(3, int(witness["witness_tracks"]))
        )
        candidates.append({
            "score": float(score), "a": a, "b": b, "geometry": geom,
            "euclidean_m": euclid, "angle_a_deg": float(aa), "angle_b_deg": float(ab),
            "cycle_closure": bool(same_component), "existing_detour_m": float(detour),
            **{k: v for k, v in witness.items() if k != "best_geometry"},
        })

    candidates.sort(key=lambda r: (float(r["score"]), -int(r["witness_tracks"]), float(r["euclidean_m"])))
    used: Set[int] = set()
    accepted_geoms: List[LineString] = []
    rows: List[Dict[str, Any]] = []
    next_eid = max([int(e["edge_id"]) for e in net.edges] or [-1]) + 1
    for c in candidates:
        if len(rows) >= int(max_repairs):
            break
        a, b = int(c["a"]), int(c["b"])
        if a in used or b in used:
            continue
        if nx.has_path(weighted, a, b):
            if not bool(allow_cycle_closure):
                continue
            ok, detour_now = cycle_closure_allowed(
                weighted, a, b, float(c["geometry"].length),
                min_detour_ratio=float(cycle_min_detour_ratio),
                min_detour_excess_m=float(cycle_min_detour_excess_m),
            )
            if not ok:
                continue
            c["cycle_closure"] = True; c["existing_detour_m"] = float(detour_now)
        # Accepted raw connectors may share modelled endpoints but cannot cross each
        # other in their interiors.
        conflict = False
        for old in accepted_geoms:
            inter = safe_intersection(
                c["geometry"], old, operation="raw_track_connector_intersection",
                context={"endpoint_a": int(a), "endpoint_b": int(b)},
            )
            if inter is None:
                conflict = True; break
            if inter.is_empty:
                continue
            p = inter.centroid
            da = min(float(c["geometry"].project(p)), float(c["geometry"].length - c["geometry"].project(p)))
            db = min(float(old.project(p)), float(old.length - old.project(p)))
            if da > 1e-6 and db > 1e-6:
                conflict = True; break
        if conflict:
            continue
        continuity_class = "raw_subpath_transition" if bool(use_raw_subpath_geometry) else "direct_transition"
        final_rescue = str(repair_mode).startswith("final_connectivity_rescue")
        edge = {
            "edge_id": int(next_eid), "u": a, "v": b, "geometry": c["geometry"],
            "length_m": float(c["geometry"].length), "pixel_count": 0,
            "source_type": "s7_final_connectivity_rescue" if final_rescue else "s7_raw_track_gap_repair",
            "primary_fraction": 0.0, "auxiliary_fraction": 0.0, "rescued_fraction": 0.0,
            "support_mean": 0.0, "support_median": 0.0, "support_max": 0.0,
            "support_ge2_fraction": 0.0, "kde_mean": 0.0, "kde_median": 0.0, "kde_max": 0.0,
            "distance_mean_m": 0.0,
            "gap_repair": True, "approved_gap_repair": True, "raw_track_gap_repair": True,
            "raw_continuity_class": continuity_class,
            "final_connectivity_rescue": bool(final_rescue),
            "raw_witness_tracks": int(c["witness_tracks"]),
            "raw_witness_track_id": str(c["best_track_id"]),
            "raw_witness_source_path_m": float(c["best_source_path_m"]),
            "raw_witness_snap_a_m": float(c["best_snap_a_m"]),
            "raw_witness_snap_b_m": float(c["best_snap_b_m"]),
            "raw_witness_corridor_p90_m": float(c["best_corridor_p90_m"]),
            "raw_gap_euclidean_m": float(c["euclidean_m"]),
            "raw_gap_angle_a_deg": float(c["angle_a_deg"]),
            "raw_gap_angle_b_deg": float(c["angle_b_deg"]),
            "raw_gap_cycle_closure": bool(c.get("cycle_closure", False)),
            "raw_gap_existing_detour_m": float(c.get("existing_detour_m", float("inf"))),
            # Persist the exact admissibility envelope used when this connector was
            # created.  Final v0.5 audit must replay these limits instead of applying
            # the ordinary short-gap limits to longer coverage/final-rescue edges.
            "raw_gap_audit_mode": str(repair_mode),
            "raw_gap_audit_max_endpoint_distance_m": float(max_endpoint_distance_m),
            "raw_gap_audit_max_track_path_m": float(max_track_path_m),
            "raw_gap_audit_max_angle_deg": float(max_angle_deg),
            "raw_gap_audit_min_witness_tracks": int(min_witness_tracks),
            "raw_gap_audit_max_path_ratio": float(max_path_ratio),
            "confidence": min(0.98, 0.90 + 0.02 * max(0, int(c["witness_tracks"]) - 1)),
            "retained_reason": "final_raw_track_continuity_rescue" if final_rescue else "original_track_continuity_witness",
            "geometry_origin": "s7_final_connectivity_rescue_v0_5" if final_rescue else "s7_raw_track_witness_v0_5",
            "max_geometry_shift_m": 0.0,
            "baseline_length_m": float(c["geometry"].length),
        }
        net.edges.append(edge)
        weighted.add_edge(a, b, weight=float(c["geometry"].length))
        used.update((a, b)); accepted_geoms.append(c["geometry"])
        rows.append({
            "edge_id": int(next_eid), "mode": str(repair_mode),
            "raw_continuity_class": continuity_class,
            "final_connectivity_rescue": bool(final_rescue),
            "endpoint_a": a, "endpoint_b": b,
            "raw_witness_tracks": int(c["witness_tracks"]),
            "raw_witness_track_id": str(c["best_track_id"]),
            "raw_witness_source_path_m": float(c["best_source_path_m"]),
            "euclidean_m": float(c["euclidean_m"]),
            "angle_a_deg": float(c["angle_a_deg"]), "angle_b_deg": float(c["angle_b_deg"]),
            "cycle_closure": bool(c.get("cycle_closure", False)),
            "existing_detour_m": float(c.get("existing_detour_m", float("inf"))),
            "audit_mode": str(repair_mode),
            "audit_max_endpoint_distance_m": float(max_endpoint_distance_m),
            "audit_max_track_path_m": float(max_track_path_m),
            "audit_max_angle_deg": float(max_angle_deg),
            "audit_min_witness_tracks": int(min_witness_tracks),
            "geometry": c["geometry"],
        })
        next_eid += 1
    if rows:
        net = reindex_network(net)
    print(f"    [raw-track-gap] candidates={len(candidates):,}, accepted={len(rows):,}", flush=True)
    return net, rows


def repair_high_support_uncovered_corridors(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    max_endpoint_distance_m: float = 90.0,
    max_track_path_m: float = 140.0,
    endpoint_snap_m: float = 8.0,
    max_angle_deg: float = 70.0,
    min_witness_tracks: int = 2,
    max_path_ratio: float = 3.5,
    max_repairs: int = 2000,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Restore high-support raw corridors missed by ordinary short-gap repair.

    v0.5 uses this as a final *coverage completeness* pass.  It deliberately
    permits longer/curved raw subpaths than ordinary endpoint repair, but requires
    repeated independent original-track support and never closes a cycle: only
    different connected components may be merged.
    """
    fixed, rows = repair_raw_track_gaps(
        network, witness_index,
        max_endpoint_distance_m=max(0.0, float(max_endpoint_distance_m)),
        max_track_path_m=max(0.0, float(max_track_path_m)),
        endpoint_snap_m=max(0.0, float(endpoint_snap_m)),
        max_angle_deg=max(1.0, float(max_angle_deg)),
        min_witness_tracks=max(2, int(min_witness_tracks)),
        max_path_ratio=max(1.0, float(max_path_ratio)),
        corridor_m=max(2.0, float(endpoint_snap_m)),
        allow_cycle_closure=False,
        max_repairs=max(0, int(max_repairs)),
        use_raw_subpath_geometry=True,
        repair_mode="final_connectivity_rescue_raw_coverage",
    )
    accepted_ids = {int(r.get("edge_id", -1)) for r in rows}
    for edge in fixed.edges:
        if int(edge.get("edge_id", -1)) in accepted_ids:
            edge["raw_track_coverage_rescue"] = True
            edge["coverage_audit"] = True
            edge["retained_reason"] = "high_support_uncovered_raw_corridor"
    for row in rows:
        row["coverage_audit"] = True
        row["coverage_status"] = "restored"
        row["cycle_closure"] = bool(row.get("cycle_closure", False))
    return fixed, rows


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    angle = _angle_deg(a, b)
    return min(float(angle), abs(180.0 - float(angle)))


def edge_axis_at_node(edge: Dict[str, Any], node_id: int, probe_m: float = 8.0) -> Optional[np.ndarray]:
    coords = np.asarray(edge["geometry"].coords, dtype=float)
    if len(coords) < 2:
        return None
    if int(edge["u"]) == int(node_id):
        arr = coords
    elif int(edge["v"]) == int(node_id):
        arr = coords[::-1]
    else:
        return None
    origin = arr[0]
    target = arr[-1]
    travelled = 0.0
    for a, b in zip(arr[:-1], arr[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg <= 1e-9:
            continue
        if travelled + seg >= float(probe_m):
            f = (float(probe_m) - travelled) / seg
            target = a + np.clip(f, 0.0, 1.0) * (b - a)
            break
        travelled += seg
    v = np.asarray(target - origin, dtype=float)
    return v if np.linalg.norm(v) > 1e-9 else None


def edge_axis_at_distance(edge: Dict[str, Any], distance: float, probe_m: float = 4.0) -> Optional[np.ndarray]:
    geom = edge["geometry"]
    if not isinstance(geom, LineString) or geom.length <= 0:
        return None
    d = float(np.clip(distance, 0.0, float(geom.length)))
    lo = max(0.0, d - float(probe_m)); hi = min(float(geom.length), d + float(probe_m))
    if hi - lo <= 1e-6:
        return None
    a = geom.interpolate(lo); b = geom.interpolate(hi)
    v = np.asarray([float(b.x - a.x), float(b.y - a.y)], dtype=float)
    return v if np.linalg.norm(v) > 1e-9 else None


def is_transverse_parallel_bridge(
    connector: LineString,
    source_edge: Dict[str, Any],
    source_node: int,
    target_edge: Dict[str, Any],
    target_distance: float,
    parallel_axis_max_angle_deg: float = 15.0,
    transverse_min_angle_deg: float = 55.0,
) -> Tuple[bool, Dict[str, float]]:
    source_axis = edge_axis_at_node(source_edge, int(source_node))
    target_axis = edge_axis_at_distance(target_edge, float(target_distance))
    c = np.asarray(connector.coords[-1], dtype=float) - np.asarray(connector.coords[0], dtype=float)
    if source_axis is None or target_axis is None or np.linalg.norm(c) <= 1e-9:
        return False, {"parallel_axis_angle_deg": 180.0, "source_cross_angle_deg": 0.0, "target_cross_angle_deg": 0.0}
    parallel = _axis_angle_deg(source_axis, target_axis)
    ca = _axis_angle_deg(source_axis, c)
    cb = _axis_angle_deg(target_axis, c)
    flagged = (
        parallel <= float(parallel_axis_max_angle_deg)
        and ca >= float(transverse_min_angle_deg)
        and cb >= float(transverse_min_angle_deg)
    )
    return bool(flagged), {
        "parallel_axis_angle_deg": float(parallel),
        "source_cross_angle_deg": float(ca),
        "target_cross_angle_deg": float(cb),
    }


def _synthetic_connector(edge: Dict[str, Any]) -> bool:
    source = str(edge.get("source_type", ""))
    return bool(
        edge.get("raw_track_gap_repair")
        or edge.get("micro_gap_repair")
        or edge.get("s7_force_gap_repair")
        or edge.get("junction_repair")
        or edge.get("approved_junction_repair")
        or edge.get("component_stitching_repair")
        or source.startswith("s7_")
        or source in {"micro_gap_repair"}
    )


def _choose_local_axis(edges: Sequence[Dict[str, Any]], node_id: int) -> Optional[np.ndarray]:
    axes = [edge_axis_at_node(e, node_id) for e in edges]
    axes = [a for a in axes if a is not None]
    if not axes:
        return None
    # All local directions are treated as unoriented axes. Pick the longest-probe
    # vector; split through-roads produce two opposite vectors with the same axis.
    return max(axes, key=lambda a: float(np.linalg.norm(a)))


def prune_raw_negative_connectors(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    max_connector_m: float = 60.0,
    raw_witness_min_tracks: int = 1,
    raw_side_min_tracks: int = 1,
    raw_endpoint_snap_m: float = 6.0,
    raw_max_path_ratio: float = 3.0,
    raw_corridor_m: float = 10.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove synthetic connectors contradicted by local original trajectories.

    This is geometry-agnostic.  The older guard only caught transverse bridges between
    nearly parallel roads.  v0.5 applies the stronger topology rule to every synthetic
    S7 connector: when raw trajectories occupy both incident road sides but no
    motion-sane contribution directly traverses the connector, the connection is
    considered invented and is removed.  Sparse/unobserved sides remain inconclusive.
    """
    idx = witness_index
    if idx is None or not idx.enabled:
        return network.copy(), []
    net = reindex_network(network.copy())
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for edge in net.edges:
        incident[int(edge["u"])].append(edge)
        incident[int(edge["v"])].append(edge)

    drop: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    for edge in net.edges:
        if not _synthetic_connector(edge):
            continue
        geom = edge.get("geometry")
        if not isinstance(geom, LineString) or geom.length <= 0 or geom.length > float(max_connector_m):
            continue
        eid = int(edge["edge_id"]); u = int(edge["u"]); v = int(edge["v"])
        left = [e for e in incident[u] if int(e["edge_id"]) != eid]
        right = [e for e in incident[v] if int(e["edge_id"]) != eid]
        if not left or not right:
            continue
        witness = idx.query_connector(
            geom, endpoint_snap_m=float(raw_endpoint_snap_m),
            max_path_m=max(float(max_connector_m), float(geom.length) * float(raw_max_path_ratio)),
            max_path_ratio=float(raw_max_path_ratio),
            corridor_m=float(raw_corridor_m),
        )
        if int(witness.get("witness_tracks", 0) or 0) >= int(raw_witness_min_tracks):
            continue
        left_tracks: Set[str] = set()
        right_tracks: Set[str] = set()
        for side in left:
            g = side.get("geometry")
            if isinstance(g, LineString):
                left_tracks.update(idx.tracks_near_geometry(g, float(raw_endpoint_snap_m)))
        for side in right:
            g = side.get("geometry")
            if isinstance(g, LineString):
                right_tracks.update(idx.tracks_near_geometry(g, float(raw_endpoint_snap_m)))
        if len(left_tracks) < int(raw_side_min_tracks) or len(right_tracks) < int(raw_side_min_tracks):
            continue
        drop.add(eid)
        rows.append({
            "edge_id": eid,
            "action": "remove_raw_negative_false_connector",
            "source_type": str(edge.get("source_type", "")),
            "length_m": float(geom.length),
            "raw_witness_tracks": int(witness.get("witness_tracks", 0) or 0),
            "raw_side_a_tracks": int(len(left_tracks)),
            "raw_side_b_tracks": int(len(right_tracks)),
            "geometry": geom,
        })
    if not drop:
        return net, []
    net.edges = [e for e in net.edges if int(e["edge_id"]) not in drop]
    print(f"    [raw-negative-topology:v0.5] removed={len(rows):,}", flush=True)
    return reindex_network(net), rows


def prune_parallel_false_bridges(
    network: NetworkData,
    witness_index: Optional[RawTrackWitnessIndex],
    max_bridge_m: float = 35.0,
    parallel_axis_max_angle_deg: float = 15.0,
    transverse_min_angle_deg: float = 55.0,
    raw_witness_min_tracks: int = 1,
    raw_parallel_min_side_tracks: int = 1,
    raw_endpoint_snap_m: float = 6.0,
    raw_max_path_ratio: float = 3.0,
    raw_corridor_m: float = 10.0,
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    idx = witness_index
    if idx is None or not idx.enabled:
        return network.copy(), []
    net = reindex_network(network.copy())
    incident: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in net.edges:
        incident[int(e["u"])].append(e); incident[int(e["v"])].append(e)
    drop: Set[int] = set()
    rows: List[Dict[str, Any]] = []
    for edge in net.edges:
        if not _synthetic_connector(edge):
            continue
        geom = edge.get("geometry")
        if not isinstance(geom, LineString) or geom.length <= 0 or geom.length > float(max_bridge_m):
            continue
        u, v = int(edge["u"]), int(edge["v"])
        left = [e for e in incident[u] if int(e["edge_id"]) != int(edge["edge_id"])]
        right = [e for e in incident[v] if int(e["edge_id"]) != int(edge["edge_id"])]
        if not left or not right:
            continue
        a = _choose_local_axis(left, u); b = _choose_local_axis(right, v)
        c = np.asarray(geom.coords[-1], dtype=float) - np.asarray(geom.coords[0], dtype=float)
        if a is None or b is None or np.linalg.norm(c) <= 1e-9:
            continue
        parallel = _axis_angle_deg(a, b)
        ca = _axis_angle_deg(a, c); cb = _axis_angle_deg(b, c)
        if not (
            parallel <= float(parallel_axis_max_angle_deg)
            and ca >= float(transverse_min_angle_deg)
            and cb >= float(transverse_min_angle_deg)
        ):
            continue
        witness = idx.query_connector(
            geom,
            endpoint_snap_m=float(raw_endpoint_snap_m),
            max_path_m=max(float(max_bridge_m) * float(raw_max_path_ratio), float(max_bridge_m)),
            max_path_ratio=float(raw_max_path_ratio),
            corridor_m=float(raw_corridor_m),
        )
        if int(witness["witness_tracks"]) >= int(raw_witness_min_tracks):
            continue
        left_tracks: Set[str] = set()
        right_tracks: Set[str] = set()
        for side_edge in left:
            left_tracks.update(idx.tracks_near_geometry(side_edge["geometry"], float(raw_endpoint_snap_m)))
        for side_edge in right:
            right_tracks.update(idx.tracks_near_geometry(side_edge["geometry"], float(raw_endpoint_snap_m)))
        # Absence of a crossing is meaningful negative evidence only when original
        # trajectories actually occupy BOTH parallel corridors. If raw data is sparse
        # here, preserve the raster-supported connector rather than over-prune it.
        if (
            len(left_tracks) < int(raw_parallel_min_side_tracks)
            or len(right_tracks) < int(raw_parallel_min_side_tracks)
        ):
            continue
        drop.add(int(edge["edge_id"]))
        rows.append({
            "edge_id": int(edge["edge_id"]),
            "action": "remove_parallel_false_bridge",
            "source_type": str(edge.get("source_type", "")),
            "length_m": float(geom.length),
            "parallel_axis_angle_deg": float(parallel),
            "source_cross_angle_deg": float(ca),
            "target_cross_angle_deg": float(cb),
            "raw_witness_tracks": int(witness["witness_tracks"]),
            "raw_side_a_tracks": int(len(left_tracks)),
            "raw_side_b_tracks": int(len(right_tracks)),
            "geometry": geom,
        })
    if not drop:
        return net, []
    net.edges = [e for e in net.edges if int(e["edge_id"]) not in drop]
    print(f"    [parallel-bridge-guard] removed={len(rows):,}", flush=True)
    return reindex_network(net), rows


def audit_raw_track_gaps(
    network: NetworkData,
    max_endpoint_distance_m: float,
    max_track_path_m: float,
    max_angle_deg: float,
    min_witness_tracks: int = 1,
) -> List[Dict[str, Any]]:
    """Audit raw-track connectors against their creation-time envelope.

    Older/ordinary connectors that predate per-edge metadata fall back to the
    function arguments.  Coverage and final-rescue connectors persist their own
    admissibility thresholds, preventing a legitimate long coverage repair from
    being reclassified as an ordinary short gap during final audit.
    """
    rows: List[Dict[str, Any]] = []
    for edge in network.edges:
        if not bool(edge.get("raw_track_gap_repair")):
            continue
        euclid_value = edge.get("raw_gap_euclidean_m")
        path_value = edge.get("raw_witness_source_path_m")
        aa_value = edge.get("raw_gap_angle_a_deg")
        ab_value = edge.get("raw_gap_angle_b_deg")
        tracks_value = edge.get("raw_witness_tracks")
        euclid = float(edge["geometry"].length if euclid_value is None else euclid_value)
        path_m = float(edge["geometry"].length if path_value is None else path_value)
        aa = float(180.0 if aa_value is None else aa_value)
        ab = float(180.0 if ab_value is None else ab_value)
        tracks = int(0 if tracks_value is None else tracks_value)

        audit_mode = str(edge.get("raw_gap_audit_mode") or (
            "final_connectivity_rescue_raw_coverage"
            if bool(edge.get("raw_track_coverage_rescue"))
            else "ordinary_raw_track_gap"
        ))
        endpoint_limit = float(edge.get(
            "raw_gap_audit_max_endpoint_distance_m", max_endpoint_distance_m
        ))
        path_limit = float(edge.get(
            "raw_gap_audit_max_track_path_m", max_track_path_m
        ))
        angle_limit = float(edge.get("raw_gap_audit_max_angle_deg", max_angle_deg))
        witness_limit = int(edge.get(
            "raw_gap_audit_min_witness_tracks", min_witness_tracks
        ))

        reasons: List[str] = []
        if euclid > endpoint_limit + 1e-9:
            reasons.append("distance")
        if path_m > path_limit + 1e-9:
            reasons.append("raw_path")
        if max(aa, ab) > angle_limit + 1e-9:
            reasons.append("angle")
        if tracks < witness_limit:
            reasons.append("witness")
        if not bool(edge.get("approved_gap_repair")):
            reasons.append("not_approved")

        rows.append({
            "edge_id": int(edge["edge_id"]),
            "gap_repair": True,
            "raw_track_gap_repair": True,
            "raw_track_coverage_rescue": bool(edge.get("raw_track_coverage_rescue")),
            "final_connectivity_rescue": bool(edge.get("final_connectivity_rescue")),
            "audit_mode": audit_mode,
            "endpoint_distance_limit_m": endpoint_limit,
            "raw_path_limit_m": path_limit,
            "angle_limit_deg": angle_limit,
            "witness_track_limit": witness_limit,
            "outside_baseline_fraction": 1.0 if reasons else 0.0,
            "audit_reason": "+".join(reasons) if reasons else "approved_raw_track_witness",
            "euclidean_m": euclid,
            "raw_source_path_m": path_m,
            "angle_a_deg": aa,
            "angle_b_deg": ab,
            "raw_witness_tracks": tracks,
            "geometry": edge["geometry"],
        })
    return rows


def rollback_failed_raw_track_gap_repairs(
    network: NetworkData,
    violations: Sequence[Dict[str, Any]],
) -> Tuple[NetworkData, List[Dict[str, Any]]]:
    """Remove only synthetic raw-gap connectors that fail their own final audit."""
    bad_ids = {int(r["edge_id"]) for r in violations if "edge_id" in r}
    if not bad_ids:
        return reindex_network(network.copy()), []

    by_id = {int(r["edge_id"]): r for r in violations if "edge_id" in r}
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for edge in network.edges:
        eid = int(edge["edge_id"])
        if eid not in bad_ids:
            kept.append(dict(edge))
            continue
        # Only a connector created by the raw-gap machinery may be removed here.
        # If an unrelated edge somehow appears in the violation table, retain it.
        if not bool(edge.get("raw_track_gap_repair")):
            kept.append(dict(edge))
            continue
        row = dict(by_id.get(eid, {}))
        row.update({
            "rollback_reason": "failed_final_raw_track_gap_audit_v0_5",
            "rolled_back": True,
            "geometry": edge.get("geometry"),
        })
        rejected.append(row)

    cleaned = NetworkData(
        [dict(n) for n in network.nodes], kept, network.crs, network.transform,
        network.width, network.height, dict(network.metadata),
    )
    return reindex_network(cleaned), rejected
