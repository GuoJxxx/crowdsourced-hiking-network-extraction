"""Verify and derive deterministic Chicago inputs from the public ZIP files."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath

import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString

from .models import PreparedInputs


EXPECTED_SHA256 = {
    "tracks": "f0e94f3734a4b1e5399bea365d93753eda25a0add3c583e846e7a3212f07c397",
    "map": "3d3ecefa2d4cc549b0a2847d1439c2602855c52e77a3c41087a4dd5c9147c077",
}
TRIP_PATTERN = re.compile(r"chicago/trips/trip_\d+\.txt\Z")
TRACK_METADATA = {
    "chicago/", "chicago/trips/", "chicago/.DS_Store",
    "__MACOSX/", "__MACOSX/chicago/", "__MACOSX/chicago/._.DS_Store",
    "__MACOSX/chicago/._LICENSE.txt",
}
MAP_METADATA = {"__MACOSX/", "__MACOSX/._LICENSE.txt"}
# Twenty trips leave Davies below its preserved threshold of 100; smaller
# prefixes can produce a fragile or empty Voronoi graph. Share the first 100
# lexicographically sorted archive trips across methods for Smoke.
SMOKE_TRIPS = 100
CELL_SIZE_M = 2.0
GRID_PADDING_M = 20.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _open_verified_archive(path: Path, role: str) -> zipfile.ZipFile:
    actual = _sha256(path)
    if actual != EXPECTED_SHA256[role]:
        raise ValueError(f"{role} archive SHA256 mismatch: {path.name}")
    archive = zipfile.ZipFile(path)
    names: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if (
            name.startswith("/") or "\\" in name or ":" in name
            or any(part in (".", "..") for part in name.split("/"))
            or pure.is_absolute()
            or ((info.external_attr >> 16) & 0o170000) == 0o120000
        ):
            archive.close()
            raise ValueError(f"unsafe ZIP member: {name}")
        if name in names:
            archive.close()
            raise ValueError(f"duplicate ZIP member: {name}")
        names.add(name)
    if role == "tracks":
        required = {"chicago/LICENSE.txt"}
        unexpected = sorted(name for name in names if name not in TRACK_METADATA
                            and name not in required and not TRIP_PATTERN.fullmatch(name))
        trips = sorted(name for name in names if TRIP_PATTERN.fullmatch(name))
        expected_trips = {f"chicago/trips/trip_{index}.txt" for index in range(889)}
        if set(trips) != expected_trips:
            missing = sorted(expected_trips - set(trips))
            archive.close()
            raise ValueError(f"missing trip member: {missing[:1]}")
    else:
        required = {"chicago_edges_osm.txt", "chicago_vertices_osm.txt", "LICENSE.txt"}
        unexpected = sorted(names - required - MAP_METADATA)
    if unexpected:
        archive.close()
        raise ValueError(f"unexpected ZIP member: {unexpected[0]}")
    missing = sorted(required - names)
    if missing:
        archive.close()
        label = "license" if any("LICENSE" in name for name in missing) else "member"
        raise ValueError(f"missing {label}: {missing[0]}")
    license_member = "chicago/LICENSE.txt" if role == "tracks" else "LICENSE.txt"
    if b"Open Database License" not in archive.read(license_member):
        archive.close()
        raise ValueError(f"invalid license: {license_member}")
    return archive


def _parse_trip(archive: zipfile.ZipFile, member: str, transformer: Transformer):
    try:
        lines = archive.read(member).decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError(f"{member}:1: invalid UTF-8") from exc
    if len(lines) < 2:
        raise ValueError(f"{member}:1: trajectory must contain at least two points")
    projected = []
    geographic = []
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        try:
            if len(fields) != 3:
                raise ValueError("expected x y timestamp")
            x, y, timestamp = map(float, fields)
            if not all(map(math.isfinite, (x, y, timestamp))):
                raise ValueError("non-finite coordinate or timestamp")
            longitude, latitude = transformer.transform(x, y)
            if not (-88.5 < longitude < -86.5 and 41.0 < latitude < 43.0):
                raise ValueError("coordinate outside Chicago EPSG:26916 region")
        except ValueError as exc:
            raise ValueError(f"{member}:{line_number}: {exc}") from exc
        projected.append((" ".join(fields), x, y))
        geographic.append((latitude, longitude, timestamp))
    return projected, geographic


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def _write_raw_union(path: Path, tracks: list[list[tuple[float, float]]]) -> dict[str, object]:
    xs = [x for track in tracks for x, _ in track]
    ys = [y for track in tracks for _, y in track]
    left = math.floor((min(xs) - GRID_PADDING_M) / CELL_SIZE_M) * CELL_SIZE_M
    right = math.ceil((max(xs) + GRID_PADDING_M) / CELL_SIZE_M) * CELL_SIZE_M
    bottom = math.floor((min(ys) - GRID_PADDING_M) / CELL_SIZE_M) * CELL_SIZE_M
    top = math.ceil((max(ys) + GRID_PADDING_M) / CELL_SIZE_M) * CELL_SIZE_M
    width = round((right - left) / CELL_SIZE_M)
    height = round((top - bottom) / CELL_SIZE_M)
    transform = from_origin(left, top, CELL_SIZE_M, CELL_SIZE_M)
    mask = rasterize(
        ((LineString(track), 1) for track in tracks),
        out_shape=(height, width), transform=transform, fill=0,
        all_touched=False, dtype="uint8",
    )
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs="EPSG:26916", transform=transform, nodata=0,
        compress="deflate",
    ) as raster:
        raster.write(mask, 1)
    return {"cell_size_m": CELL_SIZE_M, "padding_m": GRID_PADDING_M,
            "width": width, "height": height, "marked_pixels": int(mask.sum())}


def prepare_chicago_inputs(
    tracks_archive: Path, map_archive: Path, output_root: Path, profile: str
) -> PreparedInputs:
    """Validate the public archives, then write all derivatives below output_root."""
    if profile not in ("smoke", "paper"):
        raise ValueError(f"Unsupported profile: {profile}")
    root = Path(output_root).resolve()
    if root == Path(root.anchor):
        raise ValueError("output_root cannot be a filesystem root")
    tracks_path = Path(tracks_archive)
    map_path = Path(map_archive)
    transformer = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)
    with _open_verified_archive(tracks_path, "tracks") as tracks_zip, _open_verified_archive(map_path, "map") as map_zip:
        members = sorted(name for name in tracks_zip.namelist() if TRIP_PATTERN.fullmatch(name))
        selected = members if profile == "paper" else members[:SMOKE_TRIPS]
        parsed = [(member, *_parse_trip(tracks_zip, member, transformer)) for member in selected]
        if profile == "paper":
            point_count = sum(len(projected) for _, projected, _ in parsed)
            if point_count != 118_360:
                raise ValueError(f"Paper profile requires 118360 points, found {point_count}")
        input_root = root / "inputs"
        if input_root.exists():
            raise FileExistsError(f"Prepared inputs already exist: {input_root}")
        input_root.mkdir(parents=True)
        projected_dir = input_root / "tracks_projected"
        wgs_root = input_root / "trips_wgs84"
        truth_vertices = input_root / "truth/chicago_vertices_osm.txt"
        truth_edges = input_root / "truth/chicago_edges_osm.txt"
        raw_union = input_root / "raw_trajectory_union.tif"
        trip_rows = []
        raster_tracks = []
        for trip_id, (member, projected, geographic) in enumerate(parsed):
            name = f"trip_{trip_id}.txt"
            projected_path = projected_dir / name
            _write_text(projected_path, "\n".join(row[0] for row in projected) + "\n")
            raster_tracks.append([(row[1], row[2]) for row in projected])
            rows = [f"{index},{latitude:.10f},{longitude:.10f},{timestamp:.3f}"
                    for index, (latitude, longitude, timestamp) in enumerate(geographic)]
            method_hashes = {}
            for method in ("biagioni", "davies", "edelkamp"):
                target = wgs_root / method / name
                if method == "edelkamp":
                    output_rows = [
                        f"{row},{'None' if index == 0 else index - 1},"
                        f"{'None' if index == len(rows) - 1 else index + 1}"
                        for index, row in enumerate(rows)
                    ]
                else:
                    output_rows = rows
                _write_text(target, "\n".join(output_rows) + "\n")
                method_hashes[method] = _sha256(target)
            trip_rows.append({
                "trip_id": trip_id, "member": member, "point_count": len(projected),
                "projected_path": projected_path.relative_to(input_root).as_posix(),
                "projected_sha256": _sha256(projected_path),
                "wgs84_sha256": method_hashes,
            })
        truth_vertices.parent.mkdir(parents=True)
        truth_vertices.write_bytes(map_zip.read("chicago_vertices_osm.txt"))
        truth_edges.write_bytes(map_zip.read("chicago_edges_osm.txt"))
        licenses = input_root / "licenses"
        licenses.mkdir()
        (licenses / "tracks_LICENSE.txt").write_bytes(tracks_zip.read("chicago/LICENSE.txt"))
        (licenses / "map_LICENSE.txt").write_bytes(map_zip.read("LICENSE.txt"))
        grid = _write_raw_union(raw_union, raster_tracks)
    outputs = {
        "tracks_projected": {"path": projected_dir.relative_to(input_root).as_posix(), "sha256": _tree_sha256(projected_dir)},
        "trips_wgs84": {"path": wgs_root.relative_to(input_root).as_posix(), "sha256": _tree_sha256(wgs_root)},
        "raw_union": {"path": raw_union.relative_to(input_root).as_posix(), "sha256": _sha256(raw_union)},
        "truth_vertices": {"path": truth_vertices.relative_to(input_root).as_posix(), "sha256": _sha256(truth_vertices)},
        "truth_edges": {"path": truth_edges.relative_to(input_root).as_posix(), "sha256": _sha256(truth_edges)},
        "tracks_license": {"path": "licenses/tracks_LICENSE.txt", "sha256": _sha256(licenses / "tracks_LICENSE.txt")},
        "map_license": {"path": "licenses/map_LICENSE.txt", "sha256": _sha256(licenses / "map_LICENSE.txt")},
    }
    manifest = input_root / "input_manifest.json"
    document = {
        "profile": profile, "paper_counts_reproduced": profile == "paper",
        "source_crs": "EPSG:26916", "target_crs": "EPSG:4326",
        "archive_sha256": {"tracks": _sha256(tracks_path), "map": _sha256(map_path)},
        "trip_count": len(trip_rows),
        "point_count": sum(row["point_count"] for row in trip_rows),
        "trip_order": "lexicographic archive member name",
        "raw_union_grid": grid, "outputs": outputs, "trips": trip_rows,
    }
    manifest.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return PreparedInputs(projected_dir, wgs_root, raw_union, truth_vertices, truth_edges, manifest)
