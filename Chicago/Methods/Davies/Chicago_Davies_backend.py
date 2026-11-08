"""Python 3/OpenCV 4 port of the preserved Davies contour/Voronoi workflow.

The image, contour, subdivision, direction, and export stages follow the
preserved ``davies2006.py``. All intermediates are confined to this run.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from experiments.chicago_six_methods.reference_methods.common import MethodRunResult, sha256_file
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext


SOURCE_DIR = Path(__file__).resolve().parent / "Chicago_Davies_Source"
SOURCE_SCRIPT = SOURCE_DIR / "davies2006.py"
SOURCE_IDENTIFIER = SOURCE_SCRIPT.relative_to(Path(__file__).resolve().parents[3]).as_posix()
SOURCE_PATHS = (SOURCE_DIR,)
PARAMETERS = {
    "cell_size": 2,
    "mask_threshold": 100,
    "gaussian_blur": 17,
    "voronoi_sampling_interval": 10,
    "min_direction_count": 10,
}
EXECUTION_CLASS = "Python 3/OpenCV 4 compatibility"
METERS_PER_DEGREE_LATITUDE = 111070.34306591158
METERS_PER_DEGREE_LONGITUDE = 83044.98918812413

# Each mapping points to the source operation it replaces. The unchanged
# scientific operations and stage order are recorded in native/report.json.
COMPATIBILITY = {
    "PltTripLoader.get_all_trips": "read prepared WGS84 trip_N.txt files in numeric trip order; each row is index,latitude,longitude,timestamp",
    "cv.CreateMat/CV_8UC1/CV_16UC1/CV_8UC3 and cv.SetZero": "numpy zero arrays with uint8/uint16 channel-equivalent dtypes",
    "cv.Line(...,cv.CV_AA)": "cv2.line(...,lineType=cv2.LINE_AA) with the original intensity 32, width 1",
    "cv.ConvertScale and cv.Add": "uint8-to-uint16 NumPy conversion and cv2.add with the original saturation behavior",
    "cv.Smooth(...,cv.CV_GAUSSIAN,17,17)": "cv2.GaussianBlur with 17 by 17 kernel and the original default border",
    "cv.MinMaxLoc/cv.CmpS(CV_CMP_GT)": "cv2.minMaxLoc/cv2.compare(CMP_GT), preserving threshold 100 on blurred uint16 intensity",
    "cv.FindContours(CV_RETR_CCOMP,CV_CHAIN_CODE) + cv.ApproxChains(CV_CHAIN_APPROX_NONE,0,100,1)": "cv2.findContours(RETR_CCOMP,CHAIN_APPROX_NONE), retaining contours with perimeter at least 100 pixels",
    "CvSeq.h_next/v_next and cv.PointPolygonTest": "require every OpenCV LINE_8 pixel of a rounded Voronoi segment to lie in the occupied mask, matching the confirmed Python 3 compatibility run",
    "cv.CreateSubdivDelaunay2D/cv.SubdivDelaunay2DInsert/cv.CalcSubdivVoronoi2D": "cv2.Subdiv2D.insert/getVoronoiFacetList at the original one-in-ten contour samples",
    "cv.Subdiv2DGetEdge/CV_NEXT_AROUND_LEFT/RotateEdge/EdgeOrg/EdgeDst": "cv2.Subdiv2D getLeadingEdgeList, rotateEdge, getEdge, edgeOrg, edgeDst; traverse both dual facets",
    "cv.InitLineIterator(...,8)": "direction-invariant cv2.LINE_8 raster samples used by the confirmed Python 3 compatibility run",
    "cv.DrawContours/cv.SaveImage": "cv2.drawContours/cv2.imwrite under native/",
    "Python 2 reduce/izip/backtick repr and implicit integer division": "Python 3 iteration, f-string export, and explicit legacy integer raster indices",
    "tmp/cache XML": "no cross-run cache; rebuild the original intensity and sector maps for each run",
}


def _trip_files(directory: Path) -> list[Path]:
    files = list(directory.glob("trip_*.txt"))
    if not files:
        raise FileNotFoundError(f"Prepared Davies trips are missing: {directory}")
    return sorted(files, key=lambda path: int(path.stem.removeprefix("trip_")))


def _load_trips(directory: Path) -> list[list[tuple[float, float]]]:
    trips = []
    for path in _trip_files(directory):
        points = []
        for number, row in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = row.split(",")
            if len(fields) != 4:
                raise ValueError(f"Malformed Davies trip row: {path}:{number}")
            points.append((float(fields[1]), float(fields[2])))
        if len(points) > 1:
            trips.append(points)
    if not trips:
        raise ValueError("Davies needs at least one trip with two locations")
    return trips


def _bounds(trips: list[list[tuple[float, float]]]) -> dict[str, int | float]:
    latitudes = [lat for trip in trips for lat, _ in trip]
    longitudes = [lon for trip in trips for _, lon in trip]
    min_lat, max_lat = min(latitudes) - 0.003, max(latitudes) + 0.003
    min_lon, max_lon = min(longitudes) - 0.005, max(longitudes) + 0.005
    width = int((max_lon - min_lon) * METERS_PER_DEGREE_LONGITUDE / PARAMETERS["cell_size"])
    height = int((max_lat - min_lat) * METERS_PER_DEGREE_LATITUDE / PARAMETERS["cell_size"])
    if width <= 0 or height <= 0:
        raise ValueError("Davies raster has no area")
    return {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon,
            "max_lon": max_lon, "width": width, "height": height}


def _sector(start: tuple[int, int], end: tuple[int, int]) -> int:
    # Preserved getsector: 0 north, 2 east, 4 south, 6 west.
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return int(-angle / (math.pi / 4) + 2) % 8


def _pixel(lat: float, lon: float, bounds: dict[str, int | float]) -> tuple[int, int]:
    width, height = int(bounds["width"]), int(bounds["height"])
    xscale = width / (bounds["max_lon"] - bounds["min_lon"])
    yscale = height / (bounds["max_lat"] - bounds["min_lat"])
    return (int(xscale * (lon - bounds["min_lon"])),
            height - int(yscale * (lat - bounds["min_lat"])))


def _intensity(trips: list[list[tuple[float, float]]], bounds: dict[str, int | float]) -> tuple[np.ndarray, list[np.ndarray]]:
    size = (int(bounds["height"]), int(bounds["width"]))
    intensity = np.zeros(size, dtype=np.uint16)
    sectors = [np.zeros(size, dtype=np.uint16) for _ in range(8)]
    for trip in trips:
        temporary = np.zeros(size, dtype=np.uint8)
        sector_temporary = [np.zeros(size, dtype=np.uint8) for _ in range(8)]
        for origin, destination in zip(trip, trip[1:]):
            start, end = _pixel(*origin, bounds), _pixel(*destination, bounds)
            cv2.line(temporary, start, end, 32, 1, lineType=cv2.LINE_AA)
            cv2.line(sector_temporary[_sector(start, end)], start, end, 32, 1,
                     lineType=cv2.LINE_AA)
        intensity = cv2.add(intensity, temporary.astype(np.uint16))
        for index in range(8):
            sectors[index] = cv2.add(sectors[index], sector_temporary[index].astype(np.uint16))
    return intensity, sectors


def _inside(point1: tuple[float, float], point2: tuple[float, float], contours: list[np.ndarray],
            hierarchy: np.ndarray, index: int, adding: bool) -> bool:
    """Preserved CvSeq h_next/v_next alternating outer/hole test."""
    if index < 0:
        return adding
    first = cv2.pointPolygonTest(contours[index], point1, True) > 0
    second = cv2.pointPolygonTest(contours[index], point2, True) > 0
    if adding and (first or second):
        return False
    if not adding and first and second:
        return _inside(point1, point2, contours, hierarchy, int(hierarchy[index][2]), True)
    return _inside(point1, point2, contours, hierarchy, int(hierarchy[index][0]), adding)


def _line_pixels_cv8(
    start: tuple[int, int], end: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixels selected by legacy ``cv.InitLineIterator(..., 8)``.

    The paper's confirmed Python 3 compatibility runner replaced the removed C
    iterator with OpenCV's direction-invariant ``LINE_8`` rasterization.
    """
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
    cv2.line(
        mask,
        (start[0] - left, start[1] - top),
        (end[0] - left, end[1] - top),
        1,
        1,
        lineType=cv2.LINE_8,
    )
    rows, columns = np.nonzero(mask)
    return rows + top, columns + left


def _inside_occupied(
    point1: tuple[float, float], point2: tuple[float, float], occupied: np.ndarray
) -> bool:
    start = (int(round(point1[0])), int(round(point1[1])))
    end = (int(round(point2[0])), int(round(point2[1])))
    rows, columns = _line_pixels_cv8(start, end)
    valid = (
        (rows >= 0)
        & (rows < occupied.shape[0])
        & (columns >= 0)
        & (columns < occupied.shape[1])
    )
    return bool(valid.all() and np.all(occupied[rows, columns] > 0))


def _direction_count(start: tuple[int, int], end: tuple[int, int], sectors: list[np.ndarray]) -> int:
    selected = _sector(start, end)
    rows, columns = _line_pixels_cv8(start, end)
    valid = ((rows >= 0) & (rows < sectors[0].shape[0]) &
             (columns >= 0) & (columns < sectors[0].shape[1]))
    rows, columns = rows[valid], columns[valid]
    return sum(int(sectors[sector][rows, columns].astype(np.uint64).sum())
               for sector in (selected, (selected + 1) % 8, (selected + 7) % 8))


def _contours_and_samples(
    mask: np.ndarray, interval: int
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Port the legacy ``FindContours``/``ApproxChains`` traversal exactly.

    OpenCV's old ``minimal_perimeter`` argument did not discard short chains.
    The historical compatibility run also sampled CvSeq offsets 0, 10, ...;
    both details affect the Delaunay subdivision and therefore the final graph.
    """
    contours, links = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    hierarchy = (
        np.asarray(links[0], dtype=np.int32)
        if links is not None
        else np.empty((0, 4), dtype=np.int32)
    )
    samples = [
        point
        for contour in contours
        for index, point in enumerate(contour.reshape(-1, 2))
        if index % interval == 0
    ]
    return contours, hierarchy, samples


def _legacy_quad_edge_slots(sample_count: int) -> list[int]:
    """Return the CvSet edge handles visited by legacy ``subdiv.edges``."""
    slot_count = 3 * sample_count + 3
    return list(range(4, 4 * slot_count + 1, 4))


def _graph(subdivision: cv2.Subdiv2D, contours: list[np.ndarray], hierarchy: np.ndarray,
           sectors: list[np.ndarray], image: np.ndarray,
           sample_count: int,
           occupied: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lines = []
    for edge in _legacy_quad_edge_slots(sample_count):
        for rotation in (1, 3):
            first = subdivision.rotateEdge(int(edge), rotation)
            current = subdivision.getEdge(first, cv2.Subdiv2D_NEXT_AROUND_LEFT)
            visited = set()
            while current != first and current not in visited:
                visited.add(current)
                origin_id, origin = subdivision.edgeOrg(current)
                destination_id, destination = subdivision.edgeDst(current)
                if origin_id > 0 and destination_id > 0 and origin != destination:
                    start = (int(round(origin[0])), int(round(origin[1])))
                    end = (int(round(destination[0])), int(round(destination[1])))
                    if start != end and _inside_occupied(origin, destination, occupied):
                        forward = _direction_count(start, end, sectors)
                        reverse = _direction_count(end, start, sectors)
                        if (min(forward, reverse) > 0 and
                                max(forward, reverse) / min(forward, reverse) < 4):
                            lines.extend(((start, end), (end, start)))
                        elif (forward <= PARAMETERS["min_direction_count"] and
                              reverse <= PARAMETERS["min_direction_count"]):
                            lines.extend(((start, end), (end, start)))
                        elif forward > reverse:
                            lines.append((start, end))
                        else:
                            lines.append((end, start))
                current = subdivision.getEdge(current, cv2.Subdiv2D_NEXT_AROUND_LEFT)
    unique = list(dict.fromkeys(lines))
    for start, end in unique:
        cv2.line(image, start, end, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return unique


def _write_graph(native: Path, lines: list[tuple[tuple[int, int], tuple[int, int]]],
                 bounds: dict[str, int | float]) -> tuple[int, int]:
    vertex_ids: dict[tuple[int, int], int] = {}
    edges = []
    for start, end in lines:
        if start not in vertex_ids:
            vertex_ids[start] = len(vertex_ids)
        if end not in vertex_ids:
            vertex_ids[end] = len(vertex_ids)
        edges.append(f"{vertex_ids[start]} {vertex_ids[end]}")
    yscale = bounds["height"] / (bounds["max_lat"] - bounds["min_lat"])
    xscale = bounds["width"] / (bounds["max_lon"] - bounds["min_lon"])
    vertices = [f"{identifier} {(bounds['height'] - point[1]) / yscale + bounds['min_lat']} "
                f"{point[0] / xscale + bounds['min_lon']}"
                for point, identifier in vertex_ids.items()]
    (native / "vertices.txt").write_text("\n".join(vertices) + "\n", encoding="utf-8", newline="\n")
    (native / "edges.txt").write_text("\n".join(edges) + "\n", encoding="utf-8", newline="\n")
    return len(vertices), len(edges)


def _write_image(path: Path, raster: np.ndarray) -> None:
    """Write through NumPy so OpenCV works under a Windows Unicode path."""
    success, encoded = cv2.imencode(path.suffix, raster)
    if not success:
        raise OSError(f"Could not encode Davies diagnostic raster: {path.name}")
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Could not write Davies diagnostic raster: {path.name}")


def run(context: RunContext, inputs: PreparedInputs) -> MethodRunResult:
    """Execute the preserved image/Voronoi stage order on prepared Chicago trips."""
    started = time.perf_counter()
    trips_dir = (inputs.trips_wgs84 / "davies").resolve()
    trips = _load_trips(trips_dir)
    bounds = _bounds(trips)
    directory = context.output_root.resolve() / "davies"
    native = directory / "native"
    native.mkdir(parents=True)

    intensity, sectors = _intensity(trips, bounds)
    smoothed_sectors = [cv2.GaussianBlur(sector, (17, 17), 0) for sector in sectors]
    smoothed = cv2.GaussianBlur(intensity, (17, 17), 0)
    maximum = cv2.minMaxLoc(smoothed)[1]
    normalized = cv2.convertScaleAbs(smoothed, alpha=255.0 / maximum) if maximum else np.zeros_like(intensity, dtype=np.uint8)
    thresholded = cv2.compare(smoothed, PARAMETERS["mask_threshold"], cv2.CMP_GT)
    contours, hierarchy, contour_samples = _contours_and_samples(
        thresholded, PARAMETERS["voronoi_sampling_interval"]
    )
    shape = (int(bounds["height"]), int(bounds["width"]))
    contour_image = np.zeros((*shape, 3), dtype=np.uint8)
    if contours:
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1,
                         lineType=cv2.LINE_AA)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    if contours:
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1,
                         lineType=cv2.LINE_AA)
    subdivision = cv2.Subdiv2D((0, 0, shape[1], shape[0]))
    sample_count = 0
    sampled = set()
    for row in contour_samples:
        point = (float(row[0]), float(row[1]))
        if point in sampled:
            continue
        sampled.add(point)
        subdivision.insert(point)
        sample_count += 1
    lines = []
    if sample_count >= 3:
        subdivision.getVoronoiFacetList([])
        lines = _graph(
            subdivision, contours, hierarchy, smoothed_sectors, image,
            sample_count, thresholded
        )
    if not lines:
        raise RuntimeError(
            "Davies produced an empty graph with mask_threshold=100; "
            f"{len(trips)} trips yielded {len(contours)} contours and {sample_count} Voronoi samples"
        )
    vertex_count, edge_count = _write_graph(native, lines, bounds)
    rasters = {
        "map": smoothed, "mask": normalized, "thresholded": thresholded,
        "contours": contour_image, "voronoi": image,
    }
    for name, raster in rasters.items():
        _write_image(native / f"{name}.png", raster)
    report = {
        "algorithm": "preserved Davies contour/Voronoi workflow",
        "execution_class": EXECUTION_CLASS,
        "parameters": PARAMETERS,
        "bounds": bounds,
        "trip_count": len(trips),
        "contour_count": len(contours),
        "voronoi_sample_count": sample_count,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "source_path": SOURCE_IDENTIFIER,
        "source_sha256": sha256_file(SOURCE_SCRIPT),
        "compatibility_mapping": COMPATIBILITY,
        "stage_order": ["intensity", "directional intensity", "Gaussian blur", "threshold",
                        "contours", "sampled Delaunay", "Voronoi facets",
                        "directional edge selection", "graph export"],
    }
    report_path = native / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    stage_path = directory / "stage_manifest.json"
    stage_path.write_text(json.dumps({"stage": "davies_voronoi", "execution_class": EXECUTION_CLASS,
                                      "report": "native/report.json",
                                      "diagnostic_rasters": [f"native/{name}.png" for name in rasters],
                                      "graph": ["native/vertices.txt", "native/edges.txt"]},
                                     sort_keys=True, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    outputs = {name: native / filename for name, filename in
               (("vertices", "vertices.txt"), ("edges", "edges.txt"), ("report", "report.json"))}
    outputs.update({name: native / f"{name}.png" for name in rasters})
    return MethodRunResult("davies", EXECUTION_CLASS, outputs, time.perf_counter() - started, stage_path)
