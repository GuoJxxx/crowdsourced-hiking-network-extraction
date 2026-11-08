"""Execute all eight preserved Biagioni stages in a run-owned Python 3 copy."""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

from experiments.chicago_six_methods.reference_methods.common import (
    MethodRunResult, RunnerError, fingerprint_paths, read_json_object,
    regular_unlinked_file, reject_redirects, sha256_file, write_json_atomic,
)
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext


PACKAGE = Path(__file__).resolve().parent
SOURCE_DIR = Path(__file__).resolve().parent / "Chicago_Biagioni_Source"
HELPERS = {
    "biagioni_kde.py": "kde.py",
    "biagioni_skeleton.py": "skeleton.py",
    "biagioni_spatialfunclib_accel.py": "pylibs/spatialfunclib_accel.py",
    "biagioni_subiterations.py": "subiterations.py",
}
SOURCE_PATHS = (SOURCE_DIR, *(PACKAGE / "Chicago_Biagioni_Compatibility" / name for name in HELPERS))
SUPPORTS_STAGE_RESUME = True
PARAMETERS = {"cell_size_m": 1, "gaussian_blur": 17}
EXECUTION_CLASS = "Python 3/current-library compatibility"
STAGES = (
    "01_kde", "02_skeleton", "03_graph_extract", "04_matched_1",
    "05_prune_1", "06_refine", "07_matched_2", "08_final",
)
COMPATIBILITY = [
    "lib2to3 syntax conversion", "rtree.index.Index import",
    "OpenCV 4 image I/O and Gaussian blur; NumPy scalar aliases",
    "NumPy thinning replacing Cython with preserved deletion rules",
    "pure Python fast_distance with original latitude/longitude constants",
    "probability-only Viterbi comparison", "confirmed parallel-edge collapse guard",
    "confirmed duplicate segment coalescing guard", "numeric matched_trip_<integer>.txt",
    "numeric trip_<integer>.txt loading and matching",
    "original terminal-segment behavior (no synthetic final trace append)",
]


def _generate_compatibility_copy(biagioni_dir: Path) -> None:
    """Apply the proven Mainling translation to Chicago's preserved sources."""
    from workflow.conversion import make_refactoring_tool

    shutil.copytree(SOURCE_DIR, biagioni_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    refactor = make_refactoring_tool()
    for path in sorted(biagioni_dir.rglob("*.py")):
        original = path.read_text(encoding="utf-8").expandtabs(8)
        # Even parseable Python 2 needs map/list and dict snapshot conversion.
        converted = str(refactor.refactor_string(original + "\n", str(path)))
        converted = converted.replace("from rtree import Rtree", "from rtree.index import Index as Rtree")
        path.write_text(converted, encoding="utf-8", newline="\n")
    for source, target in HELPERS.items():
        shutil.copyfile(PACKAGE / "Chicago_Biagioni_Compatibility" / source, biagioni_dir / target)
    graph_extract = biagioni_dir / "graph_extract.py"
    graph_source = graph_extract.read_text(encoding="utf-8")
    graph_source = graph_source.replace(
        "from scipy.ndimage import imread\nfrom scipy.misc import imsave\n",
        "import cv2\n",
    )
    graph_source = graph_source.replace(
        "skeleton = imread(skeleton_filename)",
        "skeleton = read_image(skeleton_filename)",
    )
    graph_source = graph_source.replace(
        "import cv2\n",
        "import cv2\n\ndef read_image(filename):\n"
        "    encoded = np.fromfile(str(filename), dtype=np.uint8)\n"
        "    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)\n"
        "    if image is None:\n"
        "        raise FileNotFoundError(filename)\n"
        "    return image\n", 1,
    )
    graph_source = re.sub(r"\bnp\.bool\b", "np.bool_", graph_source)
    graph_source = re.sub(r"\bnp\.int\b", "int", graph_source)
    graph_extract.write_text(graph_source, encoding="utf-8", newline="\n")

    streetmap = biagioni_dir / "streetmap.py"
    streetmap_source = streetmap.read_text(encoding="utf-8").replace(
        "import pyximport; pyximport.install()\n", ""
    )
    streetmap.write_text(streetmap_source, encoding="utf-8", newline="\n")

    viterbi = biagioni_dir / "viterbi.py"
    viterbi_source = viterbi.read_text(encoding="utf-8").replace(
        "(max_prob,max_from) = max(from_probs)",
        "(max_prob,max_from) = max(from_probs, key=lambda item: item[0])",
    )
    viterbi.write_text(viterbi_source, encoding="utf-8", newline="\n")

    refine_topology = biagioni_dir / "refine_topology.py"
    refine_source = refine_topology.read_text(encoding="utf-8").replace(
        "                \n", "\n"
    )
    refine_helper = '''def collapse_would_create_parallel_edge(
    segment_head_node,
    segment_tail_node,
    head_edge_neighbors,
    tail_edge_neighbors,
):
    """Protect StreetMap's simple directed-edge lookup during a collapse."""
    prospective_keys = []
    for endpoint, neighbors in (
        (segment_head_node, head_edge_neighbors),
        (segment_tail_node, tail_edge_neighbors),
    ):
        for neighbor in neighbors:
            if endpoint in neighbor.out_nodes:
                prospective_keys.append(("incoming", neighbor.id))
            if endpoint in neighbor.in_nodes:
                prospective_keys.append(("outgoing", neighbor.id))
    return len(prospective_keys) != len(set(prospective_keys))


'''
    if "def collapse_would_create_parallel_edge(" not in refine_source:
        refine_source = refine_source.replace(
            "import math\n\nclass RefineTopology:",
            "import math\n\n" + refine_helper + "class RefineTopology:",
        )
    refine_guard_anchor = '''                if (segment.tail_edge.in_node in tail_edge_neighbors):
                    tail_edge_neighbors.remove(segment.tail_edge.in_node)

                for neighbor in head_edge_neighbors:
'''
    refine_guard = '''                if (segment.tail_edge.in_node in tail_edge_neighbors):
                    tail_edge_neighbors.remove(segment.tail_edge.in_node)

                if collapse_would_create_parallel_edge(
                    segment_head_node,
                    segment_tail_node,
                    head_edge_neighbors,
                    tail_edge_neighbors,
                ):
                    print("skipping collapse: parallel edge would be created")
                    del self.graphdb.nodes[new_intersection.id]
                    node_id -= 1
                    closed_segments_list.append(segment)
                    continue

                for neighbor in head_edge_neighbors:
'''
    if "                if collapse_would_create_parallel_edge(" not in refine_source:
        if refine_guard_anchor not in refine_source:
            raise RuntimeError("Could not install Biagioni topology safety guard")
        refine_source = refine_source.replace(refine_guard_anchor, refine_guard, 1)
    refine_topology.write_text(refine_source, encoding="utf-8", newline="\n")

    process_matches = biagioni_dir / "process_map_matches.py"
    process_source = process_matches.read_text(encoding="utf-8").replace(
        "                \n", "\n"
    )
    match_order_helper = '''def ordered_matched_trip_files(directory):
    """Use the historic numeric trip order independent of directory order."""
    prefix = "matched_trip_"
    suffix = ".txt"
    files = [
        name for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(suffix)
        and name[len(prefix):-len(suffix)].isdigit()
    ]
    return sorted(files, key=lambda name: int(name[len(prefix):-len(suffix)]))


'''
    process_source = process_source.replace(
        "import math\n\n", "import math\n\n" + match_order_helper, 1
    )
    unordered_matches = 'all_matched_trip_files = [x for x in os.listdir(matched_trips_directory) if x.startswith("matched_trip_") and x.endswith(".txt")]'
    if unordered_matches not in process_source:
        raise RuntimeError("Could not install Biagioni numeric matched-trip ordering")
    process_source = process_source.replace(
        unordered_matches,
        "all_matched_trip_files = ordered_matched_trip_files(matched_trips_directory)",
        1,
    )
    coalesce_helper = '''def coalesce_segment_pair(graphdb, head_segment, tail_segment):
    """Merge once and safely ignore a duplicate scheduled merge."""
    if (
        head_segment.id not in graphdb.segments
        or tail_segment.id not in graphdb.segments
        or head_segment.id == tail_segment.id
    ):
        return False
    head_segment.edges.extend(tail_segment.edges)
    for edge in tail_segment.edges:
        edge.segment = head_segment
    del graphdb.segments[tail_segment.id]
    return True


'''
    if "def coalesce_segment_pair(" not in process_source:
        process_source = process_source.replace(
            "class ProcessMapMatches:",
            coalesce_helper + "class ProcessMapMatches:",
        )
    old_coalesce = '''            for head_segment, tail_segment in merge_segments:
                head_segment.edges.extend(tail_segment.edges)

                for edge in tail_segment.edges:
                    edge.segment = head_segment #self.graphdb.segment_lookup_table[edge.id] = head_segment

                del self.graphdb.segments[tail_segment.id]
'''
    new_coalesce = '''            for head_segment, tail_segment in merge_segments:
                coalesce_segment_pair(self.graphdb, head_segment, tail_segment)
'''
    if "                coalesce_segment_pair(self.graphdb" not in process_source:
        if old_coalesce not in process_source:
            raise RuntimeError("Could not install Biagioni segment coalescing guard")
        process_source = process_source.replace(old_coalesce, new_coalesce, 1)
    process_matches.write_text(process_source, encoding="utf-8", newline="\n")

    # Both stages 05 and 08 consume the helper above; retain the historic terminal
    # segment behavior and SQLite edge/segment insertion order.
    for filename, expression in (
        ("location.py", "os.listdir(trips_path)"),
        ("graphdb_matcher_run.py", "os.listdir(trip_directory)"),
    ):
        path = biagioni_dir / filename
        source = path.read_text(encoding="utf-8")
        if expression not in source:
            raise RuntimeError(f"Missing trip enumeration anchor: {filename}")
        ordered = (
            "sorted((name for name in " + expression
            + " if name.startswith('trip_') and name.endswith('.txt')"
            + " and name[5:-4].isdigit()), key=lambda name: int(name[5:-4]))"
        )
        path.write_text(source.replace(expression, ordered), encoding="utf-8", newline="\n")


def _remove_owned_tree(path: Path, directory: Path) -> None:
    """Delete only a validated method descendant; the dispatcher checked ownership."""
    if directory not in path.resolve().parents:
        raise RunnerError("unsafe_output", f"Not a method descendant: {path}")
    reject_redirects(path)
    for child in path.rglob("*"):
        reject_redirects(child)
        if child.is_file() and not regular_unlinked_file(child):
            raise RunnerError("unsafe_output", f"Linked output: {child}")
    shutil.rmtree(path)


def _validate_artifacts(paths: list[Path]) -> None:
    for path in paths:
        reject_redirects(path)
        if not regular_unlinked_file(path) or path.stat().st_size == 0:
            raise RunnerError("invalid_output", f"Missing or empty Biagioni stage artifact: {path.name}")
        if path.suffix == ".db":
            with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
                for table in ("nodes", "edges", "segments"):
                    if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] <= 0:
                        raise RunnerError("invalid_output", f"Biagioni produced no {table}: {path.name}")


def run(context: RunContext, inputs: PreparedInputs) -> MethodRunResult:
    """Resume validated stages; the first invalid stage invalidates every successor."""
    directory = context.output_root.resolve() / "biagioni"
    owner = {"format": "chicago-reference-method-v1", "method": "biagioni"}
    if read_json_object(directory / ".chicago_reference_method.json") != owner:
        raise RunnerError("unowned_output", "Biagioni must run through the owned method dispatcher")
    trips_dir = inputs.trips_wgs84 / "biagioni"
    trips = sorted(trips_dir.glob("trip_*.txt"), key=lambda p: int(p.stem[5:]))
    if not trips:
        raise RunnerError("missing_input", "Prepared Biagioni trips are missing")
    sources = {f"source/{i}": path for i, path in enumerate(SOURCE_PATHS)}
    sources["adapter"] = Path(__file__)
    source_hash = fingerprint_paths(sources)
    compatibility = directory / "compat/biagoni2012"
    generation = directory / "compatibility_manifest.json"
    previous = read_json_object(generation)
    valid_generation = False
    if previous and previous.get("source_sha256") == source_hash and compatibility.is_dir():
        valid_generation = previous.get("generated_sha256") == fingerprint_paths({"code": compatibility})
    if not valid_generation:
        if compatibility.exists():
            _remove_owned_tree(compatibility, directory)
        _generate_compatibility_copy(compatibility)
    code_hash = fingerprint_paths({"code": compatibility})
    write_json_atomic(generation, {
        "source_sha256": source_hash, "generated_sha256": code_hash,
        "source_files": {p.relative_to(SOURCE_DIR).as_posix(): sha256_file(p)
                         for p in SOURCE_DIR.rglob("*") if p.is_file()},
        "compatibility": COMPATIBILITY,
    })
    dirs = {name: directory / "stages" / name for name in STAGES}
    match_names = ["matched_" + p.name for p in trips]
    specs = [
        ("kde.py", ["-c", "1", "-b", "17", "-p", trips_dir], trips,
         ["kde.png", "bounding_box.txt", "raw_data.png"]),
        ("skeleton.py", [dirs["01_kde"] / "kde.png", "skeleton.png"],
         [dirs["01_kde"] / "kde.png"], ["skeleton.png"]),
        ("graph_extract.py", [dirs["02_skeleton"] / "skeleton.png",
         dirs["01_kde"] / "bounding_box.txt", "skeleton_map.db"],
         [dirs["02_skeleton"] / "skeleton.png", dirs["01_kde"] / "bounding_box.txt"],
         ["skeleton_map.db"]),
        ("graphdb_matcher_run.py", ["-d", dirs["03_graph_extract"] / "skeleton_map.db",
         "-t", trips_dir, "-o", dirs["04_matched_1"]],
         [dirs["03_graph_extract"] / "skeleton_map.db", *trips], match_names),
        ("process_map_matches.py", ["-d", dirs["03_graph_extract"] / "skeleton_map.db",
         "-t", dirs["04_matched_1"], "-o", "skeleton_map_mm1.db"],
         [dirs["03_graph_extract"] / "skeleton_map.db",
          *(dirs["04_matched_1"] / name for name in match_names)],
         ["skeleton_map_mm1.db", "skeleton_map_mm1_traces.txt"]),
        ("refine_topology.py", ["-d", dirs["05_prune_1"] / "skeleton_map_mm1.db",
         "-t", dirs["05_prune_1"] / "skeleton_map_mm1_traces.txt", "-o", "skeleton_map_mm1_tr.db"],
         [dirs["05_prune_1"] / "skeleton_map_mm1.db", dirs["05_prune_1"] / "skeleton_map_mm1_traces.txt"],
         ["skeleton_map_mm1_tr.db"]),
        ("graphdb_matcher_run.py", ["-d", dirs["06_refine"] / "skeleton_map_mm1_tr.db",
         "-t", trips_dir, "-o", dirs["07_matched_2"]],
         [dirs["06_refine"] / "skeleton_map_mm1_tr.db", *trips], match_names),
        ("process_map_matches.py", ["-d", dirs["06_refine"] / "skeleton_map_mm1_tr.db",
         "-t", dirs["07_matched_2"], "-o", "skeleton_map_mm2.db"],
         [dirs["06_refine"] / "skeleton_map_mm1_tr.db",
          *(dirs["07_matched_2"] / name for name in match_names)],
         ["skeleton_map_mm2.db", "skeleton_map_mm2_traces.txt"]),
    ]
    reports = []
    outputs = {"compatibility": generation}
    rerun_following = context.force or not context.resume
    for name, (script, arguments, stage_inputs, required) in zip(STAGES, specs):
        stage = dirs[name]
        signature = {
            "input_sha256": fingerprint_paths({str(i): p for i, p in enumerate(stage_inputs)}),
            "code_sha256": code_hash, "source_sha256": source_hash,
            "parameters": PARAMETERS, "execution_class": EXECUTION_CLASS,
        }
        marker_path = stage / "completion.json"
        marker = read_json_object(marker_path)
        artifacts = [stage / p for p in required]
        valid = False
        if marker and all(marker.get(k) == v for k, v in signature.items()):
            try:
                _validate_artifacts(artifacts)
                stage_files = [p for p in stage.rglob("*") if p.is_file() and p != marker_path]
                actual = {p.relative_to(stage).as_posix(): sha256_file(p) for p in stage_files}
                valid = (actual == marker.get("outputs")
                         and isinstance(marker.get("elapsed_seconds"), (float, int))
                         and marker["elapsed_seconds"] >= 0)
            except (OSError, ValueError, sqlite3.Error, RunnerError):
                valid = False
        if rerun_following or not valid:
            rerun_following = True
            if stage.exists():
                if read_json_object(stage / ".stage_owner.json") != {"stage": name}:
                    raise RunnerError("unowned_output", f"Stage is not owned: {name}")
                _remove_owned_tree(stage, directory)
            stage.mkdir(parents=True)
            write_json_atomic(stage / ".stage_owner.json", {"stage": name})
            command = [sys.executable, "-B", str(compatibility / script), *map(str, arguments)]
            start = time.perf_counter()
            with (stage / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
            _validate_artifacts(artifacts)
            marker = {**signature, "elapsed_seconds": time.perf_counter() - start,
                      "outputs": {p.relative_to(stage).as_posix(): sha256_file(p)
                                  for p in stage.rglob("*") if p.is_file()}}
            write_json_atomic(marker_path, marker)
            status = "completed"
        else:
            status = "reused"
        reports.append({"name": name, "status": status, "elapsed_seconds": marker["elapsed_seconds"],
                        "script": script, "outputs": {p.name: p.relative_to(directory).as_posix()
                                                     for p in artifacts}})
        for artifact in [*artifacts, stage / "run.log", marker_path]:
            outputs[f"{name}/{artifact.name}"] = artifact
    outputs["graph"] = dirs["08_final"] / "skeleton_map_mm2.db"
    stage_manifest = directory / "stage_manifest.json"
    elapsed = sum(r["elapsed_seconds"] for r in reports)
    environment = {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version}
    for package in ("numpy", "scipy", "opencv-python", "rtree"):
        environment[package] = importlib.metadata.version(package)
    write_json_atomic(stage_manifest, {
        "method": "biagioni", "execution_class": EXECUTION_CLASS, "parameters": PARAMETERS,
        "compatibility": COMPATIBILITY, "environment": environment,
        "stages": reports, "runtime_seconds": elapsed,
    })
    return MethodRunResult("biagioni", EXECUTION_CLASS, outputs, elapsed, stage_manifest)
