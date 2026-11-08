"""Construction and evaluation of Chicago methods."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.chicago_six_methods.reference_methods.common import (
    OWNER_FILE, OWNER_FORMAT, MethodRunResult, RunnerError, fingerprint_paths,
    json_hash, read_json_object, regular_unlinked_file, reject_redirects,
    sha256_file, write_json_atomic,
)
from experiments.chicago_six_methods.reference_methods.models import PreparedInputs, RunContext


PACKAGE = Path(__file__).resolve().parent
from workflow.common import ROOT as REPOSITORY, intermediate, path, dataset_results, source
DEFAULT_OUTPUT = intermediate('chicago')/'references'
DEFAULT_TRACKS = path('chicago_tracks')
DEFAULT_MAP = path('chicago_map')
METHODS = ("ahmed", "biagioni", "davies", "edelkamp", "karagiorgou")
EXECUTION_CLASSES = {
    "ahmed": "official Java",
    "biagioni": "Python 3/current-library compatibility",
    "davies": "Python 3/OpenCV 4 compatibility",
    "edelkamp": "Python 3/current-library compatibility",
    "karagiorgou": "official MATLAB source",
}
PYTHON_DEPENDENCIES = {
    "ahmed": (),
    "biagioni": ("numpy", "scipy", "cv2", "skimage", "rtree"),
    "davies": ("numpy", "scipy", "cv2", "skimage"),
    "edelkamp": ("numpy", "scipy", "rtree"),
    "karagiorgou": ("numpy", "scipy", "shapely"),
}
METHOD_OWNER = ".chicago_reference_method.json"
INPUT_OWNER = ".chicago_reference_inputs.json"
PAPER_SUPPORT_OWNER = ".chicago_paper_support.json"


def resolve_cli_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY / path).resolve()


def safe_output_root(value: str | Path, repo_root: Path = REPOSITORY) -> Path:
    """Return a canonical output root after excluding broad and source targets."""
    repo = repo_root.resolve()
    source = Path(value)
    lexical = Path(os.path.abspath(source if source.is_absolute() else repo / source))
    reject_redirects(lexical)
    root = lexical.resolve()
    home = Path.home().resolve()
    forbidden = (Path(root.anchor), home, repo, repo / "outputs")
    if root in forbidden or root in repo.parents or root in home.parents:
        raise RunnerError("unsafe_output", f"Unsafe output root: {root}")
    if repo in root.parents and not any(dataset_results(group).resolve() in root.parents for group in ('chicago','mainling')):
        raise RunnerError("unsafe_output", f"Unsafe output root: repository source tree {root}")
    return root


def _owner_valid(root: Path) -> bool:
    return read_json_object(root / OWNER_FILE) == {"format": OWNER_FORMAT}


def ensure_owned_root(value: str | Path, *, force: bool = False) -> Path:
    root = safe_output_root(value)
    if root.exists():
        if not root.is_dir():
            raise RunnerError("unsafe_output", f"Output root is not a directory: {root}")
        if _owner_valid(root):
            return root
        if any(root.iterdir()):
            raise RunnerError("unowned_output", f"Output root is not owned by this runner: {root}",
                              hint="Choose an empty output directory")
    else:
        root.mkdir(parents=True)
    write_json_atomic(root / OWNER_FILE, {"format": OWNER_FORMAT})
    return root


def _validate_descendant(root: Path, path: Path, *, must_exist: bool = True) -> Path:
    lexical = Path(os.path.abspath(path))
    if root not in lexical.parents:
        raise RunnerError("unsafe_output", f"Declared output is outside the method directory: {path}")
    reject_redirects(lexical)
    if lexical.resolve() != lexical:
        raise RunnerError("unsafe_output", f"Declared output is redirected: {path}")
    if must_exist and (not regular_unlinked_file(lexical) or lexical.stat().st_size == 0):
        raise RunnerError("invalid_output", f"Declared output is missing, empty, or linked: {path}")
    return lexical


def _prepared_paths(inputs: PreparedInputs) -> dict[str, Path]:
    return {name: getattr(inputs, name) for name in (
        "tracks_projected", "trips_wgs84", "raw_union", "truth_vertices", "truth_edges", "manifest")}


def _construction_paths(inputs: PreparedInputs, method: str) -> dict[str, Path]:
    """Isolate TraceBundle inputs while preserving other methods' contracts."""
    if method == "karagiorgou":
        return {"tracks_projected": inputs.tracks_projected, "manifest": inputs.manifest}
    return _prepared_paths(inputs)


def load_prepared_inputs(output_root: Path, *,
                         construction_methods: Sequence[str] | None = None) -> PreparedInputs:
    root = output_root / "inputs"
    inputs = PreparedInputs(
        root / "tracks_projected", root / "trips_wgs84", root / "raw_trajectory_union.tif",
        root / "truth" / "chicago_vertices_osm.txt", root / "truth" / "chicago_edges_osm.txt",
        root / "input_manifest.json",
    )
    if construction_methods is None:
        paths = _prepared_paths(inputs)
    else:
        paths = {name: path for method in construction_methods
                 for name, path in _construction_paths(inputs, method).items()}
    fingerprint_paths(paths, allowed_root=root)
    return inputs


def preflight_method(method: str, *, engine: str = "native") -> None:
    method = method.lower()
    if method not in METHODS:
        raise RunnerError("unknown_method", f"Unknown method: {method}")
    if sys.version_info < (3, 11):
        raise RunnerError("missing_dependency", "Python 3.11 or newer is required")
    if method == "karagiorgou" and engine == "native":
        executable = shutil.which("matlab")
        if executable is None:
            raise RunnerError("missing_dependency", "MATLAB executable is required for the official Karagiorgou source",
                              hint="Install MATLAB with the Statistics and Machine Learning Toolbox")
        probe = (
            "assert(strncmp(version('-release'),'2021',4), 'MATLAB R2021a or R2021b is required'); "
            "assert(exist('pdist','file') ~= 0, 'Missing Statistics function: pdist'); "
            "assert(exist('squareform','file') ~= 0, 'Missing Statistics function: squareform'); "
            "squareform(pdist([0 0; 1 1]));"
        )
        try:
            result = subprocess.run([executable, "-batch", probe], cwd=PACKAGE,
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunnerError("missing_dependency", f"MATLAB Statistics dependency probe failed: {exc}",
                              hint="MATLAB with usable pdist/squareform is required") from exc
        if result.returncode:
            diagnostic = (result.stderr or result.stdout).strip()[-2000:]
            raise RunnerError("missing_dependency", f"MATLAB Statistics dependency probe failed: {diagnostic}",
                              hint="Install and license the Statistics and Machine Learning Toolbox")
        return
    if method == "ahmed":
        missing = [name for name in ("java", "javac") if shutil.which(name) is None]
        if missing:
            raise RunnerError("missing_dependency", "Ahmed requires Java 17 or newer and Javac: " + ", ".join(missing),
                              hint="Install JDK 17 or newer and put java and javac on PATH")
        for tool in ("java", "javac"):
            try:
                result = subprocess.run([tool, "-version"], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RunnerError("missing_dependency", f"Ahmed could not check {tool} version: {exc}",
                                  hint="Install JDK 17 or newer and put java and javac on PATH") from exc
            lines = (result.stdout + result.stderr).strip().splitlines()
            pattern = (r"(?:openjdk|java)(?: version)?\s+\"?(\d+)(?=[.\s\"+-]|$)" if tool == "java"
                       else r"javac\s+(\d+)(?=[.\s\"+-]|$)")
            version = next((line.strip() for line in lines
                            if re.match(pattern, line.strip(), flags=re.IGNORECASE)), "")
            match = re.match(pattern, version, flags=re.IGNORECASE)
            if result.returncode or match is None or int(match.group(1)) < 17:
                raise RunnerError("missing_dependency",
                                  f"Ahmed requires Java and Javac 17 or newer; unsupported {tool} version: "
                                  f"{version or 'unrecognized'}",
                                  hint="Install JDK 17 or newer and put java and javac on PATH")
    missing_python = [name for name in PYTHON_DEPENDENCIES[method]
                      if importlib.util.find_spec(name) is None]
    if missing_python:
        raise RunnerError("missing_dependency", f"{method} requires Python packages: {', '.join(missing_python)}",
                          hint=f"Install -r {PACKAGE / 'requirements.txt'}")


def _code_fingerprint(method: str, module: object) -> str:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise RunnerError("invalid_method", f"{method} has no source file")
    files = {"method": Path(module_path), "runner": Path(__file__),
             "common": PACKAGE / "common.py", "models": PACKAGE / "models.py",
             "prepare": PACKAGE / "prepare_inputs.py"}
    # Every method must declare the complete set of helper source files or
    # directories it executes. Relative declarations are package-relative;
    # a directory declaration tracks membership as well as file contents.
    sources = getattr(module, "SOURCE_PATHS", None)
    if not isinstance(sources, (tuple, list)):
        raise RunnerError("invalid_method", f"{method} must declare SOURCE_PATHS")
    for index, value in enumerate(sources):
        if not isinstance(value, (str, Path)):
            raise RunnerError("invalid_method", f"{method} SOURCE_PATHS[{index}] is not a path")
        path = Path(value)
        files[f"helper/{index}"] = path if path.is_absolute() else PACKAGE / path
    original = PACKAGE.parent / "vendor"
    if original.exists():
        files.update({f"original/{path.relative_to(original).as_posix()}": path
                      for path in original.rglob("*")
                      if path.is_file() and method in path.as_posix().lower()})
    return fingerprint_paths(files)


def _method_signature(context: RunContext, inputs: PreparedInputs, method: str, module: object) -> dict[str, str]:
    owned_inputs = safe_output_root(context.output_root, context.repo_root) / "inputs"
    if Path(os.path.abspath(context.inputs_root)) != owned_inputs:
        raise RunnerError("unsafe_input", f"Prepared inputs are outside the owned output root: {context.inputs_root}")
    parameters = getattr(module, "PARAMETERS", {})
    if not isinstance(parameters, dict):
        raise RunnerError("invalid_method", f"{method} PARAMETERS must be a dictionary")
    return {
        "input_sha256": fingerprint_paths(_construction_paths(inputs, method), allowed_root=owned_inputs),
        "code_sha256": _code_fingerprint(method, module),
        "parameters_sha256": json_hash(parameters),
        "execution_class": getattr(module, "EXECUTION_CLASS", EXECUTION_CLASSES[method]),
        "profile": context.profile,
    }


def _completed_result(root: Path, method: str, signature: dict[str, str]) -> MethodRunResult | None:
    directory = root / method
    document = read_json_object(directory / "completion.json")
    if document is None or any(document.get(key) != value for key, value in signature.items()):
        return None
    if document.get("method") != method or not isinstance(document.get("outputs"), dict):
        return None
    native = document.get("native_outputs")
    stage_name = document.get("stage_manifest")
    runtime = document.get("runtime_seconds")
    if not isinstance(native, dict) or not native or not isinstance(stage_name, str):
        return None
    if not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
        return None
    try:
        outputs = {}
        for name, value in native.items():
            if not isinstance(name, str) or not isinstance(value, str):
                return None
            path = _validate_descendant(directory, root / value)
            outputs[name] = path
        stage = _validate_descendant(directory, root / stage_name)
        if not read_json_object(stage):
            return None
        actual = {path.relative_to(root).as_posix(): sha256_file(path)
                  for path in (*outputs.values(), stage)}
        if actual != document["outputs"]:
            return None
    except (OSError, RunnerError, ValueError):
        return None
    return MethodRunResult(method, signature["execution_class"], outputs, float(runtime), stage)


def _remove_owned_directory(directory: Path, owner: dict[str, str], marker_name: str) -> None:
    if read_json_object(directory / marker_name) != owner:
        raise RunnerError("unowned_output", f"Existing output is not owned by this runner: {directory}")
    reject_redirects(directory)
    for path in directory.rglob("*"):
        reject_redirects(path)
    shutil.rmtree(directory)


def run_method(context: RunContext, inputs: PreparedInputs, method: str,
               *, engine: str = "native") -> MethodRunResult:
    method = method.lower()
    if method == "karagiorgou":
        preflight_method(method, engine=engine)
    else:
        preflight_method(method)
    root = safe_output_root(context.output_root, context.repo_root)
    if not _owner_valid(root):
        raise RunnerError("unowned_output", f"Run output root is not owned: {root}", hint="Run prepare first")
    try:
        module_name = ("karagiorgou_native" if method == "karagiorgou" and engine == "native"
                       else method)
        module = importlib.import_module(f"{__package__ or 'experiments.chicago_six_methods.reference_methods'}.methods.{module_name}")
    except ModuleNotFoundError as exc:
        raise RunnerError("method_unavailable", f"Method implementation is unavailable: {method}") from exc
    signature = _method_signature(context, inputs, method, module)
    directory = root / method
    stage_resume = getattr(module, "SUPPORTS_STAGE_RESUME", False) is True
    preserve_stages = False
    if directory.exists():
        if context.resume and not context.force and not stage_resume:
            completed = _completed_result(root, method, signature)
            if completed is not None:
                return completed
        if not (context.force or context.resume):
            raise RunnerError("output_exists", f"Method output already exists: {directory}",
                              hint="Use --resume or --force")
        if stage_resume and context.resume and not context.force:
            if read_json_object(directory / METHOD_OWNER) != {"format": "chicago-reference-method-v1", "method": method}:
                raise RunnerError("unowned_output", f"Existing output is not owned by this runner: {directory}")
            reject_redirects(directory)
            for path in directory.rglob("*"):
                reject_redirects(path)
                if path.is_file() and not regular_unlinked_file(path):
                    raise RunnerError("unsafe_output", f"Linked method output: {path}")
            preserve_stages = read_json_object(directory / "run_signature.json") == signature
        if not preserve_stages:
            _remove_owned_directory(directory, {"format": "chicago-reference-method-v1", "method": method}, METHOD_OWNER)
    directory.mkdir(exist_ok=preserve_stages)
    write_json_atomic(directory / METHOD_OWNER,
                      {"format": "chicago-reference-method-v1", "method": method})
    if stage_resume:
        (directory / "completion.json").unlink(missing_ok=True)
        write_json_atomic(directory / "run_signature.json", signature)
    result = module.run(context, inputs)
    if not isinstance(result, MethodRunResult) or result.method.lower() != method:
        raise RunnerError("invalid_result", f"{method} returned an invalid MethodRunResult")
    if result.execution_class != signature["execution_class"]:
        raise RunnerError("invalid_result", f"{method} returned the wrong execution class")
    if not math.isfinite(result.runtime_seconds) or result.runtime_seconds < 0:
        raise RunnerError("invalid_result", f"{method} returned an invalid runtime")
    if not result.native_outputs:
        raise RunnerError("invalid_result", f"{method} declared no native outputs")
    native = {key: _validate_descendant(directory, Path(value))
              for key, value in result.native_outputs.items()}
    stage = _validate_descendant(directory, Path(result.stage_manifest))
    if not read_json_object(stage):
        raise RunnerError("invalid_output", f"Invalid stage manifest: {stage}")
    outputs = {path.relative_to(root).as_posix(): sha256_file(path)
               for path in (*native.values(), stage)}
    document = {
        "method": method, **signature,
        "native_outputs": {key: path.relative_to(root).as_posix() for key, path in native.items()},
        "stage_manifest": stage.relative_to(root).as_posix(),
        "runtime_seconds": result.runtime_seconds,
        "outputs": outputs,
    }
    write_json_atomic(directory / "completion.json", document)
    return result


def prepare(context: RunContext, tracks_archive: Path, map_archive: Path) -> PreparedInputs:
    try:
        from experiments.chicago_six_methods.reference_methods.prepare_inputs import prepare_chicago_inputs
    except ImportError as exc:
        raise RunnerError("missing_dependency", f"Input preparation dependency is unavailable: {exc}",
                          hint=f"Install -r {PACKAGE / 'requirements.txt'}") from exc
    root = ensure_owned_root(context.output_root, force=context.force)
    tracks = resolve_cli_path(tracks_archive)
    map_path = resolve_cli_path(map_archive)
    for role, path in (("tracks", tracks), ("map", map_path)):
        if not regular_unlinked_file(path):
            raise RunnerError("missing_input", f"Missing or redirected {role} archive: {path}")
    signature = {"profile": context.profile, "tracks_sha256": sha256_file(tracks),
                 "map_sha256": sha256_file(map_path), "code_sha256": sha256_file(PACKAGE / "prepare_inputs.py")}
    input_dir = root / "inputs"
    if input_dir.exists():
        if context.resume and not context.force:
            marker = read_json_object(input_dir / INPUT_OWNER)
            if marker and all(marker.get(key) == value for key, value in signature.items()):
                try:
                    existing = load_prepared_inputs(root)
                    if marker.get("outputs_sha256") == fingerprint_paths(_prepared_paths(existing), allowed_root=input_dir):
                        return existing
                except RunnerError:
                    pass
        if not (context.force or context.resume):
            raise RunnerError("output_exists", f"Prepared inputs already exist: {input_dir}",
                              hint="Use --resume or --force")
        _remove_owned_directory(input_dir, {"format": "chicago-reference-inputs-v1"}, INPUT_OWNER)
    inputs = prepare_chicago_inputs(tracks, map_path, root, context.profile)
    write_json_atomic(input_dir / INPUT_OWNER,
                      {"format": "chicago-reference-inputs-v1", **signature,
                       "outputs_sha256": fingerprint_paths(_prepared_paths(inputs), allowed_root=input_dir)})
    return inputs


def prepare_paper_support(context: RunContext, inputs: PreparedInputs) -> Path:
    """Build the paper-only S2--S5 evaluation raster once per prepared input set."""
    if context.profile != "paper":
        raise RunnerError("invalid_profile", "Paper support is only defined for the Paper profile")
    from experiments.chicago_six_methods.reference_methods.paper_support import (
        build_paper_support, load_projected_trips,
    )

    root = safe_output_root(context.output_root, context.repo_root)
    support_root = root / "support"
    raster = support_root / "S5_KDE" / "raw_trajectory_union.tif"
    source = REPOSITORY/'Chicago/Evaluation/Chicago_Assessment_Source'
    signature = {
        "inputs_sha256": fingerprint_paths({"tracks": inputs.tracks_projected},
                                            allowed_root=inputs.manifest.parent),
        "code_sha256": fingerprint_paths({
            "support_builder": PACKAGE / "paper_support.py",
            "configuration": source / "configs/chicago_v0_5_strict_mainroad.json",
            "schema": source / "configs/parameter_schema.json",
            "s3": source / "code/S3_activity_zone_filter.py",
            "s4": source / "code/S4_uturn_consensus_gravity.py",
            "s5": source / "code/S5_directional_kde.py",
        }),
    }
    owner = {"format": "chicago-paper-support-v1"}
    if support_root.exists():
        marker = read_json_object(support_root / PAPER_SUPPORT_OWNER)
        if marker is None or marker.get("format") != owner["format"]:
            raise RunnerError("unowned_output", f"Paper support output is not owned: {support_root}")
        if (not context.force and marker.get("signature") == signature
                and regular_unlinked_file(raster)
                and marker.get("raster_sha256") == sha256_file(raster)):
            return raster
        if not (context.force or context.resume):
            raise RunnerError("output_exists", f"Paper support already exists: {support_root}")
        reject_redirects(support_root)
        for path in support_root.rglob("*"):
            reject_redirects(path)
        shutil.rmtree(support_root)
    support_root.mkdir()
    write_json_atomic(support_root / PAPER_SUPPORT_OWNER, owner)
    trips = load_projected_trips(inputs.manifest)
    raster, counts = build_paper_support(trips, support_root)
    if counts != {"points": 118_307, "parts": 1_021}:
        raise RunnerError("invalid_output", f"Paper S2 counts differ: {counts}")
    write_json_atomic(support_root / PAPER_SUPPORT_OWNER, {
        **owner, "signature": signature, "s2": counts,
        "raster_sha256": sha256_file(raster),
    })
    return raster


def _load_postprocessor():
    """Import geospatial postprocessing only for commands that require it."""
    try:
        from experiments.chicago_six_methods.reference_methods import normalize_and_evaluate
    except ImportError as exc:
        raise RunnerError(
            "missing_dependency",
            f"Normalization/evaluation dependency is unavailable: {exc}",
            hint=f"Install -r {PACKAGE / 'requirements.txt'}",
        ) from exc
    return normalize_and_evaluate


def _require_owned_run(root: Path) -> None:
    if not _owner_valid(root):
        raise RunnerError("unowned_output", f"Run output root is not owned: {root}", hint="Run prepare first")




class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RunnerError("invalid_argument", message)


def _require_native_karagiorgou(root: Path) -> None:
    completion = read_json_object(root / "karagiorgou/completion.json")
    if completion is None or completion.get("execution_class") != EXECUTION_CLASSES["karagiorgou"]:
        raise RunnerError("non_authoritative_output",
                          "Karagiorgou normalization and evaluation require a completed native MATLAB run")


def _validate_engine(command: str, selected: Sequence[str], engine: str, profile: str) -> None:
    if engine == "compatibility" and (command != "run" or tuple(selected) != ("karagiorgou",)):
        raise RunnerError("invalid_engine", "The incomplete Python compatibility engine is only available for the isolated run command")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="Run Chicago reference methods")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "support", "run", "normalize", "evaluate", "all"):
        item = commands.add_parser(name)
        item.add_argument("--method", choices=(*[name.title() for name in METHODS], "All"), default="All")
        item.add_argument("--profile", type=str.lower, choices=("smoke", "paper"), default="paper")
        item.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
        item.add_argument("--force", action="store_true")
        item.add_argument("--resume", action="store_true")
        item.add_argument("--karagiorgou-engine", choices=("native",), default="native")
        if name in ("prepare", "all"):
            item.add_argument("--tracks-archive", type=Path, default=DEFAULT_TRACKS)
            item.add_argument("--map-archive", type=Path, default=DEFAULT_MAP)
    return parser


def _run_selected_methods(context, inputs, selected, engine):
    for method in selected:
        print(f"[CONSTRUCTION] {method.title()}: reconstructing from prepared trajectories", flush=True)
        if method == "karagiorgou":
            run_method(context, inputs, method, engine=engine)
        else:
            run_method(context, inputs, method)
        print(f"[CONSTRUCTION] {method.title()}: network reconstruction complete", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        root = safe_output_root(args.output_root)
        context = RunContext(REPOSITORY, root / "inputs", root, args.profile, args.force, args.resume)
        selected = METHODS if args.method == "All" else (args.method.lower(),)
        _validate_engine(args.command, selected, args.karagiorgou_engine, args.profile)
        if args.command == "prepare":
            prepare(context, args.tracks_archive, args.map_archive)
        elif args.command == "support":
            _require_owned_run(root)
            prepare_paper_support(context, load_prepared_inputs(root))
        elif args.command == "run":
            _require_owned_run(root)
            inputs = load_prepared_inputs(root, construction_methods=selected)
            _run_selected_methods(context, inputs, selected, args.karagiorgou_engine)
        elif args.command == "normalize":
            _require_owned_run(root)
            if "karagiorgou" in selected:
                _require_native_karagiorgou(root)
            processor = _load_postprocessor()
            processor.normalize_methods(root, selected, force=args.force, resume=args.resume)
        elif args.command == "evaluate":
            _require_owned_run(root)
            if "karagiorgou" in selected:
                _require_native_karagiorgou(root)
            processor = _load_postprocessor()
            if context.profile == "paper":
                print("[ASSESSMENT INPUTS] Preparing shared trajectory coverage", flush=True)
                prepare_paper_support(context, load_prepared_inputs(root))
            print(f"[ASSESSMENT] {', '.join(method.title() for method in selected)}: calculating accuracy measures", flush=True)
            processor.evaluate_methods(root, selected, force=args.force, resume=args.resume)
        elif args.command == "all":
            for method in selected:
                if method == "karagiorgou":
                    preflight_method(method, engine=args.karagiorgou_engine)
                else:
                    preflight_method(method)
            inputs = prepare(context, args.tracks_archive, args.map_archive)
            _run_selected_methods(context, inputs, selected, args.karagiorgou_engine)
            processor = _load_postprocessor()
            print(f"[NETWORK EXPORT] {', '.join(method.title() for method in selected)}: writing normalized networks", flush=True)
            processor.normalize_methods(root, selected, force=args.force, resume=args.resume)
            if context.profile == "paper":
                print("[ASSESSMENT INPUTS] Preparing shared trajectory coverage", flush=True)
                prepare_paper_support(context, inputs)
            print(f"[ASSESSMENT] {', '.join(method.title() for method in selected)}: calculating accuracy measures", flush=True)
            processor.evaluate_methods(root, selected, force=args.force, resume=args.resume)
            if set(selected) == set(METHODS):
                processor.write_reference_comparison(root, selected)
        else:
            raise RunnerError("invalid_argument", f"Unsupported command: {args.command}")
        return 0
    except subprocess.CalledProcessError as exc:
        error = RunnerError("child_failed", f"Child process failed: {exc.cmd}",
                            exit_code=exc.returncode if exc.returncode > 0 else 1)
    except RunnerError as exc:
        error = exc
    except (OSError, ValueError) as exc:
        error = RunnerError("runner_failed", str(exc))
    except Exception as exc:
        error = RunnerError("method_failed", f"{type(exc).__name__}: {exc}")
    print(json.dumps(error.document(), sort_keys=True), file=sys.stderr)
    return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
