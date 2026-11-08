"""Stage the preserved TraceBundle source with deterministic candidate lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from experiments.chicago_six_methods.reference_methods.methods import (
    karagiorgou_native as strict_native,
)


PACKAGE = Path(__file__).resolve().parent
BUILD_HELPER = PACKAGE / "StudyArea_Karagiorgou_MATLAB" / "tracebundle_build_endpoint_grid.m"
QUERY_HELPER = PACKAGE / "StudyArea_Karagiorgou_MATLAB" / "tracebundle_query_endpoint_grid.m"
PUBLIC_EXTRACTION_SHA256 = "46e8895ecac4d9cf2a5eaa7176953c5c3900a59d45153065035e509e8cb50d42"
PUBLIC_TRACEBUNDLE_SHA256 = "533b47b00eaa70221ea03871601e148b6b56b5b795fc24eb78f4fe0ae5db7738"
EXECUTION_CLASS = "MATLAB implementation with indexed endpoint lookup"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_exact(data: bytes, old: bytes, new: bytes, expected: int, label: str) -> bytes:
    actual = data.count(old)
    if actual != expected:
        raise ValueError(
            f"Unexpected public source anchor for {label}: expected {expected}, found {actual}"
        )
    return data.replace(old, new)


def _patch_extraction(original: bytes, cell_size_m: float) -> bytes:
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    picker = strict_native._PICKER.encode("utf-8")
    patched = _replace_exact(
        original,
        picker,
        b"[fnames,path]=tracebundle_select_tracks();",
        1,
        "interactive picker",
    )
    heading = b"% finding cluster type of turn, range, cluster weight"
    index_call = (
        f"tracebundle_endpoint_grid = tracebundle_build_endpoint_grid(slk, {cell_size_m:.1f});"
        .encode("ascii")
        + newline
        + newline
        + heading
    )
    patched = _replace_exact(patched, heading, index_call, 1, "index construction")

    query = (
        b"    tracebundle_candidates = tracebundle_query_endpoint_grid("
        b"tracebundle_endpoint_grid, mb);"
        + newline
        + b"    in1 = inpolygon(slk(tracebundle_candidates,1), "
        b"slk(tracebundle_candidates,2),mb(1,:), mb(2,:));"
    )
    patched = _replace_exact(
        patched,
        b"    in1 = inpolygon(slk(:,1), slk(:,2),mb(1,:), mb(2,:));",
        query,
        2,
        "first endpoint scan",
    )
    patched = _replace_exact(
        patched,
        b"    in2 = inpolygon(slk(:,3), slk(:,4),mb(1,:), mb(2,:));",
        b"    in2 = inpolygon(slk(tracebundle_candidates,3), "
        b"slk(tracebundle_candidates,4),mb(1,:), mb(2,:));",
        2,
        "second endpoint scan",
    )
    patched = _replace_exact(
        patched,
        b"    rs1=find(in1==1);",
        b"    rs1=tracebundle_candidates(find(in1==1));",
        2,
        "first global-index mapping",
    )
    patched = _replace_exact(
        patched,
        b"    rs2=find(in2==1);",
        b"    rs2=tracebundle_candidates(find(in2==1));",
        2,
        "second global-index mapping",
    )
    return patched


def stage_accelerated_source(
    source_dir: Path,
    tracks: Path,
    native: Path,
    *,
    input_manifest: Path | None = None,
    cell_size_m: float = 200.0,
) -> Path:
    """Copy public source and apply only the reviewed deterministic patches."""
    source_dir = Path(source_dir).resolve()
    tracks = Path(tracks).resolve()
    native = Path(native).resolve()
    extraction = source_dir / "source" / "intersection_nodes_extraction.m"
    bundling = source_dir / "source" / "tracebundle.m"
    libraries = source_dir / "libraries"
    for required in (extraction, bundling, libraries, BUILD_HELPER, QUERY_HELPER):
        if not required.exists():
            raise FileNotFoundError(f"TraceBundle dependency is missing: {required}")
    if _sha256(extraction) != PUBLIC_EXTRACTION_SHA256:
        raise ValueError("Unexpected public source hash: intersection_nodes_extraction.m")
    if _sha256(bundling) != PUBLIC_TRACEBUNDLE_SHA256:
        raise ValueError("Unexpected public source hash: tracebundle.m")
    if not isinstance(cell_size_m, (int, float)) or not float(cell_size_m) > 0:
        raise ValueError("cell_size_m must be positive")
    files, track_order = strict_native._track_files(tracks, input_manifest)
    if native.exists():
        raise FileExistsError(f"Native stage already exists: {native}")

    original = extraction.read_bytes()
    patched = _patch_extraction(original, float(cell_size_m))
    native.mkdir(parents=True)
    (native / extraction.name).write_bytes(patched)
    shutil.copy2(bundling, native / bundling.name)
    shutil.copytree(libraries, native / "libraries")
    shutil.copy2(BUILD_HELPER, native / BUILD_HELPER.name)
    shutil.copy2(QUERY_HELPER, native / QUERY_HELPER.name)

    (native / "track_manifest.txt").write_text(
        "\n".join(path.name for path in files) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (native / "track_directory.txt").write_text(
        str(tracks) + "\n", encoding="utf-8", newline="\n"
    )
    (native / "tracebundle_select_tracks.m").write_text(
        "function [fnames,path] = tracebundle_select_tracks()\n"
        "root = fileparts(mfilename('fullpath'));\n"
        "fid = fopen(fullfile(root, 'track_directory.txt'), 'r');\n"
        "if fid < 0, error('Missing track directory'); end\n"
        "path = fgetl(fid); fclose(fid);\n"
        "if ~ischar(path), error('Empty track directory'); end\n"
        "path = [path filesep];\n"
        "fid = fopen(fullfile(root, 'track_manifest.txt'), 'r');\n"
        "if fid < 0, error('Missing track manifest'); end\n"
        "fnames = {};\n"
        "while ~feof(fid)\n"
        "    name = fgetl(fid);\n"
        "    if ischar(name) && ~isempty(name), fnames{end+1} = name; end\n"
        "end\n"
        "fclose(fid);\n"
        "if isempty(fnames), error('Empty track manifest'); end\n"
        "end\n",
        encoding="utf-8",
        newline="\n",
    )
    source_hashes = {
        path.relative_to(source_dir).as_posix(): _sha256(path)
        for path in sorted(source_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "execution_class": EXECUTION_CLASS,
        "parameters": {"cell_size_m": float(cell_size_m)},
        "public_source_sha256": source_hashes,
        "patched_source_sha256": {
            extraction.name: _sha256(native / extraction.name),
            bundling.name: _sha256(native / bundling.name),
        },
        "helper_sha256": {
            BUILD_HELPER.name: _sha256(BUILD_HELPER),
            QUERY_HELPER.name: _sha256(QUERY_HELPER),
        },
        "track_sha256": {path.name: _sha256(path) for path in files},
        "track_order": track_order,
        "expected_intermediates": list(strict_native.INTERMEDIATES),
        "allowed_source_changes": [
            "replace interactive file picker with deterministic manifest",
            "build deterministic endpoint grid after slk construction",
            "replace two full endpoint scans with conservative indexed candidates",
        ],
    }
    (native / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return native
