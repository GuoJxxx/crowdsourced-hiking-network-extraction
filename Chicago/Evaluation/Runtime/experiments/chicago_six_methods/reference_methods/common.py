"""Shared result, ownership, and completion-record primitives for Chicago runs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OWNER_FILE = ".chicago_reference_run.json"
OWNER_FORMAT = "chicago-reference-run-v1"


@dataclass(frozen=True)
class MethodRunResult:
    method: str
    execution_class: str
    native_outputs: dict[str, Path]
    runtime_seconds: float
    stage_manifest: Path


class RunnerError(RuntimeError):
    """A user-correctable runner failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, hint: str = "", exit_code: int = 2):
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.exit_code = exit_code

    def document(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "hint": self.hint,
                          "exit_code": self.exit_code}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_paths(paths: dict[str, Path], *, allowed_root: Path | None = None) -> str:
    """Hash regular files after checking every path and ancestor for redirects."""
    boundary = Path(os.path.abspath(allowed_root)) if allowed_root is not None else None
    if boundary is not None:
        reject_redirects(boundary)
    checked: list[tuple[str, Path]] = []

    def validate(path: Path) -> None:
        reject_redirects(path)
        lexical = Path(os.path.abspath(path))
        if path.resolve() != lexical:
            raise RunnerError("unsafe_input", f"Input path contains a redirect: {path}")
        if boundary is not None and lexical != boundary and boundary not in lexical.parents:
            raise RunnerError("unsafe_input", f"Input path is outside the owned inputs: {path}")

    def files_below(path: Path) -> list[Path]:
        validate(path)
        if path.is_dir():
            files = []
            for child in sorted(path.iterdir()):
                files.extend(files_below(child))
            return files
        if not regular_unlinked_file(path):
            raise RunnerError("unsafe_input", f"Redirected or nonregular input: {path}")
        return [path]

    # Validate the whole declared tree before opening any file for hashing.
    for label, path in sorted(paths.items()):
        validate(path)
        if not path.exists():
            raise RunnerError("missing_input", f"Missing input: {label}", hint="Run prepare first")
        files = files_below(path)
        if not files:
            raise RunnerError("missing_input", f"Empty input directory: {label}", hint="Run prepare again")
        checked.extend((f"{label}/{member.relative_to(path).as_posix() if path.is_dir() else path.name}", member)
                       for member in files)

    digest = hashlib.sha256()
    for name, member in checked:
        digest.update(f"{name}\0".encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(member)))
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def regular_unlinked_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and not (getattr(info, "st_file_attributes", 0)
                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def reject_redirects(path: Path) -> None:
    """Reject symlinks/reparse points in any existing path component."""
    absolute = Path(os.path.abspath(path))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise RunnerError("unsafe_path", f"Output path contains a redirect: {part}")


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    candidate = path.with_name(path.name + ".tmp")
    candidate.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    candidate.replace(path)


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not regular_unlinked_file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None
