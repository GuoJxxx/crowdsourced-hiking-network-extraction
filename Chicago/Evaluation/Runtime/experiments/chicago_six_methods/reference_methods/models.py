"""Shared input and run contracts for Chicago reference methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    inputs_root: Path
    output_root: Path
    profile: Literal["smoke", "paper"]
    force: bool
    resume: bool

    def __post_init__(self) -> None:
        if self.profile not in ("smoke", "paper"):
            raise ValueError(f"Unsupported profile: {self.profile}")


@dataclass(frozen=True)
class PreparedInputs:
    tracks_projected: Path
    trips_wgs84: Path
    raw_union: Path
    truth_vertices: Path
    truth_edges: Path
    manifest: Path
