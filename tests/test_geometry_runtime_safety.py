from __future__ import annotations

import sys
import warnings
from pathlib import Path

from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def test_safe_intersection_rejects_nonfinite_geometry_without_emitting_runtimewarning():
    from pipeline_modules.network_cleanup.topology_utils import (
        clear_geometry_numeric_anomalies,
        geometry_numeric_anomalies,
        safe_intersection,
    )

    # Shapely 2.x emits `RuntimeWarning: invalid value encountered in intersection`
    # for this geometry. S7 must contain that warning locally, reject the candidate,
    # and record a diagnostic instead of leaking the warning to PowerShell stderr.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = LineString([(0.0, 0.0), (float("nan"), 1.0)])
    other = LineString([(0.0, 1.0), (1.0, 0.0)])

    clear_geometry_numeric_anomalies()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inter = safe_intersection(bad, other, operation="unit_nonfinite")

    assert inter is None
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    rows = geometry_numeric_anomalies()
    assert len(rows) == 1
    assert rows[0]["operation"] == "unit_nonfinite"
    assert rows[0]["action"] == "reject_candidate"
    assert "non_finite" in rows[0]["reason"]


def test_safe_intersection_returns_real_intersection_for_normal_lines():
    from pipeline_modules.network_cleanup.topology_utils import (
        clear_geometry_numeric_anomalies,
        geometry_numeric_anomalies,
        safe_intersection,
    )

    a = LineString([(0.0, 0.0), (2.0, 2.0)])
    b = LineString([(0.0, 2.0), (2.0, 0.0)])
    clear_geometry_numeric_anomalies()
    inter = safe_intersection(a, b, operation="unit_normal")

    assert inter is not None
    assert inter.geom_type == "Point"
    assert abs(inter.x - 1.0) < 1e-9
    assert abs(inter.y - 1.0) < 1e-9
    assert geometry_numeric_anomalies() == []


def test_all_s7_candidate_intersections_use_safe_wrapper():
    targets = [
        ROOT / "code" / "pipeline_modules" / "network_cleanup" / "junction_cleanup.py",
        ROOT / "code" / "pipeline_modules" / "network_cleanup" / "s7_force_refine.py",
        ROOT / "code" / "pipeline_modules" / "network_cleanup" / "raw_track_topology.py",
        ROOT / "code" / "pipeline_modules" / "network_cleanup" / "connectivity_repair.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert ".intersection(" not in text, f"unsafe direct intersection remains in {path.name}"
        assert "safe_intersection" in text, f"safe wrapper not used in {path.name}"


def test_windows_launcher_does_not_treat_native_python_stderr_warning_as_terminating_error():
    path = ROOT / "code" / "tools" / "run_pipeline_configurable.ps1"
    text = path.read_text(encoding="utf-8-sig")

    # Windows PowerShell 5.1 can turn native stderr merged by 2>&1 into an ErrorRecord.
    # Run-Python must temporarily use Continue while the native process runs, then
    # restore the script's Stop policy and decide success solely from LASTEXITCODE.
    assert "$nativeErrorActionPreference=$ErrorActionPreference" in text
    assert "$ErrorActionPreference='Continue'" in text
    assert "$exitCode=$LASTEXITCODE" in text
    assert "$ErrorActionPreference=$nativeErrorActionPreference" in text
