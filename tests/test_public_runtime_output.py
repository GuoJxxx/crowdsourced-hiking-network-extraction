from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def _dict_string_keys(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
    return keys


def test_stage_reports_do_not_include_explanatory_note_fields():
    for name in ["S3_activity_zone_filter.py", "S5_directional_kde.py", "S6_route_locked_skeleton.py"]:
        keys = _dict_string_keys(CODE / name)
        assert "note" not in keys, f"{name} still contains a public report 'note' field"
        assert "notes" not in keys, f"{name} still contains a public report 'notes' field"


def test_stage_completion_does_not_dump_full_report_json_to_terminal():
    for name in ["S3_activity_zone_filter.py", "S5_directional_kde.py", "S6_route_locked_skeleton.py"]:
        text = (CODE / name).read_text(encoding="utf-8")
        assert "print(json.dumps(report" not in text


def test_public_runtime_messages_are_version_neutral():
    files = [
        CODE / "S6_route_locked_skeleton.py",
        CODE / "S7_network_cleanup.py",
        CODE / "pipeline_modules" / "network_cleanup" / "uturn_topology.py",
        CODE / "pipeline_modules" / "network_cleanup" / "cycle_semantics.py",
        CODE / "pipeline_modules" / "network_cleanup" / "cleaner.py",
        CODE / "pipeline_modules" / "network_cleanup" / "raw_track_topology.py",
        CODE / "tools" / "run_pipeline_configurable.ps1",
    ]
    forbidden = [
        "[polygon-cycle:v0.5]",
        "[connectivity:v0.5]",
        "[cleanup:v0.5]",
        "[final-prune:v0.5]",
        "[final-connectivity:v0.5]",
        "[raw-coverage:v0.5]",
        "[final-cycle:v0.5]",
        "[audit:junction:rollback:v0.5]",
        "[audit:raw-track-gap:rollback:v0.5]",
        "[uturn-shortcut:v0.5]",
        "[semantic-cycle:v0.5]",
        "[final-smooth:v0.5]",
        "[raw-negative-topology:v0.5]",
        "兼容旧版已有结果",
        "旧版 S5 已完整存在",
        "v0.5 包内版本",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8-sig") for p in files)
    for token in forbidden:
        assert token not in combined, f"public runtime text still contains {token!r}"
